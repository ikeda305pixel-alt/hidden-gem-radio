"""Server-side native mpv audio playback controller over UNIX domain socket JSON IPC.

Designed for Termux (Android / Python 3.14 / ARM64) so continuous background playback
runs natively inside Termux without Android Brave's background autoplay restrictions.
"""

from __future__ import annotations

import atexit
import json
import logging
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from src.adapters.audio_stream import AudioStreamResolver

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_MPV_ERRORS = 5


class MpvLocalPlayer:
    """Controls a background `mpv --idle=yes` daemon via UNIX socket JSON IPC."""

    def __init__(
        self,
        *,
        audio_resolver: AudioStreamResolver | None = None,
        next_tracks_provider: Callable[[str, int, set[str]], list[dict[str, Any]]] | None = None,
        feedback_recorder: Callable[[str, dict[str, Any], str, float | None, float | None], Any] | None = None,
        socket_path: Path | str | None = None,
        api_base_url: str = "http://127.0.0.1:8000",
        ipc_connector: Callable[[Path], socket.socket] | None = None,
        spawn_process: bool = True,
    ) -> None:
        self.audio_resolver = audio_resolver or AudioStreamResolver()
        self._next_tracks_provider = next_tracks_provider
        self._feedback_recorder = feedback_recorder
        self.api_base_url = api_base_url.rstrip("/")
        self._custom_ipc_connector = ipc_connector
        self._spawn_process = spawn_process

        if socket_path is not None:
            self.socket_path = Path(socket_path)
        else:
            # Termux lacks /tmp; prefer tempfile.gettempdir() ($TMPDIR on Termux)
            self.socket_path = Path(tempfile.gettempdir()) / f"hidden_gem_mpv_{id(self)}.sock"

        self._state_lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._sock: socket.socket | None = None
        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self._req_id = 0
        self._pending_lock = threading.Lock()
        self._pending_replies: dict[int, tuple[threading.Event, dict[str, Any]]] = {}

        # Playback state: ALWAYS starts stopped (never auto-plays on server startup)
        self._active = False
        self._user_stopped = True
        self._paused = False
        self._muted = False
        self._station_id: str | None = None
        self._current_track: dict[str, Any] | None = None
        self._queue_preview: list[dict[str, Any]] = []
        self._position_sec: float = 0.0
        self._duration_sec: float = 0.0
        self._load_generation: int = 0
        self._consecutive_errors: int = 0
        self._shutdown_called = False

        atexit.register(self.shutdown)

    def is_available(self) -> bool:
        """Return True if a custom IPC connector is injected or `mpv` binary is on PATH."""
        if self._custom_ipc_connector is not None:
            return True
        return shutil.which("mpv") is not None

    def get_state(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "mpv_available": self.is_available(),
                "active": self._active,
                "stopped": self._user_stopped,
                "station_id": self._station_id,
                "track": dict(self._current_track) if self._current_track else None,
                "paused": self._paused,
                "muted": self._muted,
                "position_sec": round(float(self._position_sec), 2),
                "duration_sec": round(float(self._duration_sec), 2),
                "queue_preview": [dict(t) for t in self._queue_preview[:5]],
            }

    def _ensure_connected(self) -> bool:
        with self._state_lock:
            if self._sock is not None and self._reader_thread is not None and self._reader_thread.is_alive():
                return True
            if not self.is_available():
                return False

            self._close_socket_locked()
            self.socket_path.parent.mkdir(parents=True, exist_ok=True)

            if self._custom_ipc_connector is not None:
                sock = self._custom_ipc_connector(self.socket_path)
            else:
                if self._spawn_process and not self._spawn_mpv_daemon_locked():
                    return False
                sock = self._connect_unix_socket(self.socket_path, timeout_sec=4.0)
                if sock is None:
                    return False

            self._sock = sock
            self._reader_stop.clear()
            self._reader_thread = threading.Thread(
                target=self._ipc_reader_loop,
                args=(sock,),
                daemon=True,
                name="mpv-ipc-reader",
            )
            self._reader_thread.start()

        # Observe core properties asynchronously so get_state() never blocks
        for obs_id, prop in ((1, "time-pos"), (2, "duration"), (3, "pause"), (4, "mute")):
            self._send_command(["observe_property", obs_id, prop], wait_reply=False)
        return True

    def _spawn_mpv_daemon_locked(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            return True
        if self.socket_path.exists():
            with suppress(OSError):
                self.socket_path.unlink()

        base_cmd = [
            "mpv",
            "--no-video",
            "--idle=yes",
            "--no-terminal",
            "--force-window=no",
            f"--input-ipc-server={self.socket_path}",
        ]
        # Try Termux default audio output first, fallback to --ao=opensles if immediate exit
        for extra_args in ([], ["--ao=opensles"]):
            cmd = base_cmd + extra_args
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                logger.warning("Failed to spawn mpv (%s): %s", cmd, exc)
                return False

            # Wait up to 2.5s for socket file creation while checking process didn't crash
            deadline = time.monotonic() + 2.5
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                if self.socket_path.exists():
                    self._proc = proc
                    return True
                time.sleep(0.05)

            if proc.poll() is None:
                self._proc = proc
                return True
            logger.warning("mpv exited immediately with rc=%s (args=%s), retrying fallback", proc.returncode, extra_args)

        return False

    @staticmethod
    def _connect_unix_socket(sock_path: Path, timeout_sec: float = 4.0) -> socket.socket | None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if sock_path.exists():
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    s.connect(str(sock_path))
                    s.settimeout(0.5)
                    return s
                except OSError:
                    s.close()
            time.sleep(0.05)
        return None

    def _send_command(
        self,
        command: list[Any],
        *,
        wait_reply: bool = False,
        timeout_sec: float = 2.0,
    ) -> dict[str, Any] | None:
        with self._state_lock:
            sock = self._sock
        if sock is None:
            return None

        with self._pending_lock:
            self._req_id += 1
            req_id = self._req_id
            ev = threading.Event()
            reply_box: dict[str, Any] = {}
            if wait_reply:
                self._pending_replies[req_id] = (ev, reply_box)

        payload = (json.dumps({"command": command, "request_id": req_id}) + "\n").encode("utf-8")
        try:
            with self._send_lock:
                sock.sendall(payload)
        except OSError as exc:
            logger.warning("mpv IPC send failed for %s: %s", command[0] if command else "?", exc)
            if wait_reply:
                with self._pending_lock:
                    self._pending_replies.pop(req_id, None)
            return None

        if not wait_reply:
            return None

        ev.wait(timeout=timeout_sec)
        with self._pending_lock:
            self._pending_replies.pop(req_id, None)
        return reply_box if reply_box else None

    def _ipc_reader_loop(self, sock: socket.socket) -> None:
        buf = b""
        while not self._reader_stop.is_set():
            try:
                chunk = sock.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line.decode("utf-8", errors="replace"))
                except ValueError:
                    continue
                if isinstance(msg, dict):
                    self._handle_ipc_message(msg)

    def _handle_ipc_message(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("request_id")
        if isinstance(req_id, int):
            with self._pending_lock:
                waiter = self._pending_replies.get(req_id)
                if waiter is not None:
                    ev, box = waiter
                    box.update(msg)
                    ev.set()

        event = msg.get("event")
        if event == "property-change":
            name = msg.get("name")
            data = msg.get("data")
            with self._state_lock:
                if name == "time-pos" and isinstance(data, (int, float)):
                    self._position_sec = max(0.0, float(data))
                    if self._position_sec > 0.3:
                        self._consecutive_errors = 0
                elif name == "duration" and isinstance(data, (int, float)) and data > 0:
                    self._duration_sec = float(data)
                elif name == "pause" and isinstance(data, bool):
                    self._paused = data
                elif name == "mute" and isinstance(data, bool):
                    self._muted = data
            return

        if event == "end-file":
            reason = str(msg.get("reason") or "").strip().lower()
            with self._state_lock:
                gen = self._load_generation
                stopped = self._user_stopped
                active = self._active
                paused = self._paused
                ended_track = dict(self._current_track) if self._current_track else None
                station_id = self._station_id
                pos_sec = self._position_sec
                dur_sec = self._duration_sec

            if stopped or not active:
                # Hard stop lock: NEVER advance or record COMPLETE when user stopped
                return

            if reason == "eof" and not paused and station_id and ended_track:
                threading.Thread(
                    target=self._handle_eof_advance,
                    args=(gen, station_id, ended_track, pos_sec, dur_sec),
                    daemon=True,
                    name="mpv-eof-advance",
                ).start()
            elif reason == "error" and station_id:
                threading.Thread(
                    target=self._handle_error_retry,
                    args=(gen, station_id, ended_track),
                    daemon=True,
                    name="mpv-error-retry",
                ).start()

    def _resolve_stream_url(self, video_id: str) -> str:
        # Prefer cached direct URL via AudioStreamResolver; fallback to local /api/audio/{video_id}
        if self.audio_resolver.is_available():
            stream = self.audio_resolver.resolve(video_id)
            if stream is not None and stream.url:
                return stream.url
        return f"{self.api_base_url}/api/audio/{video_id}"

    def _load_track_internal(self, station_id: str, track: dict[str, Any]) -> bool:
        vid = str(track.get("video_id") or "").strip()
        if not vid:
            return False
        if not self._ensure_connected():
            return False

        stream_url = self._resolve_stream_url(vid)
        with self._state_lock:
            if self._user_stopped:
                return False
            self._load_generation += 1
            self._station_id = station_id
            self._current_track = dict(track)
            self._active = True
            self._paused = False
            self._position_sec = 0.0
            self._duration_sec = float(track.get("duration_seconds") or 0.0)

        self._send_command(["loadfile", stream_url, "replace"], wait_reply=False)
        self._send_command(["set_property", "pause", False], wait_reply=False)
        self._refresh_queue_preview(station_id, exclude_track=track)
        return True

    def _refresh_queue_preview(self, station_id: str, *, exclude_track: dict[str, Any] | None = None) -> None:
        if self._next_tracks_provider is None:
            return
        exclude: set[str] = set()
        if exclude_track:
            for key in ("track_id", "video_id"):
                val = exclude_track.get(key)
                if val:
                    exclude.add(str(val))
        try:
            upcoming = self._next_tracks_provider(station_id, 4, exclude) or []
        except Exception as exc:  # noqa: BLE001
            logger.debug("mpv queue preview refresh failed: %s", exc)
            return
        with self._state_lock:
            self._queue_preview = list(upcoming)
        vids = [str(item.get("video_id") or "") for item in upcoming[:3] if item.get("video_id")]
        if vids:
            self.audio_resolver.prefetch_many(vids, limit=3)

    def _pick_next_track(self, station_id: str, exclude_ids: set[str]) -> dict[str, Any] | None:
        with self._state_lock:
            while self._queue_preview:
                cand = self._queue_preview.pop(0)
                if str(cand.get("video_id") or "") not in exclude_ids and str(cand.get("track_id") or "") not in exclude_ids:
                    return cand
        if self._next_tracks_provider is None:
            return None
        items = self._next_tracks_provider(station_id, 4, exclude_ids) or []
        if not items:
            # 1-track fallback: retry without exclude
            items = self._next_tracks_provider(station_id, 4, set()) or []
        if not items:
            return None
        with self._state_lock:
            self._queue_preview = list(items[1:])
        return items[0]

    def _handle_eof_advance(
        self,
        expected_gen: int,
        station_id: str,
        ended_track: dict[str, Any],
        pos_sec: float,
        dur_sec: float,
    ) -> None:
        with self._state_lock:
            if self._user_stopped or not self._active or self._paused or self._load_generation != expected_gen:
                return

        if self._feedback_recorder is not None:
            try:
                effective_dur = dur_sec if dur_sec > 0 else float(ended_track.get("duration_seconds") or 0.0) or None
                effective_pos = pos_sec if pos_sec > 0 else effective_dur
                self._feedback_recorder(station_id, ended_track, "COMPLETE", effective_pos, effective_dur)
            except Exception as exc:  # noqa: BLE001
                logger.warning("mpv COMPLETE feedback recording failed: %s", exc)

        with self._state_lock:
            if self._user_stopped or not self._active or self._paused or self._load_generation != expected_gen:
                return

        exclude = {str(x) for x in (ended_track.get("track_id"), ended_track.get("video_id")) if x}
        next_tr = self._pick_next_track(station_id, exclude)
        if next_tr is not None:
            self._load_track_internal(station_id, next_tr)

    def _handle_error_retry(
        self,
        expected_gen: int,
        station_id: str,
        failed_track: dict[str, Any] | None,
    ) -> None:
        with self._state_lock:
            if self._user_stopped or not self._active or self._load_generation != expected_gen:
                return
            self._consecutive_errors += 1
            if self._consecutive_errors >= MAX_CONSECUTIVE_MPV_ERRORS:
                logger.warning("mpv hit %d consecutive errors; stopping safely", self._consecutive_errors)
                self._active = False
                self._user_stopped = True
                return

        exclude = set()
        if failed_track:
            exclude = {str(x) for x in (failed_track.get("track_id"), failed_track.get("video_id")) if x}
        next_tr = self._pick_next_track(station_id, exclude)
        if next_tr is not None:
            self._load_track_internal(station_id, next_tr)

    def play(self, station_id: str, track: dict[str, Any] | None = None) -> dict[str, Any]:
        """Start or switch station playback on local mpv (clears stopped lock)."""
        with self._state_lock:
            station_changed = station_id != self._station_id
            self._user_stopped = False
            self._active = True
            self._paused = False
            self._consecutive_errors = 0
            self._station_id = station_id
            if station_changed:
                self._queue_preview = []

        target = track
        if target is None:
            target = self._pick_next_track(station_id, set())
        if target is not None:
            self._load_track_internal(station_id, target)
        return self.get_state()

    def stop(self) -> dict[str, Any]:
        """Hard stop: silence mpv and lock out autonomous EOF/error transitions until explicit play()."""
        with self._state_lock:
            self._user_stopped = True
            self._active = False
            self._paused = False
            self._load_generation += 1
            self._position_sec = 0.0
        self._send_command(["stop"], wait_reply=False)
        return self.get_state()

    def toggle_pause(self, station_id: str | None = None) -> dict[str, Any]:
        """Toggle pause/resume; if completely stopped, starts playback for current/given station."""
        with self._state_lock:
            is_stopped = self._user_stopped or not self._active
            target_station = station_id or self._station_id
            current_track = dict(self._current_track) if self._current_track else None

        if is_stopped:
            if target_station:
                return self.play(target_station, track=current_track)
            return self.get_state()

        with self._state_lock:
            self._paused = not self._paused
            new_paused = self._paused
        self._send_command(["set_property", "pause", new_paused], wait_reply=False)
        return self.get_state()

    def next_track(self, *, event_type: str = "SKIP", station_id: str | None = None) -> dict[str, Any]:
        """Record SKIP or BAD and immediately advance to next track."""
        ev = str(event_type or "SKIP").strip().upper()
        with self._state_lock:
            target_station = station_id or self._station_id
            prev_track = dict(self._current_track) if self._current_track else None
            pos_sec = self._position_sec
            dur_sec = self._duration_sec
            self._load_generation += 1
            if ev == "BAD":
                self._queue_preview = []

        if ev == "BAD":
            # Silence audio IMMEDIATELY before recording feedback / loading next track
            self._send_command(["stop"], wait_reply=False)

        if target_station and prev_track and self._feedback_recorder is not None:
            try:
                self._feedback_recorder(target_station, prev_track, ev, pos_sec, dur_sec or None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("mpv %s feedback recording failed: %s", ev, exc)

        if not target_station:
            return self.get_state()

        with self._state_lock:
            self._user_stopped = False
            self._active = True
            self._paused = False
            self._consecutive_errors = 0

        exclude = set()
        if prev_track:
            exclude = {str(x) for x in (prev_track.get("track_id"), prev_track.get("video_id")) if x}
        next_tr = self._pick_next_track(target_station, exclude)
        if next_tr is not None:
            self._load_track_internal(target_station, next_tr)
        else:
            self.stop()
        return self.get_state()

    def seek(self, position_sec: float) -> dict[str, Any]:
        pos = max(0.0, float(position_sec))
        with self._state_lock:
            self._position_sec = pos
        self._send_command(["seek", pos, "absolute"], wait_reply=False)
        return self.get_state()

    def toggle_mute(self, muted: bool | None = None) -> dict[str, Any]:
        with self._state_lock:
            self._muted = (not self._muted) if muted is None else bool(muted)
            new_muted = self._muted
        self._send_command(["set_property", "mute", new_muted], wait_reply=False)
        return self.get_state()

    def _close_socket_locked(self) -> None:
        self._reader_stop.set()
        if self._sock is not None:
            with suppress(OSError):
                self._sock.shutdown(socket.SHUT_RDWR)
            with suppress(OSError):
                self._sock.close()
            self._sock = None

    def shutdown(self) -> None:
        """Cleanly terminate mpv subprocess and remove IPC socket so no orphaned audio survives."""
        with self._state_lock:
            if self._shutdown_called:
                return
            self._shutdown_called = True
            self._user_stopped = True
            self._active = False
            self._load_generation += 1

        with suppress(Exception):
            self._send_command(["quit"], wait_reply=False)

        with self._state_lock:
            self._close_socket_locked()
            proc = self._proc
            self._proc = None

        if proc is not None and proc.poll() is None:
            with suppress(OSError):
                proc.terminate()
            try:
                proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                with suppress(OSError):
                    proc.kill()
                with suppress(Exception):
                    proc.wait(timeout=1.0)

        if self.socket_path.exists():
            with suppress(OSError):
                self.socket_path.unlink()
