"""Regression tests for server-side native mpv IPC controller, /api/local-player/* APIs, and remote UI."""

from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.adapters.audio_stream import AudioStreamResolver, CachedAudioStream
from src.adapters.mpv_player import MpvLocalPlayer
from src.models import TrackEntity
from src.radio_server import create_app

WEB_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"


def _extract_inline_js() -> str:
    html = WEB_INDEX.read_text(encoding="utf-8")
    scripts = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, flags=re.DOTALL | re.IGNORECASE)
    assert scripts, "No inline <script> block found in web/index.html"
    return "\n".join(scripts)


class _MockMpvIpcServer:
    """In-memory UNIX socketpair mock mpv IPC server that records commands & can emit events."""

    def __init__(self) -> None:
        self.client_sock, self.server_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.client_sock.settimeout(0.2)
        self.server_sock.settimeout(0.2)
        self.commands: list[list[Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self.server_sock.recv(4096)
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
                msg = json.loads(line.decode("utf-8"))
                cmd = msg.get("command") or []
                with self._lock:
                    self.commands.append(cmd)
                req_id = msg.get("request_id")
                if req_id is not None:
                    reply = json.dumps({"error": "success", "request_id": req_id}) + "\n"
                    try:
                        self.server_sock.sendall(reply.encode("utf-8"))
                    except OSError:
                        break

    def emit_event(self, payload: dict[str, Any]) -> None:
        raw = (json.dumps(payload) + "\n").encode("utf-8")
        self.server_sock.sendall(raw)

    def get_commands(self) -> list[list[Any]]:
        with self._lock:
            return list(self.commands)

    def wait_for_loadfile_count(self, expected: int, timeout: float = 2.0) -> list[list[Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            loads = [c for c in self.get_commands() if c and c[0] == "loadfile"]
            if len(loads) >= expected:
                return loads
            time.sleep(0.02)
        return [c for c in self.get_commands() if c and c[0] == "loadfile"]

    def close(self) -> None:
        self._stop.set()
        for s in (self.server_sock, self.client_sock):
            try:
                s.close()
            except OSError:
                pass


def test_mpv_local_player_ipc_eof_stop_toggle_and_bad(tmp_path: Path) -> None:
    mock_ipc = _MockMpvIpcServer()
    resolver = AudioStreamResolver(
        extractor=lambda vid: CachedAudioStream(
            video_id=vid,
            url=f"https://audio.example.test/{vid}.m4a",
            mime_type="audio/mp4",
        ),
    )
    mpv = MpvLocalPlayer(
        audio_resolver=resolver,
        socket_path=tmp_path / "mpv.sock",
        ipc_connector=lambda _path: mock_ipc.client_sock,
        spawn_process=False,
    )
    app = create_app(
        db_path=tmp_path / "radio.db",
        stations_path=tmp_path / "stations.yaml",
        model_dir=tmp_path / "models",
        auto_dig=False,
        audio_resolver=resolver,
        mpv_player=mpv,
    )
    db = app.state.runtime.db
    for idx in range(4):
        db.upsert_track(
            "pes_passion_groove",
            TrackEntity.build(
                artist=f"Mpv Artist {idx}",
                title=f"Mpv Song {idx}",
                view_count=30_000,
                video_id=f"MPV_VID_{idx:03d}",
                duration_seconds=180,
            ),
        )

    try:
        with TestClient(app) as client:
            # 0. /health reports mpv_available=True; initial state is stopped (never auto-plays)
            h = client.get("/health").json()
            assert h["mpv_available"] is True
            st0 = client.get("/api/local-player/state").json()
            assert st0["active"] is False
            assert st0["stopped"] is True

            # 1. POST /api/local-player/play starts Track 1
            st1 = client.post("/api/local-player/play", json={"station_id": "pes_passion_groove"}).json()
            assert st1["active"] is True
            assert st1["stopped"] is False
            assert st1["track"] is not None
            first_vid = st1["track"]["video_id"]
            loads = mock_ipc.wait_for_loadfile_count(1)
            assert len(loads) == 1
            assert first_vid in loads[0][1]

            # 2. Emit end-file (eof) -> server autonomously records COMPLETE and loadfiles Track 2!
            mock_ipc.emit_event({"event": "end-file", "reason": "eof"})
            loads2 = mock_ipc.wait_for_loadfile_count(2)
            assert len(loads2) == 2, f"Expected autonomous 2nd loadfile on EOF, got {loads2}"
            st2 = client.get("/api/local-player/state").json()
            assert st2["track"]["video_id"] != first_vid
            assert db.count_feedback("pes_passion_groove") == 1

            # 3. POST /api/local-player/stop -> sends stop; subsequent end-file (stop) MUST NOT advance or record COMPLETE
            st_stop = client.post("/api/local-player/stop").json()
            assert st_stop["stopped"] is True
            assert st_stop["active"] is False
            fb_before_stop = db.count_feedback("pes_passion_groove")
            mock_ipc.emit_event({"event": "end-file", "reason": "stop"})
            time.sleep(0.15)
            assert len([c for c in mock_ipc.get_commands() if c and c[0] == "loadfile"]) == 2
            assert db.count_feedback("pes_passion_groove") == fb_before_stop

            # 4. Toggle from stopped starts playback, then toggles pause -> resume
            st_resumed = client.post("/api/local-player/toggle", json={"station_id": "pes_passion_groove"}).json()
            assert st_resumed["active"] is True
            assert st_resumed["stopped"] is False
            assert st_resumed["paused"] is False

            st_paused = client.post("/api/local-player/toggle").json()
            assert st_paused["paused"] is True
            st_unpaused = client.post("/api/local-player/toggle").json()
            assert st_unpaused["paused"] is False

            # 5. BAD via /api/local-player/next silences immediately and advances to next track
            vid_before_bad = st_unpaused["track"]["video_id"]
            st_bad = client.post("/api/local-player/next", json={"event_type": "BAD"}).json()
            assert st_bad["track"]["video_id"] != vid_before_bad
            assert db.count_feedback("pes_passion_groove") >= fb_before_stop + 1

            # 6. Seek and Mute commands
            st_seek = client.post("/api/local-player/seek", json={"position_sec": 42.5}).json()
            assert st_seek["position_sec"] == 42.5
            st_mute = client.post("/api/local-player/mute", json={"muted": True}).json()
            assert st_mute["muted"] is True
    finally:
        mpv.shutdown()
        mock_ipc.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required for JS vm tests")
def test_web_ui_mpv_remote_mode_pause_stop_bad_skip_node() -> None:
    inline_js = _extract_inline_js()
    harness = r"""
const vm = require("vm");

function makeElement(id) {
  const listeners = {};
  return {
    id,
    textContent: "",
    innerHTML: "",
    value: "",
    src: "",
    style: {},
    className: "",
    classList: {
      _set: new Set(),
      add(c) { this._set.add(c); },
      remove(c) { this._set.delete(c); },
      contains(c) { return this._set.has(c); },
      toggle(c, force) {
        if (force === undefined) {
          if (this._set.has(c)) this._set.delete(c);
          else this._set.add(c);
        } else if (force) {
          this._set.add(c);
        } else {
          this._set.delete(c);
        }
      },
    },
    children: [],
    appendChild(child) { this.children.push(child); return child; },
    addEventListener(evt, fn) {
      listeners[evt] = listeners[evt] || [];
      listeners[evt].push(fn);
    },
    async trigger(evt) {
      for (const fn of (listeners[evt] || [])) {
        await fn({ target: this, type: evt });
      }
    },
  };
}

async function runAll() {
  const elements = {};
  const getEl = (id) => {
    if (!elements[id]) elements[id] = makeElement(id);
    return elements[id];
  };

  const apiCalls = [];
  let serverState = {
    mpv_available: true,
    active: true,
    stopped: false,
    paused: false,
    muted: false,
    station_id: "pes_passion_groove",
    position_sec: 12,
    duration_sec: 200,
    track: { track_id: "trk_1", video_id: "VID_MPV_1", artist: "Parcels", title: "Remote Song 1", duration_seconds: 200 },
    queue_preview: [{ track_id: "trk_2", video_id: "VID_MPV_2", artist: "Phoenix", title: "Remote Song 2" }],
  };

  const sandbox = {
    console,
    Date,
    Math,
    Number,
    String,
    Boolean,
    Array,
    Object,
    Set,
    Map,
    Promise,
    Error,
    encodeURIComponent,
    setTimeout,
    clearTimeout,
    setInterval: () => 999,
    clearInterval: () => {},
    AbortController: typeof AbortController !== "undefined" ? AbortController : undefined,
    document: {
      hidden: false,
      getElementById: getEl,
      createElement: (tag) => makeElement(tag),
      addEventListener: () => {},
    },
    window: {
      location: { origin: "http://127.0.0.1:8000" },
      addEventListener: () => {},
    },
    navigator: {},
    MediaMetadata: function (obj) { return obj; },
    YT: {
      PlayerState: { UNSTARTED: -1, ENDED: 0, PLAYING: 1, PAUSED: 2, BUFFERING: 3, CUED: 5 },
      Player: function () {
        return {
          loadVideoById() {},
          playVideo() {},
          stopVideo() {},
          pauseVideo() {},
          getPlayerState() { return -1; },
          getCurrentTime() { return 0; },
          getDuration() { return 0; },
        };
      },
    },
    fetch: async (url, opts = {}) => {
      const method = opts.method || "GET";
      const body = opts.body ? JSON.parse(opts.body) : null;
      apiCalls.push({ url, method, body });
      if (url === "/health") {
        return { ok: true, status: 200, json: async () => ({ status: "ok", mpv_available: true, audio_stream_available: true }) };
      }
      if (url === "/stations") {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            stations: [{ station_id: "pes_passion_groove", name: "PES パッション", vibe: "疾走", candidate_count: 10 }],
          }),
        };
      }
      if (url === "/api/local-player/state") {
        return { ok: true, status: 200, json: async () => ({ ...serverState }) };
      }
      if (url === "/api/local-player/stop") {
        serverState = { ...serverState, active: false, stopped: true, paused: false };
        return { ok: true, status: 200, json: async () => ({ ...serverState }) };
      }
      if (url === "/api/local-player/toggle") {
        if (serverState.stopped || !serverState.active) {
          serverState = { ...serverState, active: true, stopped: false, paused: false };
        } else {
          serverState = { ...serverState, paused: !serverState.paused };
        }
        return { ok: true, status: 200, json: async () => ({ ...serverState }) };
      }
      if (url === "/api/local-player/next") {
        serverState = {
          ...serverState,
          active: true,
          stopped: false,
          paused: false,
          track: { track_id: "trk_2", video_id: "VID_MPV_2", artist: "Phoenix", title: "Remote Song 2", duration_seconds: 190 },
        };
        return { ok: true, status: 200, json: async () => ({ ...serverState }) };
      }
      return { ok: true, status: 200, json: async () => ({ playlists: [], updates: 0, layers: [], axes: [] }) };
    },
  };
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(
    INLINE_JS + "\n;globalThis.__exports = { state, el, isMpvModeActive, loadServerCapabilities };",
    sandbox
  );

  // Wait for startup bootstrap (/health + /stations + /api/local-player/state)
  await new Promise((r) => setTimeout(r, 30));

  const { state, el, isMpvModeActive } = sandbox.__exports;
  if (!isMpvModeActive() || state.playbackMode !== "mpv") {
    throw new Error(`Expected default mpv mode when mpv_available=true, got ${state.playbackMode}`);
  }
  if (el.btnPlayMute.textContent !== "⏸ 一時停止") {
    throw new Error(`Expected main play button to show '⏸ 一時停止' while playing (NOT Mute), got '${el.btnPlayMute.textContent}'`);
  }

  // 1. Click ⏹ 停止 -> POST /api/local-player/stop -> status '停止中', button '▶ 再生'
  await el.btnStop.trigger("click");
  if (!apiCalls.some((c) => c.url === "/api/local-player/stop" && c.method === "POST")) {
    throw new Error("Expected POST /api/local-player/stop on btnStop click");
  }
  if (el.statusPill.textContent !== "停止中" || el.btnPlayMute.textContent !== "▶ 再生") {
    throw new Error(`Expected stopped UI, got status='${el.statusPill.textContent}', btn='${el.btnPlayMute.textContent}'`);
  }

  // 2. Click ▶ 再生 -> POST /api/local-player/toggle -> playing -> '⏸ 一時停止'
  await el.btnPlayMute.trigger("click");
  if (!apiCalls.some((c) => c.url === "/api/local-player/toggle" && c.method === "POST")) {
    throw new Error("Expected POST /api/local-player/toggle on btnPlayMute click");
  }
  if (el.btnPlayMute.textContent !== "⏸ 一時停止") {
    throw new Error(`Expected '⏸ 一時停止' after toggle play, got '${el.btnPlayMute.textContent}'`);
  }

  // 3. Click ⏭ 次曲 -> POST /api/local-player/next (SKIP)
  await el.btnSkip.trigger("click");
  if (!apiCalls.some((c) => c.url === "/api/local-player/next" && c.body?.event_type === "SKIP")) {
    throw new Error("Expected POST /api/local-player/next with event_type=SKIP");
  }

  // 4. Click 👎 BAD -> POST /api/local-player/next (BAD)
  await el.btnBad.trigger("click");
  if (!apiCalls.some((c) => c.url === "/api/local-player/next" && c.body?.event_type === "BAD")) {
    throw new Error("Expected POST /api/local-player/next with event_type=BAD");
  }

  console.log(JSON.stringify({ ok: true }));
  process.exit(0);
}

runAll().catch((err) => {
  console.error(err);
  process.exit(1);
});
"""
    script = f"const INLINE_JS = {json.dumps(inline_js)};\n{harness}"
    proc = subprocess.run(["node", "-e", script], check=False, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, f"Node mpv remote vm test failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    assert '"ok":true' in proc.stdout.replace(" ", "")


def test_mpv_user_agent_header_and_error_force_refresh_retry_once(tmp_path: Path) -> None:
    mock_ipc = _MockMpvIpcServer()
    resolve_log: list[tuple[str, int]] = []

    def fake_extract(vid: str) -> CachedAudioStream:
        count = sum(1 for v, _ in resolve_log if v == vid) + 1
        resolve_log.append((vid, count))
        return CachedAudioStream(
            video_id=vid,
            url=f"https://audio.example.test/{vid}_v{count}.m4a",
            mime_type="audio/mp4",
            http_headers={"User-Agent": f"CustomYTAgent/{count}.0"},
        )

    resolver = AudioStreamResolver(extractor=fake_extract)
    mpv = MpvLocalPlayer(
        audio_resolver=resolver,
        socket_path=tmp_path / "mpv.sock",
        ipc_connector=lambda _p: mock_ipc.client_sock,
        spawn_process=False,
    )
    app = create_app(
        db_path=tmp_path / "radio.db",
        stations_path=tmp_path / "stations.yaml",
        model_dir=tmp_path / "models",
        auto_dig=False,
        audio_resolver=resolver,
        mpv_player=mpv,
    )
    db = app.state.runtime.db
    db.upsert_track(
        "pes_passion_groove",
        TrackEntity.build(artist="Header Artist 1", title="Song 1", view_count=20_000, video_id="HDR_VID_001"),
    )
    db.upsert_track(
        "pes_passion_groove",
        TrackEntity.build(artist="Header Artist 2", title="Song 2", view_count=20_000, video_id="HDR_VID_002"),
    )

    try:
        with TestClient(app) as client:
            st1 = client.post("/api/local-player/play", json={"station_id": "pes_passion_groove"}).json()
            first_vid = st1["track"]["video_id"]
            mock_ipc.wait_for_loadfile_count(1)
            cmds = mock_ipc.get_commands()
            ua_idx = next(i for i, c in enumerate(cmds) if c[:2] == ["set_property", "user-agent"])
            lf_idx = next(i for i, c in enumerate(cmds) if c and c[0] == "loadfile")
            assert ua_idx < lf_idx, "User-Agent must be set before loadfile"
            assert cmds[ua_idx][2] == "CustomYTAgent/1.0"

            # 1st error on same track -> invalidates cache & retries SAME track once with v2 URL
            mock_ipc.emit_event({"event": "end-file", "reason": "error"})
            loads2 = mock_ipc.wait_for_loadfile_count(2)
            assert len(loads2) == 2
            assert f"{first_vid}_v2.m4a" in loads2[1][1]

            # 2nd error on same track -> advances to the OTHER track (not infinite loop on same track)
            mock_ipc.emit_event({"event": "end-file", "reason": "error"})
            loads3 = mock_ipc.wait_for_loadfile_count(3)
            assert len(loads3) == 3
            assert first_vid not in loads3[2][1]
    finally:
        mpv.shutdown()
        mock_ipc.close()


def test_mpv_stop_race_during_slow_resolve_and_paused_error_guard(tmp_path: Path) -> None:
    mock_ipc = _MockMpvIpcServer()
    resolve_started = threading.Event()
    release_resolve = threading.Event()
    block_first = {"flag": True}

    def slow_extract(vid: str) -> CachedAudioStream:
        if block_first["flag"]:
            block_first["flag"] = False
            resolve_started.set()
            release_resolve.wait(timeout=3.0)
        return CachedAudioStream(
            video_id=vid,
            url=f"https://audio.example.test/{vid}.m4a",
            mime_type="audio/mp4",
        )

    resolver = AudioStreamResolver(extractor=slow_extract)
    mpv = MpvLocalPlayer(
        audio_resolver=resolver,
        socket_path=tmp_path / "mpv.sock",
        ipc_connector=lambda _p: mock_ipc.client_sock,
        spawn_process=False,
    )
    app = create_app(
        db_path=tmp_path / "radio.db",
        stations_path=tmp_path / "stations.yaml",
        model_dir=tmp_path / "models",
        auto_dig=False,
        audio_resolver=resolver,
        mpv_player=mpv,
    )
    db = app.state.runtime.db
    db.upsert_track(
        "pes_passion_groove",
        TrackEntity.build(artist="Race Artist 1", title="Race Song 1", view_count=20_000, video_id="RACE_VID_01"),
    )
    db.upsert_track(
        "pes_passion_groove",
        TrackEntity.build(artist="Race Artist 2", title="Race Song 2", view_count=20_000, video_id="RACE_VID_02"),
    )

    try:
        # 1. stop() arrives while play() is blocked in slow_extract -> ZERO loadfile sent!
        play_thread = threading.Thread(target=mpv.play, args=("pes_passion_groove",), daemon=True)
        play_thread.start()
        assert resolve_started.wait(timeout=2.0)
        mpv.stop()
        release_resolve.set()
        play_thread.join(timeout=2.0)

        assert mpv.get_state()["stopped"] is True
        assert len([c for c in mock_ipc.get_commands() if c and c[0] == "loadfile"]) == 0

        # 2. Paused track receives end-file (error) -> MUST NOT auto-resume or loadfile!
        mpv.play("pes_passion_groove")
        assert len(mock_ipc.wait_for_loadfile_count(1)) == 1
        mpv.toggle_pause()
        assert mpv.get_state()["paused"] is True
        mock_ipc.emit_event({"event": "end-file", "reason": "error"})
        time.sleep(0.15)
        assert len([c for c in mock_ipc.get_commands() if c and c[0] == "loadfile"]) == 1
        assert mpv.get_state()["paused"] is True
    finally:
        mpv.shutdown()
        mock_ipc.close()


def test_mpv_socket_path_96b_fallback_and_termux_ao_priority(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PORT", "9123")
    long_path = tmp_path / ("a" * 110) / "mpv.sock"
    mpv = MpvLocalPlayer(socket_path=long_path, spawn_process=False)
    try:
        assert mpv.socket_path.name == "hgr_mpv.sock"
        assert len(str(mpv.socket_path).encode("utf-8")) < 96
        assert mpv.api_base_url == "http://127.0.0.1:9123"
        cmd = mpv._build_mpv_spawn_cmd()
        assert "--ao=opensles,audiotrack,aaudio,pulse" in cmd
    finally:
        mpv.shutdown()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required for JS vm tests")
def test_web_ui_mpv_mode_guards_yt_events_epoch_discard_and_good_listen_seconds_node() -> None:
    inline_js = _extract_inline_js()
    harness = r"""
const vm = require("vm");

function makeElement(id) {
  const listeners = {};
  return {
    id,
    textContent: "",
    innerHTML: "",
    value: "50",
    src: "",
    style: {},
    className: "",
    classList: {
      _set: new Set(),
      add(c) { this._set.add(c); },
      remove(c) { this._set.delete(c); },
      contains(c) { return this._set.has(c); },
      toggle(c, force) {
        if (force === undefined) {
          if (this._set.has(c)) this._set.delete(c);
          else this._set.add(c);
        } else if (force) {
          this._set.add(c);
        } else {
          this._set.delete(c);
        }
      },
    },
    children: [],
    appendChild(child) { this.children.push(child); return child; },
    addEventListener(evt, fn) {
      listeners[evt] = listeners[evt] || [];
      listeners[evt].push(fn);
    },
    async trigger(evt) {
      for (const fn of (listeners[evt] || [])) {
        await fn({ target: this, type: evt });
      }
    },
  };
}

async function runAll() {
  const elements = {};
  const getEl = (id) => {
    if (!elements[id]) elements[id] = makeElement(id);
    return elements[id];
  };

  let ytEvents = null;
  let ytPlayVideoCalls = 0;
  let resolveStalePoll = null;
  let holdNextPoll = false;
  const feedbackBodies = [];
  const seekBodies = [];

  const sandbox = {
    console,
    Date,
    Math,
    Number,
    String,
    Boolean,
    Array,
    Object,
    Set,
    Map,
    Promise,
    Error,
    encodeURIComponent,
    setTimeout,
    clearTimeout,
    setInterval: () => 999,
    clearInterval: () => {},
    AbortController: typeof AbortController !== "undefined" ? AbortController : undefined,
    document: {
      hidden: false,
      getElementById: getEl,
      createElement: (tag) => makeElement(tag),
      addEventListener: () => {},
    },
    window: {
      location: { origin: "http://127.0.0.1:8000" },
      addEventListener: () => {},
    },
    navigator: {},
    MediaMetadata: function (obj) { return obj; },
    YT: {
      PlayerState: { UNSTARTED: -1, ENDED: 0, PLAYING: 1, PAUSED: 2, BUFFERING: 3, CUED: 5 },
      Player: function (_id, opts) {
        ytEvents = opts.events;
        return {
          loadVideoById() {},
          playVideo() { ytPlayVideoCalls += 1; },
          stopVideo() {},
          pauseVideo() {},
          getPlayerState() { return 5; },
          getCurrentTime() { return 0; },
          getDuration() { return 0; },
        };
      },
    },
    fetch: async (url, opts = {}) => {
      const body = opts.body ? JSON.parse(opts.body) : null;
      if (url === "/health") {
        return { ok: true, status: 200, json: async () => ({ status: "ok", mpv_available: true, audio_stream_available: true }) };
      }
      if (url === "/stations") {
        return { ok: true, status: 200, json: async () => ({ stations: [{ station_id: "pes_passion_groove", name: "PES", vibe: "v", candidate_count: 5 }] }) };
      }
      if (url === "/api/local-player/state") {
        if (holdNextPoll) {
          holdNextPoll = false;
          // Stale in-flight poll that resolves AFTER user clicked Stop
          return new Promise((resolve) => {
            resolveStalePoll = () => resolve({
              ok: true,
              status: 200,
              json: async () => ({
                mpv_available: true,
                active: true,
                stopped: false,
                paused: false,
                station_id: "pes_passion_groove",
                position_sec: 88.5,
                duration_sec: 210,
                track: { track_id: "trk_1", video_id: "VID_1", artist: "Parcels", title: "Song 1", duration_seconds: 210 },
                queue_preview: [],
              }),
            });
          });
        }
        return {
          ok: true,
          status: 200,
          json: async () => ({
            mpv_available: true,
            active: true,
            stopped: false,
            paused: false,
            station_id: "pes_passion_groove",
            position_sec: 77.5,
            duration_sec: 210,
            track: { track_id: "trk_1", video_id: "VID_1", artist: "Parcels", title: "Song 1", duration_seconds: 210 },
            queue_preview: [],
          }),
        };
      }
      if (url === "/api/local-player/stop") {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            mpv_available: true,
            active: false,
            stopped: true,
            paused: false,
            station_id: "pes_passion_groove",
            position_sec: 0,
            duration_sec: 210,
            track: { track_id: "trk_1", video_id: "VID_1", artist: "Parcels", title: "Song 1", duration_seconds: 210 },
            queue_preview: [],
          }),
        };
      }
      if (url === "/api/local-player/seek") {
        seekBodies.push(body);
        return {
          ok: true,
          status: 200,
          json: async () => ({
            mpv_available: true,
            active: true,
            stopped: false,
            paused: false,
            station_id: "pes_passion_groove",
            position_sec: body.position_sec,
            duration_sec: 210,
            track: { track_id: "trk_1", video_id: "VID_1", artist: "Parcels", title: "Song 1", duration_seconds: 210 },
            queue_preview: [],
          }),
        };
      }
      if (url === "/feedback") {
        feedbackBodies.push(body);
        return { ok: true, status: 200, json: async () => ({ status: "ok", model_updates: 1 }) };
      }
      return { ok: true, status: 200, json: async () => ({ playlists: [], updates: 1, layers: [], axes: [] }) };
    },
  };
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(
    INLINE_JS + "\n;globalThis.__exports = { state, el, pollLocalPlayerStateOnce };",
    sandbox
  );
  sandbox.window.onYouTubeIframeAPIReady();
  await new Promise((r) => setTimeout(r, 30));

  const { state, el, pollLocalPlayerStateOnce } = sandbox.__exports;
  state.hasUserPlaybackStarted = true;

  // 1. YT CUED / ENDED while in mpv mode must NEVER call yt.playVideo()
  await ytEvents.onStateChange({ data: 5 }); // CUED
  await ytEvents.onStateChange({ data: 0 }); // ENDED
  if (ytPlayVideoCalls !== 0) {
    throw new Error(`Expected 0 YouTube playVideo calls in mpv mode, got ${ytPlayVideoCalls}`);
  }

  // 2. Good feedback in mpv mode sends localPlayerState.position_sec (77.5)
  await el.btnGood.trigger("click");
  const goodFb = feedbackBodies.find((b) => b.event_type === "GOOD");
  if (!goodFb || goodFb.listen_seconds !== 77.5) {
    throw new Error(`Expected GOOD listen_seconds=77.5 from localPlayerState, got ${JSON.stringify(goodFb)}`);
  }

  // 3. Seekbar drag (input) sends ZERO POSTs; release (change) sends exactly 1 POST
  el.audioSeekBar.value = "50";
  await el.audioSeekBar.trigger("input");
  if (seekBodies.length !== 0 || state.isSeekingAudioBar !== true) {
    throw new Error(`Expected 0 seek POSTs on input drag, got ${seekBodies.length}`);
  }
  await el.audioSeekBar.trigger("change");
  if (seekBodies.length !== 1 || seekBodies[0].position_sec !== 105 || state.isSeekingAudioBar !== false) {
    throw new Error(`Expected 1 seek POST (105s) on change, got ${JSON.stringify(seekBodies)}`);
  }

  // 4. Stale in-flight poll discarded when localControlEpoch incremented by btnStop
  holdNextPoll = true;
  const stalePollPromise = pollLocalPlayerStateOnce();
  await new Promise((r) => setTimeout(r, 5));
  await el.btnStop.trigger("click");
  resolveStalePoll();
  await stalePollPromise;
  if (el.statusPill.textContent !== "停止中" || state.userStopped !== true) {
    throw new Error(`Stale poll rewound stopped UI: status='${el.statusPill.textContent}', userStopped=${state.userStopped}`);
  }

  console.log(JSON.stringify({ ok: true }));
  process.exit(0);
}

runAll().catch((err) => {
  console.error(err);
  process.exit(1);
});
"""
    script = f"const INLINE_JS = {json.dumps(inline_js)};\n{harness}"
    proc = subprocess.run(["node", "-e", script], check=False, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, f"Node mpv audit vm test failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    assert '"ok":true' in proc.stdout.replace(" ", "")
