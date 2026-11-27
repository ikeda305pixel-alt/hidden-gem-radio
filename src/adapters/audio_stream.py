"""YouTube audio-only stream resolver & TTL prefetch cache using yt-dlp (module or CLI).

Designed for Termux (Android / Python 3.14 / ARM64) with zero native build dependencies:
uses `yt_dlp` Python package if installed, otherwise falls back to the `yt-dlp` CLI subprocess.
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from src.db import sanitize_video_id

logger = logging.getLogger(__name__)

DEFAULT_AUDIO_FORMAT = "bestaudio[ext=m4a]/bestaudio/best"
DEFAULT_CACHE_TTL_SEC = 1800.0  # 30 minutes
DEFAULT_MAX_CACHE_SIZE = 256
_MIME_BY_EXT = {
    "m4a": "audio/mp4",
    "mp4": "audio/mp4",
    "webm": "audio/webm",
    "opus": "audio/webm",
    "ogg": "audio/ogg",
    "mp3": "audio/mpeg",
}


@dataclass(frozen=True)
class CachedAudioStream:
    """Resolved upstream direct audio stream metadata stored in memory cache."""

    video_id: str
    url: str
    mime_type: str = "audio/mp4"
    content_length: int | None = None
    http_headers: dict[str, str] = field(default_factory=dict)
    expires_at: float = 0.0

    def is_expired(self, now: float | None = None) -> bool:
        ts = time.time() if now is None else now
        return ts >= self.expires_at


def _guess_mime_from_url_or_ext(url: str, ext: str | None = None) -> str:
    if ext:
        clean_ext = ext.strip().lower().lstrip(".")
        if clean_ext in _MIME_BY_EXT:
            return _MIME_BY_EXT[clean_ext]
    low = url.lower()
    if "mime=audio%2fwebm" in low or "mime=audio/webm" in low or ".webm" in low:
        return "audio/webm"
    if "mime=audio%2fmp4" in low or "mime=audio/mp4" in low or ".m4a" in low:
        return "audio/mp4"
    return "audio/mp4"


class AudioStreamResolver:
    """Resolves YouTube audio-only direct URLs via yt-dlp with 30-min TTL cache & background prefetch."""

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_CACHE_TTL_SEC,
        max_cache_size: int = DEFAULT_MAX_CACHE_SIZE,
        max_prefetch_workers: int = 2,
        format_selector: str = DEFAULT_AUDIO_FORMAT,
        extractor: Callable[[str], CachedAudioStream | None] | None = None,
    ) -> None:
        self.ttl_seconds = max(30.0, float(ttl_seconds))
        self.max_cache_size = max(16, int(max_cache_size))
        self.format_selector = format_selector
        self._custom_extractor = extractor

        self._cache_lock = threading.Lock()
        self._cache: dict[str, CachedAudioStream] = {}
        self._inflight_locks: dict[str, threading.Lock] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_prefetch_workers)),
            thread_name_prefix="audio-prefetch",
        )

    def is_available(self) -> bool:
        """Return True if a custom extractor, yt_dlp Python module, or yt-dlp CLI is available."""
        if self._custom_extractor is not None:
            return True
        if importlib.util.find_spec("yt_dlp") is not None:
            return True
        return shutil.which("yt-dlp") is not None

    def get_cached(self, video_id: str, *, now: float | None = None) -> CachedAudioStream | None:
        clean_vid = sanitize_video_id(video_id)
        if not clean_vid:
            return None
        with self._cache_lock:
            entry = self._cache.get(clean_vid)
            if entry is None:
                return None
            if entry.is_expired(now=now):
                self._cache.pop(clean_vid, None)
                return None
            return entry

    def invalidate(self, video_id: str) -> None:
        clean_vid = sanitize_video_id(video_id)
        if not clean_vid:
            return
        with self._cache_lock:
            self._cache.pop(clean_vid, None)

    def _put_cache(self, entry: CachedAudioStream) -> None:
        with self._cache_lock:
            if len(self._cache) >= self.max_cache_size and entry.video_id not in self._cache:
                oldest_vid = min(self._cache.items(), key=lambda kv: kv[1].expires_at)[0]
                self._cache.pop(oldest_vid, None)
            self._cache[entry.video_id] = entry

    def _get_inflight_lock(self, video_id: str) -> threading.Lock:
        with self._cache_lock:
            lock = self._inflight_locks.get(video_id)
            if lock is None:
                lock = threading.Lock()
                self._inflight_locks[video_id] = lock
            return lock

    def resolve(self, video_id: str, *, force_refresh: bool = False) -> CachedAudioStream | None:
        """Resolve and cache a direct audio URL for video_id (deduplicating concurrent resolves)."""
        clean_vid = sanitize_video_id(video_id)
        if not clean_vid:
            return None
        if not force_refresh:
            cached = self.get_cached(clean_vid)
            if cached is not None:
                return cached

        vid_lock = self._get_inflight_lock(clean_vid)
        with vid_lock:
            if not force_refresh:
                cached = self.get_cached(clean_vid)
                if cached is not None:
                    return cached

            extracted = self._extract_stream(clean_vid)
            if extracted is not None:
                if extracted.expires_at <= 0:
                    extracted = CachedAudioStream(
                        video_id=clean_vid,
                        url=extracted.url,
                        mime_type=extracted.mime_type,
                        content_length=extracted.content_length,
                        http_headers=dict(extracted.http_headers),
                        expires_at=time.time() + self.ttl_seconds,
                    )
                self._put_cache(extracted)
            return extracted

    def prefetch_many(self, video_ids: list[str], *, limit: int = 3) -> int:
        """Non-blocking background prefetch of up to `limit` uncached video_ids."""
        if not self.is_available():
            return 0
        scheduled = 0
        seen: set[str] = set()
        for raw_vid in video_ids:
            if scheduled >= limit:
                break
            clean_vid = sanitize_video_id(raw_vid)
            if not clean_vid or clean_vid in seen:
                continue
            seen.add(clean_vid)
            if self.get_cached(clean_vid) is not None:
                continue
            try:
                self._executor.submit(self._safe_prefetch_one, clean_vid)
                scheduled += 1
            except RuntimeError:
                break
        return scheduled

    def _safe_prefetch_one(self, video_id: str) -> None:
        try:
            self.resolve(video_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Background audio prefetch failed for %s: %s", video_id, exc)

    def _extract_stream(self, video_id: str) -> CachedAudioStream | None:
        if self._custom_extractor is not None:
            return self._custom_extractor(video_id)

        # 1. Primary: Python yt_dlp module
        if importlib.util.find_spec("yt_dlp") is not None:
            res = self._extract_via_python_module(video_id)
            if res is not None:
                return res

        # 2. Fallback: yt-dlp CLI subprocess
        if shutil.which("yt-dlp") is not None:
            return self._extract_via_cli(video_id)

        return None

    def _extract_via_python_module(self, video_id: str) -> CachedAudioStream | None:
        try:
            import yt_dlp  # type: ignore[import-not-found]
        except Exception:  # noqa: BLE001
            return None

        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        opts: dict[str, Any] = {
            "format": self.format_selector,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
            "socket_timeout": 10,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(watch_url, download=False)
            if not isinstance(info, dict):
                return None
            stream_url = str(info.get("url") or "").strip()
            if not stream_url:
                return None
            ext = str(info.get("ext") or "").strip() or None
            raw_headers = info.get("http_headers")
            headers = (
                {str(k): str(v) for k, v in raw_headers.items() if k and v}
                if isinstance(raw_headers, dict)
                else {}
            )
            raw_size = info.get("filesize") or info.get("filesize_approx")
            content_length = int(raw_size) if isinstance(raw_size, (int, float)) and raw_size > 0 else None
            return CachedAudioStream(
                video_id=video_id,
                url=stream_url,
                mime_type=_guess_mime_from_url_or_ext(stream_url, ext),
                content_length=content_length,
                http_headers=headers,
                expires_at=time.time() + self.ttl_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("yt_dlp module extraction failed for %s: %s", video_id, exc)
            return None

    def _extract_via_cli(self, video_id: str) -> CachedAudioStream | None:
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        cmd = [
            "yt-dlp",
            "-f",
            self.format_selector,
            "-g",
            "--no-playlist",
            "--no-warnings",
            watch_url,
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=12,
            )
            if proc.returncode != 0:
                logger.warning("yt-dlp CLI failed for %s (rc=%d): %s", video_id, proc.returncode, proc.stderr.strip())
                return None
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip().startswith("http")]
            if not lines:
                return None
            stream_url = lines[0]
            return CachedAudioStream(
                video_id=video_id,
                url=stream_url,
                mime_type=_guess_mime_from_url_or_ext(stream_url),
                content_length=None,
                http_headers={},
                expires_at=time.time() + self.ttl_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("yt-dlp CLI subprocess error for %s: %s", video_id, exc)
            return None
