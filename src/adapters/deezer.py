"""Deezer Public API adapter (no auth required) with rate limiting and SQLite caching."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from src.db import RadioDatabase
from src.models import normalize_artist, normalize_text, normalize_track_key

logger = logging.getLogger(__name__)

DEEZER_API_BASE = "https://api.deezer.com"


@dataclass(slots=True)
class DeezerTrackMeta:
    """Measured track metadata resolved from Deezer Public API."""

    deezer_id: int
    artist: str
    title: str
    bpm: float
    gain: float
    rank: int
    duration: int
    preview_url: str | None = None
    mir_a1_c4: list[float] | None = None
    analysis_source: str = "deezer_metadata"

    def to_dict(self) -> dict[str, Any]:
        return {
            "deezer_id": int(self.deezer_id),
            "artist": self.artist,
            "title": self.title,
            "bpm": float(self.bpm),
            "gain": float(self.gain),
            "rank": int(self.rank),
            "duration": int(self.duration),
            "preview_url": self.preview_url,
            "mir_a1_c4": list(self.mir_a1_c4) if self.mir_a1_c4 is not None else None,
            "analysis_source": self.analysis_source,
        }


def _match_score(target_artist: str, target_title: str, cand_artist: str, cand_title: str) -> float:
    ta = normalize_artist(target_artist)
    tt = normalize_text(target_title)
    ca = normalize_artist(cand_artist)
    ct = normalize_text(cand_title)

    score = 0.0
    if ta and ca:
        if ta == ca:
            score += 4.0
        elif ta in ca or ca in ta:
            score += 2.5
        else:
            ta_toks = set(ta.split())
            ca_toks = set(ca.split())
            if ta_toks and ca_toks:
                score += 2.0 * (len(ta_toks & ca_toks) / max(len(ta_toks), len(ca_toks)))

    if tt and ct:
        if tt == ct:
            score += 5.0
        elif tt in ct or ct in tt:
            score += 3.0
        else:
            tt_toks = set(tt.split())
            ct_toks = set(ct.split())
            if tt_toks and ct_toks:
                score += 2.5 * (len(tt_toks & ct_toks) / max(len(tt_toks), len(ct_toks)))

    return score


class DeezerClient:
    """Rate-limited, SQLite-cached client for Deezer Public API (search, track detail, 30s preview)."""

    def __init__(
        self,
        *,
        db: RadioDatabase | None = None,
        min_interval_sec: float = 0.25,
        timeout: float = 8.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.db = db
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self.timeout = float(timeout)
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=self.timeout,
            headers={"User-Agent": "HiddenGemRadio/0.3 (MIR Audio Analyzer)"},
            follow_redirects=True,
        )
        self._rate_lock = threading.Lock()
        self._last_request_ts = 0.0

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _throttle(self) -> None:
        if self.min_interval_sec <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            wait = self.min_interval_sec - (now - self._last_request_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_request_ts = time.monotonic()

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        self._throttle()
        url = f"{DEEZER_API_BASE}{path}"
        try:
            resp = self._client.get(url, params=params)
            if resp.status_code != 200:
                logger.warning("Deezer API %s returned HTTP %s", path, resp.status_code)
                return None
            payload = resp.json()
            if isinstance(payload, dict) and payload.get("error"):
                logger.warning("Deezer API error on %s: %s", path, payload.get("error"))
                return None
            return payload if isinstance(payload, dict) else None
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Deezer API request failed (%s): %s", path, exc)
            return None

    def resolve_track(
        self,
        artist: str,
        title: str,
        *,
        video_id: str | None = None,
        use_cache: bool = True,
        fetch_detail_if_missing_bpm: bool = True,
    ) -> DeezerTrackMeta | None:
        """Resolve (artist, title) via SQLite cache first, then Deezer /search (+ /track/{id} for BPM/gain)."""
        canonical_key = normalize_track_key(artist, title)
        if use_cache and self.db is not None:
            cached = self.db.get_audio_feature_cache(canonical_key=canonical_key, video_id=video_id)
            if cached is not None and cached.get("deezer_id") is not None:
                return DeezerTrackMeta(
                    deezer_id=int(cached["deezer_id"]),
                    artist=str(cached.get("artist") or artist),
                    title=str(cached.get("title") or title),
                    bpm=float(cached.get("bpm") or 0.0),
                    gain=float(cached.get("gain") if cached.get("gain") is not None else -8.0),
                    rank=int(cached.get("rank") or 0),
                    duration=int(cached.get("duration") or 0),
                    preview_url=cached.get("preview_url"),
                    mir_a1_c4=cached.get("mir_a1_c4"),
                    analysis_source=str(cached.get("analysis_source") or "deezer_metadata"),
                )

        query = f'artist:"{artist}" track:"{title}"' if artist and title else f"{artist} {title}".strip()
        data = self._get_json("/search", params={"q": query, "limit": 8})
        items = (data or {}).get("data") or []
        if not items and artist and title:
            # Fallback to looser unquoted search when strict field quotes yield 0 matches
            data = self._get_json("/search", params={"q": f"{artist} {title}", "limit": 8})
            items = (data or {}).get("data") or []

        if not items:
            return None

        best_item: dict[str, Any] | None = None
        best_score = -1.0
        for item in items:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            cand_artist = str((item.get("artist") or {}).get("name") or "")
            cand_title = str(item.get("title") or "")
            sc = _match_score(artist, title, cand_artist, cand_title)
            if item.get("preview"):
                sc += 0.25
            if sc > best_score:
                best_score = sc
                best_item = item

        if best_item is None or best_score < 2.0:
            return None

        deezer_id = int(best_item["id"])
        bpm = float(best_item.get("bpm") or 0.0)
        gain_raw = best_item.get("gain")
        rank = int(best_item.get("rank") or 0)
        duration = int(best_item.get("duration") or 0)
        preview_url = str(best_item.get("preview") or "").strip() or None
        found_artist = str((best_item.get("artist") or {}).get("name") or artist)
        found_title = str(best_item.get("title") or title)

        # Deezer /search often omits bpm/gain; fetch /track/{id} once to obtain real measured BPM & gain
        if fetch_detail_if_missing_bpm and (bpm <= 0.0 or gain_raw is None):
            detail = self._get_json(f"/track/{deezer_id}")
            if detail:
                bpm = float(detail.get("bpm") or bpm)
                if detail.get("gain") is not None:
                    gain_raw = detail.get("gain")
                rank = int(detail.get("rank") or rank)
                duration = int(detail.get("duration") or duration)
                if detail.get("preview"):
                    preview_url = str(detail["preview"]).strip() or preview_url

        gain = float(gain_raw) if gain_raw is not None else -8.0
        meta = DeezerTrackMeta(
            deezer_id=deezer_id,
            artist=found_artist,
            title=found_title,
            bpm=bpm,
            gain=gain,
            rank=rank,
            duration=duration,
            preview_url=preview_url,
            mir_a1_c4=None,
            analysis_source="deezer_metadata",
        )

        if self.db is not None:
            self.db.upsert_audio_feature_cache(
                canonical_key=canonical_key,
                video_id=video_id,
                deezer_id=meta.deezer_id,
                artist=meta.artist,
                title=meta.title,
                bpm=meta.bpm,
                gain=meta.gain,
                rank=meta.rank,
                duration=meta.duration,
                preview_url=meta.preview_url,
                mir_a1_c4=None,
                analysis_source=meta.analysis_source,
            )
        return meta

    def fetch_preview_bytes(self, preview_url: str | None) -> bytes | None:
        """Download 30-second MP3 preview bytes directly into memory (no temp files)."""
        if not preview_url:
            return None
        self._throttle()
        try:
            resp = self._client.get(preview_url)
            if resp.status_code != 200 or not resp.content:
                return None
            return bytes(resp.content)
        except httpx.HTTPError as exc:
            logger.warning("Failed to download Deezer preview %s: %s", preview_url, exc)
            return None
