"""Real Acoustic & Music Theory (MIR) 16-dimensional feature engine using numpy + scipy + optional ffmpeg pipe.

Layer A (Rhythm & Groove):
  0: a1_bpm_norm              - Measured BPM (Deezer or onset autocorrelation) normalized to [0, 1]
  1: a2_beat_clarity          - Dominant beat autocorrelation peak of onset strength envelope
  2: a3_syncopation_index     - Off-beat / 3-3-2 tresillo onset ratio vs primary pulse
  3: a4_sub_bass_drive        - 20-120 Hz low-end energy ratio
Layer B (Harmony & Tonality):
  4: b1_major_minor_score     - 12-bin chroma correlation diff vs Krumhansl-Schmuckler major/minor profiles
  5: b2_harmonic_entropy      - Normalized Shannon entropy of mean 12-bin chroma distribution
  6: b3_chord_change_rate     - Mean cosine distance between consecutive smoothed chroma frames
  7: b4_harmonic_percussive_ratio - Lightweight HPSS harmonic vs percussive spectral energy ratio
Layer C (Timbre, Loudness & Structure):
  8: c1_spectral_centroid     - Normalized spectral centroid (brightness)
  9: c2_spectral_flatness     - Spectral flatness / Wiener entropy (acoustic vs electronic/noise)
 10: c3_loudness_dynamic_punch - Deezer gain (mastering loudness) + frame RMS contrast
 11: c4_vocal_band_presence   - 300-3400 Hz vocal/melody band energy ratio
Layer D (Scene, Popularity Sweetspot & Curation):
 12: d1_cooccurrence_score    - Stage 1 collaborative graph multi-seed co-occurrence score
 13: d2_deezer_rank_sweetspot - Log bell-curve on Deezer rank (avoids slop, peaks at semi-classic/gem tier)
 14: d3_yt_views_sweetspot    - Log bell-curve on YouTube views (peaks 300k-50M, penalizes <100k, decays >500M)
 15: d4_cult_chic_similarity  - Cosine similarity of track A1-C4 vs calibrated PES 2011 / Emily in Paris centroid
"""

from __future__ import annotations

import io
import logging
import math
import shutil
import subprocess
import wave
from typing import Any

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import resample_poly, stft

from src.adapters.deezer import DeezerClient, DeezerTrackMeta
from src.db import RadioDatabase
from src.models import TrackEntity, normalize_track_key

logger = logging.getLogger(__name__)

TARGET_SR = 22050
MIR_DIM = 16

MIR_FEATURE_KEYS: tuple[str, ...] = (
    "a1_bpm_norm",
    "a2_beat_clarity",
    "a3_syncopation_index",
    "a4_sub_bass_drive",
    "b1_major_minor_score",
    "b2_harmonic_entropy",
    "b3_chord_change_rate",
    "b4_harmonic_percussive_ratio",
    "c1_spectral_centroid",
    "c2_spectral_flatness",
    "c3_loudness_dynamic_punch",
    "c4_vocal_band_presence",
    "d1_cooccurrence_score",
    "d2_deezer_rank_sweetspot",
    "d3_yt_views_sweetspot",
    "d4_cult_chic_similarity",
)

# Krumhansl-Schmuckler tonal key profiles (C, C#, D, ..., B)
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float64)
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64)
_KS_MAJOR_NORM = (_KS_MAJOR - _KS_MAJOR.mean()) / (np.linalg.norm(_KS_MAJOR - _KS_MAJOR.mean()) + 1e-12)
_KS_MINOR_NORM = (_KS_MINOR - _KS_MINOR.mean()) / (np.linalg.norm(_KS_MINOR - _KS_MINOR.mean()) + 1e-12)

# Precompute all 12 rotations of KS templates for fast vectorized key correlation
_KS_MAJOR_ROT = np.vstack([np.roll(_KS_MAJOR_NORM, k) for k in range(12)])
_KS_MINOR_ROT = np.vstack([np.roll(_KS_MINOR_NORM, k) for k in range(12)])

# Default acoustic prior for PES 2011 / Emily in Paris cult-chic anchor (12-dim A1..C4)
# Replaced automatically once calibrated from real tracks in SQLite station_profiles["__cult_chic_anchor__"]
DEFAULT_CULT_CHIC_A1_C4 = np.array(
    [0.50, 0.68, 0.54, 0.36, 0.56, 0.68, 0.52, 0.66, 0.42, 0.24, 0.64, 0.58],
    dtype=np.float64,
)

# Precomputed chroma projection matrix cache keyed by (n_fft, sr)
_CHROMA_MATRIX_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _get_chroma_matrix(n_fft: int = 2048, sr: int = TARGET_SR) -> np.ndarray:
    """Build a (12, n_fft//2 + 1) frequency-to-pitch-class projection matrix."""
    key = (n_fft, sr)
    cached = _CHROMA_MATRIX_CACHE.get(key)
    if cached is not None:
        return cached

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mat = np.zeros((12, len(freqs)), dtype=np.float64)
    for idx, f in enumerate(freqs):
        if f < 65.0 or f > 4200.0:
            continue
        midi = 69.0 + 12.0 * math.log2(f / 440.0)
        pc_float = midi % 12.0
        # Soft Gaussian weight across nearest pitch classes for smooth tuning tolerance
        for pc in range(12):
            dist = min(abs(pc_float - pc), 12.0 - abs(pc_float - pc))
            if dist <= 1.0:
                mat[pc, idx] += math.exp(-0.5 * (dist / 0.35) ** 2)
    row_sums = mat.sum(axis=1, keepdims=True)
    mat = np.divide(mat, np.maximum(1e-9, row_sums))
    _CHROMA_MATRIX_CACHE[key] = mat
    return mat


def bpm_to_norm(bpm: float) -> float:
    """Normalize musical BPM (60-180 range, octave-folded) into [0.0, 1.0]."""
    if bpm <= 0.0:
        return 0.5
    val = float(bpm)
    while val < 60.0:
        val *= 2.0
    while val > 180.0:
        val *= 0.5
    return _clip01((val - 60.0) / 120.0)


def compute_cooccurrence_score(cooccurrence_count: int) -> float:
    """Map Stage 1 multi-seed co-occurrence count to [0, 1] (count>=2 receives major boost)."""
    c = max(1, int(cooccurrence_count))
    if c == 1:
        return 0.25
    return _clip01(0.25 + 0.75 * (1.0 - math.exp(-0.70 * (c - 1))))


def compute_deezer_rank_sweetspot(rank: int | None) -> float:
    """Log bell-curve on Deezer rank: avoids unknown slop (<10k), peaks around 200k-650k."""
    if rank is None or rank <= 0:
        return 0.40
    log_r = math.log10(float(rank) + 1.0)
    # Peak around log10(350_000) ~= 5.54
    if log_r < 4.0:
        # Steep drop for ultra-low / unverified slop
        return _clip01(0.15 + 0.35 * math.exp(-((4.0 - log_r) / 0.8) ** 2))
    if log_r <= 5.85:
        return _clip01(math.exp(-((log_r - 5.50) / 1.15) ** 2))
    # Gentle decay for mega-mainstream >700k rank
    return _clip01(0.85 + 0.15 * math.exp(-((log_r - 5.85) / 0.6) ** 2))


def compute_yt_views_sweetspot(view_count: int | None) -> float:
    """Log bell-curve on YouTube views: peak 1.0 across 300k-50M, strong penalty <100k, gentle decay >500M."""
    if view_count is None or view_count < 0:
        return 0.45
    log_v = math.log10(float(view_count) + 1.0)
    low_peak = math.log10(300_000.0)      # ~5.477
    high_peak = math.log10(50_000_000.0)  # ~7.699
    if low_peak <= log_v <= high_peak:
        return 1.0
    if log_v < low_peak:
        # Steep penalty below 100k (log10=5.0) to suppress amateur/AI slop
        if log_v < 5.0:
            return _clip01(0.35 * math.exp(-((5.0 - log_v) / 0.9) ** 2))
        frac = (log_v - 5.0) / (low_peak - 5.0)
        return _clip01(0.35 + 0.65 * frac)
    # Gentle decay above 50M (at 500M log10=8.70 -> ~0.78, at 2B -> ~0.55)
    return _clip01(math.exp(-((log_v - high_peak) / 2.0) ** 2))


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0.0, 1.0] for non-negative acoustic vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.5
    cos = float(np.dot(a, b) / (na * nb))
    return _clip01(cos)


def decode_wav_bytes_to_mono_pcm(wav_bytes: bytes, target_sr: int = TARGET_SR) -> np.ndarray | None:
    """Decode WAV bytes directly with Python stdlib wave + numpy (no ffmpeg required)."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(min(n_frames, sr * 30))
    except (wave.Error, EOFError, ValueError):
        return None

    if not raw or sr <= 0:
        return None

    if sampwidth == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return None

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    if sr != target_sr and len(samples) > 16:
        gcd = math.gcd(int(sr), int(target_sr))
        up = int(target_sr // gcd)
        down = int(sr // gcd)
        samples = resample_poly(samples, up, down).astype(np.float32)

    return samples.astype(np.float32, copy=False)


def decode_audio_to_mono_pcm(
    audio_bytes: bytes | None = None,
    *,
    audio_url: str | None = None,
    target_sr: int = TARGET_SR,
    ffmpeg_bin: str | None = None,
) -> np.ndarray | None:
    """Decode WAV natively or MP3/URL via in-memory ffmpeg pipe (zero temp files).

    Gracefully returns None when ffmpeg is unavailable or decoding fails.
    """
    if audio_bytes is not None and len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF":
        pcm = decode_wav_bytes_to_mono_pcm(audio_bytes, target_sr=target_sr)
        if pcm is not None and len(pcm) >= 512:
            return pcm

    ffmpeg_path = ffmpeg_bin if ffmpeg_bin is not None else shutil.which("ffmpeg")
    if not ffmpeg_path:
        return None

    source_arg = "pipe:0" if audio_bytes is not None else audio_url
    if not source_arg:
        return None

    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        source_arg,
        "-t",
        "30",
        "-ac",
        "1",
        "-ar",
        str(int(target_sr)),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=audio_bytes if audio_bytes is not None else None,
            capture_output=True,
            timeout=12,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("ffmpeg pipe decode failed cleanly: %s", exc)
        return None

    if proc.returncode != 0 or len(proc.stdout) < 2048:
        return None

    usable = (len(proc.stdout) // 4) * 4
    pcm = np.frombuffer(proc.stdout[:usable], dtype=np.float32)
    return pcm.copy()


def analyze_pcm_a1_c4(
    samples: np.ndarray,
    *,
    sr: int = TARGET_SR,
    deezer_bpm: float | None = None,
    deezer_gain: float | None = None,
) -> tuple[np.ndarray, float]:
    """Compute real 12-dimensional acoustic MIR features (A1..C4) from mono float32 PCM.

    Returns (vec12, estimated_bpm). Designed to execute in <0.1s per 30s clip on ARM64.
    """
    y = np.asarray(samples, dtype=np.float64).ravel()
    if len(y) < 1024:
        return np.full(12, 0.5, dtype=np.float64), float(deezer_bpm or 118.0)

    # Trim silence/normalize amplitude
    peak = float(np.max(np.abs(y)))
    if peak > 1e-8:
        y = y / peak

    nperseg = 2048
    noverlap = 1536  # hop = 512 (~23.2ms @ 22050Hz, ~43.07 frames/sec)
    hop = nperseg - noverlap
    freqs, _times, Zxx = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)
    mag = np.abs(Zxx)
    power = mag * mag
    frame_rate = float(sr) / float(hop)

    if mag.shape[1] < 8:
        return np.full(12, 0.5, dtype=np.float64), float(deezer_bpm or 118.0)

    # ---------------- Layer A: Rhythm & Groove ----------------
    # Positive spectral flux onset envelope
    log_mag = np.log1p(10.0 * mag)
    flux = np.maximum(0.0, np.diff(log_mag, axis=1)).mean(axis=0)
    flux_centered = flux - flux.mean()
    flux_energy = float(np.dot(flux_centered, flux_centered))

    est_bpm = float(deezer_bpm) if (deezer_bpm is not None and deezer_bpm > 0.0) else 0.0
    beat_clarity = 0.25
    syncopation_index = 0.35

    if flux_energy > 1e-10 and len(flux_centered) > 20:
        ac_full = np.correlate(flux_centered, flux_centered, mode="full")
        ac = ac_full[len(flux_centered) - 1 :] / (flux_energy + 1e-12)

        min_lag = max(2, round(frame_rate * 60.0 / 185.0))
        max_lag = min(len(ac) - 1, round(frame_rate * 60.0 / 58.0))
        if max_lag > min_lag:
            search_ac = ac[min_lag : max_lag + 1]
            best_rel = int(np.argmax(search_ac))
            best_lag = min_lag + best_rel
            peak_val = float(search_ac[best_rel])
            # Clarify periodic beat strength in [0, 1]
            beat_clarity = _clip01(peak_val * 1.15)
            if est_bpm <= 0.0:
                est_bpm = 60.0 * frame_rate / float(best_lag)

            # Syncopation / tresillo (3-3-2) ratio: compare energy at off-beat/tresillo lags
            # (3/8, 1/2, 5/8, 3/4 of a 2-beat/1-bar span) against the primary beat lag
            bar_lag = min(len(ac) - 1, best_lag * 2)
            off_positions = [
                round(best_lag * 0.5),
                round(bar_lag * 3.0 / 8.0),
                round(bar_lag * 6.0 / 8.0),
            ]
            valid_offs = [max(0.0, float(ac[p])) for p in off_positions if 1 <= p < len(ac)]
            if valid_offs:
                off_mean = float(np.mean(valid_offs))
                syncopation_index = _clip01(off_mean / (abs(peak_val) + 0.25))

    if est_bpm <= 0.0:
        est_bpm = 118.0
    a1_bpm = bpm_to_norm(est_bpm)

    # Sub-bass drive (20-120 Hz energy ratio)
    total_power = float(power.sum()) + 1e-12
    sub_mask = (freqs >= 20.0) & (freqs <= 120.0)
    sub_ratio = float(power[sub_mask].sum()) / total_power
    a4_sub_bass = _clip01(math.pow(max(0.0, sub_ratio), 0.45))

    # ---------------- Layer B: Harmony & Tonality ----------------
    chroma_mat = _get_chroma_matrix(nperseg, sr)
    chroma_frames = chroma_mat @ mag  # shape: (12, T)
    chroma_mean = chroma_frames.mean(axis=1)
    c_sum = float(chroma_mean.sum())
    if c_sum > 1e-12:
        chroma_prob = chroma_mean / c_sum
        c_centered = chroma_prob - chroma_prob.mean()
        c_norm = float(np.linalg.norm(c_centered))
        if c_norm > 1e-12:
            c_unit = c_centered / c_norm
            max_maj = float(np.max(_KS_MAJOR_ROT @ c_unit))
            max_min = float(np.max(_KS_MINOR_ROT @ c_unit))
            b1_major_minor = _clip01(0.5 + 0.5 * math.tanh((max_maj - max_min) * 3.0))
        else:
            b1_major_minor = 0.5
        # Normalized Shannon entropy in [0, 1]
        p_pos = chroma_prob[chroma_prob > 1e-12]
        entropy = float(-np.sum(p_pos * np.log(p_pos))) / math.log(12.0)
        b2_harmonic_entropy = _clip01(entropy)
    else:
        b1_major_minor = 0.5
        b2_harmonic_entropy = 0.5

    # Chord change rate: cosine distance between consecutive ~0.5s averaged chroma windows
    win_frames = max(2, round(0.5 * frame_rate))
    n_wins = chroma_frames.shape[1] // win_frames
    if n_wins >= 2:
        trimmed = chroma_frames[:, : n_wins * win_frames].reshape(12, n_wins, win_frames).mean(axis=2)
        norms = np.linalg.norm(trimmed, axis=0) + 1e-9
        normed = trimmed / norms
        cos_adj = np.sum(normed[:, :-1] * normed[:, 1:], axis=0)
        dists = np.clip(1.0 - cos_adj, 0.0, 1.0)
        b3_chord_change = _clip01(float(np.mean(dists)) * 2.5)
    else:
        b3_chord_change = 0.35

    # Lightweight HPSS via 1D median filters along time (harmonic) vs frequency (percussive)
    h_spec = median_filter(mag, size=(1, 13))
    p_spec = median_filter(mag, size=(13, 1))
    h_energy = float(np.sum(h_spec * h_spec))
    p_energy = float(np.sum(p_spec * p_spec))
    b4_hpr = _clip01(h_energy / (h_energy + p_energy + 1e-12))

    # ---------------- Layer C: Timbre, Loudness & Structure ----------------
    frame_mag_sum = np.sum(mag, axis=0) + 1e-12
    centroid_hz = float(np.mean((freqs[:, None] * mag).sum(axis=0) / frame_mag_sum))
    # Musical brightness spans roughly 400Hz-4500Hz
    c1_centroid = _clip01((centroid_hz - 300.0) / 4200.0)

    # Spectral flatness (geometric mean / arithmetic mean of power spectrum)
    eps = 1e-10
    geo_mean = np.exp(np.mean(np.log(power + eps), axis=0))
    arith_mean = np.mean(power + eps, axis=0)
    flatness = float(np.mean(geo_mean / arith_mean))
    c2_flatness = _clip01(math.pow(max(0.0, flatness), 0.35))

    # Loudness & dynamic punch: Deezer gain (-15..-3 dB) + frame RMS contrast
    frame_rms = np.sqrt(np.mean(power, axis=0))
    rms_mean = float(np.mean(frame_rms)) + 1e-9
    rms_cv = float(np.std(frame_rms)) / rms_mean  # dynamic contrast
    dyn_score = _clip01(rms_cv / 0.85)
    if deezer_gain is not None:
        # Deezer gain is replay-gain attenuation (more negative = louder mastered track, e.g. -11dB loud, -4dB quiet)
        loud_score = _clip01((-float(deezer_gain) - 3.0) / 10.0)
        c3_loudness_punch = _clip01(0.55 * loud_score + 0.45 * dyn_score)
    else:
        c3_loudness_punch = _clip01(0.35 + 0.65 * dyn_score)

    # Vocal / melody band presence (300-3400 Hz)
    vocal_mask = (freqs >= 300.0) & (freqs <= 3400.0)
    vocal_ratio = float(power[vocal_mask].sum()) / total_power
    c4_vocal_presence = _clip01(math.pow(max(0.0, vocal_ratio), 0.55))

    vec12 = np.array(
        [
            a1_bpm,
            beat_clarity,
            syncopation_index,
            a4_sub_bass,
            b1_major_minor,
            b2_harmonic_entropy,
            b3_chord_change,
            b4_hpr,
            c1_centroid,
            c2_flatness,
            c3_loudness_punch,
            c4_vocal_presence,
        ],
        dtype=np.float64,
    )
    return vec12, round(float(est_bpm), 1)


def estimate_a1_c4_from_metadata(
    deezer_meta: DeezerTrackMeta | None = None,
    track: TrackEntity | None = None,
) -> tuple[np.ndarray, float]:
    """Graceful fallback when ffmpeg/preview PCM is unavailable: uses Deezer BPM/gain/rank + track duration."""
    bpm = 0.0
    gain: float | None = None
    dur = 195
    if deezer_meta is not None:
        bpm = float(deezer_meta.bpm or 0.0)
        gain = float(deezer_meta.gain)
        if deezer_meta.duration > 0:
            dur = int(deezer_meta.duration)
    if track is not None:
        if bpm <= 0.0 and track.deezer_bpm:
            bpm = float(track.deezer_bpm)
        if gain is None and track.deezer_gain is not None:
            gain = float(track.deezer_gain)
        if track.duration_seconds:
            dur = int(track.duration_seconds)

    eff_bpm = bpm if bpm > 0.0 else 118.0
    a1 = bpm_to_norm(eff_bpm)
    # Higher BPM & loud mastering correlate gently with tighter beat clarity & sub-bass drive
    loud_score = _clip01((-gain - 3.0) / 10.0) if gain is not None else 0.55
    dur_focus = _clip01(math.exp(-((float(dur) - 195.0) / 100.0) ** 2))
    a2 = _clip01(0.45 + 0.25 * loud_score + 0.12 * dur_focus)
    a3 = _clip01(0.42 + 0.18 * (1.0 - abs(a1 - 0.50)))
    a4 = _clip01(0.38 + 0.25 * loud_score)

    b1 = 0.52
    b2 = _clip01(0.50 + 0.12 * dur_focus)
    b3 = _clip01(0.46 + 0.10 * a1)
    b4 = 0.56

    c1 = _clip01(0.42 + 0.15 * a1)
    c2 = _clip01(0.28 + 0.18 * loud_score)
    c3 = _clip01(0.25 + 0.70 * loud_score)
    c4 = _clip01(0.52 + 0.10 * dur_focus)

    return np.array([a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4], dtype=np.float64), round(eff_bpm, 1)


def summarize_4_layers(
    mir_vec: np.ndarray,
    centroid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute human-readable 4-layer match percentages for UI display (リズム / 和声 / 音色 / カルト＆シック)."""
    v = np.asarray(mir_vec, dtype=np.float64)
    if len(v) < MIR_DIM:
        v = np.pad(v, (0, MIR_DIM - len(v)), constant_values=0.5)

    def _layer_match(sl: slice) -> float:
        if centroid is not None and len(centroid) >= MIR_DIM:
            diff = float(np.mean(np.abs(v[sl] - centroid[sl])))
            return round(_clip01(1.0 - diff * 1.4), 3)
        return round(_clip01(float(np.mean(v[sl]))), 3)

    rhythm = _layer_match(slice(0, 4))
    harmony = _layer_match(slice(4, 8))
    timbre = _layer_match(slice(8, 12))
    cult_chic = round(_clip01(float(0.40 * v[12] + 0.25 * v[13] + 0.15 * v[14] + 0.20 * v[15])), 3)

    return {
        "rhythm": {"key": "layer_a", "label_ja": "リズム＆グルーヴ", "score": rhythm},
        "harmony": {"key": "layer_b", "label_ja": "和声＆調性", "score": harmony},
        "timbre": {"key": "layer_c", "label_ja": "音色＆音圧", "score": timbre},
        "cult_chic": {"key": "layer_d", "label_ja": "カルト＆シック", "score": cult_chic},
    }


class AudioMIRExtractor:
    """Extracts and caches 16-dimensional real MIR + collaborative graph + curation vectors."""

    def __init__(
        self,
        *,
        db: RadioDatabase | None = None,
        deezer_client: DeezerClient | None = None,
        ffmpeg_bin: str | None = None,
        cult_chic_anchor_a1_c4: np.ndarray | None = None,
    ) -> None:
        self.db = db
        self.deezer = deezer_client
        self.ffmpeg_bin = ffmpeg_bin
        self._cult_chic_anchor = (
            np.asarray(cult_chic_anchor_a1_c4, dtype=np.float64)
            if cult_chic_anchor_a1_c4 is not None
            else DEFAULT_CULT_CHIC_A1_C4.copy()
        )
        if cult_chic_anchor_a1_c4 is None and db is not None:
            saved = db.get_station_profile("__cult_chic_anchor__")
            if saved and len(saved.get("centroid", [])) >= 12:
                self._cult_chic_anchor = np.asarray(saved["centroid"][:12], dtype=np.float64)

    @property
    def cult_chic_anchor(self) -> np.ndarray:
        return self._cult_chic_anchor

    def set_cult_chic_anchor(self, anchor_a1_c4: np.ndarray) -> None:
        arr = np.asarray(anchor_a1_c4, dtype=np.float64).ravel()[:12]
        if len(arr) == 12:
            self._cult_chic_anchor = arr.copy()

    def extract_a1_c4_for_track(
        self,
        artist: str,
        title: str,
        *,
        video_id: str | None = None,
        audio_bytes: bytes | None = None,
        deezer_meta: DeezerTrackMeta | None = None,
        track: TrackEntity | None = None,
        allow_network: bool = True,
    ) -> tuple[np.ndarray, float, DeezerTrackMeta | None, str]:
        """Compute or load cached 12-dim A1..C4 acoustic features + measured BPM + source tag."""
        canonical_key = normalize_track_key(artist, title)

        # 1. Fast path: explicit audio bytes passed directly (e.g. WAV test or pre-fetched MP3)
        if audio_bytes is not None:
            pcm = decode_audio_to_mono_pcm(audio_bytes, ffmpeg_bin=self.ffmpeg_bin)
            if pcm is not None:
                vec12, est_bpm = analyze_pcm_a1_c4(
                    pcm,
                    deezer_bpm=deezer_meta.bpm if deezer_meta else None,
                    deezer_gain=deezer_meta.gain if deezer_meta else None,
                )
                return vec12, est_bpm, deezer_meta, "mir_pcm"
            vec12, est_bpm = estimate_a1_c4_from_metadata(deezer_meta, track)
            return vec12, est_bpm, deezer_meta, "deezer_metadata" if deezer_meta else "fallback"

        # 2. SQLite cache lookup
        if self.db is not None and deezer_meta is None:
            cached = self.db.get_audio_feature_cache(canonical_key=canonical_key, video_id=video_id)
            if cached is not None and cached.get("mir_a1_c4") and len(cached["mir_a1_c4"]) == 12:
                meta = (
                    DeezerTrackMeta(
                        deezer_id=int(cached["deezer_id"]),
                        artist=str(cached.get("artist") or artist),
                        title=str(cached.get("title") or title),
                        bpm=float(cached.get("bpm") or 0.0),
                        gain=float(cached.get("gain") if cached.get("gain") is not None else -8.0),
                        rank=int(cached.get("rank") or 0),
                        duration=int(cached.get("duration") or 0),
                        preview_url=cached.get("preview_url"),
                        mir_a1_c4=list(cached["mir_a1_c4"]),
                        analysis_source=str(cached.get("analysis_source") or "deezer_metadata"),
                    )
                    if cached.get("deezer_id") is not None
                    else None
                )
                est_bpm = float(cached.get("bpm") or 118.0)
                return (
                    np.asarray(cached["mir_a1_c4"], dtype=np.float64),
                    est_bpm,
                    meta,
                    str(cached.get("analysis_source") or "cache"),
                )

        # 3. Resolve Deezer metadata if online allowed
        meta = deezer_meta
        if meta is None and allow_network and self.deezer is not None:
            meta = self.deezer.resolve_track(artist, title, video_id=video_id, use_cache=True)

        # 4. Attempt 30s preview PCM analysis via ffmpeg pipe (no temp files)
        source_tag = "deezer_metadata" if meta is not None else "fallback"
        vec12: np.ndarray | None = None
        est_bpm = float(meta.bpm) if (meta is not None and meta.bpm > 0.0) else 0.0

        if meta is not None and meta.preview_url and allow_network and self.deezer is not None:
            ffmpeg_avail = self.ffmpeg_bin if self.ffmpeg_bin is not None else shutil.which("ffmpeg")
            if ffmpeg_avail:
                preview_bytes = self.deezer.fetch_preview_bytes(meta.preview_url)
                if preview_bytes:
                    pcm = decode_audio_to_mono_pcm(preview_bytes, ffmpeg_bin=ffmpeg_avail)
                    if pcm is not None:
                        vec12, est_bpm = analyze_pcm_a1_c4(
                            pcm,
                            deezer_bpm=meta.bpm if meta.bpm > 0.0 else None,
                            deezer_gain=meta.gain,
                        )
                        source_tag = "mir_pcm"

        if vec12 is None:
            vec12, est_bpm = estimate_a1_c4_from_metadata(meta, track)

        # 5. Persist computed A1..C4 into SQLite cache so the same song is never re-analyzed
        if self.db is not None and (meta is not None or source_tag == "mir_pcm"):
            self.db.upsert_audio_feature_cache(
                canonical_key=canonical_key,
                video_id=video_id,
                deezer_id=meta.deezer_id if meta else None,
                artist=meta.artist if meta else artist,
                title=meta.title if meta else title,
                bpm=est_bpm,
                gain=meta.gain if meta else -8.0,
                rank=meta.rank if meta else 0,
                duration=meta.duration if meta else (track.duration_seconds or 0 if track else 0),
                preview_url=meta.preview_url if meta else None,
                mir_a1_c4=[round(float(x), 5) for x in vec12],
                analysis_source=source_tag,
            )

        return vec12, est_bpm, meta, source_tag

    def extract_16d_vector(
        self,
        track: TrackEntity,
        *,
        audio_bytes: bytes | None = None,
        deezer_meta: DeezerTrackMeta | None = None,
        allow_network: bool = False,
    ) -> tuple[np.ndarray, float, str]:
        """Return full 16-dimensional MIR + Graph + Sweetspot + Cult-Chic vector and measured BPM."""
        # Use precomputed 16-dim vector cached on TrackEntity when no override is supplied
        if audio_bytes is None and deezer_meta is None and track.mir_vector and len(track.mir_vector) == MIR_DIM:
            vec = np.asarray(track.mir_vector, dtype=np.float64).copy()
            # Recompute dynamic Layer D fields in case co-occurrence or cult-chic anchor updated
            vec[12] = compute_cooccurrence_score(track.cooccurrence_count)
            if track.deezer_rank is not None:
                vec[13] = compute_deezer_rank_sweetspot(track.deezer_rank)
            if track.view_count is not None:
                vec[14] = compute_yt_views_sweetspot(track.view_count)
            vec[15] = compute_cosine_similarity(vec[:12], self._cult_chic_anchor)
            bpm = float(track.deezer_bpm) if track.deezer_bpm else round(60.0 + 120.0 * float(vec[0]), 1)
            return vec, bpm, "track_cache"

        vec12, est_bpm, meta, source_tag = self.extract_a1_c4_for_track(
            track.artist,
            track.title,
            video_id=track.video_id,
            audio_bytes=audio_bytes,
            deezer_meta=deezer_meta,
            track=track,
            allow_network=allow_network,
        )

        d1 = compute_cooccurrence_score(track.cooccurrence_count)
        d2 = compute_deezer_rank_sweetspot(meta.rank if meta else track.deezer_rank)
        d3 = compute_yt_views_sweetspot(track.view_count)
        d4 = compute_cosine_similarity(vec12, self._cult_chic_anchor)

        vec16 = np.concatenate([vec12, np.array([d1, d2, d3, d4], dtype=np.float64)])
        return vec16, est_bpm, source_tag
