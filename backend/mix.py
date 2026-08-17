"""Render one transition between two analyzed tracks.

Takes track A's exit cue and track B's entry cue, matches B's tempo to A's,
lines their downbeats up, and crossfades. Only the window around the
transition is rendered, since that is the part being judged.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from audio_analysis.loader import load_audio

RENDER_SR = 44100

# A DJ would not push a track further than this without it sounding wrong.
COMFORTABLE_STRETCH = 0.06


def render_transition(
    a_path: Path,
    a_result: dict,
    a_time: float,
    b_path: Path,
    b_result: dict,
    b_time: float,
    crossfade_bars: int = 8,
    lead_in_bars: int = 8,
    tail_bars: int = 8,
    match_tempo: bool = True,
    out_path: Path | None = None,
) -> dict:
    tempo_a = float(a_result["tempo"])
    tempo_b = float(b_result["tempo"])

    a, sr = load_audio(a_path, sr=RENDER_SR, mono=False)
    b, _ = load_audio(b_path, sr=RENDER_SR, mono=False)
    a, b = _as_stereo(a), _as_stereo(b)

    # Stretch B onto A's tempo. Rate > 1 speeds up, which pulls every
    # timestamp in B earlier by the same factor.
    rate = tempo_a / tempo_b if match_tempo and tempo_b > 0 else 1.0
    if abs(rate - 1.0) > 0.001:
        b = np.vstack([librosa.effects.time_stretch(ch, rate=rate) for ch in b])
        b_time = b_time / rate
        tempo_b_final = tempo_b * rate
    else:
        rate = 1.0
        tempo_b_final = tempo_b

    bar = (60.0 / tempo_a) * 4.0
    lead_in = lead_in_bars * bar
    crossfade = crossfade_bars * bar
    tail = tail_bars * bar

    # A runs from lead_in before its cue through to the end of the crossfade.
    a_seg = _slice(a, sr, a_time - lead_in, a_time + crossfade)
    # B starts at its own cue and keeps going past the crossfade.
    b_seg = _slice(b, sr, b_time, b_time + crossfade + tail)

    # Match B's level to A's across the blend, the way a DJ sets trim before
    # bringing a track in. Without this the mix dips or jumps whenever the two
    # cue points sit at different energies, which they usually do.
    # Matched over the whole rendered segment rather than just the blend:
    # cue points often sit in quiet passages, and matching there would set the
    # trim off a soft moment and leave the rest of the track jumping.
    a_level = _rms(a_seg)
    b_level = _rms(b_seg)
    gain = float(np.clip(a_level / b_level, 0.25, 4.0)) if b_level > 1e-6 else 1.0
    b_seg = b_seg * gain

    total = int(round((lead_in + crossfade + tail) * sr))
    out = np.zeros((2, total), dtype=np.float32)

    # Equal-power curves; a linear crossfade dips in the middle because the
    # two tracks are not correlated.
    fade_n = int(round(crossfade * sr))
    theta = np.linspace(0.0, np.pi / 2, fade_n, dtype=np.float32)
    fade_out, fade_in = np.cos(theta), np.sin(theta)

    a_env = np.ones(a_seg.shape[1], dtype=np.float32)
    start = max(0, a_seg.shape[1] - fade_n)
    a_env[start:] = fade_out[: a_seg.shape[1] - start]
    _add(out, a_seg * a_env, 0)

    b_env = np.ones(b_seg.shape[1], dtype=np.float32)
    b_env[:fade_n] = fade_in[: b_seg.shape[1]]
    _add(out, b_seg * b_env, int(round(lead_in * sr)))

    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        out = out * (0.99 / peak)

    if out_path is not None:
        sf.write(str(out_path), out.T, sr, subtype="PCM_16")

    return {
        "duration": round(total / sr, 2),
        "tempo_a": round(tempo_a, 2),
        "tempo_b": round(tempo_b, 2),
        "tempo_played": round(tempo_b_final, 2),
        "stretch_percent": round((rate - 1.0) * 100, 2),
        "stretch_comfortable": abs(rate - 1.0) <= COMFORTABLE_STRETCH,
        "crossfade_bars": crossfade_bars,
        "crossfade_seconds": round(crossfade, 2),
        "transition_at": round(lead_in, 2),
        "peak": round(peak, 3),
        "gain_db": round(float(20 * np.log10(gain)), 2) if gain > 0 else 0.0,
    }


def _rms(seg: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(seg)))) if seg.size else 0.0


def _as_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.vstack([y, y])
    if y.shape[0] == 1:
        return np.vstack([y[0], y[0]])
    return y[:2]


def _slice(y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Cut [start, end) in seconds, zero-padding wherever the track runs out."""
    n = int(round((end - start) * sr))
    out = np.zeros((y.shape[0], n), dtype=np.float32)
    src_start = int(round(start * sr))
    src_end = src_start + n

    take_start = max(0, src_start)
    take_end = min(y.shape[1], src_end)
    if take_end > take_start:
        out[:, take_start - src_start : take_end - src_start] = y[:, take_start:take_end]
    return out


def _add(out: np.ndarray, seg: np.ndarray, offset: int) -> None:
    end = min(out.shape[1], offset + seg.shape[1])
    if end > offset:
        out[:, offset:end] += seg[:, : end - offset]
