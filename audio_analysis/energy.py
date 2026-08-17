"""Energy curve.

Loudness alone reads badly on modern pop, which is heavily limited: a quiet
verse and a loud chorus often sit within a couple of dB. Combining RMS with
spectral flux and high-frequency content separates sections much better,
because choruses add drums, cymbals and layered synths rather than raw level.
"""

import librosa
import numpy as np

HOP_LENGTH = 512


def energy_curve(y: np.ndarray, sr: int, onset_env: np.ndarray | None = None) -> dict:
    """Frame-wise energy in 0..1, plus the components it was built from."""
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)

    # Percussive activity: how much broadband transient content there is.
    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    onset_env = _resize(onset_env, len(rms))

    # Brightness: cymbals and open hats push this up in choruses/drops.
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    centroid = _resize(centroid, len(rms))

    rms_n = _normalize(_smooth(rms, 9))
    onset_n = _normalize(_smooth(onset_env, 9))
    centroid_n = _normalize(_smooth(centroid, 9))

    combined = 0.5 * rms_n + 0.3 * onset_n + 0.2 * centroid_n
    combined = _normalize(_smooth(combined, 15))

    return {
        "times": times,
        "energy": combined,
        "rms": rms_n,
        "percussive": onset_n,
        "brightness": centroid_n,
        "hop_length": HOP_LENGTH,
    }


def _resize(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) == n:
        return arr
    if len(arr) == 0:
        return np.zeros(n)
    return np.interp(np.linspace(0, len(arr) - 1, n), np.arange(len(arr)), arr)


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def sample_curve(times: np.ndarray, energy: np.ndarray, step: float = 1.0) -> list[dict]:
    """Downsample to roughly one point per `step` seconds for transport."""
    if len(times) == 0:
        return []
    grid = np.arange(0.0, float(times[-1]), step)
    values = np.interp(grid, times, energy)
    return [{"t": round(float(t), 2), "energy": round(float(v), 4)}
            for t, v in zip(grid, values)]
