"""Vocal activity detection.

Two vocals playing over each other is the most audible mistake in a mix —
more obvious than a slightly loose beatmatch, and the longer the blend the
worse it gets. Knowing where a track is singing lets the engine overlap a
vocal with an instrumental passage instead of stacking two of them.

There is no separation model here; what is needed is not isolated vocals but
a per-moment answer to "is someone singing". Three absolute measurements
answer it between them, and each rules out a different impostor:

  band fraction   how much harmonic energy sits where voice lives. A bass
                  line scores 0.00 and drums 0.23 against a voice's 0.88.
  timbre movement how fast the spectral envelope changes. Vowels move; a
                  sustained pad does not — 1.8 against a voice's 5.2, which
                  is the only one of the three that separates those two.
  harmonic share  harmonic against percussive energy in-band, which drops
                  percussion to 0.09.

All three are absolute. An earlier version normalised each per track, which
made every value relative to that track's own range and scored a plain bass
line higher than singing.
"""

import librosa
import numpy as np

HOP = 512
N_FFT = 2048

# Roughly the telephone band. Sung fundamentals and the first few formants
# sit here; kick and bass are below it, cymbals and air above.
VOCAL_LOW, VOCAL_HIGH = 300.0, 3400.0

# Below this the spectral envelope is holding still, which is a pad or a
# held synth note rather than anything articulating words.
MOVEMENT_FLOOR, MOVEMENT_CEILING = 2.0, 5.0


def vocal_activity(y: np.ndarray, sr: int) -> dict:
    """A 0..1 curve of how likely it is that something is singing."""
    if y.size < sr:
        return {"times": np.zeros(0), "activity": np.zeros(0)}

    harmonic = librosa.effects.harmonic(y, margin=2.0)
    percussive = librosa.effects.percussive(y, margin=2.0)

    spec = np.abs(librosa.stft(harmonic, n_fft=N_FFT, hop_length=HOP))
    perc = np.abs(librosa.stft(percussive, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    band = (freqs >= VOCAL_LOW) & (freqs <= VOCAL_HIGH)

    frames = min(spec.shape[1], perc.shape[1])
    spec, perc = spec[:, :frames], perc[:, :frames]

    in_band = spec[band].sum(axis=0)
    band_fraction = in_band / (spec.sum(axis=0) + 1e-9)
    harmonic_share = in_band / (in_band + perc[band].sum(axis=0) + 1e-9)

    mfcc = librosa.feature.mfcc(y=harmonic, sr=sr, n_mfcc=13,
                               hop_length=HOP, fmin=VOCAL_LOW, fmax=VOCAL_HIGH)
    movement = np.mean(np.abs(np.diff(mfcc, axis=1)), axis=0)
    movement = np.concatenate([[movement[0] if movement.size else 0.0], movement])[:frames]
    movement = np.clip(
        (movement - MOVEMENT_FLOOR) / (MOVEMENT_CEILING - MOVEMENT_FLOOR), 0.0, 1.0)

    # Multiplied, not summed: any one of the three failing should veto the
    # frame, since each rules out a different thing that is not a voice.
    activity = _smooth(band_fraction * harmonic_share * movement, 21)

    times = librosa.frames_to_time(np.arange(frames), sr=sr, hop_length=HOP)
    return {"times": times, "activity": np.clip(activity, 0.0, 1.0)}


def segment_vocal_ratio(segments: list[dict], times: np.ndarray,
                        activity: np.ndarray) -> list[dict]:
    """Attach a mean vocal score to each structural section."""
    if len(times) == 0:
        return segments
    for segment in segments:
        mask = (times >= segment["start"]) & (times < segment["end"])
        segment["vocal"] = round(float(np.mean(activity[mask])), 4) if np.any(mask) else 0.0
    return segments


def vocal_between(start: float, end: float, times: np.ndarray,
                  activity: np.ndarray) -> float:
    """Mean vocal activity over a span — how much singing a blend would cover."""
    if len(times) == 0 or end <= start:
        return 0.0
    mask = (times >= start) & (times < end)
    return float(np.mean(activity[mask])) if np.any(mask) else 0.0


def sample_curve(times: np.ndarray, activity: np.ndarray, step: float = 1.0) -> list[dict]:
    if len(times) == 0:
        return []
    grid = np.arange(0.0, float(times[-1]), step)
    values = np.interp(grid, times, activity)
    return [{"t": round(float(t), 2), "vocal": round(float(v), 4)}
            for t, v in zip(grid, values)]


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="same")
