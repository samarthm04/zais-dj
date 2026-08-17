"""Tempo, beat grid, downbeats and musical key.

Assumes 4/4, which holds for essentially all mainstream pop and dance music.
"""

import librosa
import numpy as np

# Krumhansl-Kessler key profiles, the standard correlation templates.
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Camelot wheel. Index is the pitch class, value the Camelot code.
_CAMELOT_MAJOR = {
    "B": "1B", "F#": "2B", "C#": "3B", "G#": "4B", "D#": "5B", "A#": "6B",
    "F": "7B", "C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B",
}
_CAMELOT_MINOR = {
    "G#": "1A", "D#": "2A", "A#": "3A", "F": "4A", "C": "5A", "G": "6A",
    "D": "7A", "A": "8A", "E": "9A", "B": "10A", "F#": "11A", "C#": "12A",
}


def detect_beats(y: np.ndarray, sr: int) -> dict:
    """Tempo, beat times and inferred downbeats."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tracked_tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, trim=False
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    duration = len(y) / sr

    # Beat tracking only covers stretches with audible transients, so an
    # ambient intro or outro comes back bare. Pop tempo is effectively
    # constant, so fit one uniform grid and lay it across the whole track.
    fitted = _fit_grid(beat_times, duration, float(np.atleast_1d(tracked_tempo)[0]))
    if isinstance(fitted[0], np.ndarray):  # too few beats to fit; grid is already built
        beat_times, tempo = fitted
    else:
        period, offset = _refine_grid(y, sr, fitted[0], fitted[1], duration)
        beat_times = _build_grid(period, offset, duration)
        tempo = 60.0 / period

    downbeat_times, phase = _infer_downbeats(y, sr, beat_times)

    return {
        "tempo": round(tempo, 2),
        "beat_times": [round(float(t), 4) for t in beat_times],
        "downbeat_times": [round(float(t), 4) for t in downbeat_times],
        "beats_per_bar": 4,
        "downbeat_phase": phase,
        "onset_env": onset_env,
    }


def _fit_grid(beat_times: np.ndarray, duration: float, fallback_tempo: float):
    """Fit one uniform beat grid to the tracked beats and span the track with it.

    Tracked beat times are quantised to the STFT hop (~23ms), so consecutive
    intervals alternate between neighbouring hop counts; taking their median
    locks onto whichever is more common and biases the tempo by a couple of
    BPM. A least-squares fit of time against beat index averages that
    quantisation out, and because it is a straight line it cannot accumulate
    drift the way repeated addition of a slightly-wrong period does.
    """
    period = 60.0 / fallback_tempo if fallback_tempo > 0 else 0.5
    if len(beat_times) < 8:
        grid = np.arange(0.0, duration, period)
        return grid, (60.0 / period if period > 0 else fallback_tempo)

    diffs = np.diff(beat_times)
    median = float(np.median(diffs))
    # Drop gaps left by beats the tracker missed; they are whole multiples
    # of the period and would drag the average up.
    inliers = diffs[(diffs > 0.6 * median) & (diffs < 1.4 * median)]
    period = float(np.mean(inliers)) if len(inliers) else median

    # Index each tracked beat, then fit time = period * index + offset.
    index = np.round((beat_times - beat_times[0]) / period)
    design = np.vstack([index, np.ones_like(index)]).T
    period, offset = np.linalg.lstsq(design, beat_times, rcond=None)[0]
    if not np.isfinite(period) or period <= 0:
        return np.arange(0.0, duration, 60.0 / fallback_tempo), fallback_tempo

    return period, offset


def _refine_grid(y, sr, period: float, offset: float, duration: float):
    """Lock the grid onto the audio by comb search.

    The tracker's own beats are quantised to its analysis hop, which leaves
    the fitted period off by a fraction of a percent. That is inaudible on
    one beat and half a beat wide by the end of a song. Sliding a uniform
    comb over a finer onset envelope and taking the period/phase with the
    most onset energy underneath removes that bias.
    """
    hop = 256
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if env.size == 0 or float(np.max(env)) <= 0:
        return period, offset
    env = env / float(np.max(env))
    env_times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)

    best = (-np.inf, period, offset)
    for candidate in np.linspace(period * 0.97, period * 1.03, 61):
        count = int(duration / candidate)
        if count < 4:
            continue
        for phase in np.linspace(0.0, candidate, 24, endpoint=False):
            grid = phase + candidate * np.arange(count)
            score = float(np.mean(np.interp(grid, env_times, env)))
            if score > best[0]:
                best = (score, candidate, phase)

    return best[1], best[2]


def _build_grid(period: float, offset: float, duration: float) -> np.ndarray:
    first = int(np.floor(-offset / period))
    last = int(np.ceil((duration - offset) / period))
    grid = offset + period * np.arange(first, last + 1)
    return grid[(grid >= 0.0) & (grid < duration)]


def _infer_downbeats(y: np.ndarray, sr: int, beat_times: np.ndarray):
    """Choose which of the four beat positions is beat 1.

    Onset strength alone is a trap here: pop puts a loud snare on 2 and 4,
    which regularly beats the kick on 1. Two cues that do point at beat 1 are
    low-end weight (the kick, and the bass note landing) and harmonic change,
    since chords overwhelmingly change on the downbeat.
    """
    if len(beat_times) < 8:
        return beat_times, 0

    hop = 512
    low_onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, fmax=150, n_mels=16)
    # Snare and hat noise reaches well below 150 Hz, so the low band alone
    # still favours the backbeat. Subtracting the bright band cancels it:
    # cymbals and snare rattle put far more energy up here than a kick does.
    high_onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, fmin=2000)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    harmonic_change = np.concatenate([[0.0], np.sum(np.abs(np.diff(chroma, axis=1)), axis=0)])

    frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop)
    low_at = _unit(_sample(low_onset, frames))
    high_at = _unit(_sample(high_onset, frames))
    harm_at = _unit(_sample(harmonic_change, frames))

    scores = [
        low_at[phase::4].mean()
        - 0.6 * high_at[phase::4].mean()
        + harm_at[phase::4].mean()
        for phase in range(4)
    ]
    phase = int(np.argmax(scores))
    return beat_times[phase::4], phase


def _sample(curve: np.ndarray, frames: np.ndarray) -> np.ndarray:
    if len(curve) == 0:
        return np.zeros(len(frames))
    return curve[np.clip(frames, 0, len(curve) - 1)]


def _unit(arr: np.ndarray) -> np.ndarray:
    span = float(np.max(arr) - np.min(arr))
    if span < 1e-9:
        return np.zeros_like(arr)
    return (arr - float(np.min(arr))) / span


def detect_key(y: np.ndarray, sr: int) -> dict:
    """Correlate the average chroma against major/minor profiles.

    Two things matter more than the profiles themselves. Percussion sprays
    energy across every chroma bin and flattens the picture, so the harmonic
    component is separated out first. And the comparison has to be a real
    correlation with both sides mean-centred: chroma and key profiles are
    both non-negative and broadly similar in shape, so an uncentred cosine
    scores every one of the 24 keys above 0.95 and separates nothing.
    """
    harmonic = librosa.effects.harmonic(y, margin=3.0)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)

    # Normalise per frame so loud passages do not outvote the rest.
    frame_energy = np.sum(chroma, axis=0, keepdims=True)
    chroma = np.divide(chroma, frame_energy, out=np.zeros_like(chroma),
                       where=frame_energy > 1e-9)
    mean_chroma = np.mean(chroma, axis=1)
    if float(np.sum(mean_chroma)) <= 0:
        return {"camelot": None, "name": None, "confidence": 0.0}

    centred = mean_chroma - np.mean(mean_chroma)
    norm = np.linalg.norm(centred)
    if norm < 1e-9:
        return {"camelot": None, "name": None, "confidence": 0.0}

    best = {"score": -np.inf}
    ranked = []
    for tonic in range(12):
        for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            rotated = np.roll(profile, tonic)
            rotated = rotated - np.mean(rotated)
            score = float(np.dot(centred, rotated) / (norm * np.linalg.norm(rotated)))
            ranked.append(score)
            if score > best["score"]:
                best = {"score": score, "tonic": tonic, "mode": mode}

    ranked.sort(reverse=True)
    # Pearson r spans -1..1, so the gap to the runner-up is meaningful now.
    margin = ranked[0] - ranked[1]

    name = _PITCH_NAMES[best["tonic"]]
    table = _CAMELOT_MAJOR if best["mode"] == "major" else _CAMELOT_MINOR
    return {
        "camelot": table[name],
        "name": f"{name} {'major' if best['mode'] == 'major' else 'minor'}",
        "confidence": round(float(min(1.0, max(0.0, margin * 4))), 3),
    }
