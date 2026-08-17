"""Structural segmentation.

Boundaries come from agglomerative clustering over timbre + harmony features,
then get snapped to the nearest downbeat so they land musically rather than
a few hundred milliseconds off.
"""

import librosa
import numpy as np

HOP_LENGTH = 512

# Roughly one section per 20s, kept inside sane bounds for a 3-4 minute pop song.
_TARGET_SECONDS_PER_SEGMENT = 20.0
_MIN_SEGMENTS = 4
_MAX_SEGMENTS = 12


def detect_segments(
    y: np.ndarray,
    sr: int,
    energy_times: np.ndarray,
    energy_values: np.ndarray,
    downbeats: list[float],
) -> list[dict]:
    duration = len(y) / sr
    n_segments = int(round(duration / _TARGET_SECONDS_PER_SEGMENT))
    n_segments = max(_MIN_SEGMENTS, min(_MAX_SEGMENTS, n_segments))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP_LENGTH, n_mfcc=13)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    n = min(mfcc.shape[1], chroma.shape[1])
    features = np.vstack([_stdize(mfcc[:, :n]), _stdize(chroma[:, :n])])

    if features.shape[1] <= n_segments:
        return [{"start": 0.0, "end": round(duration, 3), "label": "whole",
                 "energy": round(float(np.mean(energy_values)), 4)}]

    bounds = librosa.segment.agglomerative(features, n_segments)
    bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=HOP_LENGTH)

    edges = [0.0] + [_snap(t, downbeats) for t in bound_times if 0 < t < duration]
    edges = sorted(set(round(float(t), 3) for t in edges)) + [round(duration, 3)]

    # Nothing shorter than four bars counts as a section. Clustering likes to
    # carve a long fade into slivers, and each spurious edge would otherwise
    # hand out a "section boundary" bonus during cue scoring.
    bar = float(np.median(np.diff(downbeats))) if len(downbeats) > 2 else 2.0
    min_length = max(4.0, bar * 4)
    edges = _drop_short(edges, min_length)

    segments = []
    for start, end in zip(edges, edges[1:]):
        mask = (energy_times >= start) & (energy_times < end)
        mean_energy = float(np.mean(energy_values[mask])) if np.any(mask) else 0.0
        segments.append({"start": start, "end": end, "energy": round(mean_energy, 4)})

    return _label(segments)


def _drop_short(edges: list[float], min_length: float) -> list[float]:
    """Remove interior edges that would leave a segment under `min_length`."""
    kept = [edges[0]]
    for edge in edges[1:-1]:
        if edge - kept[-1] >= min_length:
            kept.append(edge)
    # The final segment can still be short; absorb it into the one before.
    if len(kept) > 1 and edges[-1] - kept[-1] < min_length:
        kept.pop()
    kept.append(edges[-1])
    return kept


def _label(segments: list[dict]) -> list[dict]:
    """Tag each segment by relative energy, with the first and last called out.

    Deliberately generic: these are energy tiers, not claims about verse or
    chorus, which would need lyric or repetition analysis to assert.
    """
    if not segments:
        return segments
    energies = np.array([s["energy"] for s in segments])
    lo, hi = np.percentile(energies, 33), np.percentile(energies, 67)

    for i, seg in enumerate(segments):
        if seg["energy"] <= lo:
            tier = "low"
        elif seg["energy"] >= hi:
            tier = "high"
        else:
            tier = "mid"
        if i == 0:
            seg["label"] = "intro"
        elif i == len(segments) - 1:
            seg["label"] = "outro"
        else:
            seg["label"] = tier
        seg["tier"] = tier
    return segments


def _snap(t: float, downbeats: list[float], max_distance: float = 1.5) -> float:
    """Move `t` to the nearest downbeat if one is close enough."""
    if not downbeats:
        return t
    arr = np.asarray(downbeats)
    idx = int(np.argmin(np.abs(arr - t)))
    return float(arr[idx]) if abs(arr[idx] - t) <= max_distance else t


def _stdize(m: np.ndarray) -> np.ndarray:
    mean = m.mean(axis=1, keepdims=True)
    std = m.std(axis=1, keepdims=True)
    return (m - mean) / (std + 1e-9)
