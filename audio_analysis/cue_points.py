"""Entry and exit cue point detection.

Candidates are phrase-aligned downbeats. Each is scored on how well it suits
its role, and the ranked top few are returned.

Entry = where the incoming track starts, so it wants to be early, phrase
aligned, and quiet enough to sit under the track still playing.
Exit  = where the outgoing track starts giving way, so it wants to be late,
phrase aligned, and ideally right before the track's energy falls away.
"""

import numpy as np

# How many bars of runway a cue needs before the track ends.
_MIN_END_HEADROOM_BARS = 8
_TOP_N = 5


def detect_cue_points(
    duration: float,
    tempo: float,
    downbeats: list[float],
    energy_times: np.ndarray,
    energy_values: np.ndarray,
    segments: list[dict],
) -> dict:
    if not downbeats or tempo <= 0 or duration <= 0:
        return {"entry": [], "exit": []}

    bar_seconds = (60.0 / tempo) * 4.0
    boundaries = [s["start"] for s in segments[1:]] if len(segments) > 1 else []

    candidates = []
    for bar_index, t in enumerate(downbeats):
        if bar_index % 4 != 0:  # only consider 4-bar group starts
            continue
        if t < 0.5 or t > duration - bar_seconds * _MIN_END_HEADROOM_BARS:
            continue
        candidates.append((bar_index, float(t)))

    if not candidates:
        return {"entry": [], "exit": []}

    entries = [_score(b, t, "entry", duration, bar_seconds, boundaries,
                      energy_times, energy_values) for b, t in candidates]
    exits = [_score(b, t, "exit", duration, bar_seconds, boundaries,
                    energy_times, energy_values) for b, t in candidates]

    return {
        "entry": _rank(entries),
        "exit": _rank(exits),
    }


def _rank(cues: list[dict]) -> list[dict]:
    ranked = sorted(cues, key=lambda c: c["score"], reverse=True)[:_TOP_N]
    for i, c in enumerate(ranked):
        c["rank"] = i + 1
    return ranked


def _score(bar_index, t, kind, duration, bar_seconds, boundaries,
           energy_times, energy_values) -> dict:
    position = t / duration
    energy_here = _energy_at(t, energy_times, energy_values)
    energy_before = _energy_window(t - 4 * bar_seconds, t, energy_times, energy_values)
    energy_after = _energy_window(t, t + 4 * bar_seconds, energy_times, energy_values)

    # Phrase alignment: 16-bar boundaries are the safest place to bring a
    # track in or out, 8-bar next, plain 4-bar acceptable.
    if bar_index % 16 == 0:
        phrase = 1.0
        phrase_label = "16-bar"
    elif bar_index % 8 == 0:
        phrase = 0.85
        phrase_label = "8-bar"
    else:
        phrase = 0.6
        phrase_label = "4-bar"

    # Sitting on a structural boundary is a strong signal either way.
    near_boundary = any(abs(t - b) <= bar_seconds for b in boundaries)
    boundary = 1.0 if near_boundary else 0.4

    if kind == "entry":
        position_fit = _bell(position, center=0.12, width=0.18)
        energy_fit = 1.0 - min(1.0, energy_here)          # quiet blends better
        shape_fit = _clamp01(0.5 + (energy_after - energy_before) * 2)  # rising
    else:
        position_fit = _bell(position, center=0.78, width=0.20)
        energy_fit = min(1.0, energy_here)                # still driving
        shape_fit = _clamp01(0.5 + (energy_before - energy_after) * 2)  # falling

    score = (
        0.30 * phrase
        + 0.22 * position_fit
        + 0.20 * energy_fit
        + 0.16 * shape_fit
        + 0.12 * boundary
    )

    reasons = [phrase_label]
    if near_boundary:
        reasons.append("section boundary")
    if kind == "entry" and energy_after > energy_before + 0.05:
        reasons.append("energy rising")
    if kind == "exit" and energy_before > energy_after + 0.05:
        reasons.append("energy dropping")

    return {
        "time": round(float(t), 3),
        "bar": int(bar_index),
        "type": kind,
        "score": round(float(_clamp01(score)), 4),
        "energy": round(float(energy_here), 4),
        "label": ", ".join(reasons),
    }


def _energy_at(t, times, values) -> float:
    if len(times) == 0:
        return 0.0
    return float(np.interp(t, times, values))


def _energy_window(start, end, times, values) -> float:
    if len(times) == 0 or end <= start:
        return 0.0
    mask = (times >= max(0.0, start)) & (times < end)
    if not np.any(mask):
        return _energy_at((start + end) / 2, times, values)
    return float(np.mean(values[mask]))


def _bell(x: float, center: float, width: float) -> float:
    return float(np.exp(-((x - center) ** 2) / (2 * width**2)))


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))
