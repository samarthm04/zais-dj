"""Rhythmic cycle structure: cycle length, sam, taali and khaali.

A beat grid says where the beats are. It does not say which beat the music
lands on, and that is the one a track has to enter on. In Hindustani terms
the cycle is the tala, its first beat is the sam — the point everything
resolves to — the accented positions are taali, and the one deliberately
unaccented position is khaali, where the cycle breathes before returning.

Western pop has the same shape without the vocabulary: a four- or eight-bar
phrase, landing on bar one, with a lift or drop-out partway through. The
detection below is the same either way, because both are asking where the
cycle starts and which positions inside it carry weight.
"""

import librosa
import numpy as np

HOP = 512

# Cycle lengths in beats, all multiples of four.
#
# Odd-length talas — Rupak at 7, Jhaptaal at 10, Dhamar at 14 — are
# deliberately absent. The beat tracker upstream assumes 4/4 and lays a
# four-beat bar over everything, so a genuine seven-beat cycle is already
# mis-gridded before this code sees it; "detecting" one here would be
# reporting an artefact of that assumption. The talas that do survive a 4/4
# grid are Keherwa at 8 and Teentaal at 16, which is most film and pop music
# anyway. Supporting the odd ones properly means meter detection first.
CANDIDATES = (4, 8, 16, 32)

# Named only where the length is unambiguous. This is a length match, not a
# claim about the theka being played.
TALA_NAMES = {8: "Keherwa", 16: "Teentaal"}

# A longer cycle also correlates at its divisors, so a 16-beat cycle scores
# well at 4 and 8 too. Prefer the longest length that explains the track
# about as well as the best one does.
LONGER_TOLERANCE = 0.9


def detect_cycle(y: np.ndarray, sr: int, beat_times, onset_env=None) -> dict:
    """Find the cycle length, where its sam falls, and its taali/khaali map."""
    beat_times = np.asarray(beat_times, dtype=float)
    if len(beat_times) < 24:
        return _fallback("too few beats to find a cycle")

    frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP)

    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    low = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, fmax=150, n_mels=16)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)

    accent = _at(onset_env, frames)
    bass = _at(low, frames)
    beat_chroma = chroma[:, np.clip(frames, 0, chroma.shape[1] - 1)]
    harmonic = np.concatenate(
        [[0.0], np.sum(np.abs(np.diff(beat_chroma, axis=1)), axis=0)]
    )

    scored = {}
    for period in CANDIDATES:
        if len(accent) < period * 3:
            continue
        score, similarity, contrast = _score_period(accent, beat_chroma, period)
        scored[period] = {"score": score, "similarity": similarity, "contrast": contrast}

    if not scored:
        return _fallback("track too short for cycle detection")

    top = max(scored.values(), key=lambda s: s["score"])["score"]
    cutoff = top * LONGER_TOLERANCE if top > 0 else top
    period = max(p for p, s in scored.items() if s["score"] >= cutoff)
    best = scored[period]

    # Only bar starts can carry sam, taali or khaali: within a 4/4 grid the
    # cycle's weight falls on bar lines, never mid-bar.
    positions = np.arange(0, period, 4)

    # Sam is where the cycle lands — the bar carrying the most low-end weight
    # and the most harmonic change, since chords turn over on the downbeat.
    weight = _unit(bass) + _unit(harmonic)
    landing = np.array([weight[p::period].mean() for p in positions])
    sam = int(positions[int(np.argmax(landing))])

    profile = np.array([_unit(accent)[phase::period].mean() for phase in range(period)])
    rotated = np.roll(profile, -sam)  # index 0 is now sam

    # Khaali sits at the middle of the cycle — beat 9 of 16 in Teentaal,
    # beat 5 of 8 in Keherwa — so look for the emptiest bar start there
    # rather than anywhere in the cycle.
    khaali = None
    interior = [int(p) for p in positions if p != 0]
    if interior:
        midpoint = period / 2
        near_middle = [p for p in interior if abs(p - midpoint) <= period / 8]
        khaali = int(min(near_middle or interior, key=lambda p: rotated[p]))

    # Everything else that carries weight is taali, sam included. In Teentaal
    # this lands on 0, 4 and 12 with khaali at 8 — the textbook map, arrived
    # at from the audio rather than assumed.
    taali = [0] + [p for p in interior
                   if p != khaali and rotated[p] >= float(rotated[interior].mean())]

    return {
        "beats_per_cycle": period,
        "bars_per_cycle": round(period / 4, 3) if period % 4 == 0 else None,
        "sam_beat_index": sam,
        "sam_times": [round(float(t), 3) for t in beat_times[sam::period]],
        "taali": taali,
        "khaali": khaali,
        "accent_profile": [round(float(v), 4) for v in rotated],
        "tala_hint": TALA_NAMES.get(period),
        "regular_4_4": period % 4 == 0,
        "confidence": round(float(min(1.0, max(0.0, best["score"]))), 3),
        "note": None,
    }


def _score_period(accent, beat_chroma, period):
    """How well `period` explains the track.

    A real cycle repeats *and* has internal shape. Repetition alone is not
    enough, because a 16-beat cycle also repeats at 4 and 8; what separates
    the true length is that its accent profile has structure — a sam that
    stands up and a khaali that drops away. Contrast supplies that, and
    without it detection collapses onto the shortest candidate every time.
    """
    a = (accent - accent.mean()) / (accent.std() + 1e-9)
    similarity = float(np.corrcoef(a[:-period], a[period:])[0, 1])
    if not np.isfinite(similarity):
        similarity = 0.0

    left = beat_chroma[:, :-period]
    right = beat_chroma[:, period:]
    denom = (np.linalg.norm(left, axis=0) * np.linalg.norm(right, axis=0)) + 1e-9
    harmonic_similarity = float(np.mean(np.sum(left * right, axis=0) / denom))

    profile = np.array([a[phase::period].mean() for phase in range(period)])
    contrast = float(np.std(profile))

    score = 0.45 * similarity + 0.25 * harmonic_similarity + 0.30 * contrast
    return score, similarity, contrast


def _fallback(note: str) -> dict:
    return {
        "beats_per_cycle": 16, "bars_per_cycle": 4, "sam_beat_index": 0,
        "sam_times": [], "taali": [0, 4, 12], "khaali": 8,
        "accent_profile": [], "tala_hint": None, "regular_4_4": True,
        "confidence": 0.0, "note": note,
    }


def _at(curve: np.ndarray, frames: np.ndarray) -> np.ndarray:
    if len(curve) == 0:
        return np.zeros(len(frames))
    return curve[np.clip(frames, 0, len(curve) - 1)]


def _unit(arr: np.ndarray) -> np.ndarray:
    span = float(np.max(arr) - np.min(arr))
    if span < 1e-9:
        return np.zeros_like(arr)
    return (arr - float(np.min(arr))) / span
