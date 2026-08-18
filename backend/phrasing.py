"""The phrasing agent: where a track enters, and how long a blend runs.

Everything here follows from one rule — a track enters on sam. Not on any
downbeat, on the beat the cycle resolves to. Get that wrong and the incoming
track is permanently offset against the one already playing: still in time,
still on the grid, but landing in the wrong part of the cycle, which is the
difference between a mix that sits and a mix that fights itself.

The length of a blend follows from the same rule. If it does not last a
whole number of cycles for *both* tracks, then whichever one started on sam
arrives at the far end mid-cycle. So the blend is measured in cycles, and
the number of bars is whatever that works out to — which differs per pair,
because the two tracks' cycles differ.
"""

import math

DEFAULT_CYCLE = {"beats_per_cycle": 16, "sam_times": [], "confidence": 0.0}

# Past this a blend stops reading as a transition and starts sounding like
# two tracks that happen to be playing at once. Twenty-four bars is about
# 45 seconds at house tempo, which is already a long layer; thirty-two runs
# past a minute and swallows tracks whole.
MAX_BLEND_BARS = 24
MIN_BLEND_BARS = 2


def cycle_of(result: dict) -> dict:
    cycle = (result or {}).get("cycle") or {}
    return {
        "beats_per_cycle": int(cycle.get("beats_per_cycle") or 16),
        "sam_times": cycle.get("sam_times") or [],
        "confidence": float(cycle.get("confidence") or 0.0),
        "taali": cycle.get("taali") or [],
        "khaali": cycle.get("khaali"),
        "tala_hint": cycle.get("tala_hint"),
    }


def blend_bars(out_cycle: dict, in_cycle: dict, preferred_bars: int,
               context: dict | None = None) -> dict:
    """Bars for one handover: whole cycles for both tracks, sized to the pair.

    Two separate questions. The cycles decide the *granularity* — the common
    unit is the lowest multiple satisfying both, because a blend that is not
    whole cycles for both leaves one of them arriving mid-cycle. Where that
    unit is impractically long the incoming track's cycle wins, since a track
    arriving mid-cycle is far more audible than one leaving mid-cycle.

    How many of those units to use is a different question, and it is what
    makes the length vary down a set. Two tracks that are close in tempo, key
    and energy can be layered over each other for a long time and it only
    gets richer. Push any of those apart and the same length turns into two
    records fighting, so the blend shortens.
    """
    out_beats = max(1, int(out_cycle["beats_per_cycle"]))
    in_beats = max(1, int(in_cycle["beats_per_cycle"]))

    common_beats = math.lcm(out_beats, in_beats)
    unit_bars = common_beats / 4.0
    reason = "whole cycles for both tracks"

    if unit_bars > MAX_BLEND_BARS or unit_bars != int(unit_bars):
        unit_bars = in_beats / 4.0
        reason = "incoming track's cycle; the two do not share a practical one"

    unit_bars = max(1, int(round(unit_bars)))

    target, notes = _suitable_length(preferred_bars, context)

    # Whole number of units nearest the target length.
    multiple = max(1, round(target / unit_bars))
    bars = unit_bars * multiple
    while bars > MAX_BLEND_BARS and multiple > 1:
        multiple -= 1
        bars = unit_bars * multiple
    bars = max(MIN_BLEND_BARS, bars)

    return {
        "bars": int(bars),
        "cycle_bars": unit_bars,
        "cycles": multiple,
        "out_beats_per_cycle": out_beats,
        "in_beats_per_cycle": in_beats,
        "preferred_bars": preferred_bars,
        "target_bars": round(target, 1),
        "reason": reason,
        "sizing": notes or ["nothing pulling it either way"],
    }


def _arc_factor(position: float, shape: str) -> tuple[float, str]:
    """How long this handover wants to be, given where it falls in the set.

    A long overlap is the euphoric move in a mix — two records layered for
    half a minute rather than swapped in eight bars — and it belongs late,
    once the set has built to it. Opening with one leaves nowhere to go.
    """
    p = min(1.0, max(0.0, position))
    if shape == "peak":
        # Climax around three-quarters through, easing back for the outro.
        return 0.85 + 1.25 * math.sin(math.pi * (p / 0.75) / 2) ** 1.2, "building to the peak"
    if shape == "wave":
        return 0.9 + 0.45 * p, "opening out as the set runs on"
    return 0.85 + 1.15 * (p ** 1.2), "opening out towards the end"


def _suitable_length(preferred_bars: int, context: dict | None):
    """Scale the preset's blend length to the pair and to where it falls."""
    if not context:
        return float(preferred_bars), []

    scale, arc_note = _arc_factor(
        float(context.get("position", 0.5)), context.get("energy_shape", "rise")
    )
    notes = [arc_note]

    tempo_gap = abs(float(context.get("tempo_delta_percent") or 0.0))
    if tempo_gap > 3.0:
        # Any residual tempo difference drifts the two grids apart, and the
        # longer they overlap the further apart they get by the end.
        factor = 1.0 - min(0.5, (tempo_gap - 3.0) / 18.0)
        scale *= factor
        notes.append(f"shortened for a {tempo_gap:.0f}% tempo gap")

    energy_jump = abs(float(context.get("energy_to", 0.5)) - float(context.get("energy_from", 0.5)))
    if energy_jump > 0.15:
        scale *= 1.0 - min(0.35, (energy_jump - 0.15) * 1.2)
        notes.append(f"shortened for an energy jump of {energy_jump:.2f}")

    clash = float(context.get("vocal_clash") or 0.0)
    if clash > 0.15:
        # Every extra bar is another bar of two lead vocals competing, so the
        # overlap shrinks as the clash gets worse.
        scale *= 1.0 - min(0.45, (clash - 0.15) * 1.6)
        notes.append(f"shortened — both tracks singing ({clash:.2f} overlap)")

    key_distance = float(context.get("camelot_distance") or 0.0)
    if key_distance >= 2.0:
        scale *= 1.0 - min(0.3, (key_distance - 1.0) / 12.0)
        notes.append(f"shortened for {key_distance:g} steps of key distance")
    elif key_distance <= 1.5 and tempo_gap <= 5.0:
        # Close in tempo and key: these two can sit on top of each other.
        # Graded rather than a threshold — the old version was a cliff that
        # almost never fired, which is why long overlaps never happened.
        closeness = (1.0 - key_distance / 1.5) * (1.0 - tempo_gap / 5.0)
        if closeness > 0.05:
            scale *= 1.0 + 0.4 * closeness
            notes.append("lengthened — close in tempo and key, so they layer well")

    return max(2.0, preferred_bars * scale), notes


def snap_to_sam(time: float, cycle: dict,
                tolerance: float | None = None) -> tuple[float, bool]:
    """Move a time onto the nearest sam. Returns (time, whether it moved).

    With no tolerance it always snaps, which is what cue placement wants:
    sams are one cycle apart, so the nearest is at most half a cycle away,
    and a long cycle at a slow tempo can put that well past any fixed
    tolerance in seconds. A fixed limit silently left those cues off-cycle.
    """
    sam_times = cycle.get("sam_times") or []
    if not sam_times:
        return time, False
    nearest = min(sam_times, key=lambda t: abs(t - time))
    if tolerance is not None and abs(nearest - time) > tolerance:
        return time, False
    return float(nearest), abs(nearest - time) > 1e-6


def sam_at_or_after(time: float, cycle: dict, floor_: float) -> float | None:
    """The first sam at or after `time`, staying above `floor_`.

    Used where a cue has to be pushed to a limit — capping how long a track
    plays, say — and must still land on the cycle rather than on whichever
    bar line happens to sit at the cutoff.
    """
    candidates = [t for t in (cycle.get("sam_times") or []) if t > floor_]
    if not candidates:
        return None
    later = [t for t in candidates if t >= time]
    return float(later[0]) if later else float(candidates[-1])


def describe(cycle: dict) -> str:
    """Short human description of a track's cycle."""
    beats = cycle["beats_per_cycle"]
    bars = beats / 4
    parts = [f"{beats}-beat cycle ({bars:g} bars)"]
    if cycle.get("tala_hint"):
        parts.append(f"{cycle['tala_hint']} length")
    if cycle.get("khaali") is not None:
        parts.append(f"khaali at beat {cycle['khaali'] + 1}")
    return ", ".join(parts)
