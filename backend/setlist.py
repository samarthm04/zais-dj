"""Ordering a playlist into a set, and the genre presets that shape it.

Two tracks sit well together when their tempos are close, their keys are
adjacent on the Camelot wheel, and the energy moves the way a set should.
Ordering is chosen to maximise that across the whole run, not pair by pair.
"""

import itertools
import math

# Genre presets. These shape how the set is mixed — blend lengths, which
# transition styles are reachable, and the tempo band the set aims for. They
# do not change what the tracks are: a preset makes a pop set mixed like
# house, not a house record.
PRESETS = {
    "pop": {
        "label": "Pop",
        "crossfade_bars": 8,
        "styles": ("blend", "bass_swap", "filter_sweep", "echo_out", "cut"),
        "tempo_band": (95, 130),
        "energy_shape": "rise",
    },
    "house": {
        "label": "House",
        "crossfade_bars": 16,
        "styles": ("bass_swap", "blend", "filter_sweep"),
        "tempo_band": (118, 128),
        "energy_shape": "rise",
    },
    "afrobeat": {
        "label": "Afrobeat",
        "crossfade_bars": 8,
        "styles": ("bass_swap", "blend", "echo_out"),
        "tempo_band": (100, 116),
        "energy_shape": "wave",
    },
    "hiphop": {
        "label": "Hip-hop",
        "crossfade_bars": 4,
        "styles": ("echo_out", "cut", "blend"),
        "tempo_band": (80, 105),
        "energy_shape": "wave",
    },
    "festival": {
        "label": "Festival / EDM",
        "crossfade_bars": 16,
        "styles": ("filter_sweep", "bass_swap", "blend"),
        "tempo_band": (124, 136),
        "energy_shape": "peak",
    },
}

# Searching every permutation now, not every rotation, so this is a factor
# of n more work than before. Eight is the point where it still returns
# in well under a second.
MAX_EXACT = 8


def camelot_distance(a: str | None, b: str | None) -> float:
    """Steps around the Camelot wheel. 0 is the same key, 1 a neighbour."""
    if not a or not b:
        return 2.0
    try:
        na, ma = int(a[:-1]), a[-1].upper()
        nb, mb = int(b[:-1]), b[-1].upper()
    except (ValueError, IndexError):
        return 2.0
    around = min((na - nb) % 12, (nb - na) % 12)
    if ma == mb:
        return float(around)
    # Relative major/minor at the same number is the free switch.
    return 0.5 if around == 0 else float(around) + 1.0


def tempo_delta_percent(a: float, b: float) -> float:
    """Percent change from a to b, allowing for half and double time."""
    if a <= 0 or b <= 0:
        return 100.0
    options = [b, b * 2, b / 2]
    best = min(options, key=lambda x: abs(x - a))
    return (best - a) / a * 100.0


def target_energy(position: float, shape: str) -> float:
    """The energy a set wants at this point, 0 at the start and 1 at the end.

    A straight climb is the flattest thing a set can do — with no valleys
    there is nothing for the peaks to rise out of. Each shape here builds
    overall while still dipping, so a drop into something sparse becomes a
    deliberate move rather than a cost to be avoided.
    """
    p = min(1.0, max(0.0, position))
    if shape == "peak":
        # One big arc: climb to a summit around three-quarters, then come down.
        return 0.30 + 0.65 * math.sin(math.pi * min(1.0, p / 0.75) / 2) ** 1.1
    if shape == "wave":
        # Repeated swells, for sets that live on movement rather than a climax.
        return 0.55 + 0.30 * math.sin(2 * math.pi * 1.25 * p - math.pi / 2)
    # Rising, but with contour on the way up rather than a ramp.
    return 0.32 + 0.50 * p + 0.13 * math.sin(2 * math.pi * 1.5 * p)


def placement_cost(track: dict, position: float, preset: dict) -> float:
    """How far this track sits from the energy the arc wants here."""
    wanted = target_energy(position, preset.get("energy_shape", "rise"))
    return abs(float(track.get("energy", 0.5)) - wanted) * 30.0


def sequence_cost(seq: list[dict], preset: dict, prefs: dict | None = None) -> float:
    """Total cost of an ordering: how each pair joins, plus where each sits."""
    total = sum(pair_cost(seq[i], seq[i + 1], preset, prefs) for i in range(len(seq) - 1))
    last = max(1, len(seq) - 1)
    total += sum(placement_cost(t, i / last, preset) for i, t in enumerate(seq))
    return total


def pair_cost(t1: dict, t2: dict, preset: dict, prefs: dict | None = None) -> float:
    tempo = abs(tempo_delta_percent(t1["tempo"], t2["tempo"]))
    key = camelot_distance(t1.get("camelot"), t2.get("camelot"))

    # Tempo dominates: it is the thing you cannot fake.
    cost = min(tempo, 40.0) * 1.0
    cost += key * 3.0

    # Only outright lurches are penalised now. Charging for every energy drop
    # forced the set to climb monotonically, which is exactly what made it
    # feel flat: a set with no valleys has no peaks either. The shape is set
    # by target_energy() below, which places tracks along an arc instead.
    cost += max(0.0, abs(t2["energy"] - t1["energy"]) - 0.4) * 20.0

    # Trust keys less when the detector was unsure about either end.
    confidence = min(t1.get("key_confidence", 0.0), t2.get("key_confidence", 0.0))
    if confidence < 0.15:
        cost -= key * 2.0

    # A remembered verdict on this exact handover outranks the heuristics,
    # since it came from someone actually listening to it.
    if prefs is not None:
        from backend.preferences import pair_adjustment
        cost += pair_adjustment(t1.get("filename"), t2.get("filename"), prefs)
    return cost


def order_tracks(tracks: list[dict], preset: dict, start_id: str | None = None,
                 prefs: dict | None = None) -> list[dict]:
    """Order the playlist to minimise total transition cost."""
    if len(tracks) <= 2:
        return tracks

    if start_id:
        head = [t for t in tracks if t["track_id"] == start_id]
        rest = [t for t in tracks if t["track_id"] != start_id]
    else:
        head, rest = [], list(tracks)

    if len(rest) <= MAX_EXACT and not head:
        best, best_cost = None, float("inf")
        # Every permutation is searched, first track included. Pinning it was
        # sound when only adjacency mattered and rotations were equivalent,
        # but the arc cares where a track sits: the set has to be free to open
        # on something quiet and climb from there.
        for candidate in itertools.permutations(rest):
            cost = sequence_cost(list(candidate), preset, prefs)
            if cost < best_cost:
                best, best_cost = list(candidate), cost
        return best

    ordered = head or [rest.pop(0)]
    if head:
        pass
    while rest:
        current = ordered[-1]
        nxt = min(rest, key=lambda t: pair_cost(current, t, preset, prefs))
        rest.remove(nxt)
        ordered.append(nxt)
    return _two_opt(ordered, preset, prefs)


def _two_opt(order: list[dict], preset: dict, prefs: dict | None = None) -> list[dict]:
    """Local cleanup: reverse any span that lowers the total cost."""
    def total(seq):
        return sequence_cost(seq, preset, prefs)

    best = order
    best_cost = total(best)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                cost = total(candidate)
                if cost < best_cost - 1e-9:
                    best, best_cost, improved = candidate, cost, True
    return best


def summarize(tracks: list[dict], preset: dict, prefs: dict | None = None) -> list[dict]:
    """Per-handover description of the ordering that was chosen."""
    rows = []
    for t1, t2 in zip(tracks, tracks[1:]):
        rows.append({
            "from": t1["filename"],
            "to": t2["filename"],
            "tempo_delta_percent": round(tempo_delta_percent(t1["tempo"], t2["tempo"]), 2),
            "camelot_distance": camelot_distance(t1.get("camelot"), t2.get("camelot")),
            "cost": round(pair_cost(t1, t2, preset, prefs), 2),
        })
    return rows
