"""Remembered likes and dislikes.

A set is judged by ear, and the verdict is really about its handovers: this
track into that one, mixed that way, worked. Feedback is stored per ordered
pair so that when the same songs come round again the ordering can favour
what already sounded good and avoid what did not.
"""

import json
from pathlib import Path

STORE = Path(__file__).resolve().parent / "data" / "preferences.json"

# How much a remembered verdict moves the ordering cost. Large enough to
# reorder a set, not so large that it overrides a tempo clash that would
# make the transition impossible.
LIKE_BONUS = 6.0
DISLIKE_PENALTY = 12.0
MAX_WEIGHT = 3  # repeated verdicts stop compounding past this


def _key(a: str, b: str) -> str:
    return json.dumps([a, b], ensure_ascii=False)


def load() -> dict:
    if not STORE.exists():
        return {"pairs": {}, "observations": []}
    try:
        data = json.loads(STORE.read_text())
    except (json.JSONDecodeError, OSError):
        return {"pairs": {}, "observations": []}
    data.setdefault("pairs", {})
    data.setdefault("observations", [])
    return data


def save(data: dict) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(data, indent=2))


def record_transition(row: dict, nxt: dict, verdict: str,
                      preset: str | None = None, position: float | None = None) -> dict:
    """Record a verdict on one handover.

    Two things are written. The pair aggregate is what ordering reads back,
    cheaply. The observation is the row of a training set: the verdict plus
    the features that produced it, so what the transition *was* can later be
    learned from, rather than only which two songs it joined. Counts alone
    can say "these two worked"; they cannot say "16-bar bass swaps under 5%
    tempo drift work", which is the useful generalisation.
    """
    if verdict not in ("like", "dislike"):
        raise ValueError("verdict must be 'like' or 'dislike'")

    a, b = row.get("filename"), nxt.get("filename")
    if not a or not b:
        raise ValueError("handover is missing a track name")

    data = load()
    entry = data["pairs"].setdefault(_key(a, b), {"likes": 0, "dislikes": 0, "styles": {}})
    entry["likes" if verdict == "like" else "dislikes"] += 1

    style = row.get("transition_to_next")
    if style:
        entry["styles"][style] = entry["styles"].get(style, 0) + (1 if verdict == "like" else -1)

    phrasing = row.get("phrasing") or {}
    data.setdefault("observations", []).append({
        "verdict": verdict,
        "preset": preset,
        "from": a,
        "to": b,
        # Features of the transition itself — what a model would learn from.
        "style": style,
        "bars": row.get("transition_bars"),
        "cycles": phrasing.get("cycles"),
        "cycle_bars": phrasing.get("cycle_bars"),
        "tempo_delta_percent": row.get("tempo_delta_percent"),
        "beatmatched": row.get("beatmatched"),
        "leaves_at_tempo": row.get("leaves_at_tempo"),
        "enters_at_tempo": nxt.get("enters_at_tempo"),
        "stretch_percent": max(row.get("stretch_percent") or 0,
                               nxt.get("stretch_percent") or 0),
        "from_camelot": row.get("camelot"),
        "to_camelot": nxt.get("camelot"),
        "from_cycle_beats": row.get("beats_per_cycle"),
        "to_cycle_beats": nxt.get("beats_per_cycle"),
        "into_reprise": bool(nxt.get("is_reprise")),
        "position": round(position, 3) if position is not None else None,
    })
    data["observations"] = data["observations"][-2000:]
    save(data)

    return {
        "from": a, "to": b, "style": style, "bars": row.get("transition_bars"),
        "likes": entry["likes"], "dislikes": entry["dislikes"],
        "observations": len(data["observations"]),
    }


def pair_adjustment(a: str | None, b: str | None, data: dict | None = None) -> float:
    """Cost adjustment for putting `a` before `b`. Negative means preferred."""
    if not a or not b:
        return 0.0
    pairs = (data or load())["pairs"]
    entry = pairs.get(_key(a, b))
    if not entry:
        return 0.0
    likes = min(entry.get("likes", 0), MAX_WEIGHT)
    dislikes = min(entry.get("dislikes", 0), MAX_WEIGHT)
    return dislikes * DISLIKE_PENALTY - likes * LIKE_BONUS


def preferred_style(a: str | None, b: str | None, data: dict | None = None) -> str | None:
    """The style that earned this pairing its likes, if there is a clear one."""
    if not a or not b:
        return None
    entry = (data or load())["pairs"].get(_key(a, b))
    if not entry or not entry.get("styles"):
        return None
    style, score = max(entry["styles"].items(), key=lambda kv: kv[1])
    return style if score > 0 else None


def summary() -> dict:
    data = load()
    liked, disliked = [], []
    for key, entry in data["pairs"].items():
        a, b = json.loads(key)
        net = entry.get("likes", 0) - entry.get("dislikes", 0)
        row = {"from": a, "to": b, "likes": entry.get("likes", 0),
               "dislikes": entry.get("dislikes", 0)}
        if net > 0:
            liked.append(row)
        elif net < 0:
            disliked.append(row)
    observations = data.get("observations", [])
    by_style = {}
    for o in observations:
        if not o.get("style"):
            continue
        row = by_style.setdefault(o["style"], {"likes": 0, "dislikes": 0})
        row["likes" if o["verdict"] == "like" else "dislikes"] += 1
    return {"liked": liked, "disliked": disliked,
            "rated": len(observations), "by_style": by_style}
