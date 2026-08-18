"""Barebones upload -> analyze -> listen server for the AI DJ engine.

Files and results live on local disk under backend/data/. No database.
The only contact point with the analysis engine is `run_analysis()`, which
imports `audio_analysis.analyze_track` unmodified.
"""

import json
import mimetypes
import sys
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
RESULT_DIR = BASE_DIR / "data" / "results"
MIX_DIR = BASE_DIR / "data" / "mixes"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
MIX_DIR.mkdir(parents=True, exist_ok=True)

# audio_analysis lives at the project root, next to backend/.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAX_FILES = 10
ALLOWED_SUFFIXES = {".mp3", ".wav", ".m4a", ".mp4", ".flac", ".aac", ".ogg", ".aiff", ".aif"}

app = FastAPI(title="AI DJ - cue point audition")


class ProcessRequest(BaseModel):
    track_ids: list[str] | None = None


class MixRequest(BaseModel):
    from_track_id: str
    from_time: float
    to_track_id: str
    to_time: float
    crossfade_bars: int = 8
    match_tempo: bool = True


class SetRequest(BaseModel):
    track_ids: list[str] | None = None
    preset: str = "pop"
    start_track_id: str | None = None
    max_play_seconds: float = 75.0
    use_fillers: bool = True
    use_reprises: bool = True
    reprise_seconds: float = 45.0


class FeedbackRequest(BaseModel):
    set_id: str
    index: int
    verdict: str


def _cue_times(result: dict) -> tuple[float, float] | None:
    """Best entry, and the best exit that leaves room to actually play."""
    cues = result.get("cue_points") or {}
    entries = sorted(cues.get("entry") or [], key=lambda c: -c.get("score", 0))
    exits = sorted(cues.get("exit") or [], key=lambda c: -c.get("score", 0))
    if not entries:
        return None
    entry = float(entries[0]["time"])
    usable = [float(c["time"]) for c in exits if float(c["time"]) > entry + 20.0]
    if usable:
        return entry, usable[0]
    duration = float(result.get("duration") or 0)
    return (entry, max(entry + 30.0, duration - 20.0)) if duration else None


def _reprise_start(result: dict, avoid: list[tuple[float, float]],
                   length: float) -> float | None:
    """Where to re-enter a track, given the stretches it has already played.

    Entry cues are the wrong source here. They are scored to favour early,
    quiet, rising moments — the right thing for opening a track, which means
    they all cluster in the intro. Re-entering there gives the same handful
    of seconds every time, and by the second reprise it is the intro again.

    A reprise lands mid-set, where the interesting move is dropping into a
    loud section, so the sections themselves are the candidates: the highest
    energy one that does not overlap anything already played.
    """
    duration = float(result.get("duration") or 0)

    def clear(start: float) -> bool:
        return not any(start < end and start + length > begin for begin, end in avoid)

    sections = []
    for segment in result.get("segments") or []:
        start, end = float(segment.get("start", 0)), float(segment.get("end", 0))
        if end - start < 8.0 or (duration and start + length > duration):
            continue
        if clear(start):
            sections.append((float(segment.get("energy", 0.0)), start))
    if sections:
        return max(sections)[1]

    # No untouched section long enough: fall back to any cue that is clear.
    cues = sorted((result.get("cue_points") or {}).get("entry") or [],
                  key=lambda c: -c.get("score", 0))
    for cue in cues:
        if clear(float(cue["time"])):
            return float(cue["time"])
    return None


def _plan_reprises(ordered: list[dict], max_gap: float, reprise_seconds: float,
                   max_play_seconds: float) -> list[dict]:
    """Insert reprises of already-played tracks across un-matchable jumps.

    A reprise is the same song entered somewhere new, and it is the element
    that takes the tempo strain: it enters on the outgoing track's tempo and
    leaves on the incoming one's, so neither real track has to stretch. The
    track chosen is whichever one's own tempo sits nearest the middle of the
    jump, which keeps its stretch as small as the jump allows.
    """
    from backend.setlist import tempo_delta_percent

    # Every stretch of every track that has already been heard, so a reprise
    # never replays something the set just played.
    # Trimmed to what the renderer will actually play. The exit cue can sit
    # minutes after the entry, but the render stops at max_play_seconds — and
    # treating the untrimmed span as heard blocks nearly the whole track from
    # ever being reprised.
    played: dict[str, list[tuple[float, float]]] = {}
    for track in ordered:
        entry = float(track["entry_time"])
        played.setdefault(track["track_id"], []).append(
            (entry, min(float(track["exit_time"]), entry + max_play_seconds))
        )
    reprised: set[str] = set()

    sequence = [ordered[0]]
    for i in range(len(ordered) - 1):
        a, b = ordered[i], ordered[i + 1]
        gap = abs(tempo_delta_percent(a["tempo"], b["tempo"]))
        if gap > max_gap:
            midpoint = (a["tempo"] + b["tempo"]) / 2.0
            pool = [t for t in ordered if t["track_id"] not in (a["track_id"], b["track_id"])]
            pool = pool or [t for t in ordered if t["track_id"] != b["track_id"]]
            # Spread reprises across the playlist. Picking purely on tempo
            # returns the same track for every gap, so one song comes back
            # again and again while the rest never do.
            fresh = [t for t in pool if t["track_id"] not in reprised]
            pool = fresh or pool

            if pool:
                pick = min(pool, key=lambda t: abs(t["tempo"] - midpoint))
                start = _reprise_start(
                    pick["result"], played.get(pick["track_id"], []), reprise_seconds
                )
                if start is not None:
                    reprise = dict(pick)
                    reprise["entry_time"] = start
                    reprise["exit_time"] = start + reprise_seconds
                    reprise["is_reprise"] = True
                    reprise["bridges"] = (a["tempo"], b["tempo"])
                    sequence.append(reprise)
                    reprised.add(pick["track_id"])
                    played.setdefault(pick["track_id"], []).append(
                        (start, start + reprise_seconds)
                    )
        sequence.append(b)
    return sequence


def _mean_energy(result: dict) -> float:
    curve = result.get("energy_curve") or []
    values = [p.get("energy", 0.0) for p in curve if isinstance(p, dict)]
    return float(sum(values) / len(values)) if values else 0.5


def track_dir(track_id: str) -> Path:
    """Resolve a track's upload dir, rejecting anything that isn't a plain uuid hex."""
    try:
        uuid.UUID(hex=track_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="bad track id")
    return UPLOAD_DIR / track_id


def audio_files(directory: Path) -> list[Path]:
    """Audio in a track directory, ignoring anything that is not audio."""
    return sorted(p for p in directory.glob("*")
                  if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES)


def audio_path(track_id: str) -> Path:
    files = audio_files(track_dir(track_id))
    if not files:
        raise HTTPException(status_code=404, detail="track not found")
    return files[0]


def result_path(track_id: str) -> Path:
    return RESULT_DIR / f"{track_id}.json"


def load_result(track_id: str):
    p = result_path(track_id)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def list_tracks() -> list[dict]:
    tracks = []
    for d in sorted(UPLOAD_DIR.iterdir(), key=lambda p: p.stat().st_mtime):
        if not d.is_dir():
            continue
        files = audio_files(d)
        if not files:
            continue
        tracks.append(
            {
                "track_id": d.name,
                "filename": files[0].name,
                "result": load_result(d.name),
            }
        )
    return tracks


def run_analysis(path: Path, track_id: str) -> dict:
    """Call the existing engine. Imported lazily so the server still boots
    (and reports a clear error) when audio_analysis is absent."""
    from audio_analysis import analyze_track

    return analyze_track(str(path), track_id)


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="no files given")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"max {MAX_FILES} files at once")

    uploaded = []
    for f in files:
        name = Path(f.filename or "").name
        if not name:
            raise HTTPException(status_code=400, detail="file with no name")
        if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"unsupported audio file: {name}")

        track_id = uuid.uuid4().hex
        dest_dir = UPLOAD_DIR / track_id
        dest_dir.mkdir(parents=True)
        dest = dest_dir / name
        with dest.open("wb") as out:
            while chunk := await f.read(1024 * 1024):
                out.write(chunk)

        uploaded.append({"track_id": track_id, "filename": name, "result": None})

    return {"tracks": uploaded}


@app.post("/api/process")
def process(req: ProcessRequest | None = None):
    """Run each requested track through analyze_track(). Defaults to every
    track that has no result yet."""
    if req is not None and req.track_ids:
        targets = req.track_ids
    else:
        targets = [t["track_id"] for t in list_tracks() if t["result"] is None]

    processed = []
    for track_id in targets:
        path = audio_path(track_id)
        entry = {"track_id": track_id, "filename": path.name}
        try:
            result = run_analysis(path, track_id)
        except ImportError as exc:
            entry["error"] = (
                f"audio_analysis package not importable from {PROJECT_ROOT}: {exc}"
            )
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()
        else:
            result_path(track_id).write_text(json.dumps(result, indent=2, default=str))
            entry["result"] = result
        processed.append(entry)

    return {"tracks": processed}


@app.get("/api/tracks")
def tracks():
    return {"tracks": list_tracks()}


@app.delete("/api/track/{track_id}")
def delete_track(track_id: str):
    """Drop a track from the library: its audio and its analysis."""
    directory = track_dir(track_id)
    if not directory.exists():
        raise HTTPException(status_code=404, detail="track not found")

    for f in directory.glob("*"):
        f.unlink(missing_ok=True)
    directory.rmdir()
    result_path(track_id).unlink(missing_ok=True)

    from backend import embedding
    embedding.forget(track_id)
    return {"deleted": track_id, "remaining": len(list_tracks())}


@app.get("/api/audio/{track_id}")
def audio(track_id: str):
    path = audio_path(track_id)
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.post("/api/mix")
def mix(req: MixRequest):
    """Render track A's exit cue crossfading into track B's entry cue."""
    from backend.mix import render_transition

    a_result = load_result(req.from_track_id)
    b_result = load_result(req.to_track_id)
    if a_result is None or b_result is None:
        raise HTTPException(status_code=400, detail="both tracks must be analyzed first")
    if not 1 <= req.crossfade_bars <= 64:
        raise HTTPException(status_code=400, detail="crossfade_bars must be 1-64")

    mix_id = uuid.uuid4().hex
    out_path = MIX_DIR / f"{mix_id}.wav"
    try:
        details = render_transition(
            a_path=audio_path(req.from_track_id),
            a_result=a_result,
            a_time=req.from_time,
            b_path=audio_path(req.to_track_id),
            b_result=b_result,
            b_time=req.to_time,
            crossfade_bars=req.crossfade_bars,
            match_tempo=req.match_tempo,
            out_path=out_path,
        )
    except Exception as exc:
        out_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    return {
        "mix_id": mix_id,
        "url": f"/api/mix/{mix_id}",
        "from": {"filename": a_result.get("filename"), "time": req.from_time},
        "to": {"filename": b_result.get("filename"), "time": req.to_time},
        **details,
    }


@app.get("/api/presets")
def presets():
    from backend.setlist import PRESETS

    return {"presets": [{"id": k, **{kk: vv for kk, vv in v.items() if kk != "styles"},
                         "styles": list(v["styles"])} for k, v in PRESETS.items()]}


@app.post("/api/set")
def build_set(req: SetRequest):
    """Order the playlist and render it as one continuous mix."""
    from backend.setlist import PRESETS, order_tracks, summarize
    from backend.setrender import render_set

    if req.preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"unknown preset: {req.preset}")
    preset = {**PRESETS[req.preset], "id": req.preset}

    wanted = set(req.track_ids) if req.track_ids else None
    candidates = []
    skipped = []
    for t in list_tracks():
        if wanted is not None and t["track_id"] not in wanted:
            continue
        result = t["result"]
        if not result:
            skipped.append({"filename": t["filename"], "reason": "not analyzed yet"})
            continue
        cues = _cue_times(result)
        if cues is None:
            duration = float(result.get("duration") or 0)
            skipped.append({
                "filename": t["filename"],
                "reason": f"no usable cue points ({duration:.0f}s long — too short to mix)"
                if duration < 60 else "no usable cue points",
            })
            continue
        key = result.get("key") or {}
        candidates.append({
            "track_id": t["track_id"],
            "filename": t["filename"],
            "tempo": float(result.get("tempo") or 0),
            "camelot": key.get("camelot"),
            "key_confidence": float(key.get("confidence") or 0),
            "energy": _mean_energy(result),
            "entry_time": cues[0],
            "exit_time": cues[1],
            "result": result,
            "path": audio_path(t["track_id"]),
        })

    if len(candidates) < 2:
        raise HTTPException(status_code=400,
                            detail="need at least 2 analyzed tracks with cue points")

    from backend import embedding
    from backend.preferences import load as load_prefs
    from backend.setrender import MAX_MATCHABLE_PCT

    # Learned similarity, if the encoder is installed. Vectors are cached per
    # track, so this only pays the model load on the first set after a
    # restart, and nothing at all when the encoder is absent.
    if embedding.available():
        vectors = {c["track_id"]: embedding.embed(c["track_id"], c["path"])
                   for c in candidates}
        for c in candidates:
            mine = vectors.get(c["track_id"])
            c["embedding_distance_to"] = {
                other["track_id"]: d
                for other in candidates
                if other["track_id"] != c["track_id"]
                and (d := embedding.distance(mine, vectors.get(other["track_id"]))) is not None
            }

    prefs = load_prefs()
    ordered = order_tracks(candidates, preset, start_id=req.start_track_id, prefs=prefs)
    pairing = summarize(ordered, preset, prefs)

    sequence = ordered
    if req.use_reprises and len(ordered) >= 2:
        sequence = _plan_reprises(ordered, MAX_MATCHABLE_PCT, req.reprise_seconds,
                                  req.max_play_seconds)

    set_id = uuid.uuid4().hex
    try:
        details = render_set(sequence, preset, MIX_DIR / set_id,
                             max_play_seconds=req.max_play_seconds,
                             use_fillers=req.use_fillers)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    unmatched = [r for r in details["timeline"] if r.get("beatmatched") is False]
    # Kept beside the audio so a verdict can still be recorded after a restart.
    (MIX_DIR / f"{set_id}.json").write_text(json.dumps(
        {"preset": req.preset, "timeline": details["timeline"]}, indent=2, default=str))
    return {
        "set_id": set_id,
        "url": f"/api/set/{set_id}",
        "preset": req.preset,
        "order": [t["filename"] for t in sequence],
        "pairing": pairing,
        "skipped": skipped,
        "all_beatmatched": not unmatched,
        "sonic_ordering": bool(candidates and candidates[0].get("embedding_distance_to")),
        "reprises": sum(1 for t in sequence if t.get("is_reprise")),
        **details,
    }


@app.get("/api/preferences")
def get_preferences():
    from backend.preferences import summary

    return summary()


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    """Remember a verdict on one handover of a rendered set."""
    from backend.preferences import record_transition

    if req.verdict not in ("like", "dislike"):
        raise HTTPException(status_code=400, detail="verdict must be 'like' or 'dislike'")
    try:
        uuid.UUID(hex=req.set_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="bad set id")

    meta_path = MIX_DIR / f"{req.set_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="unknown set; rebuild it and try again")

    meta = json.loads(meta_path.read_text())
    timeline = meta["timeline"]
    if not 0 <= req.index < len(timeline) - 1:
        raise HTTPException(status_code=400, detail="no handover at that index")

    handovers = max(1, len(timeline) - 2)
    result = record_transition(
        timeline[req.index], timeline[req.index + 1], req.verdict,
        preset=meta.get("preset"), position=req.index / handovers,
    )
    return {"verdict": req.verdict, "index": req.index, **result}


@app.get("/api/set/{set_id}")
def set_audio(set_id: str):
    try:
        uuid.UUID(hex=set_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="bad set id")
    for suffix, media in ((".mp3", "audio/mpeg"), (".wav", "audio/wav")):
        path = MIX_DIR / f"{set_id}{suffix}"
        if path.exists():
            return FileResponse(path, media_type=media, filename=path.name)
    raise HTTPException(status_code=404, detail="set not found")


@app.get("/api/mix/{mix_id}")
def mix_audio(mix_id: str):
    try:
        uuid.UUID(hex=mix_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="bad mix id")
    path = MIX_DIR / f"{mix_id}.wav"
    if not path.exists():
        raise HTTPException(status_code=404, detail="mix not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
