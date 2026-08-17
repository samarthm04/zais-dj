"""Top-level entry point: analyze_track()."""

import time
from pathlib import Path

from .beat_key import detect_beats, detect_key
from .cue_points import detect_cue_points
from .energy import energy_curve, sample_curve
from .loader import load_audio
from .structure import detect_segments
from .tala import detect_cycle


def analyze_track(path: str | Path, track_id: str | None = None) -> dict:
    """Analyze one audio file.

    Returns tempo, Camelot key, beat grid, energy curve, segments and ranked
    entry/exit cue points. All times are seconds from the start of the file.
    """
    started = time.perf_counter()
    path = Path(path)

    y, sr = load_audio(path)
    duration = len(y) / sr

    beats = detect_beats(y, sr)
    key = detect_key(y, sr)
    energy = energy_curve(y, sr, onset_env=beats["onset_env"])
    cycle = detect_cycle(y, sr, beats["beat_times"], onset_env=beats["onset_env"])
    segments = detect_segments(
        y, sr, energy["times"], energy["energy"], beats["downbeat_times"]
    )
    cues = detect_cue_points(
        duration=duration,
        tempo=beats["tempo"],
        downbeats=beats["downbeat_times"],
        energy_times=energy["times"],
        energy_values=energy["energy"],
        segments=segments,
    )

    return {
        "track_id": track_id,
        "filename": path.name,
        "duration": round(duration, 3),
        "sample_rate": sr,
        "tempo": beats["tempo"],
        "key": {
            "camelot": key["camelot"],
            "name": key["name"],
            "confidence": key["confidence"],
        },
        "beat_grid": {
            "beats_per_bar": beats["beats_per_bar"],
            "beat_times": beats["beat_times"],
            "downbeat_times": beats["downbeat_times"],
        },
        "cycle": cycle,
        "energy_curve": sample_curve(energy["times"], energy["energy"], step=1.0),
        "segments": segments,
        "cue_points": cues,
        "analysis_seconds": round(time.perf_counter() - started, 2),
    }
