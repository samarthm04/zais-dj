"""Learned audio similarity, via teticio's Deej-AI encoder.

Tempo, key and energy say whether two tracks *can* be mixed. They say very
little about whether they belong next to each other — two records can agree
on all three and still sit wrong, because what makes them belong is era,
production, instrumentation and mood.

This model was trained on a million Spotify playlists: songs treated as
words, playlists as sentences, so the embedding encodes which tracks humans
actually put together. A CNN then learned to predict those vectors from a
spectrogram, which is what makes it usable on arbitrary local files.

Distance is Euclidean on the raw vectors, deliberately. The vectors share an
enormous common component — the mean vector is essentially as long as any
individual one — so cosine similarity on them comes out between 0.99 and
1.00 for every pair and discriminates nothing. Mean-centring fixes cosine,
but the centre moves whenever the library changes; Euclidean needs no
reference point and gives the same neighbours.

The dependency is heavy (torch) and GPL-3.0, so everything here is optional:
`available()` reports whether it loaded, and every caller degrades to the
tempo/key/energy heuristics when it did not.
"""

import json
from pathlib import Path

import numpy as np

MODEL_ID = "teticio/audio-encoder"
CACHE = Path(__file__).resolve().parent / "data" / "embeddings"

_encoder = None
_load_failed = False

# Distances run from roughly 30 to 150 across a varied library. This scales
# them to about 0..1 so the ordering cost can weigh them against the others.
DISTANCE_SCALE = 120.0


def available() -> bool:
    """Whether the encoder can be used at all, without loading it."""
    if _load_failed:
        return False
    try:
        import audiodiffusion.audio_encoder  # noqa: F401
        return True
    except Exception:
        return False


def _load():
    """Load once and keep it. Costs about 45 seconds the first time."""
    global _encoder, _load_failed
    if _encoder is not None or _load_failed:
        return _encoder
    try:
        from audiodiffusion.audio_encoder import AudioEncoder
        _encoder = AudioEncoder.from_pretrained(MODEL_ID)
    except Exception:
        _load_failed = True
        _encoder = None
    return _encoder


def _cache_path(track_id: str) -> Path:
    return CACHE / f"{track_id}.json"


def load_cached(track_id: str) -> np.ndarray | None:
    path = _cache_path(track_id)
    if not path.exists():
        return None
    try:
        return np.asarray(json.loads(path.read_text()), dtype=np.float32)
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def embed(track_id: str, audio_path) -> np.ndarray | None:
    """Embed one track, reusing the cached vector when there is one."""
    cached = load_cached(track_id)
    if cached is not None:
        return cached

    encoder = _load()
    if encoder is None:
        return None
    try:
        vector = np.asarray(encoder.encode([str(audio_path)]), dtype=np.float32).reshape(-1)
    except Exception:
        return None

    CACHE.mkdir(parents=True, exist_ok=True)
    _cache_path(track_id).write_text(json.dumps([round(float(v), 4) for v in vector]))
    return vector


def distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Scaled Euclidean distance. Lower means the two belong together more."""
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(np.linalg.norm(a - b) / DISTANCE_SCALE)


def forget(track_id: str) -> None:
    _cache_path(track_id).unlink(missing_ok=True)
