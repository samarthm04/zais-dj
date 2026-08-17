"""Audio loading.

libsndfile handles mp3/wav/flac/aiff/ogg directly. It does not handle
m4a/aac/mp4, which is a common format for purchased pop tracks, so those
are decoded through ffmpeg into a temporary wav first.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np

# Analysis sample rate. 22.05 kHz keeps everything below ~11 kHz, which is
# plenty for beat, key and energy work, and roughly halves the compute.
ANALYSIS_SR = 22050

# Formats libsndfile cannot open; these go through ffmpeg.
FFMPEG_SUFFIXES = {".m4a", ".mp4", ".aac", ".wma"}


def _decode_with_ffmpeg(path: Path, sr: int, channels: int) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            f"{path.suffix} needs ffmpeg to decode, but ffmpeg was not found on PATH. "
            "Install it (brew install ffmpeg) or convert the file to mp3/wav."
        )
    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    proc = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
         "-ac", str(channels), "-ar", str(sr), str(tmp)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg could not decode {path.name}: {proc.stderr.strip()[:400]}")
    return tmp


def load_audio(
    path: str | Path, sr: int = ANALYSIS_SR, mono: bool = True
) -> tuple[np.ndarray, int]:
    """Load `path` as float32 at `sr`. Returns (samples, sample_rate).

    Mono at the analysis rate is what the pipeline wants. Rendering a mix to
    listen to wants stereo at full rate, hence the switches.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such audio file: {path}")

    tmp = None
    try:
        if path.suffix.lower() in FFMPEG_SUFFIXES:
            tmp = _decode_with_ffmpeg(path, sr, 1 if mono else 2)
            source = tmp
        else:
            source = path
        y, out_sr = librosa.load(str(source), sr=sr, mono=mono)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)

    if y.size == 0:
        raise ValueError(f"{path.name} decoded to zero samples")

    # Trim DC offset; leave amplitude alone so energy stays comparable.
    if y.ndim == 1:
        y = y - float(np.mean(y))
    else:
        y = y - np.mean(y, axis=1, keepdims=True)
    return y.astype(np.float32), int(out_sr)
