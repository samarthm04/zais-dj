"""Transition styles.

A plain crossfade is only one of the things a DJ does, and it is the wrong
one when two tracks have competing basslines or incompatible tempos. Each
style here shapes the overlap differently; `choose` picks one from what the
analysis says about the pair.
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

STYLES = ("blend", "bass_swap", "filter_sweep", "echo_out", "cut")


def choose(preferred: tuple[str, ...], *, energy_from: float, energy_to: float,
           bars: int = 8, key_distance: float = 0.0,
           tempo_delta_pct: float = 0.0, recent: tuple[str, ...] = ()) -> str:
    """Pick a style for one handover by scoring each against what it is for.

    Every style answers a different musical situation. A bass swap solves two
    loud records fighting over the low end. A filter sweep is how you make a
    busy track evaporate. An echo tail is a dramatic exit into an arrival. A
    cut is a slam, and only reads as one when the incoming track is loud and
    the overlap is short. A long blend is the euphoric layer, and needs the
    keys to agree or it is half a minute of dissonance.

    Scoring them rather than running an if-ladder matters because the ladder
    collapsed: once tempo mismatch was handled elsewhere its early branches
    stopped firing, and every handover fell through to the same default.
    """
    if not preferred:
        return "blend"

    # Nothing beatmatched: only the styles that never claimed to be.
    if abs(tempo_delta_pct) > 20:
        honest = [s for s in ("cut", "echo_out") if s in preferred]
        if honest:
            return honest[0] if energy_to >= energy_from else honest[-1]

    lift = energy_to - energy_from          # positive = stepping up
    both_loud = min(energy_from, energy_to)
    length = min(1.0, max(0.0, (bars - 4) / 20.0))   # 4 bars -> 0, 24 -> 1
    harmony = 1.0 - min(1.0, key_distance / 4.0)

    fit = {
        # A long layer of two harmonically close records: the euphoric one.
        "blend": 0.30 + 0.40 * length + 0.35 * harmony - 0.45 * abs(lift),
        # Both driving hard, so hand the low end over instead of stacking it.
        "bass_swap": 0.25 + 0.65 * both_loud - 0.25 * abs(lift),
        # Thin the outgoing track away — right when energy is falling off.
        "filter_sweep": 0.30 + 0.70 * max(0.0, -lift) + 0.20 * (1.0 - length),
        # Dramatic exit into an arrival; wants a shortish overlap.
        "echo_out": 0.25 + 0.55 * max(0.0, lift) + 0.30 * (1.0 - length),
        # A slam. Only reads as one going up, loud, and over quickly.
        "cut": 0.05 + 0.70 * max(0.0, lift) + 0.30 * energy_to - 0.60 * length,
    }

    # Variety is part of good mixing. The same move three times running stops
    # being a choice and starts being a tic.
    for offset, style in enumerate(reversed(recent[-2:])):
        if style in fit:
            fit[style] -= 0.40 if offset == 0 else 0.18

    options = [s for s in preferred if s in fit]
    if not options:
        return preferred[0]
    # A cut needs a short overlap; over a long one it just truncates a track.
    if bars > 8:
        options = [s for s in options if s != "cut"] or options
    return max(options, key=lambda s: fit[s])


def apply(style: str, a: np.ndarray, b: np.ndarray, sr: int, fade_n: int,
          tempo: float = 120.0) -> tuple:
    """Shape the two overlapping segments.

    `a` is the outgoing track's overlap region, `b` the incoming one's, both
    the same length as the overlap. `tempo` is the tempo the handover happens
    at, which the time-based effects need to stay in time. Returns the pair,
    processed.
    """
    if style == "cut":
        return _cut(a, b, sr, fade_n)
    if style == "bass_swap":
        return _bass_swap(a, b, sr, fade_n)
    if style == "filter_sweep":
        return _filter_sweep(a, b, sr, fade_n)
    if style == "echo_out":
        return _echo_out(a, b, sr, fade_n, tempo)
    return _blend(a, b, fade_n)


def _curves(fade_n: int):
    theta = np.linspace(0.0, np.pi / 2, fade_n, dtype=np.float32)
    return np.cos(theta), np.sin(theta)


def _envelope(length: int, curve: np.ndarray, at_start: bool) -> np.ndarray:
    env = np.ones(length, dtype=np.float32)
    n = min(length, len(curve))
    if at_start:
        env[:n] = curve[:n]
        if length > n:
            env[n:] = curve[n - 1] if n else 1.0
    else:
        env[length - n:] = curve[-n:]
    return env


def _blend(a, b, fade_n):
    """Equal-power crossfade. Constant perceived level for uncorrelated audio."""
    out_c, in_c = _curves(fade_n)
    return a * _envelope(a.shape[-1], out_c, False), b * _envelope(b.shape[-1], in_c, True)


def _bass_swap(a, b, sr, fade_n):
    """Hand the low end over at the midpoint, the way a mixer's EQ kills do.

    Both tracks stay audible throughout; only one owns the bass at a time, so
    the kicks and basslines never stack up.
    """
    swap = fade_n // 2
    a_hp = _highpass(a, sr, 240.0)
    b_hp = _highpass(b, sr, 240.0)

    a_out = a.copy()
    a_out[..., swap:] = a_hp[..., swap:]          # A loses its bass at the swap
    b_out = b.copy()
    b_out[..., :swap] = b_hp[..., :swap]          # B has none until then

    # Gentler level crossfade; the EQ is doing the work.
    out_c, in_c = _curves(fade_n)
    out_c = 0.35 + 0.65 * out_c
    in_c = 0.35 + 0.65 * in_c
    a_out = a_out * _envelope(a_out.shape[-1], out_c, False)
    b_out = b_out * _envelope(b_out.shape[-1], in_c, True)
    return a_out, b_out


def _filter_sweep(a, b, sr, fade_n):
    """Sweep a high-pass up through the outgoing track until it thins away."""
    cutoffs = np.geomspace(30.0, 4000.0, max(2, fade_n))
    a_sweep = _moving_highpass(a, sr, cutoffs)
    out_c, in_c = _curves(fade_n)
    return (a_sweep * _envelope(a_sweep.shape[-1], out_c, False),
            b * _envelope(b.shape[-1], in_c, True))


def _echo_out(a, b, sr, fade_n, tempo: float = 120.0):
    """Cut the outgoing track early and let a delay tail carry it out.

    Needs no tempo match, which is what makes it the safe move between tracks
    that cannot be beatmatched. The delay has to be derived from the tempo of
    the handover — a fixed delay drifts further out of time with every repeat
    and turns the tail into a stumble.
    """
    cut_at = fade_n // 3
    a_out = a.copy()
    tail = a_out[..., :cut_at]
    a_out[..., cut_at:] = 0.0

    eighth = 60.0 / max(tempo, 1e-6) / 2.0
    delay = max(1, int(sr * eighth))
    echo = np.zeros_like(a_out)
    level = 0.6
    for rep in range(1, 5):
        start = cut_at + delay * rep
        if start >= echo.shape[-1]:
            break
        span = min(echo.shape[-1] - start, tail.shape[-1])
        echo[..., start:start + span] += tail[..., :span] * (level ** rep)
    a_out = a_out + echo

    # The outgoing track is gone a third of the way in, so the incoming one
    # has to be up by then. A full-length fade would leave a hole where
    # neither track is carrying the mix.
    _, in_c = _curves(max(2, cut_at))
    return a_out, b * _envelope(b.shape[-1], in_c, True)


def _cut(a, b, sr, fade_n):
    """Hard switch on the downbeat, with just enough taper to avoid a click."""
    edge = max(64, int(sr * 0.008))
    a_out = a.copy()
    a_out[..., edge:] = 0.0
    a_out[..., :edge] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)
    b_out = b.copy()
    b_out[..., :edge] *= np.linspace(0.0, 1.0, edge, dtype=np.float32)
    return a_out, b_out


def _highpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    sos = butter(4, cutoff / (sr / 2), btype="highpass", output="sos")
    return sosfilt(sos, x, axis=-1).astype(np.float32)


def _moving_highpass(x: np.ndarray, sr: int, cutoffs: np.ndarray,
                     block: int = 1024) -> np.ndarray:
    """High-pass whose cutoff moves across the signal.

    Filter state is carried between blocks so the changing coefficients do
    not produce a click at every boundary.
    """
    n = x.shape[-1]
    out = np.zeros_like(x)
    channels = x.shape[0] if x.ndim > 1 else 1
    zi = None
    positions = np.linspace(0, len(cutoffs) - 1, max(1, int(np.ceil(n / block))))
    for i, pos in enumerate(positions):
        start, end = i * block, min(n, (i + 1) * block)
        if start >= end:
            break
        sos = butter(2, float(cutoffs[int(pos)]) / (sr / 2), btype="highpass", output="sos")
        if zi is None:
            zi = np.stack([sosfilt_zi(sos)] * channels, axis=-2) if x.ndim > 1 else sosfilt_zi(sos)
            zi = zi * 0.0
        chunk, zi = sosfilt(sos, x[..., start:end], axis=-1, zi=zi)
        out[..., start:end] = chunk
    return out.astype(np.float32)
