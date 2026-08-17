"""Variable-rate time stretching.

A constant stretch holds a track off its own tempo for its entire play time.
What a DJ actually does is ride the pitch fader: pull the track onto the
other one's tempo for the blend, then let it back to its native tempo once
it is playing alone. That needs a stretch rate that changes over time, which
is what this module provides.

The tempo plan is given as breakpoints in *input* time, linearly interpolated
and held flat outside the given range.
"""

import librosa
import numpy as np

N_FFT = 2048
HOP = 512


def warp(
    y: np.ndarray,
    sr: int,
    native_tempo: float,
    breakpoints: list[tuple[float, float]],
    mode: str = "resample",
    n_fft: int = N_FFT,
    hop: int = HOP,
):
    """Stretch `y` so it plays at the tempo described by `breakpoints`.

    mode="resample" moves pitch with tempo, exactly like a turntable pitch
    fader. It is sample-accurate and leaves transients untouched, which
    matters because the phase vocoder costs ~36% of onset sharpness even at
    a 2% stretch — enough to visibly soften drums.
    mode="vocoder" holds pitch constant (keylock) at that cost.

    Returns (warped_audio, input_time_to_output_time), the second being a
    callable so callers can find where a cue point ended up.
    """
    duration = y.shape[-1] / sr
    if duration <= 0:
        return y, (lambda t: np.asarray(t, dtype=float))

    times = np.array([b[0] for b in breakpoints], dtype=float)
    tempos = np.array([b[1] for b in breakpoints], dtype=float)
    order = np.argsort(times)
    times, tempos = times[order], tempos[order]

    # Fine grid over input time. np.interp holds the end values flat outside
    # the breakpoint range, which is the behaviour we want.
    step = hop / sr / 4.0
    grid = np.arange(0.0, duration + step, step)
    rate = np.interp(grid, times, tempos) / native_tempo
    rate = np.clip(rate, 0.25, 4.0)

    # Output time advances at 1/rate per unit of input time. Trapezoid rule
    # so the mapping stays accurate through the ramps.
    inv = 1.0 / rate
    tau = np.concatenate([[0.0], np.cumsum(0.5 * (inv[:-1] + inv[1:]) * np.diff(grid))])
    out_duration = float(tau[-1])

    def to_output(t):
        return np.interp(t, grid, tau)

    # Nothing to do if the plan never leaves the track's own tempo. Worth a
    # special case: most of a set is tracks playing solo, untouched.
    if np.max(np.abs(rate - 1.0)) < 1e-4:
        return y.astype(np.float32), to_output

    if mode == "resample":
        n_out = max(2, int(out_duration * sr))
        source_pos = np.interp(np.arange(n_out) / sr, tau, grid) * sr
        index = np.arange(y.shape[-1])
        if y.ndim == 1:
            y_out = np.interp(source_pos, index, y)
        else:
            y_out = np.vstack([np.interp(source_pos, index, ch) for ch in y])
        return y_out.astype(np.float32), to_output

    n_frames = 1 + y.shape[-1] // hop
    n_out = max(2, int(np.floor(out_duration * sr / hop)))
    frame_times = np.arange(n_out) * hop / sr
    t_out = np.interp(frame_times, tau, grid) * sr / hop
    t_out = np.clip(t_out, 0.0, n_frames - 1)

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    # hop_length/n_fft are deprecated here and unused when t_out is given.
    warped = librosa.phase_vocoder(D, t_out=t_out)
    y_out = librosa.istft(
        warped, hop_length=hop, n_fft=n_fft, length=int(out_duration * sr)
    )
    return y_out.astype(np.float32), to_output


def ramp_plan(
    entry: float,
    exit_: float,
    native: float,
    tempo_in: float,
    tempo_out: float,
    blend_in: float,
    blend_out: float,
    ramp: float = 8.0,
) -> list[tuple[float, float]]:
    """Tempo breakpoints for one track in a set, in input time from `entry`.

    Holds the incoming blend tempo, eases back to the track's own tempo for
    the solo stretch, then eases onto the next blend tempo in time for the
    handover. Ramps are shortened if the solo section is too short to hold
    them.
    """
    solo_start = entry + blend_in
    solo_end = max(solo_start, exit_)
    room = max(0.0, solo_end - solo_start)
    ramp = min(ramp, room / 2.0) if room > 0 else 0.0

    points = [(entry, tempo_in), (solo_start, tempo_in)]
    if ramp > 0:
        points.append((solo_start + ramp, native))
        points.append((solo_end - ramp, native))
    else:
        mid = (solo_start + solo_end) / 2.0
        points.append((mid, native))
    points.append((solo_end, tempo_out))
    points.append((exit_ + blend_out, tempo_out))

    # Keep strictly increasing so np.interp behaves.
    cleaned = []
    for t, v in points:
        if cleaned and t <= cleaned[-1][0]:
            t = cleaned[-1][0] + 1e-4
        cleaned.append((t, v))
    return cleaned
