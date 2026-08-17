"""Render an ordered playlist into one continuous set.

Each track enters on the previous one's tempo, eases back to its own while it
plays alone, then eases onto the next track's tempo in time for the handover.
Nothing is held off its native tempo longer than the mix requires.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_analysis.loader import load_audio
from backend import phrasing, transitions
from backend.filler import make_bridge
from backend.setlist import camelot_distance, tempo_delta_percent
from backend.timewarp import ramp_plan, warp

RENDER_SR = 44100
PULSE_LOW, PULSE_HIGH = 85.0, 170.0
# Meeting in the middle splits the gap across both decks, so this cap is
# twice the per-track stretch. Beyond about 6% a track audibly sags — and in
# resample mode the pitch drops with it, which is very obvious on vocals — so
# wider gaps are handed over without beatmatching rather than forced.
MAX_MATCHABLE_PCT = 12.0
RAMP_SECONDS = 8.0

# Bridge length is set by how far it has to travel: roughly this much tempo
# change per bar, so a wide gap gets a longer ramp instead of a steeper one.
FILLER_PCT_PER_BAR = 4.0
FILLER_MIN_BARS = 8
FILLER_MAX_BARS = 16
FILLER_OVERLAP_BARS = 2


def filler_bars(tempo_from: float, tempo_to: float) -> int:
    if tempo_from <= 0:
        return FILLER_MIN_BARS
    gap = abs(tempo_to - tempo_from) / tempo_from * 100.0
    return int(min(FILLER_MAX_BARS, max(FILLER_MIN_BARS, round(gap / FILLER_PCT_PER_BAR))))


def set_tempo(tempo: float) -> float:
    """Fold a tempo into a common band so tracks are comparable.

    A 186 BPM track and a 93 BPM one are already mixable; only the label
    differs. Folding both into one band makes bar maths consistent, and the
    stretch rate stays correct because rate is target/folded on both sides.
    """
    if tempo <= 0:
        return 120.0
    while tempo < PULSE_LOW:
        tempo *= 2.0
    while tempo > PULSE_HIGH:
        tempo /= 2.0
    return tempo


def render_set(items: list[dict], preset: dict, out_path: Path,
               max_play_seconds: float = 75.0, use_fillers: bool = True) -> dict:
    """items: ordered [{path, result, entry_time, exit_time, energy}]."""
    n = len(items)
    tempos = [set_tempo(float(i["result"]["tempo"])) for i in items]

    # Blend tempo per junction: meet in the middle when the two are close
    # enough, otherwise leave both alone and hand over without beatmatching.
    reprise = [bool(item.get("is_reprise")) for item in items]

    junctions = []
    for i in range(n - 1):
        delta = tempo_delta_percent(tempos[i], tempos[i + 1])
        if reprise[i] or reprise[i + 1]:
            # A reprise exists to take the strain, so the handover happens at
            # the real track's own tempo and only the reprise bends. That
            # makes these junctions matched by construction.
            common = tempos[i + 1] if reprise[i] else tempos[i]
            junctions.append({"tempo": common, "matched": True, "delta": delta,
                              "absorbed": True})
        elif abs(delta) <= MAX_MATCHABLE_PCT:
            common = (tempos[i] + tempos[i + 1]) / 2.0
            junctions.append({"tempo": common, "matched": True, "delta": delta})
        else:
            junctions.append({"tempo": None, "matched": False, "delta": delta})

    bars = int(preset["crossfade_bars"])
    cycles = [phrasing.cycle_of(item["result"]) for item in items]

    # The phrasing agent sets each handover's length from the two tracks'
    # cycles, so it varies down the set rather than being one fixed number.
    plans = []
    for i in range(n - 1):
        plans.append(phrasing.blend_bars(
            cycles[i], cycles[i + 1], bars,
            context={
                # Where this handover sits in the set, 0 at the first and 1
                # at the last, so blends can open out towards the finish.
                "position": i / max(1, n - 2),
                "energy_shape": preset.get("energy_shape", "rise"),
                "tempo_delta_percent": junctions[i]["delta"],
                "energy_from": items[i].get("energy", 0.5),
                "energy_to": items[i + 1].get("energy", 0.5),
                "camelot_distance": camelot_distance(
                    (items[i]["result"].get("key") or {}).get("camelot"),
                    (items[i + 1]["result"].get("key") or {}).get("camelot"),
                ),
            },
        ))

    buffers, timeline = [], []

    for i, item in enumerate(items):
        native = tempos[i]
        tempo_in = junctions[i - 1]["tempo"] if i > 0 and junctions[i - 1]["matched"] else native
        tempo_out = junctions[i]["tempo"] if i < n - 1 and junctions[i]["matched"] else native

        # Enter on sam, not merely on a downbeat: landing in the wrong part
        # of the cycle offsets the whole track against the one still playing.
        entry, entry_moved = phrasing.snap_to_sam(float(item["entry_time"]), cycles[i])
        exit_, exit_moved = phrasing.snap_to_sam(float(item["exit_time"]), cycles[i])
        if exit_ <= entry:
            exit_ = float(item["exit_time"])
            exit_moved = False

        # Capping how long a track plays must still leave it on the cycle.
        # Snapping the cap to a plain downbeat put the exit on the wrong beat
        # of the cycle three times in four, and since the cap fires on nearly
        # every track that was undoing the sam alignment almost everywhere.
        if exit_ - entry > max_play_seconds:
            capped = phrasing.sam_at_or_after(entry + max_play_seconds, cycles[i], entry)
            exit_ = capped if capped is not None else _snap_downbeat(
                entry + max_play_seconds, item["result"], entry)
        if exit_ <= entry + 4.0:
            fallback = phrasing.sam_at_or_after(
                entry + max(8.0, max_play_seconds / 2), cycles[i], entry)
            exit_ = fallback if fallback is not None else entry + max(8.0, max_play_seconds / 2)

        # A bridged handover only overlaps the bridge by a couple of bars;
        # the bridge itself carries the rest of the distance.
        bridged = (i < n - 1 and use_fillers and not junctions[i]["matched"])
        out_bars = FILLER_OVERLAP_BARS if bridged else (plans[i]["bars"] if i < n - 1 else bars)

        # Blend length in output seconds, then in this track's input seconds.
        blend_out_secs = out_bars * 4 * 60.0 / tempo_out if i < n - 1 else 0.0
        in_bars = (FILLER_OVERLAP_BARS
                   if (i > 0 and use_fillers and not junctions[i - 1]["matched"])
                   else (plans[i - 1]["bars"] if i > 0 else bars))
        blend_in_secs = in_bars * 4 * 60.0 / tempo_in if i > 0 else 0.0
        blend_in_input = blend_in_secs * (tempo_in / native)
        blend_out_input = blend_out_secs * (tempo_out / native)

        y, sr = load_audio(item["path"], sr=RENDER_SR, mono=False)
        y = _stereo(y)
        seg = _slice(y, sr, entry, exit_ + blend_out_input + 0.5)

        if reprise[i]:
            # One straight ramp across the whole reprise. It passes through
            # the track's own tempo on the way, and spreading the travel over
            # its full length keeps the rate of change as gentle as possible.
            span = (exit_ - entry) + blend_out_input
            plan = [(0.0, tempo_in), (max(span, 1.0), tempo_out)]
        else:
            plan = ramp_plan(
                entry=0.0,
                exit_=exit_ - entry,
                native=native,
                tempo_in=tempo_in,
                tempo_out=tempo_out,
                blend_in=blend_in_input,
                blend_out=blend_out_input,
                ramp=RAMP_SECONDS,
            )
        # Keylock on a reprise: it is bent much further than a normal track,
        # and a 20% pitch drop is far more obvious than softened transients.
        mode = "vocoder" if reprise[i] and abs(tempo_in - tempo_out) > 0.12 * native else "resample"
        warped, to_out = warp(seg, sr, native, plan, mode=mode)

        exit_out = float(to_out(exit_ - entry))
        buffers.append({"audio": warped, "exit_out": exit_out,
                        "blend_out": blend_out_secs, "sr": sr})
        timeline.append({
            "index": i,
            "filename": item["result"].get("filename"),
            "native_tempo": round(float(item["result"]["tempo"]), 2),
            "set_tempo": round(native, 2),
            "enters_at_tempo": round(tempo_in, 2),
            "leaves_at_tempo": round(tempo_out, 2),
            "entry_cue": round(entry, 2),
            "exit_cue": round(exit_, 2),
            "camelot": (item["result"].get("key") or {}).get("camelot"),
            "is_reprise": reprise[i],
            "stretch_percent": round(max(abs(tempo_in / native - 1),
                                         abs(tempo_out / native - 1)) * 100, 1),
            "keylock": mode == "vocoder",
            "cycle": phrasing.describe(cycles[i]),
            "beats_per_cycle": cycles[i]["beats_per_cycle"],
            "cycle_confidence": cycles[i]["confidence"],
            "entry_on_sam": entry_moved or bool(cycles[i]["sam_times"]),
        })

    # Chain of things to play. A bridge is just another element: it hands over
    # to whatever follows exactly like a track does, so placement and overlap
    # handling do not need to know which is which.
    chain = []
    fillers = []
    for i, buf in enumerate(buffers):
        chain.append({"audio": buf["audio"], "handover_at": buf["exit_out"],
                      "blend": buf["blend_out"], "track": i})
        if i >= n - 1:
            continue
        if not (use_fillers and not junctions[i]["matched"]):
            continue

        span = filler_bars(tempos[i], tempos[i + 1])
        bridge, beats, duration = make_bridge(
            tempo_from=tempos[i], tempo_to=tempos[i + 1], bars=span,
            sr=RENDER_SR, preset=preset.get("id", "pop"),
            camelot=(items[i + 1]["result"].get("key") or {}).get("camelot"),
            seed=i,
        )
        # Hand over an exact number of bars before the end, so the incoming
        # track lands on a bridge downbeat rather than mid-bar.
        handover_beat = max(1, len(beats) - 1 - FILLER_OVERLAP_BARS * 4)
        handover_at = float(beats[handover_beat])
        chain.append({"audio": bridge, "handover_at": handover_at,
                      "blend": max(0.05, duration - handover_at), "track": None,
                      "from_tempo": tempos[i], "to_tempo": tempos[i + 1],
                      "bars": span, "duration": duration})
        fillers.append({
            "after": timeline[i]["filename"],
            "before": timeline[i + 1]["filename"],
            "bars": span,
            "tempo_from": round(tempos[i], 2),
            "tempo_to": round(tempos[i + 1], 2),
            "seconds": round(duration, 2),
        })

    offsets = [0.0]
    for k in range(len(chain) - 1):
        offsets.append(offsets[k] + chain[k]["handover_at"])

    # Shape every overlap in the chain before summing.
    styles = []
    for k in range(len(chain) - 1):
        a, b = chain[k]["audio"], chain[k + 1]["audio"]
        fade_n = int(round(chain[k]["blend"] * RENDER_SR))
        fade_n = max(1, min(fade_n, a.shape[-1], b.shape[-1]))

        i = chain[k]["track"]
        if i is None or chain[k + 1]["track"] is None:
            # Into or out of a bridge the tempos already agree, so a plain
            # equal-power blend is the right move.
            style = "blend"
            blend_tempo = chain[k].get("to_tempo") or tempos[max(0, i or 0)]
        else:
            # Style has to be chosen from the tempo gap that remains *at the
            # handover*, not the raw gap between the two tracks' own tempos.
            # On a matched junction both sides arrive at one tempo — either
            # by meeting in the middle or because a reprise took the strain —
            # so the residual is nothing, and passing the raw gap made this
            # pick a hard cut on junctions that were already beatmatched.
            residual = 0.0 if junctions[i]["matched"] else junctions[i]["delta"]
            style = transitions.choose(
                tuple(preset["styles"]),
                energy_from=float(items[i].get("energy", 0.5)),
                energy_to=float(items[i + 1].get("energy", 0.5)),
                bars=plans[i]["bars"],
                key_distance=camelot_distance(
                    (items[i]["result"].get("key") or {}).get("camelot"),
                    (items[i + 1]["result"].get("key") or {}).get("camelot"),
                ),
                tempo_delta_pct=residual,
                recent=tuple(styles),
            )
            blend_tempo = junctions[i]["tempo"] or tempos[i]
            timeline[i]["transition_to_next"] = style
            timeline[i]["transition_bars"] = plans[i]["bars"]
            timeline[i]["phrasing"] = plans[i]
            timeline[i]["beatmatched"] = junctions[i]["matched"]
            timeline[i]["tempo_delta_percent"] = round(junctions[i]["delta"], 2)

        a_tail, b_head = transitions.apply(style, a[..., -fade_n:], b[..., :fade_n],
                                           RENDER_SR, fade_n, tempo=blend_tempo)
        a[..., -fade_n:] = a_tail
        b[..., :fade_n] = b_head
        styles.append(style)

    # Junctions that got a bridge never went through transitions.choose.
    for i in range(n - 1):
        if "transition_to_next" not in timeline[i]:
            timeline[i]["transition_to_next"] = "bridge"
            timeline[i]["transition_bars"] = filler_bars(tempos[i], tempos[i + 1])
            timeline[i]["beatmatched"] = True
            timeline[i]["bridged"] = True
            timeline[i]["tempo_delta_percent"] = round(junctions[i]["delta"], 2)

    total = int(round(max(offsets[k] * RENDER_SR + chain[k]["audio"].shape[-1]
                          for k in range(len(chain)))))
    out = np.zeros((2, total), dtype=np.float32)

    # Match every element's level to the first track so nothing jumps.
    reference = _rms(chain[0]["audio"])
    filler_index = 0
    for k, element in enumerate(chain):
        level = _rms(element["audio"])
        gain = float(np.clip(reference / level, 0.3, 3.0)) if level > 1e-6 else 1.0
        start = int(round(offsets[k] * RENDER_SR))
        end = min(total, start + element["audio"].shape[-1])
        out[:, start:end] += element["audio"][:, : end - start] * gain
        if element["track"] is not None:
            timeline[element["track"]]["starts_at"] = round(offsets[k], 2)
            timeline[element["track"]]["gain_db"] = round(float(20 * np.log10(gain)), 2)
        else:
            fillers[filler_index]["starts_at"] = round(offsets[k], 2)
            filler_index += 1

    # Overlaps stack peaks, so a whole set routinely runs hot. Scaling the
    # entire mix down to fit its loudest instant would cost several dB across
    # the board; a limiter only ducks the moments that actually overshoot.
    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        out = _limit(out, RENDER_SR)

    wav_path = out_path.with_suffix(".wav")
    sf.write(str(wav_path), out.T, RENDER_SR, subtype="PCM_16")
    final, encoded = _encode(wav_path, out_path)

    return {
        "duration": round(total / RENDER_SR, 2),
        "tracks": len(items),
        "timeline": timeline,
        "fillers": fillers,
        "styles": styles,
        "peak": round(peak, 3),
        "format": "mp3" if encoded else "wav",
        "filename": final.name,
    }


def _encode(wav_path: Path, out_path: Path):
    """A full set is minutes long; mp3 keeps it a sane download."""
    ffmpeg = shutil.which("ffmpeg")
    mp3_path = out_path.with_suffix(".mp3")
    if ffmpeg is None:
        return wav_path, False
    proc = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav_path),
         "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return wav_path, False
    wav_path.unlink(missing_ok=True)
    return mp3_path, True


def _snap_downbeat(t: float, result: dict, floor_: float) -> float:
    grid = (result.get("beat_grid") or {}).get("downbeat_times") or []
    candidates = [d for d in grid if d > floor_ + 4.0]
    if not candidates:
        return t
    return float(min(candidates, key=lambda d: abs(d - t)))


def _stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.vstack([y, y])
    return y[:2] if y.shape[0] >= 2 else np.vstack([y[0], y[0]])


def _slice(y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    n = max(1, int(round((end - start) * sr)))
    out = np.zeros((y.shape[0], n), dtype=np.float32)
    s = max(0, int(round(start * sr)))
    e = min(y.shape[1], s + n)
    if e > s:
        out[:, : e - s] = y[:, s:e]
    return out


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _limit(x: np.ndarray, sr: int, ceiling: float = 0.84) -> np.ndarray:
    """Look-ahead peak limiter.

    The minimum filter pulls the gain down before the peak arrives rather
    than after it, so the reduction is never audible as a click, and the
    smoothing keeps the gain curve from distorting the waveform.

    The ceiling sits well below full scale on purpose: lossy encoding does
    not reproduce the waveform sample for sample, and a mix limited to 0.98
    decodes back above 0 dBFS. The headroom absorbs that overshoot.
    """
    from scipy.ndimage import minimum_filter1d

    peak = np.max(np.abs(x), axis=0)
    gain = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))

    # The minimum window has to be wider than the smoothing kernel. Smoothing
    # a narrow dip lifts its floor back up, which lets the very peak the dip
    # existed for slip through — that is how a limiter ends up above 0 dBFS.
    window = max(3, int(sr * 0.01))
    gain = minimum_filter1d(gain, size=window * 4 + 1, mode="nearest")
    kernel = np.hanning(window * 2 + 1)
    gain = np.convolve(gain, kernel / kernel.sum(), mode="same")
    out = x * gain
    return np.clip(out, -ceiling, ceiling).astype(np.float32)
