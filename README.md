# zais-dj

An engine that turns a folder of songs into one continuous, beatmatched DJ set.

It analyses each track for tempo, key, energy, structure and rhythmic cycle,
finds the points where a track can be mixed in and out, orders the playlist
along an energy arc, and renders the whole thing as a single audio file with
real transitions between the tracks — bass swaps, filter sweeps, echo tails —
rather than crossfades.

Every track plays at its **own** tempo while it is playing alone. Tempo only
bends approaching a handover, and eases back once the track is on its own
again, which is what a DJ does with a pitch fader.

---

## Quick start

Requires **Python 3.13** — librosa and numba have no 3.14 wheels yet — and
`ffmpeg` on PATH for m4a decoding and mp3 output.

```bash
python3.13 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
backend/.venv/bin/python -m uvicorn backend.app:app --port 8765
```

Open <http://127.0.0.1:8765/>. Upload audio, press **Process**, then build a
set. Analysis runs about 15 seconds per track; a six-track set renders in
about 10 seconds.

Opening `frontend/index.html` directly from disk will not work — the page
needs the backend that serves it, and it will tell you so.

---

## How it works

### Analysis — `audio_analysis/`

**Beat grid.** Beat trackers return times quantised to their STFT hop, so
consecutive intervals alternate between neighbouring hop counts and the
median locks onto whichever is more common — biasing tempo by a couple of
BPM. The grid is instead fitted by least squares against beat index, then
refined by sliding a uniform comb over a finer onset envelope. On a
synthetic 120 BPM track this reads 119.96 BPM with a median beat error of
28 ms and no accumulated drift across three minutes.

**Downbeats.** Onset strength alone finds the backbeat, not beat 1 — pop
puts a loud snare on 2 and 4 that routinely outweighs the kick. Phase is
chosen from low-band onsets *minus* the bright band (snare rattle and
cymbals put far more energy up there than a kick does), plus harmonic
change, since chords turn over on the downbeat.

**Key.** Chroma and key profiles are both non-negative and broadly similar
in shape, so an uncentred cosine scores all 24 keys above 0.95 and separates
nothing. Both sides are mean-centred into a real correlation, and percussion
is stripped first because drums spray energy across every chroma bin. Top-1
against top-4 spread went from under 3% to 38–51%, and the remaining
confusions are relative and parallel keys — the same ones humans make.

**Rhythmic cycle — sam, taali, khaali.** A beat grid says where the beats
are, not which one the music lands on. This finds the cycle length, its
**sam** (the beat everything resolves to), the accented **taali** positions
and the deliberately empty **khaali**. Sam is the bar carrying the most
low-end weight and harmonic change; khaali is the emptiest bar near the
cycle's midpoint.

Cycle lengths are restricted to 4, 8, 16 and 32 beats. Odd-length talas —
Rupak at 7, Jhaptaal at 10, Dhamar at 14 — are deliberately excluded: the
beat tracker assumes 4/4 and lays a four-beat bar over everything, so a
genuine seven-beat cycle is already mis-gridded before this code sees it,
and "detecting" one would be reporting an artefact. Keherwa (8) and Teentaal
(16) survive a 4/4 grid, and those are most film and pop music anyway.

**Cue points.** Candidates are phrase-aligned downbeats scored on phrase
position, energy shape and proximity to a structural boundary. Entry cues
favour quiet, rising moments; exit cues favour falling energy late in a
track.

### Mixing — `backend/`

**Variable-rate time warp.** A constant stretch holds a track off its own
tempo for its entire play time. Here the rate follows a plan: match the
other track for the blend, ease back to native while playing solo. At
native tempo the audio is returned bit-identical.

Warping is done by **resampling**, not a phase vocoder — pitch moves with
tempo exactly like a turntable. A phase vocoder costs about 36% of onset
sharpness *even at a 2% stretch*, enough to visibly soften drums; resampling
costs 2.5% at 6%. Keylock (vocoder) is used only where a track is bent past
12%, since a large pitch drop is more obvious than softened transients.

**Transition styles.** Five, chosen by scoring each against what it is for:

| style | what it solves |
|---|---|
| `bass_swap` | two loud records fighting over the low end |
| `filter_sweep` | making a busy track evaporate as energy falls |
| `echo_out` | a dramatic exit into an arrival; needs no tempo match |
| `cut` | a slam — gated to short overlaps |
| `blend` | the long euphoric layer; needs the keys to agree |

A repetition penalty stops the same move appearing twice running.

**Blend length.** Measured in whole cycles of *both* tracks, so neither
arrives mid-cycle, then sized by how well the pair sits together and where
it falls in the set. Long overlaps belong late, once a set has built to
them. Sixteen bars over clashing keys is thirty seconds of dissonance, so
position raises the target while harmonic compatibility gates it.

**Ordering.** Tracks are placed against a target energy arc rather than a
monotonic climb — a set with no valleys has no peaks either. Presets choose
the shape: `rise` builds with contour, `peak` climbs to a summit and eases
off, `wave` swells repeatedly.

**Reprises.** Where two tracks are too far apart in tempo to beatmatch, an
already-played track returns at a *new* entry point and takes the strain: it
enters on the outgoing tempo and leaves on the incoming one, so neither real
track has to stretch. Re-entry points come from structural sections, not cue
points — cue points cluster in intros, so reprises would otherwise replay
the same few seconds every time.

**Bridges.** Where no reprise can span a gap, a synthesised percussion
bridge ramps between the two tempos. Its beat grid is solved analytically
so every hit lands exactly on the ramp.

**Feedback.** Like/dislike is recorded per handover, storing both a pair
aggregate (read back by ordering) and a feature row — style, bars, cycles,
tempo delta, key distance, stretch, position — so the data can train a model
later. Counts alone can say "these two worked"; they cannot say "16-bar bass
swaps under 5% tempo drift work", which is the useful generalisation.

---

## API

| method | path | purpose |
|---|---|---|
| `POST` | `/api/upload` | upload up to 10 audio files |
| `POST` | `/api/process` | run analysis over tracks |
| `GET` | `/api/tracks` | list the library with results |
| `DELETE` | `/api/track/{id}` | remove a track |
| `GET` | `/api/audio/{id}` | stream a track (supports Range) |
| `POST` | `/api/mix` | render one transition between two tracks |
| `GET` | `/api/mix/{id}` | stream a rendered transition |
| `POST` | `/api/set` | order the playlist and render a full set |
| `GET` | `/api/set/{id}` | stream a rendered set |
| `GET` | `/api/presets` | genre presets |
| `POST` | `/api/feedback` | rate one handover |
| `GET` | `/api/preferences` | what has been liked and disliked |

---

## Layout

```
audio_analysis/     offline analysis, no web dependencies
  loader.py         decoding, ffmpeg fallback for m4a
  beat_key.py       tempo, beat grid, downbeats, key
  tala.py           cycle length, sam, taali, khaali
  energy.py         energy curve
  structure.py      structural segmentation
  cue_points.py     ranked entry and exit cues
  pipeline.py       analyze_track()

backend/            FastAPI service and the mixing engine
  app.py            HTTP endpoints, upload and library handling
  timewarp.py       variable-rate stretch
  transitions.py    the five styles
  phrasing.py       sam alignment and blend length
  setlist.py        ordering, energy arc, presets
  setrender.py      renders a full set
  mix.py            renders a single transition
  filler.py         synthesised tempo-ramping bridges
  preferences.py    remembered verdicts

frontend/           plain HTML/CSS/JS, no build step
```

State lives in `backend/data/` — uploads, analysis JSON, rendered mixes and
`preferences.json`. It is git-ignored in full.

---

## Known limitations

- **4/4 only.** The beat tracker assumes four beats to a bar, so odd-metre
  music is mis-gridded before analysis begins.
- **Sam confidence is moderate** (0.35–0.50). Eight-bar phrases carry an
  inherent half-cycle ambiguity — is the phrase eight bars from here, or
  from four bars later?
- **The energy arc needs range.** A library where every track sits within a
  narrow energy band gives the ordering nothing to shape; the arc will ask
  for peaks and troughs that nothing available can fill.
- **Cue scoring is hand-tuned** and unvalidated against listening. This is
  what the feedback data exists to fix.
- **Tempo octave is ambiguous.** A 186 BPM track may be reported as 93. The
  grid is correct either way, but the label depends on convention.

---

## Audio

Bring your own. Nothing here downloads music, and `backend/data/` is
git-ignored so no audio ever enters the repository.

A DJ mix is a **derivative work**. If you intend to publish a set rather
than listen to it yourself, the source tracks need to permit that —
Creative Commons licences containing `ND` (NoDerivatives) forbid mixing
outright, and `NC` ones rule out commercial use.
