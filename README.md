# PHARMNIGHT 2026 — CINEMA SOIRÉE · The Grand Premiere

An 18-second cinematic logo reveal: a vintage movie theatre, a film projector switching on, golden spotlights, burgundy velvet curtains opening, and the official emblem revealed on stage.

## Deliverables (`output/`)

| File | Use |
|---|---|
| `PharmNight2026_CinemaSoiree_GrandPremiere_4K.mp4` | Master: 3840×2160, 24 fps, 18.0 s, H.264 + AAC, tagged BT.709 |
| `PharmNight2026_CinemaSoiree_GrandPremiere_1080p.mp4` | Smaller file for WhatsApp, Instagram, slides and projectors |
| `PharmNight2026_CinemaSoiree_score.wav` | The synced sound design and score on its own (48 kHz stereo) |
| `final_frame.png` | Lossless last frame, the hero shot |

### 9:16 vertical for Instagram Reels (`output/reels_9x16/`)

| File | Use |
|---|---|
| `PharmNight2026_CinemaSoiree_Reels_1080x1920.mp4` | **Upload this to Reels.** 1080×1920, 24 fps, 18.0 s, with the cinematic soundtrack (about −14.7 LUFS, close to Instagram's loudness level) |
| `PharmNight2026_CinemaSoiree_Reels_1080x1920_MUTED.mp4` | The same picture with no audio, for adding your own sound or an Instagram music track |
| `PharmNight2026_CinemaSoiree_Reels_2160x3840_master.mp4` | Vertical 4K master |
| `PharmNight2026_CinemaSoiree_Reels_cinematic_score.wav` | The cinematic soundtrack on its own |
| `final_frame_vertical.png` | Lossless last frame |

The vertical version is a separate render, not a crop. The stage is restaged for portrait: a narrower proscenium, curtains that open to frame the emblem, the projector and spotlights moved into the tall frame, and front-row seats along the bottom. The timing matches the 16:9 version exactly. The same logo rules apply, and in the 2160×3840 master the emblem's final 2 s are bit-identical to the source. In the 1080×1920 file the whole frame is scaled down to half size, so the emblem is shown at half resolution, but its colours and proportions are unchanged.

The cinematic soundtrack adds a trailer-style layer to the score: a sub-bass drone, a rising swell and a reverse-cymbal whoosh into the reveal, a large brass-like impact with a low boom at 6.0 s, a lift at 10 s and a softer brass-like impact at the 14 s finale.

```bash
LAYOUT=vertical python3 src/render_all.py   # frames → build_vertical/
bash src/encode_vertical.sh                  # Reels files → output/reels_9x16/
```

## How the logo is protected

The logo is **not** generated or redrawn by AI. The original bitmap (`assets/logo_original.webp`) is composited as one flat layer:

* It is only ever **uniformly scaled and moved** by the camera dolly. It is never rotated, warped, recoloured, cropped or stretched. During the camera arc the emblem stays front-facing, and the parallax happens in the theatre behind and in front of it.
* The emblem is cut out as a circle of radius 575 px around its ring centre. The artwork ends at the gold ring (r ≈ 548 px), so nothing in the emblem is cropped.
* The curtains, spotlights, haze, grain, vignette and bloom are all rendered **underneath** the emblem. None of them is applied to it. Dust motes are masked out of the emblem area.
* Light effects on the emblem are short and fade out completely. They are: the projection glow on arrival, the gold light running around the border, soft star glows, a light sweep, glints on the gold details and the final sparkles.
* **From 16.0 s to the end, the emblem is placed at native 1:1 scale on whole-pixel coordinates with nothing on top.** These frames are checked in code to be bit-identical to the source file (0 mismatched pixels out of 1,036,886).
* Delivery files are converted with the BT.709 matrix and tagged as BT.709, so players don't shift the colours.

## Timeline

| Time | Beat |
|---|---|
| 00:00–00:03 | Darkness. A vintage projector flickers on, reels spinning. Its beam cuts through floating dust onto the closed curtain, and the camera pushes in slowly. |
| 00:03–00:06 | Two golden spotlights strike, the footlights come on in sequence, and the velvet curtains open from the centre (the hem trails and sways). The camera dollies in, and a lavender and navy film strip sweeps across the frame. |
| 00:06–00:10 | The emblem appears whole in a wash of projector light. Champagne-gold light runs around its circular border, the stars glow, and a soft projector pool crosses it. |
| 00:10–00:14 | The spotlights frame the emblem. A light sweep crosses it and the gold details glint. The camera arcs to one side and comes back to centre. |
| 00:14–00:18 | Centred, symmetrical hero shot. Sparkles appear around the stars and the rim, and the emblem holds sharp and still for the last 2 s. |

## Re-rendering

Needs Python 3 with `numpy`, `opencv-python-headless` and `Pillow`, plus `ffmpeg`.

```bash
python3 src/audio.py build/score.wav     # score + sound design (~3 s)
python3 src/render_all.py                # 432 frames at 4K, lossless segments (~12–20 min on 4 cores)
bash src/encode.sh                       # delivery files into output/
python3 src/preview.py 3.6,6.0,12,17 preview.jpg   # quick contact sheet of chosen times
```

* `src/cinema.py`: the scene (camera, theatre, curtains, lights, projector, dust, film strip, emblem compositing)
* `src/audio.py`: procedural score (string pad, harp glissando, bells, projector, spotlight and curtain sound effects)

---

# PSA — Pharmacy Student Association crest intro (9:16)

A 16-second vertical (2160×3840, 60 fps) intro for the official PSA crest. It follows the choreography of the reference animation the club provided. Only the motion was used from that video: none of its logo, branding, text, watermark, outro or audio appears.

## Deliverables (`output/psa_intro_9x16/`)

| File | Use |
|---|---|
| `PSA_Crest_Intro_1080x1920_60fps.mp4` | **For TikTok and Reels uploads.** 1080×1920, 60 fps, with the original sound design (−14 LUFS) |
| `PSA_Crest_Intro_1080x1920_60fps_MUTED.mp4` | The same picture without sound |
| `PSA_Crest_Intro_2160x3840_60fps.mp4` | 4K vertical master |
| `PSA_Crest_Intro_sound_design.wav` | The sound design on its own |
| `final_frame.png` | Lossless last frame |

## Beat sheet (matched to the reference's sequence)

| Time | Reference beat | PSA version |
|---|---|---|
| 0.0–2.4 s | pieces fly in and snap into the outline on a light-grey wall | 38 fragments cut from the crest's own frame and side scrolls fly in, spin and lock in place, with soft wall shadows and an angled camera |
| 2.4–3.0 s | outline holds, camera drifts | same |
| 3.0–3.85 s | centre fills in like an iris | the maroon shield field closes in as 10 converging wedges |
| 3.95–5.55 s | central figure rises from the base | the hands, capsule, leaves, arcs, XXIII XXIV and Arabic lettering rise from the base upward |
| 5.65–7.7 s | lettering appears piece by piece | the grey banner unrolls, then EST 2011, PHARMACY, the dots and dashes, and STUDENT ASSOCIATION appear letter by letter |
| 7.72 s | crest complete | impact |
| 7.8–8.6 s | camera pulls back and the emblem floats above the floor | same, with a soft floor shadow |
| 8.6–11 s | floating, glints, dust | same, plus a gloss sweep |
| 11–12.6 s | push-in with a 3-D turn | same |
| 12.6–14.6 s | hero sway | same, then settles front-on |
| 14.6–16 s | — (the reference cuts to a TikTok outro here) | a still, front-on hold of the full crest |

## Logo integrity

* Every pixel of the crest comes from `assets/psa_logo_original.webp`. `src/psa/assets.py` only **masks** the original into layers (fragments, wedges, symbols, banner, letters). Recomposited, the layers match the original exactly: 0 of 2,087,481 solid pixels differ.
* The motion is perspective transforms of those original layers, like a printed acrylic slab seen by a moving camera. Depth uses darkened copies of the crest's own colours. No new artwork, lettering or colours are introduced.
* The white backdrop is removed. Enclosed white areas that are part of the design (the shield's white border and the capsules) are kept, and holes (inside letters, gaps beside the scrolls) become transparent.
* The only temporary non-original pixels are a flat fill of the field's own median maroon under the symbols while they rise. The finished symbols cover it completely.
* For the final hold (14.6–16 s), the original is placed front-on with a single high-quality downscale (0.96×) and nothing on top.

## Sound design

Everything is synthesised from scratch in `src/psa/audio.py`. The sound uses the same schedules as the renderer, so it lines up with the picture:
* each fragment has its own flight whoosh and a metallic lock tick, panned to where it lands
* the iris has a servo whine and an accelerating ratchet, then a heavy lock
* the symbols rise with a hydraulic hiss and motor hum, ending in a pneumatic release
* the banner slides left to right, and each letter has its own tick
* a riser leads into a sub-bass and metallic impact when the crest completes, and a pull-back whoosh follows
* glints get chimes, and the gloss sweeps get shimmer
* the final settle has a soft lock, a low boom and a bell chord

An original pulse-and-pad score runs underneath.

```bash
python3 src/psa/assets.py        # build layers from the original crest
python3 src/psa/render_all.py    # 960 frames → build_psa/
bash src/psa/encode.sh           # deliverables → output/psa_intro_9x16/
python3 src/psa/preview.py 1.4,3.6,6.9,9.6 sheet.jpg
```
