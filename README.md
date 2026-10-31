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
