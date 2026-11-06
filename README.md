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

A 17-second vertical (2160×3840, 60 fps) intro for the official PSA crest. It follows the choreography of the reference animation the club provided. Only the motion was used from that video: none of its logo, branding, text, watermark, outro or audio appears.

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
| 12.6–14.6 s | hero sway | same, then the crest settles front-on and large, filling about 91% of the frame width |
| 14.8–15.9 s | — (the reference cuts to a TikTok outro here) | the motto **DUCATUS \| SCIENTIA \| MINISTERIUM** rises in under the crest, word by word |
| 15.9–17 s | — | still hold of the crest and motto |

## Look

* **Studio:** mid-grey with soft light falloff, faint volumetric haze, two overhead light beams, a floor shadow and a soft floor reflection.
* **Metal badge crest:** lighting is added on top of the original artwork. A height map built from the crest's own colour regions bevels every boundary and rounds the outer edge:
  * the silver frame, scrolls and hands are raised, polished brushed metal, with reflections that slide as the camera turns
  * the maroon field is recessed glossy enamel
  * the white and black areas are glossy enamel
  * the beige Arabic lettering and arcs are gold
  * creases get ambient-occlusion shading, and the badge has an extruded side wall
  Flat areas keep their original colour; the lighting mostly shows on edges and highlights. The realistic lighting stays on in the final hold, as requested.
* **Motto:** DUCATUS | SCIENTIA | MINISTERIUM in brushed-silver Cinzel Bold, with a bevel, a thin dark rim and a drop shadow.
* **Camera:** short camera shakes on the heavy hits (outline slam, iris lock, symbol release and crest complete). The final hold is rock-steady.

## Logo integrity

* The crest's artwork always comes from `assets/psa_logo_original.webp`. `src/psa/assets.py` only **masks** the original into layers (fragments, wedges, symbols, banner, letters). Recomposited, the layers match the original exactly: 0 of 2,087,481 solid pixels differ.
* Nothing is redrawn and no new artwork or lettering is added to the crest. The metal and enamel look comes only from lighting and shading applied on top.
* The white backdrop is removed. Enclosed white areas that are part of the design (the shield's white border and the capsules) are kept, and holes (inside letters, gaps beside the scrolls) become transparent.
* While the symbols rise, a flat fill of the field's own median maroon sits under them temporarily. The finished symbols cover it completely.

## Sound design (epic trailer)

Everything is synthesised from scratch in `src/psa/audio.py`, with no samples and nothing taken from the reference. It is driven by the same schedules as the picture:
* **Opening:** a dark low drone and air. The fragments fly in on a deep swirling whoosh and lock with a dense metallic rattle, then an outline slam (sub drop, low tom, dark metal).
* **Iris:** a heavy servo grind and ratchet, then a big lock hit with a braam.
* **Symbols:** they rise on hydraulic pressure and a rising rumble, ending in a pneumatic release and a clank.
* **Lettering:** the banner slides in, and the letters stamp in as muted thuds (heavier for PHARMACY).
* **Crest complete:** a riser leads into the main trailer hit (sub drop, braam, low tom, crack and dark metal), followed by deep camera whooshes.
* **Finale:** a riser leads into the settle, then three trailer title hits on DUCATUS, SCIENTIA and MINISTERIUM, the last one the biggest.
* **Score:** an original D-minor low-string ostinato with taiko drums that builds twice, sustained dark strings while the crest floats, and a final low strings-and-brass chord.
* **Mastering:** a hall reverb per bus and soft limiting, then the encode normalises to −14 LUFS.

```bash
python3 src/psa/assets.py        # build layers from the original crest
python3 src/psa/render_all.py    # 1020 frames → build_psa/
bash src/psa/encode.sh           # deliverables → output/psa_intro_9x16/
python3 src/psa/preview.py 1.4,3.6,6.9,9.6 sheet.jpg
```

---

# PSA — clean premium assembly on white (9:16, 10 s)

A 10-second vertical intro (2160×3840, 60 fps) that follows the second reference animation and the brief "clean, elegant opening on a white background". Only motion and timing were taken from the reference: none of its logo, text, watermark, TikTok outro or audio appears.

## Deliverables (`output/psa_white_10s/`)

| File | Use |
|---|---|
| `PSA_White_Reveal_1080x1920_60fps.mp4` | **For TikTok and Reels uploads.** With sound, −14 LUFS |
| `PSA_White_Reveal_1080x1920_60fps_MUTED.mp4` | The same picture without sound |
| `PSA_White_Reveal_2160x3840_60fps.mp4` | 4K vertical master |
| `PSA_White_Reveal_sound_design.wav` | The sound on its own |
| `final_frame.png` | Lossless last frame |

## Build order (inside out, as in the reference)

| Time | What happens |
|---|---|
| 0.15–0.85 s | the maroon shield field pops up from the centre (the icon's base) |
| 0.6–2.3 s | the hands rise, then the capsule and leaves pop in, the outer arc draws around, XXIII XXIV appears, and the Arabic letters arrive right to left |
| 1.85–2.75 s | the silver and white frame sweeps clockwise around the shield |
| 2.45–3.7 s | the four scroll ornaments glide in from outside |
| 3.5–6.05 s | EST 2011, then PHARMACY, then the banner, dots, dashes and STUDENT ASSOCIATION, letter by letter |
| 6.05 s | complete: a gentle pulse |
| 6.05–9.6 s | slight float, soft push-in, subtle tilt and depth separation between the scrolls, the shield and the lettering, a shimmer sweep, then sparkle glints at the edge points and travelling along the outline |
| 9.6–10 s | a clean, front-on, still hold |

## Look

* **Studio:** clean white with soft falloff and soft shadows. The crest sits slightly raised off the wall, as in the reference.
* **Crest:** lit as a cast-metal badge at the client's request ("ultra realistic, mature"). The silver frame and scrolls are brushed metal, the maroon and white are clear-coated enamel with a soft studio highlight and window reflection, the beige details read as gold, and the edges are bevelled with an extruded side wall.
* **Artwork:** this is all lighting on top of the original. The layers recomposite the original artwork exactly, and no lettering, symbols or shapes are changed or added.

## Sound

Everything is synthesised from scratch in `src/psa_white/audio.py` and timed from the renderer's element schedule:
* **Opening:** a deep soft boom as the shield appears, then muted thumps as the symbols pop in.
* **Arcs and lettering:** airy sweeps for the arcs, and a swish that travels right to left with the Arabic letters.
* **Frame:** a circular whoosh that pans with the frame sweep, ending in a firm lock.
* **Scrolls:** whooshes panned to each side, landing on soft thuds.
* **Letters:** soft texture ticks, heavier for PHARMACY.
* **Reveal:** a reverse-cymbal rise into the reveal boom with a soft braam.
* **Finish:** an airy shimmer for the shimmer sweep and filtered-noise sparkles for the glints. There are no tonal pings.
* **Score:** a warm string progression (Bm, G, A, then D add9 at the reveal) over a low pulse.

```bash
python3 src/psa_white/layers.py      # frame / ornament / symbol split
python3 src/psa_white/render_all.py  # 600 frames → build_psa_white/
bash src/psa_white/encode.sh         # deliverables → output/psa_white_10s/
python3 src/psa_white/preview.py 1,3,5,7,9.9 sheet.jpg
```
