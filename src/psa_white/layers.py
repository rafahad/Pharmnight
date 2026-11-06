"""
Layer split for the white "premium assembly" intro, built on top of src/psa/assets.py.
Everything is a mask over the ORIGINAL crest pixels; nothing is redrawn.

Groups, in the order the reference builds them (inside out):
  field      maroon shield field (the icon's base)
  sym_*      central symbols: hands, capsules/leaves, arcs, XXIII XXIV, Arabic letters
  frame      white stripe + silver frame + thin maroon band around the field
  orn_*      the four ornament groups (side scrolls, bottom scrolls)
  items      banner, EST 2011, PHARMACY, dots, dashes, STUDENT ASSOCIATION (from assets.py)
"""
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'psa'))
import assets

FRAME_R = 155.0


def build():
    L = assets.load()
    a = L['alpha']
    field = L['field'].astype(bool)
    items = L['item_ids'] >= 0
    outline = (a > 0) & ~field & ~items
    dist = cv2.distanceTransform((~field).astype(np.uint8), cv2.DIST_L2, 5)
    frame = outline & (dist <= FRAME_R)
    orn = outline & (dist > FRAME_R)
    n, lab, st, cen = cv2.connectedComponentsWithStats(orn.astype(np.uint8), 8)
    big = [i for i in range(1, n) if st[i, 4] > 20000]
    orn_lab = np.full(a.shape, -1, np.int16)
    for k, i in enumerate(big):
        orn_lab[lab == i] = k
    # small leftover bits (scroll tips that poke past the frame) travel with the frame
    for i in range(1, n):
        if i not in big:
            frame |= lab == i
    orn_c = np.array([cen[j] for j in big], np.float32)

    # symbols: connected parts, classified by position / size
    sym = L['sym']
    n2, l2, s2, c2 = cv2.connectedComponentsWithStats(sym.astype(np.uint8), 8)
    sym_lab = np.full(a.shape, -1, np.int16)
    kinds, cents = [], []
    for i in range(1, n2):
        x, y, w, h, area = s2[i]
        cx, cy = c2[i]
        if area < 30:
            continue
        if cy < 560:
            kind = 'arabic'
        elif cy > 1040 and area > 4000:
            kind = 'hands'
        elif 960 < cy < 1080 and h < 70 and area < 5000:
            kind = 'roman'
        elif (w > 250 or h > 250) and area < 40000:
            kind = 'arc'
        elif area < 600:
            kind = 'dotpart'
        else:
            kind = 'icon'
        sym_lab[l2 == i] = len(kinds)
        kinds.append(kind)
        cents.append((cx, cy))
    out = os.path.join(os.path.dirname(assets.CACHE), 'white_layers.npz')
    np.savez_compressed(out, frame=frame, orn_lab=orn_lab, orn_c=orn_c, sym_lab=sym_lab,
                        sym_kind=np.array(kinds), sym_c=np.array(cents, np.float32))
    return out


def load():
    p = os.path.join(os.path.dirname(assets.CACHE), 'white_layers.npz')
    if not os.path.exists(p):
        build()
    z = np.load(p)
    return {k: z[k] for k in z.files}


if __name__ == '__main__':
    build()
    W = load()
    k, c = np.unique(W['sym_kind'], return_counts=True)
    print('symbols', dict(zip(k, c)), 'ornament groups', len(W['orn_c']), W['orn_c'].round())
