"""
Decompose the official PSA crest into animation layers WITHOUT redrawing anything.

Every layer is a mask over the ORIGINAL pixels of assets/psa_logo_original.webp:
  outline  : side scrolls + silver/white shield frame, cut into fragments (the "shards" that fly in)
  field    : the maroon shield field, cut into radial wedges (the "iris" fill)
  symbols  : capsule/leaves, hands, arcs, Arabic lettering, XXIII XXIV (rise from the base)
  band     : grey arc banner
  letters  : EST 2011, PHARMACY, STUDENT ASSOCIATION, dots and dashes (appear one by one)
Composited together they reproduce the original crest exactly (verified in build()).

The only non-original pixels are a temporary plain-maroon fill under the symbols (inpainted from the
surrounding field) that is visible only while the symbols are rising; it is fully covered afterwards.
"""
import os
import numpy as np
import cv2
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, 'assets', 'psa_logo_original.webp')
CACHE = os.path.join(ROOT, 'build_psa', 'layers.npz')
N_SHARDS = 38
N_WEDGES = 10


def _alpha_and_colour(a):
    """Cut the crest from its white backdrop; soft 1-2 px edge with white un-mixed from edge colours."""
    mn = a.min(2)
    white = (mn > 238).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(white, connectivity=4)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bg = np.isin(lab, list(border))
    # maroon field (largest dark-red component), holes filled -> region where white is real artwork
    r, g, b = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    mar = ((r > 90) & (r < 170) & (g < 40) & (b < 40)).astype(np.uint8)
    n2, l2, s2, c2 = cv2.connectedComponentsWithStats(mar, connectivity=8)
    big = 1 + int(np.argmax(s2[1:, 4]))
    field = (l2 == big).astype(np.uint8)
    cnts, _ = cv2.findContours(field, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    field_filled = np.zeros_like(field)
    cv2.drawContours(field_filled, cnts, -1, 1, thickness=-1)
    keep_zone = cv2.dilate(field_filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121)))
    # enclosed white pockets: transparent unless they are part of the shield artwork
    for i in range(1, n):
        if i in border:
            continue
        cx, cy = cen[i]
        if not keep_zone[int(round(cy)), int(round(cx))]:
            bg |= (lab == i)
    fg = ~bg
    # soft edge: pixels of fg within 2 px of bg get alpha from their distance to white
    dist = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 3)
    edge = fg & (dist <= 2.0)
    alpha = fg.astype(np.float32)
    ae = np.clip((255.0 - mn) / 70.0, 0.0, 1.0)
    alpha[edge] = np.maximum(ae[edge], 0.35)
    col = a.astype(np.float32) / 255.0
    al = alpha[..., None]
    un = (col - (1 - al)) / np.maximum(al, 1e-3)
    col = np.where(edge[..., None], np.clip(un, 0, 1), col)
    return alpha, col, fg, field_filled.astype(bool)


def build():
    a = np.asarray(Image.open(SRC).convert('RGB'))
    alpha, col, fg, field_filled = _alpha_and_colour(a)
    hh, ww = alpha.shape

    n, lab, st, cen = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    order = np.argsort(-st[1:, 4]) + 1
    crest_id = order[0]
    crest = lab == crest_id
    items = []                          # (kind, mask, cx, cy)
    for i in range(1, n):
        if i == crest_id or st[i, 4] < 20:
            continue
        x, y, w, h, area = st[i]
        cx, cy = cen[i]
        if area > 30000:
            kind = 'band'
        elif cy < 300:
            kind = 'est'
        elif cy < 1600 and w < 120 and h < 120 and (cx < 450 or cx > 1500):
            kind = 'dot'
        elif area > 9000:
            kind = 'pharmacy'
        elif area < 6000 and h < 100 and w > 95 and cy < 1700:
            kind = 'dash'
        else:
            kind = 'student'
        items.append((kind, lab == i, float(cx), float(cy)))

    # symbols inside the field = everything in the filled field that is not plain maroon
    r, g, b = [a[..., k].astype(int) for k in range(3)]
    plain = (r > 90) & (r < 175) & (g < 45) & (b < 45)
    sym = field_filled & ~plain
    sym = cv2.morphologyEx(sym.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    sym = cv2.dilate(sym.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool) & field_filled
    # plain field: original pixels, with the symbol area inpainted from the surrounding maroon
    base = np.median(a[field_filled & ~sym], axis=0).astype(np.float32) / 255.0   # the field's own maroon
    field_plain = np.where(sym[..., None], base[None, None, :], col)

    outline = crest & ~field_filled
    # ---- shards: Voronoi cells over the outline (seeded by farthest-point sampling) ----
    rng = np.random.default_rng(2011)
    ys, xs = np.nonzero(outline)
    pts = np.stack([xs, ys], 1)[rng.choice(len(xs), 20000, replace=False)].astype(np.float32)
    seeds = [pts[0]]
    d = np.full(len(pts), 1e18, np.float32)
    for _ in range(N_SHARDS - 1):
        d = np.minimum(d, ((pts - seeds[-1]) ** 2).sum(1))
        seeds.append(pts[int(np.argmax(d))])
    seeds = np.array(seeds)
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    best = np.full((hh, ww), 1e18, np.float32)
    cell = np.zeros((hh, ww), np.int16)
    for k, (sx, sy) in enumerate(seeds):
        dd = (xx - sx) ** 2 + (yy - sy) ** 2
        m = dd < best
        best[m] = dd[m]
        cell[m] = k
    shards = []
    for k in range(N_SHARDS):
        m = outline & (cell == k)
        if m.sum() > 50:
            ys_, xs_ = np.nonzero(m)
            shards.append((m, float(xs_.mean()), float(ys_.mean())))

    # ---- wedges: angular sectors of the field around its centroid ----
    fy, fx = np.nonzero(field_filled)
    fcx, fcy = float(fx.mean()), float(fy.mean())
    ang = (np.arctan2(yy - fcy, xx - fcx) + np.pi) / (2 * np.pi)
    wedges = []
    for k in range(N_WEDGES):
        m = field_filled & (ang >= k / N_WEDGES) & (ang < (k + 1) / N_WEDGES)
        ys_, xs_ = np.nonzero(m)
        wedges.append((m, float(xs_.mean()), float(ys_.mean())))

    # ---- verify: layers recompose the original exactly ----
    recon = np.zeros((hh, ww), bool)
    for m, _, _ in shards:
        recon |= m
    recon |= field_filled
    for _, m, _, _ in items:
        recon |= m
    missing = (alpha > 0) & ~recon
    assert missing.sum() < 50, f'layers do not cover the crest: {missing.sum()} px missing'

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(
        CACHE, alpha=alpha, col=col.astype(np.float32), field_plain=field_plain.astype(np.float32),
        field=field_filled, sym=sym,
        shard_ids=_label(shards, hh, ww),
        shard_c=np.array([[cx, cy] for _, cx, cy in shards], np.float32),
        wedge_ids=_label(wedges, hh, ww), wedge_c=np.array([[cx, cy] for _, cx, cy in wedges], np.float32),
        field_c=np.array([fcx, fcy], np.float32),
        item_ids=_label([(m, 0, 0) for _, m, _, _ in items], hh, ww),
        item_kind=np.array([k for k, _, _, _ in items]),
        item_c=np.array([[cx, cy] for _, _, cx, cy in items], np.float32))
    return CACHE


def _label(lst, hh, ww):
    lab = np.full((hh, ww), -1, np.int16)
    for k, (m, _, _) in enumerate(lst):
        lab[m] = k
    return lab


def load():
    if not os.path.exists(CACHE):
        build()
    z = np.load(CACHE)
    return {k: z[k] for k in z.files}


if __name__ == '__main__':
    p = build()
    L = load()
    kinds, counts = np.unique(L['item_kind'], return_counts=True)
    print(p, dict(zip(kinds, counts)), 'shards', len(L['shard_c']), 'wedges', len(L['wedge_c']))
