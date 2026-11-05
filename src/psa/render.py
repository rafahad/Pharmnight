"""
PSA crest intro renderer (9:16, 2160x3840, 60 fps) — "metal badge" look.

The crest is NEVER redrawn: its artwork is always the original image (assets/psa_logo_original.webp),
masked into layers by assets.py.  Realism is added as LIGHTING on top of that artwork:
  * a height map from the artwork's own colour regions (raised silver frame & scrolls, recessed maroon
    enamel field, raised lettering) gives bevels at every boundary and rounded outer edges,
  * silver/grey regions are treated as polished metal (moving reflections + brushed grain),
    maroon/white/black as glossy enamel, the beige/gold details as gold,
  * a key light fixed in the studio, so highlights slide across the badge as the camera turns,
  * ambient occlusion in crevices, an extruded side wall, contact/floor shadows and a soft floor reflection.
Flat areas keep their original colour; the lighting shows mainly on bevels, edges and highlights.
"""
import math
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assets
import timeline as TL

W, H = TL.W, TL.H
F = 3200.0
LIGHT = np.array([-0.55, -0.70, 0.75])
LIGHT /= np.linalg.norm(LIGHT)


class R:
    pass


# ---------------------------------------------------------------- material
def build_material(col, alpha, field_mask=None):
    """Height/normal/material maps derived from the artwork's own colour regions."""
    u8 = np.clip(np.round(col * 255), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((hh < 12) | (hh > 165)) & (ss > 110) & (vv > 50)
    pink = ((hh < 14) | (hh > 160)) & (ss > 35) & (ss <= 110) & (vv > 140)
    grey = (ss < 45) & (vv > 85) & (vv < 228)
    white = (ss < 45) & (vv >= 228)
    black = vv < 75
    gold = (hh >= 10) & (hh <= 35) & (ss >= 35) & (vv > 120)
    height = np.full(alpha.shape, 0.8, np.float32)
    height[red] = 0.92
    if field_mask is not None:
        height[red & field_mask] = 0.55                    # recessed enamel field
    height[pink] = 0.72
    height[grey] = 1.0                                     # raised polished silver
    height[white] = 0.78
    height[black] = 0.95
    height[gold] = 0.74
    metal = np.zeros(alpha.shape, np.float32)
    metal[grey] = 1.0
    metal[gold] = 0.85
    # rounded outer edge: fall-off towards the silhouette
    inside = (alpha > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    edge = np.clip(dist / 11.0, 0, 1)
    edge = edge * edge * (3 - 2 * edge)
    height = height * (0.25 + 0.75 * edge)
    height = cv2.GaussianBlur(height, (0, 0), 2.2)
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    k = 9.0
    nx, ny, nz = -gx * k, -gy * k, np.ones_like(gx)
    nn = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nn, ny / nn, nz / nn
    ao = 1.0 - 0.45 * np.clip(cv2.GaussianBlur(height, (0, 0), 9) - height, 0, 0.6) / 0.6
    metal = cv2.GaussianBlur(metal, (0, 0), 1.0)
    ks = (0.42 * metal + 0.30 * (1 - metal)).astype(np.float32)          # enamel is glossy, metal is bright
    kp = (28.0 * metal + 90.0 * (1 - metal)).astype(np.float32)
    rng = np.random.default_rng(12)
    brush = cv2.GaussianBlur(rng.standard_normal(alpha.shape).astype(np.float32), (0, 0), sigmaX=9, sigmaY=0.7)
    brush /= brush.std() + 1e-6
    return dict(nx=nx.astype(np.float32), ny=ny.astype(np.float32), nz=nz.astype(np.float32), ao=ao.astype(np.float32),
                metal=metal, ks=ks, kp=kp, brush=brush.astype(np.float32))


def light(col, M, cam, t):
    """Lit RGB of the artwork for the current camera (light fixed in the studio)."""
    Rm = cam['R']
    Lb = Rm.T @ LIGHT
    Vb = Rm.T @ np.array([0.0, 0.0, 1.0])
    d = M['nx'] * Lb[0] + M['ny'] * Lb[1] + M['nz'] * Lb[2]
    shade = np.clip(1.0 + 1.05 * (d - Lb[2]), 0.30, 1.65)
    Hh = Lb + Vb
    Hh /= np.linalg.norm(Hh)
    nh = np.clip(M['nx'] * Hh[0] + M['ny'] * Hh[1] + M['nz'] * Hh[2], 1e-4, 1.0)
    spec = M['ks'] * np.exp(M['kp'] * np.log(nh))
    # polished metal: broad reflections of the studio that slide as the badge turns, plus brushed grain
    ph = 2.6 * Vb[0] + 1.9 * Vb[1] + 0.15 * t
    env = 1.0 + M['metal'] * (0.20 * np.sin(R.uu * 0.0023 + R.vv * 0.0011 + ph) +
                              0.08 * np.sin(R.uu * 0.0061 - R.vv * 0.0042 + 1.7 * ph) + 0.035 * M['brush'])
    grad = 1.0 + 0.07 * (-(R.uu - 1000) / 1000 * Lb[0] * 1.5 - (R.vv - 1000) / 1000 * Lb[1] * 1.5)
    base = col * (shade * M['ao'] * env * grad)[..., None]
    tint = col * M['metal'][..., None] + (1 - M['metal'][..., None])          # metal speculars take its colour
    out = base + (spec[..., None] * tint)
    return np.clip(out, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- setup
def init():
    L = assets.load()
    R.col = L['col']
    R.alpha = L['alpha']
    R.field = L['field']
    R.sym = L['sym']
    R.fp = L['field_plain']
    R.shard_ids, R.wedge_ids, R.item_ids = L['shard_ids'], L['wedge_ids'], L['item_ids']
    R.shard_c, R.wedge_c, R.item_c = L['shard_c'], L['wedge_c'], L['item_c']
    R.field_c = L['field_c']
    R.item_kind = [str(k) for k in L['item_kind']]
    R.uu = np.arange(2000, dtype=np.float32)[None, :]
    R.vv = np.arange(2000, dtype=np.float32)[:, None]
    R.mat = build_material(R.col, R.alpha, R.field)
    R.mat_f = build_material(R.fp, R.field.astype(np.float32), R.field)
    R.masks = {'shard': _masks(R.shard_ids, len(R.shard_c)), 'wedge': _masks(R.wedge_ids, len(R.wedge_c)),
               'item': _masks(R.item_ids, len(R.item_c))}
    R.shard_sched = TL.shard_schedule(R.shard_c)
    R.wedge_sched = TL.wedge_schedule(len(R.wedge_c))
    R.item_sched = TL.letter_schedule(R.item_kind, R.item_c)
    ys, xs = np.nonzero(R.sym)
    R.sym_box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    x0, y0, x1, y1 = R.sym_box
    R.sym_m = R.sym[y0:y1, x0:x1, None].astype(np.float32)
    R.gy = np.arange(H, dtype=np.float32)[:, None]
    R.gx = np.arange(W, dtype=np.float32)[None, :]
    cnts, _ = cv2.findContours((R.alpha > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea)).reshape(-1, 2).astype(np.float32)
    pick = [hull[np.argmin(hull[:, 1])]]
    for _ in range(9):
        d = np.min(np.linalg.norm(hull[:, None, :] - np.array(pick)[None], axis=2), 1)
        pick.append(hull[int(np.argmax(d))])
    R.glint_pts = np.array(pick)
    rng = np.random.default_rng(5)
    R.dust = np.stack([rng.uniform(0, W, 46), rng.uniform(0, H, 46), rng.uniform(6, 20, 46),
                       rng.uniform(0.04, 0.12, 46), rng.uniform(0, 100, 46)], 1)
    # haze noise (quarter resolution), larger than the frame so it can drift
    hq, wq = H // 4, W // 4
    n = np.zeros((hq + 200, wq + 200), np.float32)
    for sc, wt in ((48, 0.35), (96, 0.4), (160, 0.25)):
        r = rng.standard_normal(((hq + 200) // sc + 3, (wq + 200) // sc + 3)).astype(np.float32)
        n += wt * cv2.resize(r, ((wq + 200) // sc * sc + 3 * sc, (hq + 200) // sc * sc + 3 * sc), interpolation=cv2.INTER_CUBIC)[:hq + 200, :wq + 200]
    R.haze = (n - n.mean()) / (n.std() + 1e-6)
    R.qy = np.arange(hq, dtype=np.float32)[:, None] * 4
    R.qx = np.arange(wq, dtype=np.float32)[None, :] * 4
    R.final_cache = None
    init_tagline()


def _masks(ids, n):
    out = []
    for k in range(n):
        m = ids == k
        ys, xs = np.nonzero(m)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        out.append((np.ascontiguousarray(m[y0:y1, x0:x1, None], np.float32), x0, y0))
    return out


def crop(kind, k, pm, fpm):
    m, x0, y0 = R.masks[kind][k]
    src = fpm if kind == 'wedge' else pm
    return src[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] * m, x0, y0


# ---------------------------------------------------------------- geometry
def rot_matrix(yaw, pitch, roll):
    y, p, r = map(math.radians, (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def hcam(cam, z, bob=0.0):
    Rm = cam['R']
    D = F / cam['scale']
    A = np.array([[1, 0, -1000.0], [0, 1, -1000.0 + bob], [0, 0, 1]])
    M = np.array([[F * Rm[0, 0], F * Rm[0, 1], F * Rm[0, 2] * z],
                  [F * Rm[1, 0], F * Rm[1, 1], F * Rm[1, 2] * z],
                  [-Rm[2, 0], -Rm[2, 1], D - Rm[2, 2] * z]])
    S = np.array([[1, 0, cam['cx']], [0, 1, cam['cy']], [0, 0, 1]])
    return S @ M @ A


def T(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1.0]])


def RS(deg, s):
    a = math.radians(deg)
    return np.array([[s * math.cos(a), -s * math.sin(a), 0], [s * math.sin(a), s * math.cos(a), 0], [0, 0, 1.0]])


def warp_over(dst, src, Hm, shade=None, opacity=1.0, scale_hint=1.0):
    h, w = src.shape[:2]
    if scale_hint < 0.8 and min(h, w) > 64:
        pre = max(scale_hint * 1.2, 0.2)
        nw, nh = max(int(w * pre), 2), max(int(h * pre), 2)
        src = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        Hm = Hm @ np.diag([w / nw, h / nh, 1.0])
        h, w = nh, nw
    pts = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float).T
    q = Hm @ pts
    if np.any(q[2] <= 1e-6):
        return None
    q = q[:2] / q[2]
    bx0, by0 = int(max(math.floor(q[0].min()) - 2, 0)), int(max(math.floor(q[1].min()) - 2, 0))
    bx1, by1 = int(min(math.ceil(q[0].max()) + 2, dst.shape[1])), int(min(math.ceil(q[1].max()) + 2, dst.shape[0]))
    if bx1 <= bx0 or by1 <= by0:
        return None
    out = cv2.warpPerspective(src, T(-bx0, -by0) @ Hm, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if shade is not None:
        out[..., :3] *= shade
    if opacity < 1.0:
        out *= opacity
    reg = dst[by0:by1, bx0:bx1]
    a = out[..., 3:4]
    reg[..., :3] = out[..., :3] + reg[..., :3] * (1 - a)
    reg[..., 3:4] = a + reg[..., 3:4] * (1 - a)
    return (bx0, by0, bx1, by1)


def warp_alpha_add(dst, a_src, Hm, gain):
    h, w = a_src.shape
    pts = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float).T
    q = Hm @ pts
    if np.any(q[2] <= 1e-6):
        return
    q = q[:2] / q[2]
    bx0, by0 = int(max(q[0].min() - 2, 0)), int(max(q[1].min() - 2, 0))
    bx1, by1 = int(min(q[0].max() + 2, dst.shape[1])), int(min(q[1].max() + 2, dst.shape[0]))
    if bx1 <= bx0 or by1 <= by0:
        return
    out = cv2.warpPerspective(a_src, T(-bx0, -by0) @ Hm, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR)
    reg = dst[by0:by1, bx0:bx1]
    np.maximum(reg, out * gain, out=reg)


# ---------------------------------------------------------------- scene state
def settled_canvas(t, pm, fpm):
    if t >= TL.T_COMPLETE + 0.02:
        return pm, []
    flying = []
    canvas = np.zeros_like(pm)
    landed = np.array([t >= s['land'] for s in R.shard_sched])
    m = np.r_[False, landed][R.shard_ids + 1]
    canvas[m] = pm[m]
    flying += [('shard', k) for k, s in enumerate(R.shard_sched) if s['start'] <= t < s['land']]
    landed_w = np.array([t >= s['land'] for s in R.wedge_sched])
    m = np.r_[False, landed_w][R.wedge_ids + 1]
    canvas[m] = fpm[m]
    flying += [('wedge', k) for k, s in enumerate(R.wedge_sched) if s['start'] <= t < s['land']]
    if t > TL.SYM_START and landed_w.all():
        e = TL.s5(TL.SYM_START, TL.SYM_END, t)
        yw = TL.SYM_Y0 + (TL.SYM_Y1 - TL.SYM_Y0) * e
        dy = 34.0 * (1 - e)
        x0, y0, x1, y1 = R.sym_box
        sub = pm[y0:y1, x0:x1] * R.sym_m
        if t < TL.SYM_END:
            if dy > 0.01:
                sub = cv2.warpAffine(sub, np.float32([[1, 0, 0], [0, 1, dy]]), (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
            rows = np.arange(y0, y1, dtype=np.float32)[:, None]
            sub = sub * np.clip((rows - yw) / 46.0 + 0.5, 0, 1)[..., None]
        reg = canvas[y0:y1, x0:x1]
        reg[:] = sub + reg * (1 - sub[..., 3:4])
    for k, (st, ld) in enumerate(R.item_sched):
        if R.item_kind[k] == 'band':
            if t <= st:
                continue
            c, x0, y0 = crop('item', k, pm, fpm)
            xw = x0 + (c.shape[1] + 40) * TL.s5(st, ld, t)
            cols = np.arange(x0, x0 + c.shape[1], dtype=np.float32)[None, :]
            c = c * np.clip((xw - cols) / 30.0, 0, 1)[..., None]
            reg = canvas[y0:y0 + c.shape[0], x0:x0 + c.shape[1]]
            reg[:] = c + reg * (1 - c[..., 3:4])
        elif t >= ld:
            c, x0, y0 = crop('item', k, pm, fpm)
            reg = canvas[y0:y0 + c.shape[0], x0:x0 + c.shape[1]]
            reg[:] = c + reg * (1 - c[..., 3:4])
        elif t >= st:
            flying.append(('item', k))
    return canvas, flying


def element_pose(kind, k, t):
    if kind == 'shard':
        s = R.shard_sched[k]
        cx, cy = R.shard_c[k]
        u = (t - s['start']) / (s['land'] - s['start'])
        e = TL.ease_out_cubic(u)
        rot = s['rot'] * (1 - TL.ease_out_back(u, 1.1))
        sc = s['sc'] + (1 - s['sc']) * TL.ease_out_cubic(u * 1.1)
        z = s['z'] * (1 - u) ** 2.2
        dx, dy = s['dx'] * (1 - e), s['dy'] * (1 - e)
        op = TL.clamp(u * 4)
    elif kind == 'wedge':
        s = R.wedge_sched[k]
        cx, cy = R.field_c
        wx, wy = R.wedge_c[k]
        u = (t - s['start']) / (s['land'] - s['start'])
        e = TL.ease_out_cubic(u)
        ang = math.atan2(wy - cy, wx - cx)
        rot = s['rot'] * (1 - e)
        sc = 1.0
        z = s['z'] * (1 - u) ** 2
        dx, dy = math.cos(ang) * s['push'] * (1 - e), math.sin(ang) * s['push'] * (1 - e)
        op = TL.clamp(u * 3)
    else:
        st, ld = R.item_sched[k]
        cx, cy = R.item_c[k]
        u = (t - st) / (ld - st)
        e = TL.ease_out_cubic(u)
        rot = 0.0
        sc = 1 + 0.45 * (1 - TL.ease_out_back(u, 1.6))
        z = 150.0 * (1 - u) ** 2
        dx, dy = 0.0, -40.0 * (1 - e)
        op = TL.clamp(u * 3.5)
    m, x0, y0 = R.masks[kind][k]
    M = T(cx + dx, cy + dy) @ RS(rot, sc) @ T(-cx, -cy) @ T(x0, y0)
    return M, z, op


# ---------------------------------------------------------------- environment
GREY = np.array([0.985, 0.99, 1.0], np.float32)


def background(t, cam):
    w = TL.wall_amount(t)
    gy, gx = R.gy, R.gx
    out = np.zeros((H, W, 3), np.float32)
    rr = ((gx - W / 2) / (W * 0.75)) ** 2 + ((gy - H / 2) / (H * 0.62)) ** 2
    if w > 0:
        p = hcam(cam, -60.0) @ np.array([500.0, 120.0, 1.0])
        px, py = p[0] / p[2], p[1] / p[2]
        pool = np.exp(-(((gx - px) / 2000.0) ** 2 + ((gy - py) / 2200.0) ** 2))
        v = 0.50 + 0.15 * pool - 0.13 * np.clip(rr, 0, 1.6)
        v = v + 0.04 * math.sin(math.radians(cam['yaw'])) * (gx - W / 2) / W
        out += w * v[..., None] * GREY
    if w < 1:
        yh = cam['cy'] + cam['scale'] * 1200.0
        above = 0.60 - 0.15 * np.clip((yh - gy) / yh, 0, 1) ** 1.2
        below = 0.58 - 0.12 * np.clip((gy - yh) / (H - yh), 0, 1) ** 0.8
        k = 1 / (1 + np.exp(-(gy - yh) / 90.0))
        v = above * (1 - k) + below * k
        pool = np.exp(-(((gx - W / 2) / 1400.0) ** 2 + ((gy - cam['cy']) / 1600.0) ** 2))
        v = v + 0.07 * pool - 0.10 * np.clip(rr, 0, 1.6)
        out += (1 - w) * np.broadcast_to(v, (H, W))[..., None] * GREY
    return out


def haze(img, t, cam):
    """Soft volumetric beams from two overhead lights, through slowly drifting haze."""
    ox, oy = int(30 + 18 * t) % 200, int(100 - 9 * t) % 200
    hz = R.haze[oy:oy + H // 4, ox:ox + W // 4]
    acc = np.zeros((H // 4, W // 4), np.float32)
    tx, ty = cam['cx'], cam['cy']
    for sx in (W * 0.12, W * 0.88):
        sy = -250.0
        ax, ay = tx - sx, ty - sy
        ln = math.hypot(ax, ay)
        ax, ay = ax / ln, ay / ln
        vx, vy = R.qx - sx, R.qy - sy
        along = vx * ax + vy * ay
        perp = np.abs(vx * ay - vy * ax)
        width = 120 + 0.33 * np.maximum(along, 0)
        acc += np.exp(-(perp / width) ** 2) * np.clip(along / 600.0, 0, 1) * np.exp(-np.maximum(along - ln, 0) / 700.0)
    amb = 0.35 + 0.65 * np.exp(-((R.qy - 600) / 1500.0) ** 2)
    v = (0.045 * acc + 0.016 * amb) * (0.93 + 0.07 * hz)
    v *= 0.6 + 0.4 * TL.sstep(0.0, 2.0, t)
    img += cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * np.array([1.0, 0.99, 0.97], np.float32)


def draw_dust(img, t):
    amt = TL.sstep(7.9, 8.8, t)
    if amt <= 0:
        return
    for x, y, r, a, ph in R.dust:
        yy = (y - 26 * t) % H
        xx = x + 40 * math.sin(t * 0.4 + ph)
        tw = 0.6 + 0.4 * math.sin(t * 1.7 + ph)
        x0, x1 = int(max(xx - 3 * r, 0)), int(min(xx + 3 * r, W))
        y0, y1 = int(max(yy - 3 * r, 0)), int(min(yy + 3 * r, H))
        if x1 <= x0 or y1 <= y0:
            continue
        g = np.exp(-(((np.arange(x0, x1)[None, :] - xx) ** 2 + (np.arange(y0, y1)[:, None] - yy) ** 2) / (2 * r * r)))
        img[y0:y1, x0:x1] += (g * a * tw * amt * 0.5)[..., None]


def flare(img, x, y, size, amp):
    ext = int(size * 1.3) + 3
    x0, x1 = int(max(x - ext, 0)), int(min(x + ext, W))
    y0, y1 = int(max(y - ext, 0)), int(min(y + ext, H))
    if x1 <= x0 or y1 <= y0:
        return
    xx = np.arange(x0, x1, dtype=np.float32)[None, :] - x
    yy = np.arange(y0, y1, dtype=np.float32)[:, None] - y
    rays = np.exp(-np.abs(yy) / 3.0 - np.abs(xx) / (size * 0.42)) + np.exp(-np.abs(xx) / 3.0 - np.abs(yy) / (size * 0.42))
    r2 = xx * xx + yy * yy
    core = np.exp(-r2 / (2 * (size * 0.06) ** 2)) + 0.35 * np.exp(-r2 / (2 * (size * 0.2) ** 2))
    v = np.clip((rays + core) * amp, 0, 1.2)
    reg = img[y0:y1, x0:x1]
    reg[:] = np.clip(reg + v[..., None] * np.array([1.0, 0.98, 0.94], np.float32), 0, 1)


def gloss(canvas, t):
    g, p = 0.0, None
    for a, b in TL.GLOSS:
        if a < t < b:
            g = TL.bump(t, a, a + 0.15, b - 0.15, b)
            p = -900 + 3900 * TL.s5(a, b, t)
    if g <= 0:
        return canvas
    band = np.exp(-(((R.uu + 0.65 * R.vv) - p) / 120.0) ** 2) * 0.26 * g
    out = canvas.copy()
    out[..., :3] += band[..., None] * (out[..., 3:4] - out[..., :3])
    return out


def floor_geom(cam, t, bob):
    sc = cam['scale']
    lift = -bob / 14.0 if bob else 0.0
    yb = cam['cy'] + sc * 1000.0
    return yb + sc * TL.shadow_offset(t), sc, lift


def floor_shadow(img, cam, t, bob):
    amt = 1 - TL.wall_amount(t)
    if amt <= 0:
        return
    cy, sc, lift = floor_geom(cam, t, bob)
    rx, ry = sc * 760.0 * (1 - 0.05 * lift), sc * 70.0
    x0, x1 = int(max(cam['cx'] - rx * 2, 0)), int(min(cam['cx'] + rx * 2, W))
    y0, y1 = int(max(cy - ry * 5, 0)), int(min(cy + ry * 5, H))
    xx = (np.arange(x0, x1, dtype=np.float32)[None, :] - cam['cx']) / rx
    yy = (np.arange(y0, y1, dtype=np.float32)[:, None] - cy) / ry
    g = np.exp(-(xx ** 2 + yy ** 2) * 1.6) * 0.75 + np.exp(-(xx ** 2 * 0.5 + yy ** 2 * 0.35)) * 0.25
    img[y0:y1, x0:x1] *= (1 - 0.42 * amt * (1 - 0.18 * lift) * g)[..., None]


def floor_reflection(img, buf, cam, t, bob):
    """Faint, blurred reflection of the badge in the polished studio floor."""
    amt = 1 - TL.wall_amount(t)
    if amt <= 0:
        return
    cy, sc, _ = floor_geom(cam, t, bob)
    ym = int(cy)
    span = min(900, ym, H - ym)
    if span <= 10:
        return
    src = buf[ym - span:ym][::-1]
    src = cv2.GaussianBlur(src, (0, 0), 4)
    fall = (np.exp(-np.arange(span, dtype=np.float32) / 220.0) * 0.10 * amt)[:, None, None]
    reg = img[ym:ym + span]
    reg[:] = reg * (1 - src[..., 3:4] * fall) + src[..., :3] * fall


def composite(img, buf):
    img *= (1 - buf[..., 3:4])
    img += buf[..., :3]


# ---------------------------------------------------------------- frame
def render_frame(t):
    cam = TL.camera(t)
    cam['R'] = rot_matrix(cam['yaw'], cam['pitch'], cam['roll'])
    bob = TL.bob(t)
    img = background(t, cam)
    haze(img, t, cam)
    draw_dust(img, t)

    if t >= TL.T_SETTLE:
        return render_hold(img, cam, t)

    lit = light(R.col, R.mat, cam, t)
    pm = np.dstack([lit * R.alpha[..., None], R.alpha])
    if t < TL.T_COMPLETE + 0.02:
        flit = light(R.fp, R.mat_f, cam, t)
        fpm = np.dstack([flit * R.field[..., None], R.field.astype(np.float32)])
    else:
        fpm = None
    canvas, flying = settled_canvas(t, pm, fpm)
    w = TL.wall_amount(t)
    thick = TL.thickness(t)

    if w > 0:
        sh = np.zeros((H // 2, W // 2), np.float32)
        half = np.diag([0.5, 0.5, 1.0])
        a_small = cv2.resize(canvas[..., 3], (500, 500), interpolation=cv2.INTER_AREA)
        off = 14.0 + thick * 0.5
        warp_alpha_add(sh, a_small, half @ hcam(cam, -thick - 6, bob) @ T(off * 0.35, off * 0.6) @ np.diag([4, 4, 1.0]), 1.0)
        for kind, k in flying:
            M, z, op = element_pose(kind, k, t)
            m, _, _ = R.masks[kind][k]
            o = 14.0 + z * 0.9
            warp_alpha_add(sh, np.ascontiguousarray(m[..., 0]), half @ hcam(cam, -thick - 6, bob) @ T(o * 0.35, o * 0.6) @ M,
                           op * max(0.25, 1 - z / 900))
        sh = cv2.GaussianBlur(sh, (0, 0), 9)
        sh = cv2.resize(sh, (W, H), interpolation=cv2.INTER_LINEAR)
        img *= (1 - 0.42 * w * sh)[..., None]
    if w < 1:
        floor_shadow(img, cam, t, bob)

    buf = np.zeros((H, W, 4), np.float32)
    canvas = gloss(canvas, t)
    sc = cam['scale']
    nside = 10 if thick > 1 else 0
    for i in range(nside, 0, -1):
        z = -thick * i / nside
        # side wall: darker, with a soft highlight band near the front edge (bevelled metal)
        shade = 0.30 + 0.30 * (1 - i / nside) ** 2
        warp_over(buf, canvas, hcam(cam, z, bob), shade=shade, scale_hint=sc)
    warp_over(buf, canvas, hcam(cam, 0.0, bob), scale_hint=sc)

    flying.sort(key=lambda e: -element_pose(*e, t)[1])
    for kind, k in flying:
        M, z, op = element_pose(kind, k, t)
        c, _, _ = crop(kind, k, pm, fpm)
        tk = max(thick, 14.0)
        for i in ((3, 2, 1) if kind != 'wedge' else ()):
            warp_over(buf, c, hcam(cam, z - tk * i / 3, bob) @ M, shade=0.32 + 0.1 * (3 - i), opacity=op, scale_hint=sc)
        warp_over(buf, c, hcam(cam, z, bob) @ M, opacity=op, scale_hint=sc)

    floor_reflection(img, buf, cam, t, bob)
    composite(img, buf)
    for j, tg in enumerate(TL.GLINTS):
        k = (t - tg) / 0.5
        if 0 < k < 1:
            u, v = R.glint_pts[j % len(R.glint_pts)]
            p = hcam(cam, 2.0, bob) @ np.array([u, v, 1.0])
            flare(img, p[0] / p[2], p[1] / p[2], 260 * sc, math.sin(math.pi * k) ** 2 * 0.9)
    return to_u8(img)


def render_hold(img, cam, t):
    if R.final_cache is None:
        lit = light(R.col, R.mat, cam, TL.T_SETTLE)
        pm = np.dstack([lit * R.alpha[..., None], R.alpha]).astype(np.float32)
        n = R.final_n
        fin = np.clip(cv2.resize(pm, (n, n), interpolation=cv2.INTER_LANCZOS4), 0, 1)
        fin[..., :3] = np.minimum(fin[..., :3], fin[..., 3:4])
        R.final_cache = fin
    fin = R.final_cache
    n, x0, y0 = R.final_n, R.final_x0, R.final_y0
    floor_shadow(img, cam, t, 0.0)
    buf = np.zeros((H, W, 4), np.float32)
    # thin extruded edge kept for the solid look, drawn straight behind the face
    for i in range(6, 0, -1):
        dy = i * 2
        reg = buf[y0 + dy:y0 + dy + n, x0:x0 + n]
        hgt = reg.shape[0]
        s = fin[:hgt] * np.array([0.32, 0.32, 0.32, 1.0], np.float32)
        reg[:] = s + reg * (1 - s[..., 3:4])
    reg = buf[y0:y0 + n, x0:x0 + n]
    reg[:] = fin + reg * (1 - fin[..., 3:4])
    floor_reflection(img, buf, cam, t, 0.0)
    composite(img, buf)
    draw_tagline(img, t)
    return to_u8(img)


def to_u8(img):
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- motto: brushed silver
def init_tagline():
    from PIL import Image, ImageDraw, ImageFont
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    font = ImageFont.truetype(os.path.join(root, 'assets', 'fonts', 'Cinzel-Bold.ttf'), TL.TAG_FONT_PX)
    track = TL.TAG_TRACK * TL.TAG_FONT_PX
    space = font.getlength(' ') + track
    rng = np.random.default_rng(3)
    pieces, x = [], 0.0
    for k, (txt, t0) in enumerate(TL.TAGLINE):
        if k:
            x += space
        w = sum(font.getlength(c) for c in txt) + track * (len(txt) - 1)
        im = Image.new('L', (int(math.ceil(w)) + 16, TL.TAG_FONT_PX + 48), 0)
        d = ImageDraw.Draw(im)
        cx = 8.0
        for c in txt:
            d.text((cx, 14), c, font=font, fill=255)
            cx += font.getlength(c) + track
        a = np.asarray(im, np.float32) / 255.0
        hgt = np.clip(cv2.distanceTransform((a > 0.5).astype(np.uint8), cv2.DIST_L2, 3) / 4.0, 0, 1)
        hgt = cv2.GaussianBlur(hgt, (0, 0), 1.0)
        gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3) / 8.0 * 5
        gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3) / 8.0 * 5
        nz = 1 / np.sqrt(1 + gx * gx + gy * gy)
        nx, ny = -gx * nz, -gy * nz
        dd = nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]
        hv = (LIGHT + [0, 0, 1]) / np.linalg.norm(LIGHT + [0, 0, 1])
        spec = 0.55 * np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0, 1) ** 26
        rows = np.linspace(0, 1, a.shape[0], dtype=np.float32)[:, None]
        brush = cv2.GaussianBlur(rng.standard_normal(a.shape).astype(np.float32), (0, 0), sigmaX=12, sigmaY=0.6)
        brush /= brush.std() + 1e-6
        base = (0.99 - 0.24 * rows) * (1 + 0.04 * brush) * np.clip(1 + 1.1 * (dd - LIGHT[2]), 0.4, 1.6)
        rgb = np.clip(base[..., None] * np.array([0.94, 0.95, 0.98], np.float32) + spec[..., None], 0, 1)
        # thin dark rim so the silver reads cleanly against the grey studio
        rim = np.clip(cv2.dilate(a, np.ones((5, 5), np.uint8)) - a, 0, 1)
        rgb = rgb * a[..., None] + np.array([0.16, 0.16, 0.18], np.float32) * rim[..., None]
        a = np.clip(a + rim, 0, 1)
        rgb = np.where(a[..., None] > 1e-3, rgb / np.maximum(a[..., None], 1e-3), 0)
        sha = cv2.GaussianBlur(np.roll(a, 8, axis=0), (0, 0), 6) * 0.62
        pieces.append(dict(a=a, rgb=rgb.astype(np.float32), sh=sha, x=x, t0=t0, sep=(txt == '|')))
        x += w
    s = TL.FINAL_SCALE
    n = int(round(2000 * s))
    R.final_n = n
    R.final_x0 = int(round(W / 2 - n / 2))
    R.final_y0 = int(round(TL.FINAL_CY - n / 2))
    R.tag_y = TL.FINAL_CY + s * (1963 - 1000) + 112
    R.tag_x0 = W / 2 - x / 2
    R.tag = pieces


def _place(img, arr, x, y, mode):
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    M = np.float32([[1, 0, x - x0], [0, 1, y - y0]])
    hh, ww = arr.shape[:2]
    arr = cv2.warpAffine(arr, M, (ww + 1, hh + 1), flags=cv2.INTER_LINEAR)
    return arr, x0, y0


def draw_tagline(img, t):
    for p in R.tag:
        u = TL.clamp((t - p['t0']) / TL.TAG_ANIM)
        if u <= 0:
            continue
        e = TL.ease_out_cubic(u)
        op = TL.clamp(u * 1.6)
        a, rgb, sh = p['a'], p['rgb'], p['sh']
        h, w = a.shape
        if p['sep']:
            sy = max(int(h * (0.15 + 0.85 * e)), 1)
            a = cv2.resize(a, (w, sy), interpolation=cv2.INTER_AREA)
            rgb = cv2.resize(rgb, (w, sy), interpolation=cv2.INTER_AREA)
            sh = cv2.resize(sh, (w, sy), interpolation=cv2.INTER_AREA)
            dy = (h - sy) / 2
        else:
            dy = 30.0 * (1 - e)
        x, y = R.tag_x0 + p['x'], R.tag_y + dy
        stack = np.dstack([rgb, a, sh])
        stack, x0, y0 = _place(img, stack, x, y, None)
        reg = img[y0:y0 + stack.shape[0], x0:x0 + stack.shape[1]]
        stack = stack[:reg.shape[0], :reg.shape[1]]
        reg *= (1 - stack[..., 4:5] * op)
        aa = stack[..., 3:4] * op
        reg[:] = reg * (1 - aa) + stack[..., :3] * aa
