"""
PSA crest intro renderer (9:16, 2160x3840, 60 fps).

The crest is NEVER redrawn: every pixel of it comes from assets/psa_logo_original.webp through the layer
masks in assets.py.  Motion = perspective transforms of those original layers (a flat textured slab seen
by a moving camera).  Depth/extrusion uses darkened copies of the crest's own colours.  For the final hold
(t >= 14.6 s) the original is placed front-on with a single high-quality downscale and nothing on top.
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
F = 3200.0                          # focal length (px) — moderate perspective like the reference


class R:
    pass


def init():
    L = assets.load()
    R.col = L['col']
    R.alpha = L['alpha']
    R.pm = np.dstack([R.col * R.alpha[..., None], R.alpha]).astype(np.float32)      # premultiplied RGBA
    fp = L['field_plain']
    R.field = L['field']
    R.field_pm = np.dstack([fp * R.field[..., None], R.field.astype(np.float32)]).astype(np.float32)
    R.sym = L['sym']
    R.shard_ids, R.wedge_ids, R.item_ids = L['shard_ids'], L['wedge_ids'], L['item_ids']
    R.shard_c, R.wedge_c, R.item_c = L['shard_c'], L['wedge_c'], L['item_c']
    R.field_c = L['field_c']
    R.item_kind = [str(k) for k in L['item_kind']]
    R.shards = _crops(R.pm, R.shard_ids, len(R.shard_c))
    R.wedges = _crops(R.field_pm, R.wedge_ids, len(R.wedge_c))
    R.items = _crops(R.pm, R.item_ids, len(R.item_c))
    R.shard_sched = TL.shard_schedule(R.shard_c)
    R.wedge_sched = TL.wedge_schedule(len(R.wedge_c))
    R.item_sched = TL.letter_schedule(R.item_kind, R.item_c)
    # symbols layer (original pixels) over the plain field
    ys, xs = np.nonzero(R.sym)
    R.sym_box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    x0, y0, x1, y1 = R.sym_box
    R.sym_pm = (R.pm[y0:y1, x0:x1] * R.sym[y0:y1, x0:x1, None]).astype(np.float32)
    # screen grids for backgrounds
    R.gy = np.arange(H, dtype=np.float32)[:, None]
    R.gx = np.arange(W, dtype=np.float32)[None, :]
    # glint anchor points: well-spread convex-hull vertices of the crest silhouette
    cnts, _ = cv2.findContours((R.alpha > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(big).reshape(-1, 2).astype(np.float32)
    pick = [hull[np.argmin(hull[:, 1])]]
    for _ in range(9):
        d = np.min(np.linalg.norm(hull[:, None, :] - np.array(pick)[None], axis=2), 1)
        pick.append(hull[int(np.argmax(d))])
    R.glint_pts = np.array(pick)
    init_tagline()
    rng = np.random.default_rng(5)
    R.dust = np.stack([rng.uniform(0, W, 38), rng.uniform(0, H, 38), rng.uniform(7, 22, 38),
                       rng.uniform(0.05, 0.14, 38), rng.uniform(0, 100, 38)], 1)


def _crops(src, ids, n):
    out = []
    for k in range(n):
        m = ids == k
        ys, xs = np.nonzero(m)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        c = src[y0:y1, x0:x1] * m[y0:y1, x0:x1, None]
        out.append((np.ascontiguousarray(c, np.float32), x0, y0))
    return out


# ---------------------------------------------------------------- geometry
def rot_matrix(yaw, pitch, roll):
    y, p, r = map(math.radians, (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def hcam(cam, z, bob=0.0):
    """Homography: logo pixel (u, v) on the plane at depth z (toward camera +) -> screen."""
    Rm = cam['R']
    D = F / cam['scale']
    A = np.array([[1, 0, -1000.0], [0, 1, -1000.0 + bob], [0, 0, 1]])
    M = np.array([[F * Rm[0, 0], F * Rm[0, 1], F * Rm[0, 2] * z],
                  [F * Rm[1, 0], F * Rm[1, 1], F * Rm[1, 2] * z],
                  [-Rm[2, 0], -Rm[2, 1], D - Rm[2, 2] * z]])
    S = np.array([[1, 0, W / 2], [0, 1, cam['cy']], [0, 0, 1]])
    return S @ M @ A


def T(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1.0]])


def RS(deg, s):
    a = math.radians(deg)
    return np.array([[s * math.cos(a), -s * math.sin(a), 0], [s * math.sin(a), s * math.cos(a), 0], [0, 0, 1.0]])


def warp_over(dst, src, Hm, shade=None, opacity=1.0, scale_hint=1.0):
    """Perspective-warp premultiplied RGBA `src` with Hm and composite OVER dst (in place)."""
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
    Ho = T(-bx0, -by0) @ Hm
    out = cv2.warpPerspective(src, Ho, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if out.ndim == 2:
        out = out[..., None]
    if shade is not None:
        out = out.copy()
        out[..., :3] *= shade
    if opacity < 1.0:
        out = out * opacity
    reg = dst[by0:by1, bx0:bx1]
    a = out[..., 3:4]
    reg[..., :3] = out[..., :3] + reg[..., :3] * (1 - a)
    if dst.shape[2] == 4:
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
def settled_canvas(t):
    """Premultiplied RGBA (2000x2000) of every element that has already landed, plus the rising symbols."""
    if t >= TL.T_COMPLETE + 0.02:
        return R.pm, []
    flying = []
    canvas = np.zeros_like(R.pm)
    # outline fragments
    landed = np.array([t >= s['land'] for s in R.shard_sched])
    lut = np.r_[False, landed]
    m = lut[R.shard_ids + 1]
    canvas[m] = R.pm[m]
    for k, s in enumerate(R.shard_sched):
        if s['start'] <= t < s['land']:
            flying.append(('shard', k))
    # field wedges (iris)
    landed_w = np.array([t >= s['land'] for s in R.wedge_sched])
    m = np.r_[False, landed_w][R.wedge_ids + 1]
    canvas[m] = R.field_pm[m]
    for k, s in enumerate(R.wedge_sched):
        if s['start'] <= t < s['land']:
            flying.append(('wedge', k))
    # symbols rising out of the field
    if t > TL.SYM_START and landed_w.all():
        e = TL.s5(TL.SYM_START, TL.SYM_END, t)
        yw = TL.SYM_Y0 + (TL.SYM_Y1 - TL.SYM_Y0) * e
        dy = 34.0 * (1 - e)
        x0, y0, x1, y1 = R.sym_box
        sub = R.sym_pm
        if dy > 0.01:
            sub = cv2.warpAffine(sub, np.float32([[1, 0, 0], [0, 1, dy]]), (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        rows = np.arange(y0, y1, dtype=np.float32)[:, None]
        rev = np.clip((rows - yw) / 46.0 + 0.5, 0, 1)
        if t >= TL.SYM_END:
            rev = np.ones_like(rev)
            sub = R.sym_pm
        src = sub * rev[..., None]
        reg = canvas[y0:y1, x0:x1]
        reg[:] = src + reg * (1 - src[..., 3:4])
    # lettering + banner
    for k, (st, ld) in enumerate(R.item_sched):
        kind = R.item_kind[k]
        if kind == 'band':
            if t <= st:
                continue
            c, x0, y0 = R.items[k]
            e = TL.s5(st, ld, t)
            xw = x0 + (c.shape[1] + 40) * e
            cols = np.arange(x0, x0 + c.shape[1], dtype=np.float32)[None, :]
            rev = np.clip((xw - cols) / 30.0, 0, 1)
            src = c * rev[..., None]
            reg = canvas[y0:y0 + c.shape[0], x0:x0 + c.shape[1]]
            reg[:] = src + reg * (1 - src[..., 3:4])
        elif t >= ld:
            c, x0, y0 = R.items[k]
            reg = canvas[y0:y0 + c.shape[0], x0:x0 + c.shape[1]]
            reg[:] = c + reg * (1 - c[..., 3:4])
        elif t >= st:
            flying.append(('item', k))
    return canvas, flying


def element_pose(kind, k, t):
    """(crop, crop_x0, crop_y0, 3x3 crop->logo transform, z, opacity)."""
    if kind == 'shard':
        s = R.shard_sched[k]
        c, x0, y0 = R.shards[k]
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
        c, x0, y0 = R.wedges[k]
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
        c, x0, y0 = R.items[k]
        cx, cy = R.item_c[k]
        u = (t - st) / (ld - st)
        e = TL.ease_out_cubic(u)
        rot = 0.0
        sc = 1 + 0.45 * (1 - TL.ease_out_back(u, 1.6))
        z = 150.0 * (1 - u) ** 2
        dx, dy = 0.0, -40.0 * (1 - e)
        op = TL.clamp(u * 3.5)
    M = T(cx + dx, cy + dy) @ RS(rot, sc) @ T(-cx, -cy) @ T(x0, y0)
    return c, M, z, op


# ---------------------------------------------------------------- environment
def background(t, cam):
    w = TL.wall_amount(t)
    gy, gx = R.gy, R.gx
    out = np.zeros((H, W, 3), np.float32)
    rr = ((gx - W / 2) / (W * 0.75)) ** 2 + ((gy - H / 2) / (H * 0.62)) ** 2
    if w > 0:
        Hw = hcam(cam, -60.0)
        p = Hw @ np.array([500.0, 120.0, 1.0])          # light pool on the wall, upper-left of crest
        px, py = p[0] / p[2], p[1] / p[2]
        pool = np.exp(-(((gx - px) / 2100.0) ** 2 + ((gy - py) / 2300.0) ** 2))
        v = 0.845 + 0.085 * pool - 0.09 * np.clip(rr, 0, 1.6)
        # wall is angled to the camera: gentle gradient across it
        v = v + 0.03 * math.sin(math.radians(cam['yaw'])) * (gx - W / 2) / W
        out += w * v[..., None] * np.array([0.985, 0.99, 1.0], np.float32)
    if w < 1:
        yh = cam['cy'] + cam['scale'] * 1200.0
        above = 0.935 - 0.11 * np.clip((yh - gy) / yh, 0, 1) ** 1.3
        below = 0.925 - 0.085 * np.clip((gy - yh) / (H - yh), 0, 1) ** 0.9
        k = 1 / (1 + np.exp(-(gy - yh) / 70.0))
        v = above * (1 - k) + below * k
        pool = np.exp(-(((gx - W / 2) / 1500.0) ** 2 + ((gy - cam['cy']) / 1700.0) ** 2))
        v = v + 0.035 * pool - 0.06 * np.clip(rr, 0, 1.6)
        out += (1 - w) * np.broadcast_to(v, (H, W))[..., None] * np.array([0.985, 0.99, 1.0], np.float32)
    return out


def draw_dust(img, t):
    amt = TL.sstep(7.9, 8.8, t)
    if amt <= 0:
        return
    D = R.dust
    for x, y, r, a, ph in D:
        yy = (y - 26 * t) % H
        xx = x + 40 * math.sin(t * 0.4 + ph)
        tw = 0.6 + 0.4 * math.sin(t * 1.7 + ph)
        x0, x1 = int(max(xx - 3 * r, 0)), int(min(xx + 3 * r, W))
        y0, y1 = int(max(yy - 3 * r, 0)), int(min(yy + 3 * r, H))
        if x1 <= x0 or y1 <= y0:
            continue
        g = np.exp(-(((np.arange(x0, x1)[None, :] - xx) ** 2 + (np.arange(y0, y1)[:, None] - yy) ** 2) / (2 * r * r)))
        reg = img[y0:y1, x0:x1]
        reg += (g * a * tw * amt * 0.5)[..., None]


def flare(img, x, y, size, amp):
    ext = int(size * 1.3) + 3
    x0, x1 = int(max(x - ext, 0)), int(min(x + ext, W))
    y0, y1 = int(max(y - ext, 0)), int(min(y + ext, H))
    if x1 <= x0 or y1 <= y0:
        return
    xx = np.arange(x0, x1, dtype=np.float32)[None, :] - x
    yy = np.arange(y0, y1, dtype=np.float32)[:, None] - y
    th = 3.0
    rays = np.exp(-np.abs(yy) / th - np.abs(xx) / (size * 0.42)) + np.exp(-np.abs(xx) / th - np.abs(yy) / (size * 0.42))
    r2 = xx * xx + yy * yy
    core = np.exp(-r2 / (2 * (size * 0.06) ** 2)) + 0.35 * np.exp(-r2 / (2 * (size * 0.2) ** 2))
    v = np.clip((rays + core) * amp, 0, 1.2)
    reg = img[y0:y1, x0:x1]
    reg[:] = np.clip(reg + v[..., None] * np.array([1.0, 0.98, 0.94], np.float32), 0, 1)


def gloss(canvas, t):
    g = 0.0
    p = None
    for a, b in TL.GLOSS:
        if a < t < b:
            g = TL.bump(t, a, a + 0.15, b - 0.15, b)
            p = -900 + 3900 * TL.s5(a, b, t)
    if g <= 0:
        return canvas
    u = np.arange(2000, dtype=np.float32)[None, :]
    v = np.arange(2000, dtype=np.float32)[:, None]
    band = np.exp(-(((u + 0.65 * v) - p) / 130.0) ** 2) * 0.30 * g
    out = canvas.copy()
    out[..., :3] += band[..., None] * (out[..., 3:4] - out[..., :3])
    return out


# ---------------------------------------------------------------- frame
def render_frame(t):
    cam = TL.camera(t)
    cam['R'] = rot_matrix(cam['yaw'], cam['pitch'], cam['roll'])
    bob = TL.bob(t)
    img = background(t, cam)
    draw_dust(img, t)

    if t >= TL.T_SETTLE:
        floor_shadow(img, cam, t, 0.0)
        n, x0, y0 = R.final_n, R.final_x0, R.final_y0
        reg = img[y0:y0 + n, x0:x0 + n]
        reg[:] = R.final_pm[..., :3] + reg * (1 - R.final_pm[..., 3:4])
        draw_tagline(img, t)
        return to_u8(img)

    canvas, flying = settled_canvas(t)
    w = TL.wall_amount(t)
    thick = TL.thickness(t)

    # ---- shadows ----
    if w > 0:
        sh = np.zeros((H // 2, W // 2), np.float32)
        half = np.diag([0.5, 0.5, 1.0])
        a_small = cv2.resize(canvas[..., 3], (500, 500), interpolation=cv2.INTER_AREA)
        off = 14.0 + thick * 0.5
        warp_alpha_add(sh, a_small, half @ hcam(cam, -thick - 6, bob) @ T(off * 0.35, off * 0.6) @ np.diag([4, 4, 1.0]), 1.0)
        for kind, k in flying:
            c, M, z, op = element_pose(kind, k, t)
            o = 14.0 + z * 0.9
            warp_alpha_add(sh, np.ascontiguousarray(c[..., 3]), half @ hcam(cam, -thick - 6, bob) @ T(o * 0.35, o * 0.6) @ M,
                           op * max(0.25, 1 - z / 900))
        sh = cv2.GaussianBlur(sh, (0, 0), 9 + 0.0 * thick)
        sh = cv2.resize(sh, (W, H), interpolation=cv2.INTER_LINEAR)
        img *= (1 - 0.30 * w * sh)[..., None]
    if w < 1:
        floor_shadow(img, cam, t, bob)

    # ---- crest slab: extrusion layers, then front face ----
    canvas = gloss(canvas, t)
    sc = cam['scale']
    nside = 9 if thick > 1 else 0
    for i in range(nside, 0, -1):
        z = -thick * i / nside
        shade = 0.46 + 0.16 * (1 - i / nside)
        warp_over(img, canvas, hcam(cam, z, bob), shade=shade, scale_hint=sc)
    warp_over(img, canvas, hcam(cam, 0.0, bob), scale_hint=sc)

    # ---- elements in flight (closer to camera, drawn on top) ----
    flying.sort(key=lambda e: -element_pose(*e, t)[2])
    for kind, k in flying:
        c, M, z, op = element_pose(kind, k, t)
        tk = max(thick, 14.0)
        for i in ((3, 2, 1) if kind != 'wedge' else ()):
            warp_over(img, c, hcam(cam, z - tk * i / 3, bob) @ M, shade=0.5 + 0.05 * (3 - i), opacity=op, scale_hint=sc)
        warp_over(img, c, hcam(cam, z, bob) @ M, opacity=op, scale_hint=sc)

    # ---- glints on the crest's edge points ----
    for j, tg in enumerate(TL.GLINTS):
        k = (t - tg) / 0.5
        if 0 < k < 1:
            u, v = R.glint_pts[j % len(R.glint_pts)]
            p = hcam(cam, 2.0, bob) @ np.array([u, v, 1.0])
            flare(img, p[0] / p[2], p[1] / p[2], 260 * sc, math.sin(math.pi * k) ** 2 * 0.95)
    return to_u8(img)


def floor_shadow(img, cam, t, bob):
    amt = 1 - TL.wall_amount(t)
    if amt <= 0:
        return
    sc = cam['scale']
    lift = -bob / 14.0 if bob else 0.0                   # -1..1
    yb = cam['cy'] + sc * 1000.0                          # bottom of the crest
    cy = yb + sc * TL.shadow_offset(t)
    rx = sc * 760.0 * (1 - 0.05 * lift)
    ry = sc * 70.0
    x0, x1 = int(max(W / 2 - rx * 2, 0)), int(min(W / 2 + rx * 2, W))
    y0, y1 = int(max(cy - ry * 5, 0)), int(min(cy + ry * 5, H))
    xx = (np.arange(x0, x1, dtype=np.float32)[None, :] - W / 2) / rx
    yy = (np.arange(y0, y1, dtype=np.float32)[:, None] - cy) / ry
    g = np.exp(-(xx ** 2 + yy ** 2) * 1.6) * 0.75 + np.exp(-(xx ** 2 * 0.5 + yy ** 2 * 0.35)) * 0.25
    dens = 0.30 * amt * (1 - 0.18 * lift)
    reg = img[y0:y1, x0:x1]
    reg *= (1 - dens * g)[..., None]


def to_u8(img):
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- motto (final hold)
def init_tagline():
    from PIL import Image, ImageDraw, ImageFont
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    font = ImageFont.truetype(os.path.join(root, 'assets', 'fonts', 'Cinzel-Bold.ttf'), TL.TAG_FONT_PX)
    maroon = tuple(int(round(v * 255)) for v in np.median(R.col[R.field & ~R.sym], axis=0))   # the crest's own maroon
    silver = (150, 152, 158)
    track = TL.TAG_TRACK * TL.TAG_FONT_PX
    space = font.getlength(' ') + track
    pieces, x = [], 0.0
    for k, (txt, t0) in enumerate(TL.TAGLINE):
        if k:
            x += space
        w = sum(font.getlength(c) for c in txt) + track * (len(txt) - 1)
        img = Image.new('L', (int(math.ceil(w)) + 8, TL.TAG_FONT_PX + 40), 0)
        d = ImageDraw.Draw(img)
        cx = 4.0
        for c in txt:
            d.text((cx, 10), c, font=font, fill=255)
            cx += font.getlength(c) + track
        a = np.asarray(img, np.float32) / 255.0
        col = np.array(silver if txt == '|' else maroon, np.float32) / 255.0
        pieces.append(dict(a=a, col=col, x=x, t0=t0, sep=(txt == '|')))
        x += w
    total = x
    s = TL.FINAL_SCALE
    n = int(round(2000 * s))
    R.final_n = n
    R.final_x0 = int(round(W / 2 - n / 2))
    R.final_y0 = int(round(TL.FINAL_CY - n / 2))
    up = cv2.INTER_LANCZOS4 if s > 1 else cv2.INTER_AREA
    R.final_pm = np.clip(cv2.resize(R.pm, (n, n), interpolation=up), 0, 1)
    R.final_pm[..., :3] = np.minimum(R.final_pm[..., :3], R.final_pm[..., 3:4])
    crest_bottom = TL.FINAL_CY + s * (1963 - 1000)
    R.tag_y = crest_bottom + 120                     # top of the text box
    R.tag_x0 = W / 2 - total / 2
    R.tag = pieces


def draw_tagline(img, t):
    for p in R.tag:
        u = TL.clamp((t - p['t0']) / TL.TAG_ANIM)
        if u <= 0:
            continue
        e = TL.ease_out_cubic(u)
        a = p['a'] * TL.clamp(u * 1.6)
        h, w = a.shape
        if p['sep']:
            sy = 0.15 + 0.85 * e                        # separators grow from the centre line
            a = cv2.resize(a, (w, max(int(h * sy), 1)), interpolation=cv2.INTER_AREA)
            dy = (h - a.shape[0]) / 2
        else:
            dy = 30.0 * (1 - e)                          # words rise into place
        x = R.tag_x0 + p['x']
        y = R.tag_y + dy
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        M = np.float32([[1, 0, x - x0], [0, 1, y - y0]])
        hh, ww = a.shape
        a = cv2.warpAffine(a, M, (ww + 1, hh + 1), flags=cv2.INTER_LINEAR)
        reg = img[y0:y0 + a.shape[0], x0:x0 + a.shape[1]]
        a = a[:reg.shape[0], :reg.shape[1], None]
        reg[:] = reg * (1 - a) + p['col'] * a
