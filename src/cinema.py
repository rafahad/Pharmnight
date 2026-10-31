"""
PHARMNIGHT 2026 — CINEMA SOIRÉE : "The Grand Premiere"
Procedural 18-second cinematic logo reveal.

The vintage theatre (velvet curtains, proscenium, spotlights, projector, dust,
film strip) is rendered procedurally.  The official logo is NEVER regenerated:
the original bitmap is composited as a flat, untouched layer.  Only uniform
scaling/translation is ever applied to it (camera dolly), and for every frame
from t = 16.0 s onward it is placed at native 1:1 scale on an integer pixel
offset with no effect on top, so those pixels are bit-identical to the source.
"""
import math
import numpy as np
import cv2
from PIL import Image

# --------------------------------------------------------------------------
# Output / world constants
# --------------------------------------------------------------------------
W, H = 3840, 2160          # 4K UHD master
ES = 2                     # environment is rendered at 1/ES resolution
EW, EH = W // ES, H // ES
FPS = 24
DUR = 18.0
NF = int(DUR * FPS)        # 432 frames
F = 10.0                   # focal constant: screen_scale = F / (Z - cam_z)

ZL = 10.0                  # logo plane
ZC = 9.2                   # main traveller curtain
ZV = 8.85                  # valance (swag border)
ZP = 8.8                   # proscenium wall / gold frame
ZA = 8.5                   # apron (front edge of stage)
ZB = 13.0                  # back wall of stage
YS = 760.0                 # stage floor height (y down)
YAF = 1000.0               # auditorium floor
YCEIL = -1700.0
XWALL = 2600.0
PRO_IN, PRO_OUT = 1550.0, 1720.0
PRO_TOP, HEAD_TOP = -1100.0, -1300.0
CURT_OUT = 1640.0

# Logo placement: circle centre in the source bitmap and mask radius
LOGO_PATH = None
LOGO_CX, LOGO_CY = 625.66, 612.30
LOGO_R = 575.0             # includes the outer gold ring (r~548) + cream margin
LOGO_TL_FINAL = (1294, 468)  # integer top-left in the final 4K frame

# layout-dependent staging (landscape defaults; see configure())
LAYOUT = 'landscape'
CURT_OPEN = 1190.0         # |x| of the curtains' inner edge when fully open
CURT_N = 8.0               # pleats per panel
SWAG_SEG, SWAG_SAG, SWAG_OFF = 620.0, 255.0, 1860.0
SPOT_X, SPOT_Y = 2300.0, -1550.0
SCONCE_X = 2150.0
FOOT_X = 1450.0
EXTRA_ROWS = ()
PJ_POS = (-560.0, 70.0, -9.4)
STARS = [(316, 234), (941, 234), (617, 377), (767, 358), (484, 422), (377, 567),
         (989, 623), (950, 697), (903, 713), (426, 1067), (823, 1062), (625, 1129)]
RING_R = 545.0

GOLD = np.array([0.86, 0.70, 0.46], np.float32)
WARM = np.array([1.00, 0.84, 0.60], np.float32)
VELVET = np.array([0.50, 0.040, 0.070], np.float32)
SHEEN = np.array([0.95, 0.28, 0.30], np.float32)


def configure(layout):
    """Switch between the 16:9 landscape staging and the 9:16 vertical (Reels) staging."""
    global LAYOUT, W, H, EW, EH, LOGO_TL_FINAL, PRO_IN, PRO_OUT, PRO_TOP, HEAD_TOP, CURT_OUT, YCEIL
    global CURT_OPEN, CURT_N, SWAG_SEG, SWAG_SAG, SWAG_OFF, SPOT_X, SPOT_Y, SCONCE_X, FOOT_X, EXTRA_ROWS, PJ_POS, _cz
    LAYOUT = layout
    if layout == 'vertical':
        W, H = 2160, 3840
        PRO_IN, PRO_OUT = 820.0, 950.0
        PRO_TOP, HEAD_TOP = -1400.0, -1600.0
        YCEIL = -2000.0
        CURT_OUT = PRO_IN + 90.0
        CURT_OPEN = 700.0
        CURT_N = 6.0
        SWAG_SEG, SWAG_SAG, SWAG_OFF = 410.0, 175.0, PRO_IN
        SPOT_X, SPOT_Y = 900.0, -1700.0
        SCONCE_X = 1350.0
        FOOT_X = 725.0
        EXTRA_ROWS = (4.0, 5.0, 6.0)
        PJ_POS = (-230.0, 70.0, -7.0)
        # start closer: the tall frame needs the stage bigger in the opening wide shot
        _cz = pchip([(0, -10.5), (3, -9.3), (6, -6.2), (10, -2.4), (15.5, 0.0), (16, 0.0), (18, 0.0)])
    EW, EH = W // ES, H // ES
    # integer top-left so the emblem's ring centre sits at the frame centre at 1:1
    LOGO_TL_FINAL = (int(round(W / 2 - LOGO_CX)), int(round(H / 2 - LOGO_CY)))


# --------------------------------------------------------------------------
# Small math helpers
# --------------------------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return np.minimum(np.maximum(x, a), b)


def sstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def s5(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0))
    return t * t * t * (t * (t * 6 - 15) + 10)


def bump(t, a, b, c, d):
    """0 before a, rises to 1 at b, holds, falls to 0 at d (from c)."""
    return float(sstep(a, b, t) * (1 - sstep(c, d, t)))


def pchip(keys):
    """Monotone cubic (Fritsch–Carlson) interpolation through (t, v) keys."""
    ts = np.array([k[0] for k in keys], float)
    vs = np.array([k[1] for k in keys], float)
    h = np.diff(ts)
    dl = np.diff(vs) / h
    m = np.zeros_like(vs)
    for i in range(1, len(vs) - 1):
        if dl[i - 1] * dl[i] <= 0:
            m[i] = 0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / dl[i - 1] + w2 / dl[i])
    m[0] = dl[0] * 0.5
    m[-1] = 0.0

    def f(t):
        if t <= ts[0]:
            return vs[0]
        if t >= ts[-1]:
            return vs[-1]
        i = np.searchsorted(ts, t) - 1
        hh = h[i]
        s = (t - ts[i]) / hh
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1
        h10 = s ** 3 - 2 * s ** 2 + s
        h01 = -2 * s ** 3 + 3 * s ** 2
        h11 = s ** 3 - s ** 2
        return h00 * vs[i] + h10 * hh * m[i] + h01 * vs[i + 1] + h11 * hh * m[i + 1]
    return f


def hash1(n):
    n = np.asarray(n, np.float64)
    return (np.sin(n * 127.1 + 311.7) * 43758.5453) % 1.0


def vnoise1(x, seed=0.0):
    """1-D value noise, smooth, in [0,1]."""
    i = np.floor(x)
    f = x - i
    a = hash1(i + seed * 17.0)
    b = hash1(i + 1 + seed * 17.0)
    u = f * f * (3 - 2 * f)
    return a + (b - a) * u


# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------
_cz = pchip([(0, -14.0), (3, -12.5), (6, -7.3), (10, -2.4), (15.5, 0.0), (16, 0.0), (18, 0.0)])
_cy = pchip([(0, -240.0), (6, -120.0), (10, -25.0), (14.5, 0.0), (16, 0.0), (18, 0.0)])


class Cam:
    def __init__(self, t):
        self.t = t
        self.cz = float(_cz(t)) if t < 15.5 else 0.0
        self.cy = float(_cy(t)) if t < 14.5 else 0.0
        s = clamp((t - 10.0) / 5.0)
        self.cx = float(560.0 * math.sin(math.pi * s) ** 2) if 10.0 < t < 15.0 else 0.0
        # look-at: keep the emblem horizontally centred during the arc
        self.lx = self.cx * F / (ZL - self.cz)

    def scale(self, Z):
        return F / (Z - self.cz)

    def proj(self, X, Y, Z):
        s = F / (Z - self.cz)
        return W / 2 + (X - self.cx) * s + self.lx, H / 2 + (Y - self.cy) * s, s

    def unproj_plane(self, xs, ys, Z):
        s = F / (Z - self.cz)
        return self.cx + (xs - W / 2 - self.lx) / s, self.cy + (ys - H / 2) / s


def projector_level(t):
    if t < 0.55:
        return 0.0
    if t < 1.3:
        # start-up flicker of the lamp
        pat = [(0.55, 0.62, 0.9), (0.62, 0.70, 0.0), (0.70, 0.76, 0.6), (0.76, 0.83, 0.1),
               (0.83, 0.95, 0.85), (0.95, 1.0, 0.45), (1.0, 1.3, 1.0)]
        v = 0.0
        for a, b, lv in pat:
            if a <= t < b:
                v = lv
        flick = 1.0
    else:
        v = 1.0
        flick = 0.93 + 0.07 * float(vnoise1(t * 24.0, 3.0))
    fade = 1.0 - 0.45 * float(sstep(6.0, 8.0, t)) - 0.25 * float(sstep(9.0, 11.0, t))
    return v * flick * fade


def spot_level(t, t0):
    if t < t0:
        return 0.0
    k = t - t0
    v = float(sstep(0.0, 0.10, k))
    v *= 1.0 + 0.25 * math.exp(-k * 9) * math.sin(k * 60)        # arc strike
    return v * (0.97 + 0.03 * float(vnoise1(t * 9.0, t0)))


def curtain_open(t):
    return s5(3.45, 6.25, t)


def logo_alpha(t):
    return float(s5(5.82, 6.42, t))


# --------------------------------------------------------------------------
# Static resources (per process)
# --------------------------------------------------------------------------
class Res:
    pass


R = Res()


def fbm(h, w, seed, scales=(4, 8, 16, 32, 64), weights=(0.12, 0.18, 0.25, 0.25, 0.2)):
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    for sc, wt in zip(scales, weights):
        n = rng.standard_normal((h // sc + 3, w // sc + 3)).astype(np.float32)
        n = cv2.resize(n, (w + 3 * sc, h + 3 * sc), interpolation=cv2.INTER_CUBIC)[:h, :w]
        acc += wt * n
    acc = (acc - acc.mean()) / (acc.std() + 1e-6)
    return acc


def init(logo_path):
    global LOGO_PATH
    LOGO_PATH = logo_path
    img = np.asarray(Image.open(logo_path).convert('RGB'))
    R.logo_u8 = img
    R.logo = img.astype(np.float32) / 255.0
    hh, ww = img.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    rr = np.hypot(xx + 0.5 - LOGO_CX - 0.5, yy + 0.5 - LOGO_CY - 0.5)
    R.logo_rr = rr
    R.logo_mask = clamp(LOGO_R + 0.5 - rr).astype(np.float32)          # 1px AA edge
    R.logo_ang = np.arctan2(yy - LOGO_CY, xx - LOGO_CX).astype(np.float32)
    # gold-detail mask (for subtle specular glints on existing gold artwork)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32)
    g = ((hsv[..., 0] >= 9) & (hsv[..., 0] <= 26) & (hsv[..., 1] > 60) & (hsv[..., 2] > 110))
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.2)
    R.gold = clamp(g * 1.5).astype(np.float32) * R.logo_mask
    # haze / texture noise
    R.haze1 = fbm(EH + 400, EW + 400, 11, scales=(16, 32, 64, 128), weights=(0.15, 0.3, 0.3, 0.25))
    R.haze2 = fbm(EH + 400, EW + 400, 23, scales=(32, 64, 128, 256), weights=(0.2, 0.3, 0.3, 0.2))
    R.nap = fbm(EH, EW, 5, scales=(1, 2, 4), weights=(0.5, 0.3, 0.2))
    rng = np.random.default_rng(77)
    R.grain = []
    for i in range(8):
        g = rng.standard_normal((EH, EW)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 0.6)
        R.grain.append(g / g.std())
    # dust motes (world space)
    n = 1400
    R.dust = np.stack([rng.uniform(-2300, 2300, n), rng.uniform(-1600, 900, n),
                       rng.uniform(-13.5, 12.5, n), rng.uniform(0, 1000, n),
                       rng.uniform(0.4, 1.0, n)], 1)
    # screen-space env pixel grid (full-res coordinates of half-res pixel centres)
    R.gx = ((np.arange(EW, dtype=np.float32) + 0.5) * ES)[None, :].repeat(EH, 0)
    R.gy = ((np.arange(EH, dtype=np.float32) + 0.5) * ES)[:, None].repeat(EW, 1)
    # film strip texture (for the transition)
    R.strip_rng = np.random.default_rng(99)


# --------------------------------------------------------------------------
# Lighting helpers (world-space light pools on surfaces)
# --------------------------------------------------------------------------
def ell_pool(X, Y, cx, cy, rx, ry, soft=0.25):
    d = np.sqrt(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2)
    core = 1 - sstep(1 - soft, 1.0, d)
    return core * (0.75 + 0.25 * (1 - d * d).clip(0, 1))


def rect_pool(X, Y, cx, cy, hx, hy, soft):
    ax = 1 - sstep(hx - soft, hx + soft, np.abs(X - cx))
    ay = 1 - sstep(hy - soft, hy + soft, np.abs(Y - cy))
    vig = 1 - 0.35 * (((X - cx) / hx) ** 2 + ((Y - cy) / hy) ** 2).clip(0, 2) / 2
    return ax * ay * vig


class Lights:
    """Per-frame light intensities."""

    def __init__(self, t):
        self.t = t
        self.proj = projector_level(t)
        self.sL = spot_level(t, 3.0)
        self.sR = spot_level(t, 3.28)
        self.open = float(curtain_open(t))
        self.logo = logo_alpha(t)
        self.foot = float(sstep(3.55, 4.4, t))
        self.house = 0.35 * float(sstep(0.9, 2.8, t)) * (1 - 0.5 * float(sstep(4, 7, t)))
        # projector picture area aims at the closed curtain, then at the emblem
        self.ptarget = (0.0, -60.0)


def surface_light(L, X, Y, Z, front_n=None):
    """Return (I_spotL, I_spotR, I_proj, I_foot, ambient) scalar fields for a surface."""
    # spots aim at the emblem; their pools grow with distance behind the curtain line
    k = (Z - 6.0) / (ZL - 6.0)
    sl = L.sL * ell_pool(X, Y, -230 * k, -20, 760 * k, 720 * k, 0.12)
    sr = L.sR * ell_pool(X, Y, 230 * k, -20, 760 * k, 720 * k, 0.12)
    pj = L.proj * rect_pool(X, Y, 0, -60, 620 * (Z / ZC) ** 0.6, 452 * (Z / ZC) ** 0.6, 70)
    ft = L.foot * np.exp(-np.maximum(YS - Y, 0) / 420.0) * (np.abs(X) < PRO_OUT - 20)
    return sl, sr, pj, ft


# --------------------------------------------------------------------------
# Environment layers (rendered at EH x EW, in full-res screen coordinates)
# --------------------------------------------------------------------------
def render_box(cam, L, img):
    """Auditorium side walls, ceiling, floor + proscenium wall, everything behind."""
    gx, gy = R.gx, R.gy
    dx = (gx - W / 2 - cam.lx) / F
    dy = (gy - H / 2) / F
    big = 1e9
    with np.errstate(divide='ignore', invalid='ignore'):
        tl = np.where(dx < 0, (-XWALL - cam.cx) / dx, big)
        tr = np.where(dx > 0, (XWALL - cam.cx) / dx, big)
        tc = np.where(dy < 0, (YCEIL - cam.cy) / dy, big)
        tf = np.where(dy > 0, (YAF - cam.cy) / dy, big)
    tw = np.minimum(tl, tr)
    tfront = ZP - cam.cz
    tmin = np.minimum(np.minimum(tw, tc), np.minimum(tf, tfront))
    Z = cam.cz + tmin
    X = cam.cx + dx * tmin
    Y = cam.cy + dy * tmin
    near_stage = np.exp(-np.clip(ZP - Z, 0, None) / 5.5)
    stage_glow = 0.08 * (L.sL + L.sR) / 2 + 0.05 * L.proj + 0.04 * L.logo
    col = np.zeros((EH, EW, 3), np.float32)
    # walls
    m = (tw <= tmin + 1e-6)
    pil = sstep(0.0, 0.03, np.abs(((Z / 2.6) % 1.0) - 0.5) - 0.40)
    wall = np.array([0.075, 0.022, 0.028], np.float32)
    sh = (0.25 + 1.4 * stage_glow * 6 * near_stage) * (0.8 + 0.35 * pil)
    # sconce wash (scalloped light pools on the walls)
    wash = np.zeros_like(Z)
    for zs in (-12.0, -8.0, -4.0, 0.0, 4.0):
        dz = (Z - zs) / 0.75
        dyy = (Y + 520) / 420
        up = np.exp(-dz * dz - np.maximum(-dyy, 0) ** 2 * 0.6 - np.maximum(dyy, 0) ** 2 * 2.5)
        wash += up * (1 - 0.5 * np.abs(dz).clip(0, 1) * (dyy < 0))
    sh = sh + 2.2 * L.house * wash
    col[m] = (wall[None, :] * sh[m][:, None])
    # ceiling
    m2 = (tc <= tmin + 1e-6) & ~m
    col[m2] = np.array([0.030, 0.010, 0.013], np.float32) * (0.3 + 4 * stage_glow * near_stage[m2])[:, None]
    # auditorium floor (carpet)
    m3 = (tf <= tmin + 1e-6) & ~m & ~m2
    col[m3] = np.array([0.07, 0.012, 0.018], np.float32) * (0.25 + 5 * stage_glow * near_stage[m3])[:, None]
    # proscenium wall (front)
    m4 = ~(m | m2 | m3)
    Xp, Yp = X[m4], Y[m4]
    dam = 0.5 + 0.5 * np.sin(Xp / 55.0) * np.sin(Yp / 55.0 + np.sin(Xp / 110.0))
    base = np.array([0.105, 0.026, 0.034], np.float32)
    lit = 0.30 + 2.6 * stage_glow + 1.6 * L.house * np.exp(-((np.abs(Xp) - SCONCE_X) / 320) ** 2 - ((Yp + 420) / 520) ** 2)
    col[m4] = base[None, :] * (lit * (0.88 + 0.12 * dam))[:, None]
    img[:] = col
    return X, Y, Z, m4


def render_stage_interior(cam, L, img, logo_scr):
    """Back wall + stage floor (only meaningful inside the proscenium opening)."""
    gx, gy = R.gx, R.gy
    # --- back wall (cyclorama in dark plum velvet) ---
    Xb, Yb = cam.unproj_plane(gx, gy, ZB)
    fold = np.sin(Xb / 75.0 + 1.3 * np.sin(Xb / 260.0)) * 0.5 + 0.5
    sl, sr, pj, ft = surface_light(L, Xb, Yb, ZB)
    pj = pj * (1 - 0.0 * L.open)
    I = 0.06 + 0.55 * (sl + sr) + 0.8 * pj * (1 - 0.7 * L.logo) + 0.30 * ft
    back = np.array([0.20, 0.026, 0.048], np.float32)
    col = back[None, None, :] * (I * (0.55 + 0.45 * fold))[..., None]
    # warm champagne halo behind the emblem
    if L.logo > 0:
        lx, ly, ls = logo_scr
        d = np.hypot(gx - lx, gy - ly) / (LOGO_R * ls)
        halo = np.exp(-np.maximum(d - 0.95, 0) ** 2 / 0.10) * (d > 0.9)
        col += (np.array([0.55, 0.40, 0.24], np.float32) * (0.42 * L.logo) * halo[..., None])
    # --- stage floor ---
    dy = (gy - H / 2) / F
    with np.errstate(divide='ignore', invalid='ignore'):
        tz = np.where(dy > 0, (YS - cam.cy) / dy, 1e9)
    Zf = cam.cz + tz
    floor = (Zf > ZA) & (Zf < ZB)
    Xf = cam.cx + (gx - W / 2 - cam.lx) / F * tz
    plank = np.floor(Xf / 95.0)
    pv = hash1(plank)
    grain = 0.88 + 0.12 * np.sin(Zf * 2.2 + pv * 20 + 1.5 * np.sin(Xf * 0.02)) * np.sin(Xf * 0.31 + pv * 7)
    foot = (Zf - cam.cz) / F * ES                        # world units per env pixel
    gap = 1 - 0.55 * np.exp(-((Xf / 95.0 - np.round(Xf / 95.0)) * 95.0) ** 2 / (2 * np.maximum(foot, 3) ** 2)) * np.clip(6 / np.maximum(foot, 1e-3), 0, 1)
    fl = np.array([0.15, 0.075, 0.050], np.float32) * ((0.8 + 0.4 * pv) * grain * gap)[..., None]
    k = (Zf - 6.0) / (ZL - 6.0)
    spot_pool = (L.sL * ell_pool(Xf, Zf, -150, 10.3, 1300, 2.2, 0.3) +
                 L.sR * ell_pool(Xf, Zf, 150, 10.3, 1300, 2.2, 0.3))
    footwash = L.foot * np.exp(-(Zf - ZA) / 0.9)
    If = 0.05 + 0.9 * spot_pool + 0.6 * footwash + 0.35 * L.logo * np.exp(-((Xf / 900) ** 2) - ((Zf - ZL) / 1.4) ** 2)
    fl = fl * If[..., None]
    col = np.where(floor[..., None], fl, col)
    img[:] = col
    return floor, Zf


def render_floor_reflection(cam, L, img, floor, logo_small, logo_scr):
    """Soft, dim mirror reflection of the emblem on the polished stage floor."""
    if L.logo <= 0 or logo_small is None:
        return
    lx, ly, ls = logo_scr
    yfl = cam.proj(0, YS, ZL)[1]                        # floor contact line under logo
    rgb, a = logo_small                                  # env-res premultiplied logo + alpha
    h, w = a.shape
    x0 = int(round(lx / ES - w / 2))
    y_axis = yfl / ES
    flipped = rgb[::-1]
    fa = a[::-1]
    top = int(round(2 * y_axis - (ly / ES + h / 2)))
    y1, y2 = max(top, 0), min(top + h, EH)
    x1, x2 = max(x0, 0), min(x0 + w, EW)
    if y2 <= y1 or x2 <= x1:
        return
    sub = flipped[y1 - top:y2 - top, x1 - x0:x2 - x0]
    sub = cv2.GaussianBlur(sub, (0, 0), 5)
    yy = (np.arange(y1, y2, dtype=np.float32)[:, None] - y_axis) / (h * 0.45)
    fall = np.exp(-np.maximum(yy, 0) * 2.2) * (yy > 0)
    region = img[y1:y2, x1:x2]
    fm = floor[y1:y2, x1:x2]
    region += sub * (0.20 * L.logo * fall * fm)[..., None]


def render_curtains(cam, L, img, t):
    gx, gy = R.gx, R.gy
    X, Y = cam.unproj_plane(gx, gy, ZC)
    s = cam.scale(ZC)
    v = clamp((Y - HEAD_TOP) / (YS - HEAD_TOP))
    # opening with a lagging, swinging hem
    o = s5(3.45, 6.25, t - 0.32 * v ** 1.6)
    sway = 14.0 * np.sin((t - 3.4) * 2.4 - v * 2.0) * v ** 2 * (t > 3.4) * np.exp(-max(t - 6.2, 0) * 1.1)
    e_in = -10.0 + CURT_OPEN * o + sway                     # inner edge |x|
    ax = np.abs(X)
    width = CURT_OUT - e_in
    u = clamp((CURT_OUT - ax) / width)                    # 0 outer .. 1 inner edge
    aa = ES * 0.8 / s
    cover = sstep(e_in - aa, e_in + aa, ax) * (ax < CURT_OUT + 60) * (Y < YS + 2)
    if not cover.any():
        return
    N = CURT_N
    comp = 1.0 + 1.3 * o                                 # deeper folds when gathered
    ph = 2 * np.pi * N * u + 0.9 * np.sin(2 * np.pi * 1.7 * u + 1.1) + 0.35 * np.sin(2 * np.pi * 4.3 * u + 0.4) + 0.25 * v * np.sin(5 * u)
    hgt = np.sin(ph) + 0.28 * np.sin(2 * ph + 0.8)
    dh = (np.cos(ph) + 0.56 * np.cos(2 * ph + 0.8)) * (2 * np.pi * N)
    depth = 0.20 * comp * (0.75 + 0.35 * v)               # fold depth / panel width ratio
    slope = depth * dh / N * np.sign(X) * -1.0
    nz = 1.0 / np.sqrt(1 + slope * slope)
    nx = slope * nz
    sl, sr, pj, ft = surface_light(L, X, Y, ZC)
    dL = clamp(nx * -0.55 + nz * 0.84)
    dR = clamp(nx * 0.55 + nz * 0.84)
    dF = nz
    wash = 0.16 * (L.sL + L.sR) / 2 + 0.10 * L.logo
    illum = 0.62 * (sl * dL + sr * dR) + 1.0 * pj * dF + 0.45 * ft * dF + wash * (0.6 + 0.4 * dF)
    amb = 0.035 + 0.05 * L.house
    ao = 0.30 + 0.70 * np.clip(hgt / 1.28 * 0.5 + 0.5, 0, 1) ** 0.8
    ao = ao * (0.55 + 0.45 * np.sin(np.pi * np.clip(u, 0, 1)) ** 0.5 * (o > 0.5) + 0.45 * (o <= 0.5))
    sheen = (1 - nz) ** 1.3
    nap = 1 + 0.05 * R.nap
    top_shadow = 0.35 + 0.65 * sstep(PRO_TOP + 20, PRO_TOP + 500, Y)
    col = (VELVET[None, None, :] * ((amb + 0.95 * illum) * ao * nap * top_shadow)[..., None] +
           SHEEN[None, None, :] * (0.55 * sheen * (illum + 0.02) * top_shadow)[..., None] +
           np.array([0.30, 0.12, 0.07], np.float32)[None, None, :] * (np.clip(illum - 0.45, 0, None) * ao * top_shadow)[..., None])
    # projector "picture": faint scratches & specks while it lights the curtain
    if L.proj > 0 and L.open < 0.6:
        rng = np.random.default_rng(int(t * FPS) + 5)
        sc = np.zeros_like(X)
        for _ in range(3):
            xs = rng.uniform(-520, 520)
            sc += np.exp(-((X - xs) / 3.0) ** 2) * rng.uniform(0.1, 0.35)
        for _ in range(5):
            px, py = rng.uniform(-560, 560), rng.uniform(-430, 330)
            sc -= np.exp(-((X - px) ** 2 + (Y - py) ** 2) / (rng.uniform(4, 14) ** 2)) * 0.6
        col += (WARM * 0.25)[None, None, :] * (sc * pj * (1 - L.open))[..., None]
    # inner leading edge: slight rim highlight + seam shadow
    edge = np.exp(-((ax - e_in) / (18 / 1.0)) ** 2)
    col *= (1 - 0.35 * edge)[..., None]
    # gold bullion fringe at the hem
    fr = sstep(YS - 70, YS - 62, Y)
    strand = 0.55 + 0.45 * np.sin(X * 0.55 + v * 3) ** 2
    fcol = GOLD[None, None, :] * ((0.12 + 1.1 * (illum + 0.1 * L.foot)) * strand)[..., None]
    col = col * (1 - fr[..., None]) + fcol * fr[..., None]
    a = cover[..., None]
    img[:] = img * (1 - a) + col * a


def render_valance(cam, L, img, t):
    gx, gy = R.gx, R.gy
    X, Y = cam.unproj_plane(gx, gy, ZV)
    inside = np.abs(X) < PRO_IN + 5
    seg = SWAG_SEG
    sx = ((X + SWAG_OFF) / seg) % 1.0
    yb = PRO_TOP + 40.0 + SWAG_SAG * np.sin(np.pi * sx)
    v = clamp((Y - PRO_TOP) / (yb - PRO_TOP))
    cover = inside & (Y < yb + 1) & (Y > HEAD_TOP)
    fringe = inside & (Y >= yb) & (Y < yb + 48)
    sl, sr, pj, ft = surface_light(L, X, Y, ZV)
    gl = 0.06 + 0.25 * (L.sL + L.sR) + 0.2 * L.proj + 0.25 * L.logo
    folds = np.sin(2 * np.pi * (5.0 * v ** 0.85) + 0.3)
    shade = (0.45 + 0.35 * folds) * (0.45 + 0.55 * np.sin(np.pi * sx) ** 0.6) * (0.45 + 0.55 * v)
    sheen = np.clip(-np.cos(2 * np.pi * (5.0 * v ** 0.85) + 0.3), 0, 1) ** 3
    col = (VELVET[None, None, :] * (gl * 1.6 * shade)[..., None] +
           SHEEN[None, None, :] * (0.35 * gl * sheen * np.sin(np.pi * sx))[..., None])
    a = sstep(-0.5, 0.5, (yb - Y) * cam.scale(ZV) / ES)[..., None] * cover[..., None]
    img[:] = img * (1 - a) + col * a
    # bullion fringe
    swing = 2.0 * np.sin(t * 2.0 + X / 300.0)
    strand = 0.5 + 0.5 * np.sin((X + swing * (Y - yb) / 48) * 0.6) ** 2
    fcol = GOLD[None, None, :] * ((0.15 + 1.4 * gl) * strand * (1.1 - 0.5 * (Y - yb) / 48))[..., None]
    fa = (fringe * sstep(48, 40, Y - yb))[..., None]
    img[:] = img * (1 - fa) + fcol * fa


def render_proscenium(cam, L, img, t):
    gx, gy = R.gx, R.gy
    X, Y = cam.unproj_plane(gx, gy, ZP)
    ax = np.abs(X)
    pillar = (ax >= PRO_IN) & (ax <= PRO_OUT) & (Y > HEAD_TOP) & (Y < YS)
    header = (ax <= PRO_OUT) & (Y >= HEAD_TOP) & (Y <= PRO_TOP)
    gl = 0.10 + 0.45 * (L.sL + L.sR) / 2 + 0.15 * L.proj + 0.25 * L.logo + 0.6 * L.house
    # pillar: fluted gold column with bevelled edges
    q = (ax - PRO_IN) / (PRO_OUT - PRO_IN)
    flute = 0.70 + 0.30 * np.cos(2 * np.pi * q * 5)
    bevel = 1 + 0.6 * np.exp(-(q / 0.05) ** 2) - 0.4 * np.exp(-((1 - q) / 0.05) ** 2)
    vert = 0.75 + 0.25 * np.sin(Y / 400.0 + 1.0)
    spec = np.exp(-((q - 0.30) / 0.06) ** 2) * (0.5 + 0.5 * np.sin(Y / 230.0 + t * 0.4))
    pc = GOLD[None, None, :] * (gl * flute * bevel * vert * 1.1)[..., None] + (WARM * 0.5)[None, None, :] * (gl * spec)[..., None]
    a = pillar[..., None]
    img[:] = img * (1 - a) + pc * a
    # header: gold moulding with a row of rosettes
    hq = (Y - HEAD_TOP) / (PRO_TOP - HEAD_TOP)
    band = 0.75 + 0.25 * np.cos(2 * np.pi * hq * 3)
    ros = np.exp(-(((X % 160) - 80) ** 2 + ((Y - (HEAD_TOP + PRO_TOP) / 2) * 1.0) ** 2) / 900.0)
    hc = GOLD[None, None, :] * (gl * (band + 0.6 * ros) * (0.8 + 0.4 * hq))[..., None]
    a = header[..., None]
    img[:] = img * (1 - a) + hc * a


def render_apron(cam, L, img, t):
    gx, gy = R.gx, R.gy
    X, Y = cam.unproj_plane(gx, gy, ZA)
    m = (np.abs(X) < PRO_OUT) & (Y >= YS) & (Y < YAF)
    if not m.any():
        return
    q = (Y - YS) / (YAF - YS)
    panel = 0.85 + 0.15 * np.cos(2 * np.pi * X / 340.0)
    col = np.array([0.10, 0.022, 0.03], np.float32)[None, None, :] * ((0.25 + 0.6 * L.foot * np.exp(-q * 3)) * panel)[..., None]
    trim = (Y < YS + 18)
    col = np.where(trim[..., None], GOLD[None, None, :] * (0.12 + 0.6 * L.foot + 0.3 * (L.sL + L.sR) / 2), col)
    a = m[..., None]
    img[:] = img * (1 - a) + col * a


def footlights_glow(cam, L, add):
    if L.foot <= 0:
        return
    for i, xb in enumerate(np.arange(-FOOT_X, FOOT_X + 1, 145)):
        on = float(sstep(3.55 + abs(xb) / FOOT_X * 0.55, 3.7 + abs(xb) / FOOT_X * 0.55, L.t))
        if on <= 0:
            continue
        x, y, s = cam.proj(xb, YS - 6, ZA)
        splat(add, x, y, WARM * (0.9 * on), 9 * s, 0.9)
        splat(add, x, y - 20 * s, WARM * (0.25 * on), 60 * s, 0.6)


def splat(buf, x, y, color, radius, gain=1.0, stretch=1.0):
    """Additive Gaussian glow at full-res screen coords into an env-res buffer."""
    xe, ye, r = x / ES, y / ES, max(radius / ES, 0.6)
    ext = int(r * 3.5) + 2
    x0, x1 = int(xe - ext * stretch), int(xe + ext * stretch) + 1
    y0, y1 = int(ye - ext), int(ye + ext) + 1
    if x1 < 0 or y1 < 0 or x0 >= EW or y0 >= EH:
        return
    x0c, x1c, y0c, y1c = max(x0, 0), min(x1, EW), max(y0, 0), min(y1, EH)
    xx = np.arange(x0c, x1c, dtype=np.float32)[None, :] + 0.5 - xe
    yy = np.arange(y0c, y1c, dtype=np.float32)[:, None] + 0.5 - ye
    g = np.exp(-((xx / stretch) ** 2 + yy ** 2) / (2 * r * r)) * gain
    buf[y0c:y1c, x0c:x1c] += g[..., None] * np.asarray(color, np.float32)[None, None, :]


def render_seats(cam, L, t):
    """Rows of cinema seats (silhouettes with rim light). Returns (near, far) RGBA layers."""
    gx, gy = R.gx, R.gy
    layers = []
    rim_l = 0.03 + 0.15 * L.proj + 0.30 * (L.sL + L.sR) / 2
    for group in ((-5.0, -4.0, -3.0, -2.0), (-1.0, 0.0, 1.0, 2.0, 3.0) + tuple(EXTRA_ROWS)):
        col = np.zeros((EH, EW, 3), np.float32)
        alpha = np.zeros((EH, EW), np.float32)
        for zr in sorted(group, reverse=True):
            d = zr - cam.cz
            if d < 0.7:
                continue
            s = F / d
            ytop = H / 2 + (YAF - 300 - cam.cy) * s
            if ytop > H + 50:
                continue
            r0 = int(max((ytop - 60 * s) / ES, 0))
            if r0 >= EH:
                continue
            X = cam.cx + (gx[r0:] - W / 2 - cam.lx) / s
            Y = cam.cy + (gy[r0:] - H / 2) / s
            off = 70.0 if (round(zr * 10) // 10) % 2 else 0.0
            lx_ = ((X + off) % 140.0) - 70.0
            # rounded seat back: width 120, top radius 40
            hx = np.abs(lx_) - 60.0
            yt = (YAF - 300) + 40.0
            qx = np.maximum(hx + 40.0, 0)
            qy = np.maximum(yt - Y, 0)
            sd = np.where(Y < yt, np.hypot(qx, qy) - 40.0, hx)
            aisle = (np.abs(X) > 160) & (np.abs(X) < XWALL - 120)
            cov = clamp(0.5 - sd * s / ES) * aisle * (Y < YAF)
            rim = np.exp(-np.maximum(sd + 6, 0) ** 2 / 40.0) * np.exp(-np.maximum(Y - yt, 0) / 40.0)
            c = np.array([0.075, 0.010, 0.016], np.float32)[None, None, :] * (0.25 + 1.2 * rim_l * (1 - 0.6 * np.clip((Y - yt) / 300, 0, 1)))[..., None]
            c = c + np.array([0.9, 0.45, 0.3], np.float32)[None, None, :] * (0.25 * rim * rim_l)[..., None]
            a = cov[..., None]
            col[r0:] = col[r0:] * (1 - a) + c * a
            alpha[r0:] = alpha[r0:] * (1 - cov) + cov
        layers.append((col, alpha, group))
    return layers


# --------------------------------------------------------------------------
# Vintage projector (foreground silhouette, opening shot)
# --------------------------------------------------------------------------


def sd_box(px, py, cx, cy, hx, hy, r=0.0):
    qx = np.abs(px - cx) - hx + r
    qy = np.abs(py - cy) - hy + r
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def sd_circle(px, py, cx, cy, r):
    return np.hypot(px - cx, py - cy) - r


def render_projector(cam, L, t):
    d = PJ_POS[2] - cam.cz
    if d < 0.35:
        return None
    s = F / d
    ox, oy, _ = cam.proj(PJ_POS[0], PJ_POS[1], PJ_POS[2])
    # bounding box in env pixels (local extents roughly x:-170..190, y:-230..320)
    x0 = int(max((ox - 190 * s) / ES, 0)); x1 = int(min((ox + 230 * s) / ES, EW))
    y0 = int(max((oy - 240 * s) / ES, 0)); y1 = int(min((oy + 330 * s) / ES, EH))
    if x1 <= x0 or y1 <= y0:
        return None
    gx = R.gx[y0:y1, x0:x1]; gy = R.gy[y0:y1, x0:x1]
    px = (gx - ox) / s; py = (gy - oy) / s
    body = sd_box(px, py, 0, 0, 82, 52, 14)
    lamp = sd_box(px, py, -40, -58, 30, 16, 6)
    lens = sd_box(px, py, 118, -2, 40, 17, 4)
    hood = sd_box(px, py, 160, -2, 8, 25, 3)
    col_ = sd_box(px, py, 0, 110, 16, 60, 4)
    foot = sd_box(px, py, 0, 178, 70, 10, 5)
    arm1 = sd_box(px, py, -48, -100, 5, 50, 2)
    arm2 = sd_box(px, py, 56, -96, 5, 46, 2)
    sd = np.minimum.reduce([body, lamp, lens, hood, col_, foot, arm1, arm2])
    ang = t * 3.2 if t > 0.6 else 0.0
    for (rcx, rcy, rr, dirn) in ((-50, -150, 74, 1), (62, -140, 68, 1)):
        disc = sd_circle(px, py, rcx, rcy, rr)
        a = np.arctan2(py - rcy, px - rcx) - dirn * ang * (74.0 / rr)
        rad = np.hypot(px - rcx, py - rcy)
        k = 5
        sect = np.cos(k * a)
        hole = (rad > 0.25 * rr) & (rad < 0.82 * rr) & (sect > 0.35)
        disc = np.where(hole, np.maximum(disc, 1.5), disc)
        sd = np.minimum(sd, disc)
    cov = clamp(0.5 - sd * s / ES)
    # rim light from the stage (right/up) and from the lamp
    eps = 1.0
    gxs = np.gradient(sd, axis=1); gys = np.gradient(sd, axis=0)
    gn = np.hypot(gxs, gys) + 1e-6
    nxp, nyp = gxs / gn, gys / gn
    rimband = np.exp(-np.maximum(-sd, 0) * s / ES / 3.0)
    rim = rimband * clamp(0.25 + nxp * 0.70 - nyp * 0.60)
    lvl = 0.15 + 0.85 * L.proj
    col = np.zeros(gx.shape + (3,), np.float32)
    col += np.array([0.02, 0.013, 0.014], np.float32)
    col += np.array([1.0, 0.78, 0.52], np.float32)[None, None, :] * (0.85 * rim * lvl)[..., None]
    # faint metal body sheen
    col += np.array([0.10, 0.08, 0.07], np.float32)[None, None, :] * (clamp(-py / 160.0 + 0.3) * lvl * (sd < 0))[..., None]
    # lamp-house light leak (vents)
    vents = (np.abs(py + 58) < 9) & (np.abs(px + 40) < 22) & ((np.floor((px + 40) / 6) % 2) == 0)
    col += np.array([1.0, 0.6, 0.25], np.float32)[None, None, :] * (vents * 0.9 * L.proj)[..., None]
    out = np.zeros((EH, EW, 3), np.float32)
    alpha = np.zeros((EH, EW), np.float32)
    out[y0:y1, x0:x1] = col
    alpha[y0:y1, x0:x1] = cov
    lens_xy = (ox + 168 * s, oy - 2 * s)
    return out, alpha, lens_xy, s


# --------------------------------------------------------------------------
# Volumetric beams + dust
# --------------------------------------------------------------------------
def cone(S, T, rS, rT, gain, color, haze, hard=0.0, tail=0.25, endfade=0.35):
    gx, gy = R.gx, R.gy
    ax, ay = T[0] - S[0], T[1] - S[1]
    L2 = ax * ax + ay * ay + 1e-6
    vx, vy = gx - S[0], gy - S[1]
    tt = (vx * ax + vy * ay) / L2
    px_, py_ = vx - tt * ax, vy - tt * ay
    perp = np.sqrt(px_ * px_ + py_ * py_)
    r = rS + (rT - rS) * np.clip(tt, 0, 1.3)
    q = perp / np.maximum(r, 1)
    prof = np.exp(-q * q * 2.2) * (1 - hard) + hard * (1 - sstep(0.8, 1.05, q))
    along = sstep(0.0, 0.04, tt) * (1 - sstep(1.0, 1.0 + tail, tt)) * (1.0 - endfade * np.clip(tt, 0, 1))
    I = prof * along * gain * haze
    return I[..., None] * np.asarray(color, np.float32)[None, None, :], I


def render_beams(cam, L, t, pj_lens):
    ox1 = int(t * 9) % 300; oy1 = int(t * 4) % 300
    ox2 = int(200 - t * 13) % 300; oy2 = int(t * 6) % 300
    h1 = R.haze1[oy1:oy1 + EH, ox1:ox1 + EW]
    h2 = R.haze2[oy2:oy2 + EH, ox2:ox2 + EW]
    haze = np.clip(0.86 + 0.09 * h1 + 0.08 * h2, 0.5, 1.3)
    add = np.zeros((EH, EW, 3), np.float32)
    inten = np.zeros((EH, EW), np.float32)
    # projector beam
    if L.proj > 0:
        tgt_y = -60.0 * (1 - L.open)
        tz = ZC + (ZL - ZC) * L.open
        lens = np.array([PJ_POS[0] + 168, PJ_POS[1] - 2, PJ_POS[2]])
        tgt = np.array([0.0, tgt_y, tz])
        lam = 0.0
        if lens[2] - cam.cz < 0.5:
            # lens is behind the camera: start the beam where it crosses the near plane
            lam = (cam.cz + 0.5 - lens[2]) / (tgt[2] - lens[2])
        P = lens + lam * (tgt - lens)
        Sx, Sy, ssrc = cam.proj(P[0], P[1], P[2])
        Tx, Ty, ts = cam.proj(tgt[0], tgt[1], tgt[2])
        rS = (12.0 + 548.0 * lam) * ssrc
        c, I = cone((Sx, Sy), (Tx, Ty), rS, 560.0 * ts, 0.50 * L.proj, [1.0, 0.90, 0.74], haze, hard=0.25, tail=0.05, endfade=0.7)
        add += c; inten += I
    # two golden follow-spots from front-of-house positions
    for lvl, sx in ((L.sL, -SPOT_X), (L.sR, SPOT_X)):
        if lvl <= 0:
            continue
        Sx, Sy, ss = cam.proj(sx, SPOT_Y, 6.5)
        Tx, Ty, ts = cam.proj(-sx * 0.06, -30.0, ZL)
        c, I = cone((Sx, Sy), (Tx, Ty), 14.0 * ss, 430 * ts, 0.26 * lvl, [1.0, 0.82, 0.55], haze, hard=0.6, tail=0.05)
        add += c; inten += I
        # fixture lens glare
        splat(add, Sx, Sy, np.array([1.0, 0.85, 0.6]) * lvl, 26 * ss + 6, 1.2)
        splat(add, Sx, Sy, np.array([1.0, 0.7, 0.4]) * lvl, 140 * ss + 30, 0.25)
    return add, inten


def render_dust(cam, L, t, inten, logo_scr):
    D = R.dust
    X = D[:, 0] + 40 * np.sin(t * 0.21 + D[:, 3]) + t * 6
    Y = D[:, 1] - t * 9 + 30 * np.sin(t * 0.37 + D[:, 3] * 1.7)
    Y = (Y + 1600) % 2500 - 1600
    Z = D[:, 2]
    d = Z - cam.cz
    ok = d > 0.25
    X, Y, Z, d, b = X[ok], Y[ok], Z[ok], d[ok], D[ok, 4]
    s = F / d
    xs = W / 2 + (X - cam.cx) * s + cam.lx
    ys = H / 2 + (Y - cam.cy) * s
    xe = (xs / ES).astype(np.int32); ye = (ys / ES).astype(np.int32)
    inb = (xe >= 0) & (xe < EW) & (ye >= 0) & (ye < EH)
    xe, ye, s, b, d = xe[inb], ye[inb], s[inb], b[inb], d[inb]
    light = inten[ye, xe] * 6.0 + 0.012 * L.house
    tw = 0.6 + 0.4 * np.sin(t * 3 + b * 50)
    val = light * b * tw
    # mask motes off the emblem disc (never cover the logo)
    if L.logo > 0:
        lx, ly, ls = logo_scr
        rr = np.hypot(xe * ES - lx, ye * ES - ly)
        val = val * (rr > (LOGO_R + 30) * ls)
    out = np.zeros((EH, EW), np.float32)
    sizes = s * 1.2
    for lo, hi, sig in ((0, 2.5, 0.7), (2.5, 7, 1.8), (7, 1e9, 5.0)):
        m = (sizes >= lo) & (sizes < hi)
        if not m.any():
            continue
        buf = np.zeros((EH, EW), np.float32)
        np.add.at(buf, (ye[m], xe[m]), val[m] * (1.0 if sig < 1 else 0.35 if sig < 3 else 0.12) * 2.0)
        out += cv2.GaussianBlur(buf, (0, 0), sig) * (2 * np.pi * sig * sig if sig > 1 else 2.5)
    return out[..., None] * np.array([1.0, 0.88, 0.70], np.float32)[None, None, :]


# --------------------------------------------------------------------------
# Logo layer (full-res)
# --------------------------------------------------------------------------
def logo_transform(cam):
    """Screen-space scale and top-left of the logo bitmap."""
    s = cam.scale(ZL)
    cxs, cys, _ = cam.proj(0.0, 0.0, ZL)   # world origin of logo plane
    if abs(s - 1.0) < 1e-9 and abs(cxs - W / 2) < 1e-9 and abs(cys - H / 2) < 1e-9:
        return 1.0, LOGO_TL_FINAL
    # final position: circle centre at (TLx + LOGO_CX, TLy + LOGO_CY)
    fcx = LOGO_TL_FINAL[0] + LOGO_CX
    fcy = LOGO_TL_FINAL[1] + LOGO_CY
    ccx = cxs + (fcx - W / 2) * s
    ccy = cys + (fcy - H / 2) * s
    return s, (ccx - LOGO_CX * s, ccy - LOGO_CY * s)


def logo_screen(cam):
    s, (tx, ty) = logo_transform(cam)
    return tx + LOGO_CX * s, ty + LOGO_CY * s, s


def place_logo(s, tl):
    """Return (rgb, alpha, x0, y0) of the logo resampled to screen space."""
    if s == 1.0 and float(tl[0]).is_integer() and float(tl[1]).is_integer():
        return R.logo, R.logo_mask, int(tl[0]), int(tl[1]), True
    hh, ww = R.logo_mask.shape
    nw, nh = max(int(round(ww * s)), 2), max(int(round(hh * s)), 2)
    sx, sy = nw / ww, nh / hh
    pm = np.dstack([R.logo * R.logo_mask[..., None], R.logo_mask])
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4
    small = cv2.resize(pm, (nw, nh), interpolation=interp)
    x0, y0 = int(math.floor(tl[0])), int(math.floor(tl[1]))
    fx, fy = tl[0] - x0, tl[1] - y0
    if fx > 1e-4 or fy > 1e-4:
        M = np.float32([[1, 0, fx], [0, 1, fy]])
        small = cv2.warpAffine(small, M, (nw + 2, nh + 2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = np.clip(small[..., 3], 0, 1)
    rgb = np.clip(small[..., :3] / np.maximum(a, 1e-4)[..., None], 0, 1)
    return rgb, a, x0, y0, False


# --------------------------------------------------------------------------
# Film-strip transition (foreground, full-res)
# --------------------------------------------------------------------------
def film_strip(frame, t, x_shift, y_center, angle_deg, height, alpha_gain, scroll):
    """Draw a classic 35mm-style film strip as an overlay (navy/lavender)."""
    ang = math.radians(angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    # bounding rows
    reach = height * 0.5 / max(ca, 0.2) + abs(sa) * W * 0.5 + 60
    y0 = int(max(y_center - reach, 0)); y1 = int(min(y_center + reach, H))
    if y1 <= y0:
        return
    yy = np.arange(y0, y1, dtype=np.float32)[:, None] - y_center
    xx = np.arange(W, dtype=np.float32)[None, :] - W / 2
    u = xx * ca + yy * sa + scroll + x_shift      # along strip
    v = -xx * sa + yy * ca                        # across strip
    hv = height / 2
    inside = 1 - sstep(hv - 2, hv + 2, np.abs(v))
    if inside.max() <= 0:
        return
    pitch = height * 0.11
    sp_w, sp_h = pitch * 0.42, height * 0.065
    rail = hv - height * 0.115
    du = (u % pitch) - pitch / 2
    sprocket = (np.abs(du) < sp_w / 2) & (np.abs(np.abs(v) - rail) < sp_h / 2)
    fw = height * 0.95
    fu = (u % fw) - fw / 2
    frame_win = (np.abs(fu) < fw * 0.44) & (np.abs(v) < hv * 0.70)
    base = np.array([0.13, 0.13, 0.21], np.float32)
    lav = np.array([0.80, 0.72, 0.90], np.float32)
    idx = np.floor(u / fw)
    glow = 0.55 + 0.45 * np.sin(idx * 1.7) ** 2
    rad = np.exp(-((fu / (fw * 0.35)) ** 2 + (v / (hv * 0.6)) ** 2))
    win_col = lav[None, None, :] * (0.55 + 0.45 * rad * glow)[..., None] + np.array([0.35, 0.22, 0.05], np.float32) * rad[..., None] * glow[..., None]
    col = np.where(frame_win[..., None], win_col, base[None, None, :])
    a = inside * np.where(sprocket, 0.0, np.where(frame_win, 0.55, 0.94)) * alpha_gain
    region = frame[y0:y1]
    region[:] = region * (1 - a[..., None]) + col * a[..., None]


# --------------------------------------------------------------------------
# Frame
# --------------------------------------------------------------------------
def tonemap(x):
    knee = 0.78
    y = np.where(x < knee, x, knee + (1 - knee) * (1 - np.exp(-(x - knee) / (1 - knee))))
    return np.clip(y, 0, 1)


def blur_layer(col, alpha, sigma):
    if sigma < 0.4:
        return col, alpha
    pm = col * alpha[..., None]
    pm = cv2.GaussianBlur(pm, (0, 0), sigma)
    al = cv2.GaussianBlur(alpha, (0, 0), sigma)
    return pm / np.maximum(al[..., None], 1e-4), al


def coc(d, fd, k=55.0):
    return k * abs(1.0 / max(d, 0.2) - 1.0 / fd) * 3.0


def render_frame(t, debug=False):
    cam = Cam(t)
    L = Lights(t)
    lx, ly, ls = logo_screen(cam)
    logo_scr = (lx, ly, ls)
    # focus distance: projector first, rack to the stage
    fd_proj = PJ_POS[2] - cam.cz
    fd_stage = ZL - cam.cz
    rack = float(s5(1.6, 3.2, t))
    fd = 1.0 / ((1 - rack) / max(fd_proj, 0.4) + rack / fd_stage)

    env = np.zeros((EH, EW, 3), np.float32)
    render_box(cam, L, env)
    # stage interior goes only where the opening is: build it separately & mask
    stage = np.zeros_like(env)
    floor, Zf = render_stage_interior(cam, L, stage, logo_scr)
    logo_small = None
    if L.logo > 0:
        s, tl = logo_transform(cam)
        se = s / ES
        nw = max(int(round(R.logo.shape[1] * se)), 2)
        nh = max(int(round(R.logo.shape[0] * se)), 2)
        pm = cv2.resize(R.logo * R.logo_mask[..., None], (nw, nh), interpolation=cv2.INTER_AREA)
        am = cv2.resize(R.logo_mask, (nw, nh), interpolation=cv2.INTER_AREA)
        logo_small = (pm, am)
    render_floor_reflection(cam, L, stage, floor, logo_small, logo_scr)
    render_curtains(cam, L, stage, t)
    render_valance(cam, L, stage, t)
    Xo, Yo = cam.unproj_plane(R.gx, R.gy, ZP)
    opening = ((np.abs(Xo) < PRO_IN + 1) & (Yo > HEAD_TOP) & (Yo < YS + 60)).astype(np.float32)
    env = env * (1 - opening[..., None]) + stage * opening[..., None]
    render_proscenium(cam, L, env, t)
    render_apron(cam, L, env, t)
    # the cinema starts in near-total darkness
    env *= 0.2 + 0.8 * float(sstep(0.3, 2.0, t))
    glow = np.zeros_like(env)
    footlights_glow(cam, L, glow)
    # sconces on the side walls / proscenium wall
    if L.house > 0:
        for side in (-1, 1):
            for zs in (-12.0, -8.0, -4.0, 0.0, 4.0):
                if zs - cam.cz > 0.4:
                    x, y, s = cam.proj(side * (XWALL - 30), -520.0, zs)
                    splat(glow, x, y, np.array([1.0, 0.72, 0.42]) * L.house, 14 * s, 1.6)
                    splat(glow, x, y, np.array([1.0, 0.6, 0.3]) * L.house, 70 * s, 0.22)
            x, y, s = cam.proj(side * SCONCE_X, -470.0, ZP - 0.05)
            splat(glow, x, y, np.array([1.0, 0.72, 0.42]) * L.house, 16 * s, 1.6)
            splat(glow, x, y, np.array([1.0, 0.6, 0.3]) * L.house, 90 * s, 0.25)
    # depth of field on the theatre (far group)
    sig_far = min(coc(ZP - cam.cz, fd), 10.0) / ES
    if sig_far > 0.4:
        env = cv2.GaussianBlur(env, (0, 0), sig_far)
    env += glow
    # beams + dust (in front of the stage, behind the foreground)
    pj = render_projector(cam, L, t)
    beams, inten = render_beams(cam, L, t, pj[2] if pj else None)
    env += beams
    env += render_dust(cam, L, t, inten, logo_scr)
    # seats
    for col, alpha, group in render_seats(cam, L, t):
        dd = np.mean(group) - cam.cz
        sig = min(coc(max(dd, 0.3), fd), 24.0) / ES
        col, alpha = blur_layer(col, alpha, sig)
        env = env * (1 - alpha[..., None]) + col * alpha[..., None]
    # projector silhouette + lens
    if pj:
        col, alpha, lens, ps = pj
        sig = min(coc(PJ_POS[2] - cam.cz, fd), 24.0) / ES
        col, alpha = blur_layer(col, alpha, sig)
        env = env * (1 - alpha[..., None]) + col * alpha[..., None]
        if L.proj > 0:
            splat(env, lens[0], lens[1], np.array([1.0, 0.95, 0.85]) * L.proj, 10 * ps, 3.0)
            splat(env, lens[0], lens[1], np.array([1.0, 0.85, 0.6]) * L.proj, 55 * ps, 0.5)
            splat(env, lens[0], lens[1], np.array([0.9, 0.75, 0.55]) * L.proj, 30 * ps, 0.35, stretch=9.0)
    # bloom
    br = np.clip(env - 0.55, 0, None)
    bl = cv2.GaussianBlur(br, (0, 0), 18) * 0.55 + cv2.GaussianBlur(br, (0, 0), 5) * 0.35
    env += bl
    # flicker in the opening seconds, vignette
    if t < 3.2:
        fl = 1.0 + 0.04 * (float(vnoise1(t * 24, 9)) - 0.5) * (1 - t / 3.2)
        env *= fl
    rr = ((R.gx - W / 2) / (W / 2)) ** 2 + ((R.gy - H / 2) / (H / 2)) ** 2
    env *= (1 - 0.38 * np.clip(rr, 0, 2) ** 1.2 / 1.5)[..., None]
    env = tonemap(env)
    # film grain (environment only)
    g = R.grain[int(t * FPS) % len(R.grain)]
    lum = env.mean(2)
    env += (g * (0.018 + 0.03 * lum * (1 - lum)))[..., None]
    env = np.clip(env, 0, 1)

    frame = cv2.resize(env, (W, H), interpolation=cv2.INTER_LINEAR)

    # ---------------- emblem ----------------
    exact = False
    if L.logo > 0:
        s, tl = logo_transform(cam)
        rgb, a, x0, y0, exact = place_logo(s, tl)
        draw_bezel(frame, cam, t, lx, ly, ls, L)
        h, w = a.shape
        X0, Y0 = max(x0, 0), max(y0, 0)
        X1, Y1 = min(x0 + w, W), min(y0 + h, H)
        sub_rgb = rgb[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        sub_a = a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] * L.logo
        # transient light effects ON the emblem (all zero from t = 15.9 s)
        fx = logo_effects(t, s, sub_rgb.shape[:2], (x0, y0), (X0, Y0), lx, ly)
        reg = frame[Y0:Y1, X0:X1]
        if fx is not None:
            lit = 1 - (1 - sub_rgb) * (1 - np.clip(fx, 0, 1))      # screen blend
        else:
            lit = sub_rgb
        reg[:] = reg * (1 - sub_a[..., None]) + lit * sub_a[..., None]
        # projection bloom: the emblem "arrives" in a wash of projector light
        bloom = 0.65 * (1 - float(s5(5.9, 7.3, t))) * float(sstep(5.75, 6.0, t))
        if bloom > 0:
            reg[:] = 1 - (1 - reg) * (1 - bloom * np.array([1.0, 0.93, 0.80], np.float32) * sub_a[..., None])
    rim_trace(frame, t, lx, ly, ls)
    add_sparkles(frame, t, lx, ly, ls)

    # ---------------- film-strip transition ----------------
    if 5.25 < t < 6.75:
        k = (t - 5.25) / 1.5
        e = 0.5 - 0.5 * math.cos(math.pi * k)
        yc = H + 700 - e * (H + 1400)
        film_strip(frame, t, 0, yc, -7.0, 640, 1.0, t * 2200)
        k2 = clamp((t - 5.45) / 1.3)
        if 0 < k2 < 1:
            e2 = 0.5 - 0.5 * math.cos(math.pi * k2)
            yc2 = -500 + e2 * (H + 1000)
            film_strip(frame, t, 0, yc2, 5.0, 360, 0.75, -t * 1600)
    # light leak during transition
    leak = bump(t, 5.6, 6.0, 6.1, 6.8)
    if leak > 0:
        xx = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
        yy = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
        lk = np.exp(-((xx - 0.9) ** 2 / 0.25 + (yy + 0.6) ** 2 / 0.6)) * 0.35 * leak
        frame[:] = 1 - (1 - frame) * (1 - lk[..., None] * np.array([1.0, 0.6, 0.3], np.float32))

    out = np.clip(np.round(frame * 255.0), 0, 255).astype(np.uint8)
    if exact and t >= 15.9:
        # hard guarantee: the emblem pixels are bit-identical to the source
        x0, y0 = LOGO_TL_FINAL
        hh, ww = R.logo_u8.shape[:2]
        m = R.logo_mask >= 1.0
        sub = out[y0:y0 + hh, x0:x0 + ww]
        sub[m] = R.logo_u8[m]
    return out


def draw_bezel(frame, cam, t, lx, ly, ls, L):
    """Slim champagne-gold bezel + coin edge around (outside) the emblem disc."""
    r_in, r_out = (LOGO_R - 2) * ls, (LOGO_R + 16) * ls
    ext = int(r_out + 40 * ls + 4)
    x0, x1 = int(max(lx - ext, 0)), int(min(lx + ext, W))
    y0, y1 = int(max(ly - ext, 0)), int(min(ly + ext, H))
    yy = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5 - ly
    xx = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5 - lx
    # coin edge (thickness), shifted against the camera arc for a 3-D read
    off = -cam.cx * 0.035 * ls
    re = np.hypot(xx - off, yy - 6 * ls)
    rr = np.hypot(xx, yy)
    edge = clamp(r_out + 0.5 - re) * clamp(re - r_in + 0.5)
    reg = frame[y0:y1, x0:x1]
    reg[:] = reg * (1 - edge[..., None] * L.logo) + (GOLD * 0.42)[None, None, :] * edge[..., None] * L.logo
    ring = clamp(r_out + 0.5 - rr) * clamp(rr - r_in + 0.5)
    ang = np.arctan2(yy, xx)
    q = (rr - r_in) / (r_out - r_in)
    bevel = 0.72 + 0.28 * np.cos(np.pi * q)
    light = 0.78 + 0.22 * np.cos(ang + 2.3) + 0.15 * np.cos(2 * ang)
    spec = np.exp(-((ang + 2.2 - 0.25 * np.sin(t * 0.6)) % (2 * np.pi) - np.pi) ** 2 / 0.08) * 0.35
    col = GOLD[None, None, :] * (bevel * light)[..., None] + spec[..., None] * np.array([1, 0.95, 0.85], np.float32)
    reg[:] = reg * (1 - ring[..., None] * L.logo) + np.clip(col, 0, 1) * ring[..., None] * L.logo
    # soft contact shadow of the disc onto the stage air (outside the bezel)
    sh = np.exp(-np.maximum(rr - r_out, 0) / (22 * ls)) * (rr > r_out) * 0.35 * L.logo
    reg[:] = reg * (1 - sh[..., None])


def logo_effects(t, s, shape, xy0, XY0, lx, ly):
    """Screen-blend light (0..1 RGB) applied on the emblem; None when inactive."""
    active = (5.9 < t < 15.9)
    if not active:
        return None
    h, w = shape
    x0, y0 = xy0
    X0, Y0 = XY0
    # pixel coordinates in source-bitmap space
    xs = (np.arange(X0, X0 + w, dtype=np.float32)[None, :] + 0.5 - x0) / s - 0.5
    ys = (np.arange(Y0, Y0 + h, dtype=np.float32)[:, None] + 0.5 - y0) / s - 0.5
    dx = xs - LOGO_CX
    dy = ys - LOGO_CY
    rr = np.sqrt(dx * dx + dy * dy)
    fx = np.zeros((h, w, 3), np.float32)
    warm = np.array([1.0, 0.90, 0.70], np.float32)
    # 1) champagne-gold light tracing the existing circular border (6.6 – 10.2 s)
    if 6.5 < t < 10.3:
        k = clamp((t - 6.6) / 2.9)
        head = -math.pi / 2 + 2 * math.pi * (k * k * (3 - 2 * k))
        ang = np.arctan2(dy, dx)
        behind = (head - ang) % (2 * np.pi)
        tail = np.exp(-behind / 0.9) * (behind < 2 * math.pi)
        band = np.exp(-((rr - RING_R) / 5.0) ** 2) + 0.45 * np.exp(-((rr - RING_R) / 18.0) ** 2)
        g = bump(t, 6.5, 6.8, 9.3, 9.6)
        full = bump(t, 9.2, 9.6, 9.7, 10.3) * 0.35
        head_glow = np.exp(-behind / 0.08) * np.exp(-((rr - RING_R) / 30.0) ** 2) * 0.6
        trace = (band * (tail * g + full) + head_glow * g) * 1.1
        fx += trace[..., None] * np.array([1.0, 0.86, 0.58], np.float32)
    # 2) subtle illumination of the decorative stars (7.6 – 10.4 s)
    if 7.4 < t < 10.6:
        for i, (sx, sy) in enumerate(STARS):
            g = bump(t, 7.5 + i * 0.12, 7.9 + i * 0.12, 9.4 + i * 0.05, 10.3)
            if g <= 0:
                continue
            d2 = (xs - sx) ** 2 + (ys - sy) ** 2
            fx += (np.exp(-d2 / (2 * 20.0 ** 2)) * 0.75 * g)[..., None] * warm
    # 3) soft projector pool sweeping across the emblem (7.0 – 9.6 s)
    g = bump(t, 7.0, 7.5, 9.0, 9.6)
    if g > 0:
        px = -700 + 1400 * float(s5(7.0, 9.6, t))
        d2 = ((dx - px) / 330.0) ** 2 + (dy / 520.0) ** 2
        fx += (np.exp(-d2) * 0.10 * g)[..., None] * warm
    # 4) diagonal sheen sweep + glints on the gold details (10.5 – 13.2 s)
    g = bump(t, 10.5, 10.8, 12.7, 13.2)
    if g > 0:
        p = -900 + 1800 * float(s5(10.5, 13.0, t))
        dd = (dx + dy * 0.55) - p
        fx += (np.exp(-(dd / 160.0) ** 2) * 0.20 * g)[..., None] * warm
        p2 = p - 120
        dd2 = (dx + dy * 0.55) - p2
        gm = gold_mask_window(xs, ys)
        fx += (np.exp(-(dd2 / 50.0) ** 2) * 0.85 * g * gm)[..., None] * np.array([1.0, 0.95, 0.82], np.float32)
    # 5) spotlights warming both sides (10 – 14 s), very gentle and symmetric
    g = bump(t, 10.0, 10.8, 13.2, 14.2)
    if g > 0:
        side = np.exp(-((np.abs(dx) - 520) / 160.0) ** 2) * (rr < RING_R + 20)
        fx += (side * 0.05 * g)[..., None] * warm
    return fx


def gold_mask_window(xs, ys):
    xi = np.clip(np.round(xs[0]).astype(np.int32), 0, R.gold.shape[1] - 1)
    yi = np.clip(np.round(ys[:, 0]).astype(np.int32), 0, R.gold.shape[0] - 1)
    return R.gold[np.ix_(yi, xi)]


def add_sparkles(frame, t, lx, ly, ls):
    """Final delicate twinkles near the existing decorative stars and on the rim (14.0 – 15.85 s)."""
    if not (14.0 < t < 15.9):
        return
    # (kind, index/angle, start time, size)
    picks = [('star', 1, 14.05, 120), ('rim', -0.95, 14.25, 190), ('star', 10, 14.45, 110),
             ('rim', 2.55, 14.6, 170), ('star', 0, 14.8, 120), ('rim', 0.55, 14.95, 160),
             ('star', 12 - 1, 15.1, 130), ('rim', -2.35, 15.25, 180)]
    for kind, ref, t0, size in picks:
        k = (t - t0) / 0.55
        if not (0 < k < 1):
            continue
        amp = math.sin(math.pi * k) ** 2
        if kind == 'star':
            sx, sy = STARS[ref]
            cx = lx + (sx - LOGO_CX) * ls
            cy = ly + (sy - LOGO_CY) * ls
            col = [0.85, 0.62, 0.30]          # warm gold so it reads on the ivory
            # on ivory, add a gentle golden glint (subtractive of blue keeps it visible)
            ext = int(size * ls * 0.8)
            x0, x1 = int(cx - ext), int(cx + ext)
            y0, y1 = int(cy - ext), int(cy + ext)
            yy = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5 - cy
            xx = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5 - cx
            th = 1.6 * ls
            Lr = size * ls * 0.4
            rays = np.exp(-np.abs(yy) / th - np.abs(xx) / Lr) + np.exp(-np.abs(xx) / th - np.abs(yy) / Lr)
            val = np.clip(rays * amp * 0.8, 0, 1)
            reg = frame[y0:y1, x0:x1]
            tint = np.array([1.0, 0.86, 0.55], np.float32)
            reg[:] = reg * (1 - val[..., None]) + np.minimum(reg + 0.25, 1) * tint * val[..., None]
        else:
            rc = (LOGO_R + 7) * ls
            cx, cy = lx + rc * math.cos(ref), ly + rc * math.sin(ref)
            _flare(frame, cx, cy, size * ls, amp, 0.0, [1.0, 0.90, 0.68], 2.0 * ls + 0.5)

def _flare(frame, cx, cy, size, amp, rot, color, thick):
    ext = int(size * 1.6) + 4
    x0, x1 = int(max(cx - ext, 0)), int(min(cx + ext, W))
    y0, y1 = int(max(cy - ext, 0)), int(min(cy + ext, H))
    if x1 <= x0 or y1 <= y0:
        return
    yy = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5 - cy
    xx = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5 - cx
    c, s_ = math.cos(rot), math.sin(rot)
    u = xx * c + yy * s_
    v = -xx * s_ + yy * c
    rays = np.exp(-np.abs(v) / thick - np.abs(u) / (size * 0.45)) + np.exp(-np.abs(u) / thick - np.abs(v) / (size * 0.45))
    rays += 0.35 * (np.exp(-np.abs(u + v) / thick - np.abs(u - v) / (size * 0.3)) + np.exp(-np.abs(u - v) / thick - np.abs(u + v) / (size * 0.3)))
    r2 = xx * xx + yy * yy
    core = np.exp(-r2 / (2 * (size * 0.07) ** 2)) + 0.4 * np.exp(-r2 / (2 * (size * 0.22) ** 2))
    val = np.clip((rays * 0.9 + core) * amp, 0, 1.5)
    reg = frame[y0:y1, x0:x1]
    reg[:] = np.clip(reg + val[..., None] * np.asarray(color, np.float32), 0, 1)


def rim_trace(frame, t, lx, ly, ls):
    """Champagne-gold light running around the medallion's circular border (outside the artwork)."""
    if not (6.5 < t < 10.4):
        return
    k = float(clamp((t - 6.6) / 2.9))
    e = k * k * (3 - 2 * k)
    head = -math.pi / 2 + 2 * math.pi * e
    g = bump(t, 6.5, 6.8, 9.3, 9.6)
    full = bump(t, 9.2, 9.6, 9.7, 10.4)
    rc = (LOGO_R + 7) * ls
    ext = int(rc + 60 * ls)
    x0, x1 = int(max(lx - ext, 0)), int(min(lx + ext, W))
    y0, y1 = int(max(ly - ext, 0)), int(min(ly + ext, H))
    yy = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5 - ly
    xx = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5 - lx
    rr = np.sqrt(xx * xx + yy * yy)
    near = np.abs(rr - rc) < 60 * ls
    ang = np.arctan2(yy, xx)
    behind = (head - ang) % (2 * np.pi)
    tail = np.exp(-behind / 1.1)
    outside = sstep((LOGO_R - 1) * ls, (LOGO_R + 1) * ls, rr)       # never on the artwork
    line = np.exp(-((rr - rc) / (3.0 * ls + 0.8)) ** 2)
    halo = np.exp(-((rr - rc) / (20.0 * ls)) ** 2)
    I = (line * 0.9 + halo * 0.35) * (tail * g + 0.55 * full) * outside * near
    reg = frame[y0:y1, x0:x1]
    reg[:] = np.clip(reg + I[..., None] * np.array([1.0, 0.82, 0.52], np.float32), 0, 1)
    if g > 0.05:
        hx, hy = lx + rc * math.cos(head), ly + rc * math.sin(head)
        _flare(frame, hx, hy, 150 * ls, 0.9 * g, head, [1.0, 0.88, 0.62], 2.0 * ls + 0.5)
