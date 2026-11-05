"""
Shared choreography for the PSA crest intro (9:16, 16 s, 60 fps).
Both the renderer and the sound design read THIS file, so every sound lands on the frame of its motion.

Beat sheet (follows the reference's sequence, stretched to 16 s; its TikTok outro is NOT reproduced):
  0.00 - 2.40  light-grey studio wall; crest fragments fly in, spin and snap into the outline
  2.40 - 3.00  outline holds, camera drifts
  3.00 - 3.85  maroon field closes in like an iris (wedges converge on the centre)
  3.95 - 5.55  central symbols rise from the base upward (hands, capsule, arcs, Arabic lettering)
  5.65 - 7.70  banner unrolls, EST 2011 / PHARMACY / STUDENT ASSOCIATION appear letter by letter
  7.72         crest complete — impact
  7.80 - 8.60  camera pulls back, crest lifts off the wall and floats above a floor
  8.60 - 11.0  floating, gentle bob, edge glints, drifting dust, gloss sweep
  11.0 - 12.6  push-in with a 3-D turn
  12.6 - 14.6  hero sway, settles front-on
  14.6 - 16.0  final hold, still and frontal
"""
import math
import numpy as np

FPS = 60
DUR = 16.0
NF = int(round(FPS * DUR))
W, H = 2160, 3840

T_COMPLETE = 7.72
T_SETTLE = 14.6


def clamp(x, a=0.0, b=1.0):
    return min(max(x, a), b)


def sstep(a, b, x):
    t = clamp((x - a) / (b - a))
    return t * t * (3 - 2 * t)


def s5(a, b, x):
    t = clamp((x - a) / (b - a))
    return t * t * t * (t * (t * 6 - 15) + 10)


def ease_out_cubic(u):
    u = clamp(u)
    return 1 - (1 - u) ** 3


def ease_out_back(u, k=1.4):
    u = clamp(u)
    return 1 + (k + 1) * (u - 1) ** 3 + k * (u - 1) ** 2


def bump(t, a, b, c, d):
    return sstep(a, b, t) * (1 - sstep(c, d, t))


def pchip(keys):
    ts = np.array([k[0] for k in keys], float)
    vs = np.array([k[1] for k in keys], float)
    h = np.diff(ts)
    dl = np.diff(vs) / h
    m = np.zeros_like(vs)
    for i in range(1, len(vs) - 1):
        if dl[i - 1] * dl[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / dl[i - 1] + w2 / dl[i])

    def f(t):
        if t <= ts[0]:
            return float(vs[0])
        if t >= ts[-1]:
            return float(vs[-1])
        i = int(np.searchsorted(ts, t) - 1)
        hh = h[i]
        s = (t - ts[i]) / hh
        return float((2 * s ** 3 - 3 * s ** 2 + 1) * vs[i] + (s ** 3 - 2 * s ** 2 + s) * hh * m[i] +
                     (-2 * s ** 3 + 3 * s ** 2) * vs[i + 1] + (s ** 3 - s ** 2) * hh * m[i + 1])
    return f


# ---------------------------------------------------------------- camera
_CAM = {
    #        t     scale  yaw   pitch  roll  centre_y
    'k': [(0.0,   0.70, -16.0, 12.0, -4.0, 1920),
          (2.4,   0.80,  -9.0,  9.0, -2.0, 1920),
          (3.85,  0.85,   3.0,  7.0,  0.0, 1920),
          (5.6,   0.89,  13.0,  6.0,  1.5, 1920),
          (7.72,  0.93,  -5.0,  3.0,  0.0, 1920),
          (8.6,   0.80,   0.0, -3.0,  0.0, 1840),
          (11.0,  0.82,   7.0, -4.0,  0.0, 1840),
          (12.6,  0.99,  18.0, -5.0,  0.0, 1890),
          (13.6,  0.98, -11.0, -2.0,  0.0, 1910),
          (T_SETTLE, 0.96, 0.0,  0.0,  0.0, 1920),
          (DUR,   0.96,   0.0,  0.0,  0.0, 1920)],
}
_cam_f = [pchip([(k[0], k[j]) for k in _CAM['k']]) for j in range(1, 6)]
FINAL_SCALE = 0.96


def camera(t):
    if t >= T_SETTLE:
        return dict(scale=FINAL_SCALE, yaw=0.0, pitch=0.0, roll=0.0, cy=1920.0)
    s, yaw, pitch, roll, cy = [f(t) for f in _cam_f]
    return dict(scale=s, yaw=yaw, pitch=pitch, roll=roll, cy=cy)


def bob(t):
    """vertical float offset in logo pixels (negative = up)."""
    return -14.0 * math.sin(2 * math.pi * (t - 8.4) / 2.6) * bump(t, 8.3, 9.0, 13.8, T_SETTLE)


def wall_amount(t):
    """1 = studio wall stage, 0 = floating cyclorama stage."""
    return 1.0 - s5(7.85, 8.6, t)


def thickness(t):
    """extrusion depth in logo px (faded to 0 for the clean frontal hold)."""
    return (22.0 + 36.0 * s5(7.8, 8.8, t)) * (1 - s5(14.0, T_SETTLE - 0.05, t))


# ---------------------------------------------------------------- element schedules
def shard_schedule(centres, seed=7):
    """Per-fragment (start, land, start_dx, start_dy, start_rot, start_z, start_scale)."""
    rng = np.random.default_rng(seed)
    c = np.asarray(centres) - 1000.0
    ang = np.arctan2(c[:, 1], c[:, 0])
    rank = np.argsort(np.argsort((ang + 2.2) % (2 * np.pi)))
    out = []
    n = len(c)
    for i in range(n):
        # reference: pieces scattered close around the final shape converge from all sides and lock
        # almost together; a slight angular stagger keeps the snap readable
        land = 1.80 + 0.32 * rank[i] / (n - 1) + rng.uniform(-0.04, 0.04)
        fl = rng.uniform(1.45, 1.75)
        dirn = ang[i] + rng.uniform(-0.55, 0.55)
        dist = rng.uniform(380, 820)
        out.append(dict(start=land - fl, land=land, dx=math.cos(dirn) * dist, dy=math.sin(dirn) * dist,
                        rot=rng.choice([-1, 1]) * rng.uniform(90, 240), z=rng.uniform(240, 520),
                        sc=rng.uniform(0.18, 0.38)))
    return out


def wedge_schedule(n):
    return [dict(start=3.0 + 0.012 * k, land=3.70 + 0.016 * k, rot=58.0, push=420.0, z=260.0) for k in range(n)]


SYM_START, SYM_END = 3.95, 5.55
SYM_Y0, SYM_Y1 = 1420.0, 230.0          # wipe travels from below the hands to above the Arabic lettering
BAND_START, BAND_END = 5.65, 6.05


def letter_schedule(kinds, centres):
    """Return per-item (start, land) for the lettering; order follows reading order."""
    kinds = [str(k) for k in kinds]
    c = np.asarray(centres)
    out = [None] * len(kinds)
    def seq(kind, t0, step):
        idx = [i for i, k in enumerate(kinds) if k == kind]
        idx.sort(key=lambda i: c[i, 0])
        for j, i in enumerate(idx):
            out[i] = (t0 + step * j, t0 + step * j + 0.24)
    seq('est', 5.72, 0.06)
    seq('pharmacy', 6.12, 0.075)
    seq('dot', 6.74, 0.0)
    seq('dash', 6.80, 0.0)
    seq('student', 6.86, 0.0375)
    for i, k in enumerate(kinds):
        if k == 'band':
            out[i] = (BAND_START, BAND_END)
    return out


GLINTS = [8.95, 9.35, 9.85, 10.35, 10.75, 12.95, 13.4, 13.95]
GLOSS = [(9.15, 10.25), (11.15, 12.45), (13.0, 14.0)]
