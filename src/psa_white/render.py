"""
PSA crest — clean premium assembly on white (9:16, 2160x3840, 60 fps, 10 s).

Follows the reference's build order (icon first, then the ring/frame, outer pieces, then text, then
secondary text; then pulse, float, push-in, parallax and travelling glints).  The crest keeps its
ORIGINAL colours: every element is a mask over the original pixels, animated by position/scale/reveal
only.  When assembled, the layers recompose the original artwork exactly.
"""
import math
import os
import sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'psa'))
sys.path.insert(0, HERE)
import importlib.util
import assets
import layers
import timeline as PT                       # shared easing helpers (src/psa/timeline.py)
_spec = importlib.util.spec_from_file_location('psa_render', os.path.join(os.path.dirname(HERE), 'psa', 'render.py'))
PR = importlib.util.module_from_spec(_spec)  # geometry helpers (hcam, warp_over, flare, ...)
_spec.loader.exec_module(PR)

W, H = 2160, 3840
FPS = 60
DUR = 10.0
NF = int(FPS * DUR)
T_COMPLETE = 6.05
T_CLEAN = 9.6                               # from here on: front-on, no glints, exact layout
FINAL_SCALE = 0.96

s5, sstep, clamp, bump = PT.s5, PT.sstep, PT.clamp, PT.bump
ease_out = PT.ease_out_cubic
ease_back = PT.ease_out_back


class R:
    pass


_cam = [PT.pchip([(k[0], k[j]) for k in [
    #  t     scale  yaw   pitch
    (0.0,  0.84, -7.0,  6.0),
    (3.0,  0.86, -3.0,  3.0),
    (T_COMPLETE, 0.87, 0.0, 0.0),
    (7.0,  0.89,  3.0, -1.5),
    (8.3,  0.93, -3.0,  1.0),
    (T_CLEAN, 0.96, 0.0, 0.0),
    (DUR,  0.96,  0.0,  0.0)]]) for j in (1, 2, 3)]


def camera(t):
    if t >= T_CLEAN:
        s, yaw, pitch = FINAL_SCALE, 0.0, 0.0
    else:
        s, yaw, pitch = [f(t) for f in _cam]
    pulse = 1 + 0.022 * math.sin(math.pi * clamp((t - T_COMPLETE) / 0.6)) ** 2 if t > T_COMPLETE else 1.0
    bob = -10.0 * math.sin(2 * math.pi * (t - T_COMPLETE) / 2.4) * bump(t, T_COMPLETE, 6.7, 9.0, T_CLEAN)
    cam = dict(scale=s * pulse, yaw=yaw, pitch=pitch, roll=0.0, cx=W / 2, cy=H / 2)
    cam['R'] = PR.rot_matrix(yaw, pitch, 0.0)
    return cam, bob


def depth(t):
    """parallax depth separation (logo px) between ornaments / shield / lettering."""
    return 16.0 * s5(6.2, 7.0, t) * (1 - s5(9.0, T_CLEAN, t))


# ---------------------------------------------------------------- setup
def init():
    PR.init_tagline = lambda: None
    L = assets.load()
    Wl = layers.load()
    R.alpha = L['alpha']
    R.pm = np.dstack([L['col'] * L['alpha'][..., None], L['alpha']]).astype(np.float32)
    fld = L['field']
    R.fpm = np.dstack([L['field_plain'] * fld[..., None], fld.astype(np.float32)]).astype(np.float32)
    R.uu = np.arange(2000, dtype=np.float32)[None, :]
    R.vv = np.arange(2000, dtype=np.float32)[:, None]
    # ultra-real "cast metal badge" material (lighting on top of the original artwork, shared with src/psa)
    PR.R.uu, PR.R.vv = R.uu, R.vv
    R.col, R.fp, R.fld = L['col'], L['field_plain'], fld
    R.mat = PR.build_material(R.col, R.alpha, fld)
    R.mat_f = PR.build_material(R.fp, fld.astype(np.float32), fld)
    fc = L['field_c']
    els = []

    def add(group, mask, src, anim, t0, t1, **kw):
        ys, xs = np.nonzero(mask)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        m = mask[y0:y1, x0:x1, None].astype(np.float32)
        e = dict(group=group, m=m, src=src, x0=x0, y0=y0, anim=anim, t0=t0, t1=t1,
                 c=(float(xs.mean()), float(ys.mean())), full=mask, **kw)
        if anim == 'angular':
            cx, cy = kw['centre']
            ang = (np.arctan2(np.arange(y0, y1)[:, None] - cy, np.arange(x0, x1)[None, :] - cx) - kw['start']) % (2 * np.pi)
            e['ang'] = (ang / (2 * np.pi)).astype(np.float32)[..., None]
        els.append(e)

    # 1) icon base: the maroon field pops up from the centre
    add('shield', fld, 'f', 'pop', 0.15, 0.85, s0=0.25, centre=tuple(fc))
    # 2) symbols, bottom to top: hands rise, icon pops, outer arc draws, numerals, Arabic right-to-left
    sl, sk, sc = Wl['sym_lab'], [str(k) for k in Wl['sym_kind']], Wl['sym_c']
    hands = sl == -2
    for k, kind in enumerate(sk):
        if kind == 'hands':
            hands |= sl == k
    add('shield', hands, 'p', 'wipe_up', 0.60, 1.20)
    icon_idx = sorted([k for k, kd in enumerate(sk) if kd == 'icon'], key=lambda k: sc[k][0])
    for j, k in enumerate(icon_idx):
        add('shield', sl == k, 'p', 'pop', 0.85 + 0.07 * j, 1.30 + 0.07 * j, s0=0.45)
    icon_c = np.mean([sc[k] for k in icon_idx], axis=0) if icon_idx else fc
    for k, kd in enumerate(sk):
        if kd == 'arc':
            add('shield', sl == k, 'p', 'angular', 1.05, 1.60, centre=tuple(icon_c), start=math.radians(150))
        elif kd == 'dotpart':
            add('shield', sl == k, 'p', 'pop', 1.50, 1.80, s0=0.3)
    rom = sorted([k for k, kd in enumerate(sk) if kd == 'roman'], key=lambda k: sc[k][0])
    for j, k in enumerate(rom):
        add('shield', sl == k, 'p', 'pop', 1.35 + 0.05 * j, 1.70 + 0.05 * j, s0=0.7)
    ara = sorted([k for k, kd in enumerate(sk) if kd == 'arabic'], key=lambda k: -sc[k][0])     # right-to-left
    for j, k in enumerate(ara):
        add('shield', sl == k, 'p', 'pop', 1.45 + 0.07 * j, 1.85 + 0.07 * j, s0=0.6)
    # 3) the frame sweeps round the shield, clockwise from the top
    add('shield', Wl['frame'], 'p', 'angular', 1.85, 2.75, centre=tuple(fc), start=-math.pi / 2, s0=1.03)
    # 4) ornaments glide in from outside
    oc = Wl['orn_c']
    order = np.argsort(oc[:, 1] * 10 + oc[:, 0] / 1000)
    for j, k in enumerate(order):
        cx, cy = oc[k]
        dx, dy = cx - 1000, cy - 1000
        n = math.hypot(dx, dy) + 1e-6
        add('orn', Wl['orn_lab'] == k, 'p', 'fly', 2.45 + 0.12 * j, 3.30 + 0.12 * j,
            off=(dx / n * 260, dy / n * 260), rot=(-22 if dx < 0 else 22), s0=0.86)
    # 5) lettering: EST 2011, PHARMACY, then the secondary line with its banner, dots and dashes
    kinds = [str(k) for k in L['item_kind']]
    ic = L['item_c']
    def seq(kind, t0, step, dur):
        idx = sorted([i for i, k in enumerate(kinds) if k == kind], key=lambda i: ic[i][0])
        for j, i in enumerate(idx):
            add('text', L['item_ids'] == i, 'p', 'letter', t0 + step * j, t0 + step * j + dur)
    seq('est', 3.50, 0.06, 0.38)
    seq('pharmacy', 3.95, 0.075, 0.42)
    for i, k in enumerate(kinds):
        if k == 'band':
            add('text', L['item_ids'] == i, 'p', 'wipe_lr', 4.70, 5.15)
    seq('dot', 5.00, 0.0, 0.30)
    seq('dash', 5.08, 0.0, 0.30)
    seq('student', 5.15, 0.035, 0.32)
    R.els = els
    # top-most element per pixel (later elements override the field), for fast 'landed' lookups
    R.top_id = np.full(R.alpha.shape, len(els), np.int32)
    for i, e in enumerate(els):
        R.top_id[e['full']] = i
    R.field_idx = 0
    R.in_field = fld.astype(bool)
    R.group_mask = {g: np.zeros(fld.shape, bool) for g in ('shield', 'orn', 'text')}
    for e in els:
        R.group_mask[e['group']] |= e['full']
    R.static = {g: R.pm * R.group_mask[g][..., None] for g in R.group_mask}
    # glint anchors + a path along the outer silhouette for travelling glints
    cnts, _ = cv2.findContours((R.alpha > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    R.path = big[::8]
    hull = cv2.convexHull(big).reshape(-1, 2)
    pick = [hull[np.argmin(hull[:, 1])]]
    for _ in range(7):
        d = np.min(np.linalg.norm(hull[:, None, :] - np.array(pick)[None], axis=2), 1)
        pick.append(hull[int(np.argmax(d))])
    R.glints = np.array(pick, np.float32)
    R.gy = np.arange(H, dtype=np.float32)[:, None]
    R.gx = np.arange(W, dtype=np.float32)[None, :]


# ---------------------------------------------------------------- element states
def draw_element(canvas, e, t):
    src = R.pm if e['src'] == 'p' else R.fpm
    x0, y0 = e['x0'], e['y0']
    h, w = e['m'].shape[:2]
    if t >= e['t1']:
        return False
    u = clamp((t - e['t0']) / (e['t1'] - e['t0']))
    if u <= 0:
        return True
    crop = src[y0:y0 + h, x0:x0 + w] * e['m']
    a = e['anim']
    # room to move: pad the crop so motion/overshoot is never clipped by its own box
    P = 330 if a == 'fly' else (90 if a in ('pop', 'letter', 'angular') else 0)
    if P:
        crop = cv2.copyMakeBorder(crop, P, P, P, P, cv2.BORDER_CONSTANT, value=0)
        x0, y0 = x0 - P, y0 - P
        h, w = h + 2 * P, w + 2 * P
        if 'ang' in e:
            e = dict(e, ang=cv2.copyMakeBorder(e['ang'], P, P, P, P, cv2.BORDER_CONSTANT, value=1.0)[..., None])
    cx, cy = e['c'][0] - x0, e['c'][1] - y0
    if a == 'pop':
        if 'centre' in e:
            cx, cy = e['centre'][0] - x0, e['centre'][1] - y0
        s = e['s0'] + (1 - e['s0']) * ease_back(u, 1.25)
        M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])
        out = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR) * clamp(u * 2.2)
    elif a == 'wipe_up':
        e_ = s5(0, 1, u)
        yw = h * (1 - e_)
        rows = np.arange(h, dtype=np.float32)[:, None]
        out = crop * np.clip((rows - yw) / 30.0 + 0.5, 0, 1)[..., None]
        out = cv2.warpAffine(out, np.float32([[1, 0, 0], [0, 1, 22 * (1 - e_)]]), (w, h), flags=cv2.INTER_LINEAR)
    elif a == 'wipe_lr':
        xw = (w + 40) * s5(0, 1, u)
        cols = np.arange(w, dtype=np.float32)[None, :]
        out = crop * np.clip((xw - cols) / 30.0, 0, 1)[..., None]
    elif a == 'angular':
        sweep = s5(0, 1, u) * 1.04
        out = crop * np.clip((sweep - e['ang']) / 0.035, 0, 1)
        if e.get('s0'):
            ccx, ccy = e['centre'][0] - x0, e['centre'][1] - y0
            s = e['s0'] + (1 - e['s0']) * ease_out(u)
            out = cv2.warpAffine(out, np.float32([[s, 0, ccx - s * ccx], [0, s, ccy - s * ccy]]), (w, h), flags=cv2.INTER_LINEAR)
    elif a == 'fly':
        ee = ease_out(u)
        s = e['s0'] + (1 - e['s0']) * ee
        rot = math.radians(e['rot'] * (1 - ee))
        dx, dy = e['off'][0] * (1 - ee), e['off'][1] * (1 - ee)
        ca, sa = s * math.cos(rot), s * math.sin(rot)
        M = np.float32([[ca, -sa, cx - ca * cx + sa * cy + dx], [sa, ca, cy - sa * cx - ca * cy + dy]])
        out = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR) * clamp(u * 2.5)
    else:  # letter: drops in from slightly above, settling to scale 1, fading in
        ee = ease_out(u)
        s = 1.16 - 0.16 * ee
        dy = -16 * (1 - ee)
        M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy + dy]])
        out = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR) * clamp(u * 2.4)
    if canvas is None:                     # free-flying element: returned for its own screen-space warp
        return out, x0, y0
    X0, Y0 = max(x0, 0), max(y0, 0)
    X1, Y1 = min(x0 + w, canvas.shape[1]), min(y0 + h, canvas.shape[0])
    out = out[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    reg = canvas[Y0:Y1, X0:X1]
    reg[:] = out + reg * (1 - out[..., 3:4])
    return True


def group_canvases(t):
    if t >= T_COMPLETE:
        return {g: R.pm * R.group_mask[g][..., None] for g in R.group_mask}, []
    landed = np.array([t >= e['t1'] for e in R.els] + [False])
    top = landed[R.top_id]                                   # top-most element at each pixel has landed
    field_ok = R.in_field & landed[R.field_idx]
    out = {}
    for g in ('shield', 'orn', 'text'):
        gm = R.group_mask[g]
        c = np.zeros_like(R.pm)
        m = top & gm
        c[m] = R.pm[m]
        if g == 'shield':
            m2 = field_ok & ~top
            c[m2] = R.fpm[m2]
        out[g] = c
    free = []
    for e in R.els:                                          # animated elements, in build order
        if e['t0'] < t < e['t1']:
            if e['anim'] == 'fly':
                r = draw_element(None, e, t)
                if isinstance(r, tuple):
                    free.append((e['group'],) + r)
            else:
                draw_element(out[e['group']], e, t)
    return out, free


def shimmer(canvas, t):
    a, b = 7.0, 8.1
    if not (a < t < b):
        return canvas
    g = bump(t, a, a + 0.15, b - 0.2, b)
    p = -1100 + 4300 * s5(a, b, t)
    d = (R.uu + 0.7 * R.vv) - p
    band = (np.exp(-(d / 85.0) ** 2) * 0.45 + np.exp(-(d / 300.0) ** 2) * 0.12) * g
    out = canvas.copy()
    out[..., :3] += band[..., None] * (out[..., 3:4] - out[..., :3])
    return out


def clearcoat(lit, M, cam):
    """Glossy clear-coat on the enamel: a soft studio highlight and a faint window reflection that drift
    with the camera (metal parts already carry their own reflections)."""
    yaw, pitch = cam['yaw'], cam['pitch']
    ux, vy = 640.0 - 70.0 * yaw, 520.0 + 45.0 * pitch
    spot = np.exp(-(((R.uu - ux) / 520.0) ** 2 + ((R.vv - vy) / 420.0) ** 2))
    d = (R.uu * 0.8 - R.vv) - (-250.0 + 55.0 * yaw)
    band = np.exp(-(d / 110.0) ** 2) * 0.6 + np.exp(-((d - 260) / 60.0) ** 2) * 0.4
    k = (1 - M['metal']) * (0.11 * spot + 0.045 * band) * M['ao']
    return np.clip(lit + k[..., None] * (1 - lit) , 0, 1)


def background():
    rr = ((R.gx - W / 2) / (W * 0.8)) ** 2 + ((R.gy - H / 2) / (H * 0.62)) ** 2
    v = 0.99 - 0.085 * np.clip(rr, 0, 1.6) ** 1.1
    return np.broadcast_to(v, (H, W))[..., None] * np.array([1.0, 1.0, 1.0], np.float32) + np.zeros((H, W, 3), np.float32)


# ---------------------------------------------------------------- frame
def render_frame(t):
    cam, bob = camera(t)
    lit = clearcoat(PR.light(R.col, R.mat, cam, t), R.mat, cam)
    R.pm = np.dstack([lit * R.alpha[..., None], R.alpha]).astype(np.float32)
    if t < T_COMPLETE:
        flit = clearcoat(PR.light(R.fp, R.mat_f, cam, t), R.mat_f, cam)
        R.fpm = np.dstack([flit * R.fld[..., None], R.fld.astype(np.float32)]).astype(np.float32)
    img = background()
    cans, free = group_canvases(t)
    dz = depth(t)
    zs = {'orn': -dz, 'shield': 0.0, 'text': dz}

    # soft drop shadows on the white wall (raised-emblem look of the reference)
    allA = np.maximum(np.maximum(cans['shield'][..., 3], cans['orn'][..., 3]), cans['text'][..., 3])
    a_small = cv2.resize(allA, (500, 500), interpolation=cv2.INTER_AREA)
    half = np.diag([0.5, 0.5, 1.0])
    for off, sig, op in ((34.0, 16.0, 0.26), (10.0, 4.0, 0.20)):
        sh = np.zeros((H // 2, W // 2), np.float32)
        PR.warp_alpha_add(sh, a_small, half @ PR.hcam(cam, -40.0, bob) @ PR.T(0, off) @ np.diag([4, 4, 1.0]), 1.0)
        for g, o, fx, fy in free:
            PR.warp_alpha_add(sh, np.ascontiguousarray(o[..., 3]), half @ PR.hcam(cam, -40.0, bob) @ PR.T(fx, fy + off * 1.6), 1.0)
        sh = cv2.resize(cv2.GaussianBlur(sh, (0, 0), sig), (W, H), interpolation=cv2.INTER_LINEAR)
        img *= (1 - op * sh)[..., None]

    buf = np.zeros((H, W, 4), np.float32)
    thick = 18.0 * (1 - s5(9.0, T_CLEAN, t))
    sc = cam['scale']
    for g in ('orn', 'shield', 'text'):
        c = shimmer(cans[g], t)
        if thick > 0.5:
            for i in (5, 4, 3, 2, 1):
                PR.warp_over(buf, c, PR.hcam(cam, zs[g] - thick * i / 5, bob), shade=0.28 + 0.32 * (1 - i / 5) ** 2, scale_hint=sc)
        PR.warp_over(buf, c, PR.hcam(cam, zs[g], bob), scale_hint=sc)
        for gg, o, fx, fy in free:
            if gg == g:
                for i in (5, 4, 3, 2, 1):
                    PR.warp_over(buf, o, PR.hcam(cam, zs[g] - 18.0 * i / 5, bob) @ PR.T(fx, fy), shade=0.28 + 0.32 * (1 - i / 5) ** 2, scale_hint=sc)
                PR.warp_over(buf, o, PR.hcam(cam, zs[g], bob) @ PR.T(fx, fy), scale_hint=sc)
    img *= (1 - buf[..., 3:4])
    img += buf[..., :3]

    # sparkling glints: twinkles at the crest's edge points + two glints travelling along its outline
    if 8.15 < t < 9.55:
        for j, tg in enumerate([8.2, 8.45, 8.7, 8.95, 9.15, 9.3]):
            k = (t - tg) / 0.45
            if 0 < k < 1:
                u, v = R.glints[j % len(R.glints)]
                p = PR.hcam(cam, 2.0, bob) @ np.array([u, v, 1.0])
                PR.flare(img, p[0] / p[2], p[1] / p[2], 230 * sc, math.sin(math.pi * k) ** 2 * 0.85)
        for ph in (0.0, 0.5):
            k = (t - 8.15) / 1.4
            idx = int(((k * 0.45 + ph) % 1.0) * len(R.path))
            u, v = R.path[idx]
            p = PR.hcam(cam, 2.0, bob) @ np.array([u, v, 1.0])
            PR.flare(img, p[0] / p[2], p[1] / p[2], 150 * sc, 0.75 * math.sin(math.pi * clamp(k)) ** 1.5)
    return np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
