"""
Original sound design for the PSA crest intro — synthesised from scratch (no samples, nothing taken from the
reference video).  Every effect is scheduled from timeline.py / the same element schedules the renderer uses,
so each fragment lock, wedge, letter and glint gets its own sound on the exact frame it happens.
"""
import os
import sys
import math
import wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timeline as TL
import assets

SR = 48000
N = int(SR * TL.DUR)
t_all = np.arange(N) / SR
rng = np.random.default_rng(1953)
L = np.zeros(N)
Rr = np.zeros(N)
FX_L, FX_R = np.zeros(N), np.zeros(N)       # dry effects bus (gets less reverb)


def add(sig, start, gain=1.0, pan=0.0, bus='fx'):
    i0 = int(round(start * SR))
    if i0 < 0:
        sig = sig[-i0:]
        i0 = 0
    i1 = min(i0 + len(sig), N)
    if i1 <= i0:
        return
    s = sig[:i1 - i0] * gain
    pan = max(-1.0, min(1.0, pan))
    l, r = math.sqrt(0.5 * (1 - pan)), math.sqrt(0.5 * (1 + pan))
    if bus == 'fx':
        FX_L[i0:i1] += s * l
        FX_R[i0:i1] += s * r
    else:
        L[i0:i1] += s * l
        Rr[i0:i1] += s * r


def fft_filter(x, lo=None, hi=None, order=2):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    g = np.ones_like(f)
    if lo:
        g /= np.sqrt(1 + (lo / np.maximum(f, 1.0)) ** (2 * order))
    if hi:
        g /= np.sqrt(1 + (f / hi) ** (2 * order))
    return np.fft.irfft(X * g, len(x))


def env(n, a, d):
    tt = np.arange(n) / SR
    return np.minimum(tt / max(a, 1e-4), 1.0) * np.exp(-tt / d)


def norm(x):
    return x / (np.abs(x).max() + 1e-9)


# ---------------------------------------------------------------- building blocks
def tick(freq=3000.0, dur=0.12, weight=0.5):
    """Metallic lock tick: transient + inharmonic ring + small body thud."""
    n = int(dur * SR)
    tt = np.arange(n) / SR
    ring = sum(a * np.sin(2 * np.pi * freq * r * tt + rng.uniform(0, 6)) * np.exp(-tt / (0.035 / r ** 0.5))
               for r, a in ((1.0, 1.0), (1.52, 0.6), (2.31, 0.4), (3.17, 0.25)))
    click = fft_filter(rng.standard_normal(n), 2500, 12000) * np.exp(-tt / 0.004)
    body = np.sin(2 * np.pi * (140 + 60 * weight) * tt) * np.exp(-tt / 0.03) * weight
    return norm(ring * 0.5 + click * 0.6 + body)


def whoosh(dur, lo, hi, peak=0.7, sweep=True):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    x = rng.standard_normal(n)
    out = np.zeros(n)
    blk = 2048
    for i in range(0, n, blk):
        u = i / n
        c = lo + (hi - lo) * (u if sweep else math.sin(math.pi * u))
        seg = fft_filter(x[i:i + blk + 512], c * 0.5, c * 1.8, 1)[:min(blk, n - i)]
        out[i:i + len(seg)] = seg
    e = np.where(tt < peak * dur, (tt / (peak * dur)) ** 2, np.exp(-(tt - peak * dur) / (0.25 * dur)))
    return norm(out * e)


def thump(f0=50.0, dur=1.2, drop=0.6, decay=0.35):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    f = f0 * (1 + drop * np.exp(-tt * 22))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / decay) * np.minimum(tt / 0.003, 1)


def metal_hit(base=110.0, dur=2.4):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    for r, a, d in ((1.0, 1.0, 0.9), (2.61, 0.7, 0.6), (4.11, 0.5, 0.45), (6.65, 0.35, 0.3), (10.3, 0.2, 0.2), (13.9, 0.12, 0.12)):
        out += a * np.sin(2 * np.pi * base * r * tt + rng.uniform(0, 6)) * np.exp(-tt / d)
    crack = fft_filter(rng.standard_normal(n), 800, 9000) * np.exp(-tt / 0.03)
    return norm(out + 0.8 * crack)


def bell(freq, dur=2.5):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    for r, a, d in ((1, 1.0, 1.2), (2.0, 0.4, 0.8), (2.76, 0.3, 0.6), (5.4, 0.15, 0.3), (8.93, 0.07, 0.2)):
        out += a * np.sin(2 * np.pi * freq * r * tt) * np.exp(-tt / d)
    return norm(out * np.minimum(tt / 0.002, 1))


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def pad(notes, start, dur, gain, att=1.0, rel=1.2, bright=7, pan=0.0):
    n = int((dur + rel) * SR)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    for m in notes:
        for det in (-0.08, 0.0, 0.07):
            f = midi(m) * 2 ** (det / 12)
            ph = rng.uniform(0, 6.28)
            for h in range(1, bright + 1):
                out += np.sin(2 * np.pi * f * h * tt + ph * h) / h ** 1.5
    e = np.ones(n)
    na, nr = int(att * SR), int(rel * SR)
    e[:na] = np.linspace(0, 1, na) ** 2
    e[-nr:] *= np.linspace(1, 0, nr) ** 1.7
    out = fft_filter(out * e, None, 3000)
    add(out / (len(notes) * 3), start, gain, pan - 0.2, 'mus')
    add(out / (len(notes) * 3), start + 0.011, gain, pan + 0.2, 'mus')


# ================================================================ schedule-driven effects
A = assets.load()
shard_s = TL.shard_schedule(A['shard_c'])
wedge_s = TL.wedge_schedule(len(A['wedge_c']))
kinds = [str(k) for k in A['item_kind']]
item_s = TL.letter_schedule(kinds, A['item_c'])


def pan_of(x):
    return float(np.clip((x - 1000.0) / 1000.0, -0.85, 0.85))


# room tone + low drone through the build
n = N
room = fft_filter(rng.standard_normal(n), 60, 900) * 0.012
add(room * np.clip(t_all / 1.5, 0, 1), 0.0, 1.0, 0.0, 'mus')
drone = (np.sin(2 * np.pi * midi(26) * t_all) + 0.4 * np.sin(2 * np.pi * midi(38) * t_all + 1)) * \
    np.clip(t_all / 6.0, 0, 1) ** 1.5 * (1 - np.clip((t_all - 7.6) / 0.15, 0, 1))
add(drone, 0.0, 0.10, 0.0, 'mus')

# 1) fragments: individual flight whooshes + a lock tick per fragment, panned to where it lands
for k, s in enumerate(shard_s):
    cx = float(A['shard_c'][k][0])
    fl = s['land'] - s['start']
    add(whoosh(fl, 300, 2600 + 1500 * rng.random(), peak=0.85), s['start'], 0.045, pan_of(cx + s['dx']) * 0.6 + pan_of(cx) * 0.4)
    add(tick(rng.uniform(2200, 4200), 0.14, rng.uniform(0.3, 0.7)), s['land'], 0.11, pan_of(cx))
t_outline = max(s['land'] for s in shard_s)
add(thump(70, 0.6, 0.4, 0.12), t_outline + 0.02, 0.35, 0.0)
add(metal_hit(240, 1.2), t_outline + 0.02, 0.07, 0.0)

# 2) iris: servo whine + accelerating ratchet + tick per wedge + heavy lock
t0, t1 = 3.0, max(s['land'] for s in wedge_s)
nn = int((t1 - t0 + 0.05) * SR)
tt = np.arange(nn) / SR
u = tt / (t1 - t0)
fsv = 380 + 520 * u ** 1.3
servo = sum(np.sin(2 * np.pi * np.cumsum(fsv * h) / SR) / h for h in (1, 2, 3, 5)) * (0.4 + 0.6 * u) * np.minimum(tt / 0.05, 1)
add(fft_filter(servo, 200, 4000), t0, 0.06, 0.0)
rate = 14 + 30 * u
ph = np.cumsum(rate) / SR
for i in np.nonzero(np.diff(np.floor(ph)) > 0)[0]:
    add(tick(1800, 0.05, 0.2), t0 + i / SR, 0.05, rng.uniform(-0.3, 0.3))
for k, s in enumerate(wedge_s):
    add(tick(2600 - 60 * k, 0.1, 0.5), s['land'], 0.12, pan_of(float(A['wedge_c'][k][0])))
add(thump(55, 1.0, 0.7, 0.22), t1, 0.6, 0.0)
add(metal_hit(160, 1.8), t1, 0.16, 0.0)
add(fft_filter(rng.standard_normal(int(0.2 * SR)), 300, 4000) * env(int(0.2 * SR), 0.001, 0.03), t1, 0.12, 0.0)

# 3) symbols rising: hydraulic hiss + motor hum rising, pneumatic release + clack at the top
d = TL.SYM_END - TL.SYM_START
nn = int(d * SR)
tt = np.arange(nn) / SR
u = tt / d
hiss = fft_filter(rng.standard_normal(nn), 900, 6000) * (0.3 + 0.7 * np.sin(np.pi * np.clip(u * 1.1, 0, 1))) * np.minimum(tt / 0.08, 1)
add(hiss, TL.SYM_START, 0.035, -0.1)
fm = 85 + 70 * TL.s5(0, 1, 0) + 70 * u
hum = sum(np.sin(2 * np.pi * np.cumsum(fm * h) / SR) / h for h in range(1, 7)) * np.minimum(tt / 0.1, 1) * (1 - np.clip((u - 0.92) / 0.08, 0, 1))
add(fft_filter(hum, 60, 1600), TL.SYM_START, 0.07, 0.1)
add(fft_filter(rng.standard_normal(int(0.5 * SR)), 2500, 11000) * env(int(0.5 * SR), 0.004, 0.12), TL.SYM_END, 0.12, 0.0)
add(tick(1500, 0.2, 0.9), TL.SYM_END, 0.3, 0.0)
add(thump(80, 0.5, 0.3, 0.1), TL.SYM_END, 0.3, 0.0)

# 4) lettering: banner slide follows its wipe left->right; a tick per letter, panned
for k, (st, ld) in enumerate(item_s):
    kind = kinds[k]
    x = float(A['item_c'][k][0])
    if kind == 'band':
        nn = int((ld - st + 0.1) * SR)
        sl = fft_filter(rng.standard_normal(nn), 700, 3500)
        tt = np.arange(nn) / SR
        e = np.sin(np.pi * np.clip(tt / (ld - st + 0.1), 0, 1))
        for j, p in enumerate(np.linspace(-0.6, 0.6, 6)):
            a_, b_ = j * nn // 6, (j + 1) * nn // 6
            seg = np.zeros(nn)
            seg[a_:b_] = (sl * e)[a_:b_]
            add(seg, st, 0.07, p)
        continue
    f, g, w = {'est': (3000, 0.16, 0.5), 'pharmacy': (2000, 0.22, 0.9), 'dot': (3800, 0.12, 0.3),
               'dash': (3400, 0.12, 0.3), 'student': (4000, 0.12, 0.35)}[kind]
    add(tick(f * rng.uniform(0.92, 1.08), 0.12, w), ld, g, pan_of(x))
    if kind == 'pharmacy':
        add(whoosh(ld - st, 600, 4000, 0.9), st, 0.03, pan_of(x))

# 5) build-up riser into completion, then the impact
nn = int(1.6 * SR)
tt = np.arange(nn) / SR
ris = whoosh(1.6, 300, 7000, 0.98) * (tt / 1.6) ** 2
tone = np.sin(2 * np.pi * np.cumsum(110 * 2 ** (2.2 * tt / 1.6)) / SR) * (tt / 1.6) ** 2.5
add(ris + 0.3 * tone, TL.T_COMPLETE - 1.6, 0.16, 0.0)
add(thump(36, 3.0, 0.8, 0.7), TL.T_COMPLETE, 0.95, 0.0)
add(metal_hit(98, 3.0), TL.T_COMPLETE, 0.35, -0.1)
add(metal_hit(131, 3.0), TL.T_COMPLETE + 0.004, 0.3, 0.1)

# 6) camera pull-back / lift-off whoosh
add(whoosh(0.9, 2500, 300, 0.35), 7.82, 0.16, -0.3)
add(whoosh(0.9, 2300, 280, 0.38), 7.84, 0.16, 0.3)

# 7) glints: crystalline chimes panned to the glint position
render_pts = None
for j, tg in enumerate(TL.GLINTS):
    f = midi([86, 90, 93, 98, 93, 90, 95, 98][j % 8])
    add(bell(f, 2.0), tg + 0.15, 0.06, [-0.5, 0.4, -0.2, 0.6, -0.6, 0.3, -0.4, 0.5][j % 8])

# 8) gloss sweeps: airy shimmer that travels across the stereo field
for a, b in TL.GLOSS:
    nn = int((b - a) * SR)
    sh = fft_filter(rng.standard_normal(nn), 5000, 14000)
    tt = np.arange(nn) / SR
    e = np.sin(np.pi * tt / (b - a)) ** 2
    half = nn // 2
    add(sh[:half] * e[:half], a, 0.025, -0.5)
    add(sh[half:] * e[half:], a + half / SR, 0.025, 0.5)

# 9) push-in and hero turns
add(whoosh(1.5, 250, 3000, 0.8), 11.0, 0.14, 0.0)
add(whoosh(0.8, 1800, 400, 0.4), 12.6, 0.09, 0.4)
add(whoosh(0.8, 1800, 400, 0.4), 13.6, 0.09, -0.4)

# 10) final settle: soft lock + sub + bell chord, long tail
add(tick(1300, 0.25, 1.0), TL.T_SETTLE, 0.22, 0.0)
add(thump(32, 3.5, 0.5, 0.9), TL.T_SETTLE, 0.7, 0.0)
for m, p in ((62, -0.3), (66, 0.0), (69, 0.3), (74, 0.1), (76, -0.1)):
    add(bell(midi(m), 3.0), TL.T_SETTLE + 0.02, 0.05, p)

# ================================================================ music (original)
# pulse ostinato under the build (D minor, 120 bpm eighths), growing into the impact
bt = 0.25
for i in range(int((TL.T_COMPLETE - 3.0) / bt)):
    st = 3.0 + i * bt
    m = [38, 38, 50, 38, 41, 38, 50, 45][i % 8]
    nn = int(0.22 * SR)
    tt = np.arange(nn) / SR
    f = midi(m)
    saw = sum(np.sin(2 * np.pi * f * h * tt) / h for h in range(1, 14))
    cut = 500 + 2200 * (st - 3.0) / (TL.T_COMPLETE - 3.0)
    note = fft_filter(saw, 30, cut) * np.exp(-tt / 0.09) * np.minimum(tt / 0.004, 1)
    add(note, st, 0.035 + 0.05 * (st - 3.0) / (TL.T_COMPLETE - 3.0), 0.0, 'mus')
# floating section and resolve
pad([50, 57, 62, 65, 69], 8.0, 2.6, 0.22, att=0.8)            # Dm9 colour
pad([46, 53, 58, 62, 69], 10.6, 2.0, 0.22, att=0.6)           # Bbmaj7
pad([43, 50, 58, 62, 67], 12.6, 2.0, 0.24, att=0.5)           # Gm
pad([38, 45, 50, 54, 57, 64], TL.T_SETTLE, 1.5, 0.32, att=0.08, rel=1.0)   # D major add9 — resolve

# ================================================================ mix
def reverb(x, seed, secs):
    r = np.random.default_rng(seed)
    n = int(secs * SR)
    ir = r.standard_normal(n) * np.exp(-np.arange(n) / SR * 6.9 / secs)
    ir = fft_filter(ir, 150, 6000)
    ir /= np.sqrt((ir ** 2).sum())
    m = len(x) + n
    return np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(ir, m), m)[:len(x)]


outL = FX_L * 0.85 + L * 0.8 + reverb(FX_L, 1, 1.6) * 0.25 + reverb(L, 2, 2.8) * 0.45
outR = FX_R * 0.85 + Rr * 0.8 + reverb(FX_R, 3, 1.6) * 0.25 + reverb(Rr, 4, 2.8) * 0.45
mix = np.stack([outL, outR], 1)
mix = fft_filter(mix[:, 0], 25, None)[:, None] * [1, 0] + fft_filter(mix[:, 1], 25, None)[:, None] * [0, 1]   # tidy sub-rumble
fade = np.clip((TL.DUR - t_all) / 0.9, 0, 1) ** 1.3
mix *= fade[:, None]
mix = mix / np.abs(mix).max() * 0.9
mix = np.tanh(mix * 1.15) / np.tanh(1.15)
pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
out = sys.argv[1] if len(sys.argv) > 1 else 'psa_score.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, pcm.shape)
