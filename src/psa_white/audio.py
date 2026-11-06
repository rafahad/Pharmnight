"""
Elegant cinematic sound design for the white 10 s PSA assembly — synthesised from scratch (no samples,
nothing from the reference video).  Timed from the renderer's own element schedule.

Palette: deep soft booms, warm low strings, airy whooshes, soft muted thumps, filtered-noise sparkles.
Bold low end, refined top; no pitched pings, bells or toy-like sounds.
"""
import math
import os
import sys
import wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render as WR

WR.init()
SR = 48000
DUR = WR.DUR
N = int(SR * DUR)
t_all = np.arange(N) / SR
rng = np.random.default_rng(606)
BUS = {k: np.zeros((N, 2)) for k in ('fx', 'hit', 'mus')}


def add(sig, start, gain=1.0, pan=0.0, bus='fx'):
    i0 = int(round(start * SR))
    if i0 < 0:
        sig, i0 = sig[-i0:], 0
    i1 = min(i0 + len(sig), N)
    if i1 <= i0:
        return
    s = sig[:i1 - i0] * gain
    pan = max(-1.0, min(1.0, pan))
    BUS[bus][i0:i1, 0] += s * math.sqrt(0.5 * (1 - pan))
    BUS[bus][i0:i1, 1] += s * math.sqrt(0.5 * (1 + pan))


def filt(x, lo=None, hi=None, order=2):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    g = np.ones_like(f)
    if lo:
        g /= np.sqrt(1 + (lo / np.maximum(f, 1.0)) ** (2 * order))
    if hi:
        g /= np.sqrt(1 + (f / hi) ** (2 * order))
    return np.fft.irfft(X * g, len(x))


def norm(x):
    return x / (np.abs(x).max() + 1e-9)


def tt(d):
    return np.arange(int(d * SR)) / SR


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def boom(f0=46.0, f1=29.0, dur=2.2):
    t = tt(dur)
    f = f1 + (f0 - f1) * np.exp(-t * 3.5)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (dur * 0.4)) * np.minimum(t / 0.006, 1)


def thump(f0=120.0, dur=0.3, soft=1.0):
    t = tt(dur)
    f = f0 * (1 + 0.8 * np.exp(-t * 40))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.07)
    tex = filt(rng.standard_normal(len(t)), 200, 1400) * np.exp(-t / 0.012)
    return norm(body + 0.25 * soft * norm(tex)) * np.minimum(t / 0.003, 1)


def whoosh(dur, lo, hi, peak=0.6, top=6000):
    n = int(dur * SR)
    x = rng.standard_normal(n)
    out = np.zeros(n)
    for i in range(0, n, 2048):
        u = i / n
        c = lo + (hi - lo) * math.sin(math.pi * min(u / peak, 1) / 2) if u < peak else hi + (lo - hi) * (u - peak) / (1 - peak)
        seg = filt(x[i:i + 2560], c * 0.45, min(c * 2.2, top), 1)[:min(2048, n - i)]
        out[i:i + len(seg)] = seg
    t = np.arange(n) / SR
    e = np.where(t < peak * dur, (t / (peak * dur)) ** 1.6, np.exp(-(t - peak * dur) / (0.3 * dur)))
    return norm(out * e)


def air_sparkle(dur=0.12):
    t = tt(dur)
    return norm(filt(rng.standard_normal(len(t)), 5000, 12000) * np.sin(np.pi * t / dur) ** 2)


def dark_metal(base=60.0, dur=1.6):
    t = tt(dur)
    out = sum(a * np.sin(2 * np.pi * base * r * t + rng.uniform(0, 6)) * np.exp(-t / d)
              for r, a, d in ((1.0, 1.0, 0.8), (1.83, 0.7, 0.6), (2.71, 0.5, 0.45), (4.07, 0.3, 0.3)))
    return norm(filt(out, 40, 1800))


def strings(notes, dur, att=0.8, rel=0.8, cut=1500):
    t = tt(dur + rel)
    out = np.zeros(len(t))
    for m in notes:
        for det in (-0.09, -0.03, 0.04, 0.1):
            f = midi(m) * 2 ** (det / 12) * (1 + 0.003 * np.sin(2 * np.pi * 5.2 * t + rng.uniform(0, 6)))
            ph = 2 * np.pi * np.cumsum(f) / SR + rng.uniform(0, 6)
            for h in range(1, 22):
                if midi(m) * h > 7000:
                    break
                out += np.sin(h * ph) / h
    e = np.minimum(t / att, 1) ** 2 * np.clip((dur + rel - t) / rel, 0, 1)
    return norm(filt(out, 40, cut) * e)


def braam_soft(notes, dur=2.6):
    t = tt(dur)
    out = np.zeros(len(t))
    for m in notes:
        for det in (-0.12, 0.0, 0.11):
            ph = 2 * np.pi * np.cumsum(np.full(len(t), midi(m) * 2 ** (det / 12))) / SR + rng.uniform(0, 6)
            for h in range(1, 40):
                if midi(m) * h > 5000:
                    break
                out += np.sin(h * ph) / h
    br, dk = filt(out, 30, 2600), filt(out, 30, 420)
    k = np.exp(-t / 0.25)
    e = np.minimum(t / 0.03, 1) * np.exp(-t / 1.1)
    return norm((br * k + dk * (1 - k)) * e)


def pan_x(x):
    return float(np.clip((x - 1000) / 1000, -0.8, 0.8))


els = WR.R.els

# ---------------------------------------------------------------- the build
add(whoosh(0.5, 300, 2500, 0.95), -0.35, 0.18)                     # inhale into the first shape
add(boom(46, 30, 2.2), 0.15, 0.55, 0, 'hit')                      # field appears
add(thump(70, 0.5, 0.5), 0.15, 0.35, 0, 'hit')
add(whoosh(0.9, 200, 1600, 0.5), 0.12, 0.16)
for e in els:
    x, kind = e['c'][0], e['anim']
    if e['group'] == 'shield' and e['src'] == 'p' and kind == 'wipe_up':
        add(whoosh(e['t1'] - e['t0'] + 0.1, 150, 900, 0.85), e['t0'], 0.14)
        add(thump(110, 0.3), e['t1'] - 0.05, 0.16)
    elif e['group'] == 'shield' and kind == 'pop' and 'centre' not in e:
        add(thump(150 + 40 * rng.random(), 0.25), e['t0'] + 0.18, 0.10, pan_x(x))
    elif e['group'] == 'shield' and kind == 'angular' and not e.get('s0'):
        d = e['t1'] - e['t0']
        w = whoosh(d + 0.1, 800, 4500, 0.6)
        add(w, e['t0'], 0.10, -0.5)
    elif kind == 'angular':                                            # the frame sweeps round
        d = e['t1'] - e['t0']
        n = int((d + 0.15) * SR)
        sw = whoosh(d + 0.15, 180, 2200, 0.55)
        ang = np.linspace(0, 2 * np.pi, n)
        for j in range(8):                                             # pan follows the sweep (top -> right -> bottom -> left)
            a_, b_ = j * n // 8, (j + 1) * n // 8
            seg = np.zeros(n)
            seg[a_:b_] = sw[a_:b_]
            add(seg, e['t0'], 0.26, math.sin(ang[(a_ + b_) // 2]) * 0.7)
        add(boom(52, 34, 1.4), e['t1'], 0.40, 0, 'hit')
        add(dark_metal(70, 1.4), e['t1'], 0.10, 0, 'hit')
        add(thump(90, 0.4), e['t1'], 0.30, 0, 'hit')
    elif kind == 'fly':
        d = e['t1'] - e['t0']
        add(whoosh(d, 200, 2600, 0.7), e['t0'], 0.20, pan_x(x) * 1.2)
        add(thump(95, 0.35), e['t1'] - 0.08, 0.22, pan_x(x))
    elif kind == 'letter':
        heavy = e['full'].sum() > 9000
        add(filt(rng.standard_normal(int(0.05 * SR)), 700, 3500) * np.exp(-tt(0.05) / 0.010), e['t0'] + 0.12, 0.05, pan_x(x))
        if heavy:
            add(thump(130, 0.3), e['t0'] + 0.2, 0.12, pan_x(x))
    elif kind == 'wipe_lr':
        add(whoosh(e['t1'] - e['t0'] + 0.2, 300, 2400, 0.6), e['t0'], 0.12, -0.4)
        add(whoosh(e['t1'] - e['t0'] + 0.2, 300, 2400, 0.6), e['t0'] + 0.1, 0.12, 0.4)
# Arabic letters arrive right-to-left: soft swish that travels R -> L
ara = [e for e in els if e['group'] == 'shield' and e['anim'] == 'pop' and e['c'][1] < 560]
if ara:
    t0, t1 = min(e['t0'] for e in ara), max(e['t1'] for e in ara)
    w = whoosh(t1 - t0, 1500, 6000, 0.5)
    h = len(w) // 2
    add(w[:h], t0, 0.06, 0.5)
    add(w[h:], t0 + h / SR, 0.06, -0.5)

# ---------------------------------------------------------------- completion + premium motion
T = WR.T_COMPLETE
rt = tt(1.5)
cym = filt(rng.standard_normal(len(rt)), 3000, 13000) * (rt / 1.5) ** 4
add(norm(cym), T - 1.5, 0.12, 0)
add(whoosh(1.5, 200, 5000, 0.98), T - 1.5, 0.16, 0)
add(boom(44, 27, 3.0), T, 0.85, 0, 'hit')
add(braam_soft((26, 38, 45, 50, 54)), T, 0.30, -0.2, 'hit')
add(braam_soft((26, 38, 45, 50, 54)), T + 0.01, 0.30, 0.2, 'hit')
add(dark_metal(55, 2.2), T, 0.14, 0, 'hit')
add(whoosh(1.8, 150, 1400, 0.4), 6.4, 0.12, 0)                    # soft push-in
n = int(1.1 * SR)                                                   # shimmer sweep travels L -> R
sh = filt(rng.standard_normal(n), 4500, 13000) * np.sin(np.pi * np.arange(n) / n) ** 2
add(sh[:n // 2], 7.0, 0.05, -0.5)
add(sh[n // 2:], 7.0 + (n // 2) / SR, 0.05, 0.5)
for j, tg in enumerate([8.2, 8.45, 8.7, 8.95, 9.15, 9.3]):        # glints: airy sparkles, no tones
    add(air_sparkle(0.14), tg + 0.18, 0.05, [-0.5, 0.5, -0.3, 0.3, -0.6, 0.6][j])

# ---------------------------------------------------------------- score
add(filt((np.sin(2 * np.pi * midi(26) * t_all) + 0.4 * np.sin(2 * np.pi * midi(33) * t_all)) *
         np.clip(t_all / 1.2, 0, 1) * (1 - np.clip((t_all - T) / 0.1, 0, 1)), 20, 300), 0, 0.14, 0, 'mus')
add(strings([47, 54, 59, 62], 2.2, att=1.0, rel=0.6, cut=1100), 0.3, 0.16, 0, 'mus')          # Bm
add(strings([43, 50, 59, 62, 66], 2.2, att=0.6, rel=0.5, cut=1400), 2.6, 0.18, 0, 'mus')      # G
add(strings([45, 52, 57, 61, 64], 1.4, att=0.4, rel=0.3, cut=1800), 4.7, 0.20, 0, 'mus')      # A
add(strings([38, 45, 50, 54, 57, 64, 69], DUR - T - 0.3, att=0.12, rel=0.3, cut=2200), T, 0.30, 0, 'mus')   # D add9
for i, st in enumerate(np.arange(2.6, T, 0.3)):                   # gentle low pulse that grows into the reveal
    u = (st - 2.6) / (T - 2.6)
    add(thump(62, 0.5, 0.2), st, 0.10 + 0.20 * u, 0, 'mus')
for st in (7.25, 8.45):
    add(boom(40, 30, 1.6), st, 0.25, 0, 'mus')


# ---------------------------------------------------------------- mix + master
def reverb(x, seed, secs):
    r = np.random.default_rng(seed)
    n = int(secs * SR)
    ir = r.standard_normal(n) * np.exp(-np.arange(n) / SR * 6.9 / secs)
    ir = filt(ir, 150, 6000)
    ir /= np.sqrt((ir ** 2).sum())
    m = len(x) + n
    return np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(ir, m), m)[:len(x)]


mix = np.zeros((N, 2))
for name, (dry, wet, secs) in {'fx': (0.9, 0.25, 1.4), 'hit': (0.9, 0.35, 2.8), 'mus': (0.85, 0.40, 2.6)}.items():
    b = BUS[name]
    for ch in (0, 1):
        mix[:, ch] += b[:, ch] * dry + reverb(b[:, ch], 31 + 7 * ch + len(name), secs) * wet
for ch in (0, 1):
    mix[:, ch] = filt(mix[:, ch], 24, 16000)
mix *= (np.clip((DUR - t_all) / 0.6, 0, 1) ** 1.3)[:, None]
mix = mix / np.abs(mix).max()
mix = np.tanh(mix * 1.5) / np.tanh(1.5) * 0.95
pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
out = sys.argv[1] if len(sys.argv) > 1 else 'psa_white_score.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, pcm.shape)
