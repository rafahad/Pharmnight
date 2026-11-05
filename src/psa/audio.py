"""
Epic-trailer sound design + score for the PSA crest intro — synthesised from scratch (no samples, nothing
from the reference video).  Effects are scheduled from timeline.py and the same element schedules the
renderer uses, so every lock, slam, rise and title hit lands on its frame.

Palette: sub-bass drops, layered trailer impacts (sub + low tom + crack + dark metal), detuned "braam"
brass, low taiko-style drums, low-string ostinato, risers / reverse swells, deep whooshes, hydraulics and
heavy servo grinds.  No high-pitched pings or chimes.
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
rng = np.random.default_rng(2026)
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


def tt_(dur):
    return np.arange(int(dur * SR)) / SR


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# ---------------------------------------------------------------- instruments
def sub_drop(f0=52.0, f1=27.0, dur=2.6):
    t = tt_(dur)
    f = f1 + (f0 - f1) * np.exp(-t * 3.2)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (dur * 0.42)) * np.minimum(t / 0.004, 1)


def low_tom(f0=62.0, dur=1.1):
    t = tt_(dur)
    f = f0 * (1 + 1.4 * np.exp(-t * 30))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.28)
    skin = filt(rng.standard_normal(len(t)), 90, 900) * np.exp(-t / 0.035)
    return norm(body + 0.5 * norm(skin))


def crack(dur=0.6):
    t = tt_(dur)
    return norm(filt(rng.standard_normal(len(t)), 500, 5000) * np.exp(-t / 0.05))


def dark_metal(base=55.0, dur=2.2):
    t = tt_(dur)
    out = np.zeros(len(t))
    for r, a, d in ((1.0, 1.0, 0.9), (1.83, 0.8, 0.7), (2.71, 0.6, 0.5), (4.07, 0.4, 0.35), (5.93, 0.25, 0.25), (8.1, 0.15, 0.15)):
        out += a * np.sin(2 * np.pi * base * r * t + rng.uniform(0, 6)) * np.exp(-t / d)
    return norm(filt(out, 40, 2500))


def braam(notes=(26, 38, 45, 50), dur=3.2, bright=5200, dark=380, drop=-0.35):
    """Detuned saw brass cluster with a bright attack that closes down (trailer 'BRAAM')."""
    t = tt_(dur)
    out = np.zeros(len(t))
    bend = 2 ** ((drop * np.exp(-t * 6)) / 12)
    for m in notes:
        for det in (-0.16, -0.06, 0.05, 0.15):
            f = midi(m) * 2 ** (det / 12) * bend
            ph = 2 * np.pi * np.cumsum(f) / SR + rng.uniform(0, 6)
            for h in range(1, 60):
                if midi(m) * h > 7000:
                    break
                out += np.sin(h * ph) / h
    br = filt(out, 30, bright)
    dk = filt(out, 30, dark)
    k = np.exp(-t / 0.28)
    out = br * k + dk * (1 - k)
    e = np.minimum(t / 0.015, 1) * (0.55 + 0.45 * np.exp(-t / 0.5)) * np.clip((dur - t) / 0.8, 0, 1)
    return norm(np.tanh(1.6 * norm(out)) * e)


def trailer_hit(t0, size=1.0, notes=(26, 38, 45, 50), pan=0.0, with_braam=True):
    add(sub_drop(48 + 6 * size, 27, 2.0 + 1.2 * size), t0, 0.95 * size, 0.0, 'hit')
    add(low_tom(58, 1.2), t0, 0.55 * size, 0.0, 'hit')
    add(crack(0.6), t0, 0.22 * size, pan, 'hit')
    add(dark_metal(52 + 10 * rng.random(), 2.4), t0, 0.30 * size, -0.25 + pan, 'hit')
    add(dark_metal(71 + 10 * rng.random(), 2.4), t0 + 0.004, 0.26 * size, 0.25 + pan, 'hit')
    if with_braam:
        b = braam(notes, 2.6 + 0.8 * size)
        add(b, t0, 0.42 * size, -0.3, 'hit')
        add(b[::1], t0 + 0.012, 0.42 * size, 0.3, 'hit')


def clank(t0, size=0.5, pan=0.0):
    """Heavy mechanical lock: low metal + thud + short grit (no pitched ping)."""
    add(dark_metal(95 + 40 * rng.random(), 0.9), t0, 0.35 * size, pan)
    add(low_tom(80, 0.35), t0, 0.45 * size, pan)
    add(filt(rng.standard_normal(int(0.12 * SR)), 300, 2500) * np.exp(-tt_(0.12) / 0.018), t0, 0.35 * size, pan)


def taiko(t0, g=1.0, pan=0.0):
    t = tt_(0.9)
    f = 54 * (1 + 0.9 * np.exp(-t * 35))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.22)
    slap = filt(rng.standard_normal(len(t)), 120, 1400) * np.exp(-t / 0.025)
    add(norm(body + 0.35 * norm(slap)), t0, 0.5 * g, pan, 'mus')


def whoosh(dur, lo, hi, peak=0.6, top=2400):
    n = int(dur * SR)
    x = rng.standard_normal(n)
    out = np.zeros(n)
    blk = 2048
    for i in range(0, n, blk):
        u = i / n
        c = lo + (hi - lo) * math.sin(math.pi * min(u / peak, 1) / 2) if u < peak else hi + (lo - hi) * (u - peak) / (1 - peak)
        seg = filt(x[i:i + blk + 512], c * 0.45, min(c * 2.2, top), 1)[:min(blk, n - i)]
        out[i:i + len(seg)] = seg
    t = np.arange(n) / SR
    e = np.where(t < peak * dur, (t / (peak * dur)) ** 1.6, np.exp(-(t - peak * dur) / (0.28 * dur)))
    return norm(out * e)


def riser(dur, f0=90, f1=520):
    t = tt_(dur)
    u = t / dur
    nz = whoosh(dur, 200, 5000, 0.97, 7000) * u ** 2
    f = f0 * (f1 / f0) ** u
    saw = sum(np.sin(2 * np.pi * np.cumsum(f * h * (1 + d)) / SR) / h for h in range(1, 12) for d in (-0.004, 0.004))
    saw = filt(saw, 60, 3000) * u ** 2.2
    cym = filt(rng.standard_normal(len(t)), 3500, 14000) * u ** 4
    return norm(nz * 0.8 + norm(saw) * 0.6 + norm(cym) * 0.35)


def strings(notes, dur, att=0.8, rel=0.8, cut=1400):
    t = tt_(dur + rel)
    out = np.zeros(len(t))
    for m in notes:
        for det in (-0.1, -0.03, 0.04, 0.11):
            f = midi(m) * 2 ** (det / 12) * (1 + 0.003 * np.sin(2 * np.pi * 5.3 * t + rng.uniform(0, 6)))
            ph = 2 * np.pi * np.cumsum(f) / SR + rng.uniform(0, 6)
            for h in range(1, 24):
                if midi(m) * h > 6000:
                    break
                out += np.sin(h * ph) / h
    e = np.minimum(t / att, 1) * np.clip((dur + rel - t) / rel, 0, 1)
    return norm(filt(out, 40, cut) * e)


def spiccato(m, dur=0.11, cut=1800):
    t = tt_(dur + 0.08)
    out = np.zeros(len(t))
    for det in (-0.08, 0.0, 0.08):
        ph = 2 * np.pi * midi(m) * 2 ** (det / 12) * t + rng.uniform(0, 6)
        for h in range(1, 18):
            out += np.sin(h * ph) / h
    e = np.minimum(t / 0.006, 1) * np.exp(-t / 0.07)
    return norm(filt(out, 50, cut) * e)


# ================================================================ sound effects from the schedules
A = assets.load()
shard_s = TL.shard_schedule(A['shard_c'])
wedge_s = TL.wedge_schedule(len(A['wedge_c']))
kinds = [str(k) for k in A['item_kind']]
item_s = TL.letter_schedule(kinds, A['item_c'])


def pan_of(x):
    return float(np.clip((x - 1000.0) / 1000.0, -0.8, 0.8))


# opening: dark low drone and air
dr = (np.sin(2 * np.pi * midi(26) * t_all) + 0.5 * np.sin(2 * np.pi * midi(33) * t_all + 1.0)) * \
     np.clip(t_all / 3.0, 0, 1) * (1 - np.clip((t_all - 7.65) / 0.1, 0, 1))
add(filt(dr, 20, 400), 0, 0.16, 0, 'mus')
add(filt(rng.standard_normal(N), 80, 700) * np.clip(t_all / 2, 0, 1) * (t_all < 8), 0, 0.010, 0, 'mus')

# fragments: one big swirling fly-in, a dense metallic assembly rattle as they lock, then a slam
add(whoosh(2.1, 140, 1300, 0.85), 0.05, 0.30, -0.35)
add(whoosh(2.1, 160, 1200, 0.87), 0.08, 0.30, 0.35)
for k, s in enumerate(shard_s):
    cx = float(A['shard_c'][k][0])
    add(filt(rng.standard_normal(int(0.08 * SR)), 250, 2200) * np.exp(-tt_(0.08) / 0.012), s['land'], 0.10, pan_of(cx))
    add(low_tom(110 + 40 * rng.random(), 0.18), s['land'], 0.10, pan_of(cx))
trailer_hit(TL.T_OUTLINE, 0.55, with_braam=False)

# iris: heavy servo grind building, ratchet, then a big lock
t0 = 3.0
d = TL.T_IRIS - t0
t = tt_(d)
u = t / d
f = 55 + 95 * u ** 1.4
grind = sum(np.sin(2 * np.pi * np.cumsum(f * h) / SR) / h for h in range(1, 16))
grind = filt(grind * (1 + 0.4 * filt(rng.standard_normal(len(t)), 20, 120) * 30), 40, 1800) * (0.35 + 0.65 * u)
add(norm(grind), t0, 0.16, 0.0)
rate = 9 + 22 * u
ph = np.cumsum(rate) / SR
for i in np.nonzero(np.diff(np.floor(ph)) > 0)[0]:
    add(filt(rng.standard_normal(int(0.04 * SR)), 400, 2500) * np.exp(-tt_(0.04) / 0.008), t0 + i / SR, 0.09, rng.uniform(-0.3, 0.3))
trailer_hit(TL.T_IRIS, 0.8, notes=(26, 33, 38, 45), with_braam=True)

# symbols rising: hydraulic push + rising rumble, pneumatic release + clank at the top
d = TL.SYM_END - TL.SYM_START
t = tt_(d)
u = t / d
hyd = filt(rng.standard_normal(len(t)), 250, 2200) * (0.4 + 0.6 * np.sin(np.pi * np.clip(u * 1.05, 0, 1)))
add(norm(hyd), TL.SYM_START, 0.07, -0.15)
f = 45 + 40 * u
rum = sum(np.sin(2 * np.pi * np.cumsum(f * h) / SR) / h for h in range(1, 9)) * np.minimum(t / 0.2, 1) * (1 - np.clip((u - 0.93) / 0.07, 0, 1))
add(filt(rum, 30, 700), TL.SYM_START, 0.12, 0.15)
add(filt(rng.standard_normal(int(0.7 * SR)), 1500, 8000) * np.exp(-tt_(0.7) / 0.18), TL.SYM_END, 0.10, 0.0)
clank(TL.SYM_END, 0.9)

# lettering: banner slides, letters stamp in (muted thuds, heavier for PHARMACY)
for k, (st, ld) in enumerate(item_s):
    kind = kinds[k]
    x = float(A['item_c'][k][0])
    if kind == 'band':
        sl = whoosh(ld - st + 0.15, 200, 1400, 0.6)
        add(sl, st, 0.14, -0.4)
        add(sl, st + 0.08, 0.14, 0.4)
        continue
    g = {'pharmacy': 0.95, 'est': 0.6, 'dot': 0.45, 'dash': 0.45, 'student': 0.42}[kind]
    add(low_tom(95 if kind == 'pharmacy' else 120, 0.3), ld, 0.30 * g, pan_of(x))
    add(filt(rng.standard_normal(int(0.06 * SR)), 300, 2000) * np.exp(-tt_(0.06) / 0.012), ld, 0.16 * g, pan_of(x))

# riser into the completed crest, then the main trailer hit
add(riser(2.2), TL.T_COMPLETE - 2.2, 0.28, 0.0)
trailer_hit(TL.T_COMPLETE, 1.0, notes=(26, 38, 45, 50, 53))

# camera pull-back + push-in + hero turns
add(whoosh(1.1, 900, 120, 0.25), 7.85, 0.30, 0.0)
add(whoosh(1.6, 120, 1500, 0.85), 11.0, 0.26, 0.0)
add(whoosh(0.9, 700, 140, 0.35), 12.6, 0.16, 0.45)
add(whoosh(0.9, 700, 140, 0.35), 13.6, 0.16, -0.45)
for a, b in TL.GLOSS:                     # soft air swells, no tones
    add(whoosh(b - a, 2500, 6000, 0.5, 9000), a, 0.03, 0.0)

# settle + the motto title hits (classic trailer title-card rhythm), final tail
add(riser(1.2, 110, 440), TL.T_SETTLE - 1.2, 0.20, 0.0)
add(sub_drop(44, 28, 2.0), TL.T_SETTLE, 0.6, 0.0, 'hit')
add(whoosh(0.8, 600, 120, 0.2), TL.T_SETTLE, 0.18, 0.0)
sizes = {'DUCATUS': 0.62, 'SCIENTIA': 0.72, 'MINISTERIUM': 1.0}
pans = {'DUCATUS': -0.35, 'SCIENTIA': 0.0, 'MINISTERIUM': 0.35}
for txt, t0 in TL.TAGLINE:
    if txt == '|':
        continue
    big = txt == 'MINISTERIUM'
    trailer_hit(t0 + 0.12, sizes[txt], notes=(26, 38, 45, 50, 54, 57) if big else (26, 38, 45), pan=pans[txt] * 0.4,
                with_braam=True)

# ================================================================ score
# low-string ostinato (D minor, 120 bpm 16ths) with taiko through the build
bt = 0.125
pat = [38, 38, 50, 38, 41, 38, 50, 45, 38, 38, 50, 38, 43, 41, 40, 38]
for i in range(int((TL.T_COMPLETE - 2.9) / bt)):
    st = 2.9 + i * bt
    u = (st - 2.9) / (TL.T_COMPLETE - 2.9)
    add(spiccato(pat[i % 16], cut=900 + 1800 * u), st, 0.05 + 0.10 * u, -0.25 if i % 2 else 0.25, 'mus')
    if i % 4 == 0 and st > 3.8:
        taiko(st, 0.5 + 0.7 * u, 0.0)
    if i % 8 == 6 and st > 5.5:
        taiko(st, 0.35 + 0.5 * u, 0.3)
for j, st in enumerate(np.arange(TL.T_COMPLETE - 0.75, TL.T_COMPLETE, 0.125)):   # drum fill into the hit
    taiko(st, 0.5 + 0.12 * j, (-1) ** j * 0.4)

# floating: sustained dark strings + heartbeat drum
add(strings([38, 45, 50, 53], 3.0, att=0.4, rel=0.9, cut=900), 7.9, 0.22, 0.0, 'mus')
add(strings([34, 41, 46, 50], 4.0, att=0.6, rel=0.7, cut=1000), 10.6, 0.26, 0.0, 'mus')
for st in (8.6, 9.6, 10.6):
    taiko(st, 0.45, 0.0)
# build #2 into the settle
for i in range(int((TL.T_SETTLE - 11.0) / bt)):
    st = 11.0 + i * bt
    u = (st - 11.0) / (TL.T_SETTLE - 11.0)
    add(spiccato(pat[i % 16] + (5 if i // 16 % 2 else 0), cut=1000 + 2200 * u), st, 0.10 + 0.13 * u, -0.25 if i % 2 else 0.25, 'mus')
    if i % 4 == 0:
        taiko(st, 0.7 + 0.8 * u, 0.0)
    if i % 8 == 6:
        taiko(st, 0.45 + 0.5 * u, 0.3)
for j, st in enumerate(np.arange(TL.T_SETTLE - 0.5, TL.T_SETTLE, 0.0625)):
    taiko(st, 0.35 + 0.08 * j, (-1) ** j * 0.35)
# final chord: low strings + brass on D (major colour for the resolve), rings out
add(strings([26, 38, 45, 50, 54, 57], TL.DUR - TL.T_SETTLE - 0.4, att=0.25, rel=1.0, cut=1300), TL.T_SETTLE, 0.30, 0.0, 'mus')


# ================================================================ mix + master
def reverb(x, seed, secs):
    r = np.random.default_rng(seed)
    n = int(secs * SR)
    ir = r.standard_normal(n) * np.exp(-np.arange(n) / SR * 6.9 / secs)
    ir = filt(ir, 120, 5000)
    ir /= np.sqrt((ir ** 2).sum())
    m = len(x) + n
    return np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(ir, m), m)[:len(x)]


mix = np.zeros((N, 2))
for name, (dry, wet, secs) in {'fx': (0.9, 0.22, 1.4), 'hit': (0.9, 0.38, 3.2), 'mus': (0.85, 0.40, 2.6)}.items():
    b = BUS[name]
    mix[:, 0] += b[:, 0] * dry + reverb(b[:, 0], 11 + len(name), secs) * wet
    mix[:, 1] += b[:, 1] * dry + reverb(b[:, 1], 23 + len(name), secs) * wet
mix[:, 0] = filt(mix[:, 0], 24, 16000)
mix[:, 1] = filt(mix[:, 1], 24, 16000)
fade = np.clip((TL.DUR - t_all) / 1.0, 0, 1) ** 1.2
mix *= fade[:, None]
mix = mix / np.abs(mix).max()
mix = np.tanh(mix * 1.6) / np.tanh(1.6)          # glue / soft limiting for a dense, punchy master
mix *= 0.95
pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
out = sys.argv[1] if len(sys.argv) > 1 else 'psa_score.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, pcm.shape)
