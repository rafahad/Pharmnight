"""Procedural sound design + score for the 18 s Cinema Soirée reveal (48 kHz stereo WAV)."""
import sys
import wave
import numpy as np

SR = 48000
DUR = 18.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(7)
L = np.zeros(N)
R = np.zeros(N)


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def env_adsr(n, a, r, sus=1.0):
    e = np.ones(n) * sus
    na, nr = int(a * SR), int(r * SR)
    e[:na] = np.linspace(0, 1, na) ** 2 * sus if na else e[:na]
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr) ** 1.5
    return e


def add(sig, start, gain=1.0, pan=0.0):
    i0 = int(start * SR)
    i1 = min(i0 + len(sig), N)
    if i1 <= i0:
        return
    s = sig[:i1 - i0] * gain
    L[i0:i1] += s * np.sqrt(0.5 * (1 - pan))
    R[i0:i1] += s * np.sqrt(0.5 * (1 + pan))


def lowpass_fft(x, fc, order=2):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X /= np.sqrt(1 + (f / fc) ** (2 * order))
    return np.fft.irfft(X, len(x))


def bandpass_fft(x, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / np.sqrt(1 + (lo / np.maximum(f, 1)) ** 4) / np.sqrt(1 + (f / hi) ** 4)
    return np.fft.irfft(X, len(x))


# ---------------------------------------------------------------- pad (strings)
def pad(notes, start, dur, gain, attack=1.2, release=1.5, bright=8):
    n = int((dur + release) * SR)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    for m in notes:
        f0 = midi(m)
        for det in (-0.09, 0.0, 0.08):
            f = f0 * 2 ** (det / 12)
            ph = rng.uniform(0, 2 * np.pi)
            vib = 0.0025 * np.sin(2 * np.pi * 5.2 * tt + ph)
            for h in range(1, bright + 1):
                out += np.sin(2 * np.pi * f * h * tt * (1 + vib) + ph * h) / h ** 1.45
    e = np.ones(n)
    na = int(attack * SR)
    e[:na] = (np.linspace(0, 1, na)) ** 2
    nr = int(release * SR)
    e[-nr:] *= np.linspace(1, 0, nr) ** 1.8
    out *= e
    out = lowpass_fft(out, 2600)
    add(out / (len(notes) * 3), start, gain, -0.15)
    add(out / (len(notes) * 3), start + 0.012, gain, 0.15)


# ---------------------------------------------------------------- harp (Karplus–Strong)
def pluck(m, dur=2.5, bright=0.5):
    f = midi(m)
    p = int(SR / f)
    n = int(dur * SR)
    buf = rng.uniform(-1, 1, p)
    buf = lowpass_fft(buf, 2000 + 6000 * bright, 1) if p > 8 else buf
    out = np.zeros(n)
    idx = 0
    b = buf.copy()
    decay = 0.996
    for i in range(n):
        out[i] = b[idx]
        nxt = (idx + 1) % p
        b[idx] = decay * 0.5 * (b[idx] + b[nxt])
        idx = nxt
    return out * np.exp(-np.arange(n) / SR * 1.2)


def bell(m, dur=3.5):
    f = midi(m)
    n = int(dur * SR)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    for ratio, amp, dec in ((1, 1.0, 1.4), (2.0, 0.45, 2.0), (2.76, 0.35, 2.8), (5.4, 0.18, 4.5), (8.93, 0.08, 6.0)):
        out += amp * np.sin(2 * np.pi * f * ratio * tt) * np.exp(-tt * dec)
    out *= np.minimum(tt / 0.003, 1)
    return out


def noise_burst(dur, lo, hi, a=0.002, d=0.08):
    n = int(dur * SR)
    x = bandpass_fft(rng.standard_normal(n), lo, hi)
    tt = np.arange(n) / SR
    e = np.minimum(tt / a, 1) * np.exp(-tt / d)
    return x * e / (np.abs(x).max() + 1e-9)


def thump(f0=55, dur=1.2, drop=0.5):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    f = f0 * (1 + drop * np.exp(-tt * 18))
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-tt * 4.0) * np.minimum(tt / 0.004, 1)


# ================================================================ timeline
# 0.45 s  projector switch
add(noise_burst(0.12, 800, 6000, d=0.02), 0.45, 0.35, -0.4)
add(thump(90, 0.3, 0.3), 0.45, 0.25, -0.4)

# projector motor whir + 24 fps sprocket clatter, fading as the film "becomes" the stage
seg = t >= 0.55
tt = t[seg] - 0.55
spin = np.minimum(tt / 0.8, 1) ** 0.7
fade = np.clip(1 - (t[seg] - 6.0) / 3.0, 0, 1) ** 1.5
motor = (np.sin(2 * np.pi * 50 * tt) * 0.4 + np.sin(2 * np.pi * 100 * tt) * 0.25 +
         np.sin(2 * np.pi * np.cumsum(180 * spin + 20) / SR) * 0.15) * spin
clk = np.zeros(seg.sum())
rate = 24 * spin
phase = np.cumsum(rate) / SR
ticks = np.nonzero(np.diff(np.floor(phase)) > 0)[0]
click = noise_burst(0.03, 1500, 7000, a=0.0005, d=0.006)
for k in ticks:
    e = min(len(click), len(clk) - k)
    clk[k:k + e] += click[:e] * (0.8 + 0.4 * rng.random())
proj = (motor * 0.10 + clk * 0.16) * fade
add(lowpass_fft(proj, 7000), 0.55, 1.0, -0.35)

# 3.00 / 3.28 s  spotlights strike
for st, pan in ((3.0, -0.6), (3.28, 0.6)):
    add(thump(48, 1.6, 0.8), st, 0.32, pan)
    add(noise_burst(0.25, 300, 3000, d=0.05), st, 0.25, pan)

# 3.45 – 6.4 s  velvet curtain swoosh (filtered noise, opening outwards)
n = int(3.4 * SR)
sw = bandpass_fft(rng.standard_normal(n), 150, 1800)
tt = np.arange(n) / SR
e = np.sin(np.pi * np.clip(tt / 3.2, 0, 1)) ** 1.5
add(sw * e * 0.05, 3.45, 1.0, -0.5)
add(np.roll(sw, 4000) * e * 0.05, 3.45, 1.0, 0.5)

# score — D major
pad([47, 54, 59, 62, 66], 1.4, 2.8, 0.45, attack=2.0, release=1.2)          # Bm add9
pad([43, 50, 59, 62, 66], 4.0, 2.1, 0.55, attack=1.0, release=0.8)          # Gmaj7
pad([38, 50, 57, 61, 64, 66, 69], 5.95, 4.2, 0.85, attack=0.25, release=1.0)  # Dmaj9 — reveal
pad([47, 55, 59, 62, 66, 71], 10.0, 2.1, 0.75, attack=0.6, release=0.6)     # G/B
pad([45, 52, 57, 62, 64, 69], 12.0, 2.1, 0.75, attack=0.5, release=0.5)     # Asus4 -> A
pad([45, 52, 57, 61, 64, 69], 13.1, 1.0, 0.6, attack=0.3, release=0.4)
pad([38, 45, 50, 57, 61, 64, 66, 69, 74], 14.0, 2.6, 0.95, attack=0.35, release=1.4)  # final Dmaj9

# harp glissando into the reveal
gl = [62, 64, 66, 69, 71, 74, 76, 78, 81, 83, 86]
for i, m in enumerate(gl):
    add(pluck(m, 2.2), 5.45 + i * 0.05, 0.22, -0.6 + 1.2 * i / len(gl))
# reveal accent: low boom + bells
add(thump(41, 2.5, 0.25), 5.98, 0.7, 0.0)
add(bell(86, 4.0), 6.02, 0.10, 0.25)
add(bell(81, 4.0), 6.05, 0.08, -0.25)
# ring trace shimmer (6.6 – 9.6): soft rising chime arpeggio
for i, m in enumerate([74, 78, 81, 83, 86, 90, 93]):
    add(bell(m, 2.0), 6.7 + i * 0.42, 0.035, np.cos(i * 0.9) * 0.7)
# light sweep (10.6 – 12.6): airy whoosh
n = int(2.4 * SR)
wh = bandpass_fft(rng.standard_normal(n), 2500, 9000)
tt = np.arange(n) / SR
add(wh * np.sin(np.pi * tt / 2.4) ** 2 * 0.02, 10.5, 1.0, -0.5)
add(np.roll(wh, 9000) * np.sin(np.pi * tt / 2.4) ** 2 * 0.02, 10.6, 1.0, 0.5)
# final sparkles
for st, m, pan in ((14.05, 93, 0.4), (14.25, 90, -0.5), (14.45, 98, 0.1), (14.6, 86, -0.3),
                   (14.8, 95, 0.6), (14.95, 91, 0.3), (15.1, 98, -0.6), (15.25, 93, -0.1)):
    add(bell(m, 2.0), st + 0.12, 0.04, pan)
add(thump(36, 3.0, 0.15), 14.0, 0.45, 0.0)

# ---------------------------------------------------------------- reverb + master
def reverb(x, seed, secs=2.6):
    r = np.random.default_rng(seed)
    n = int(secs * SR)
    ir = r.standard_normal(n) * np.exp(-np.arange(n) / SR * 3.0 / secs * 2.3)
    ir = lowpass_fft(ir, 5000)
    ir /= np.sqrt((ir ** 2).sum())
    m = len(x) + n
    y = np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(ir, m), m)[:len(x)]
    return y


wetL, wetR = reverb(L, 1), reverb(R, 2)
outL = L * 0.8 + wetL * 0.45
outR = R * 0.8 + wetR * 0.45
mix = np.stack([outL, outR], 1)
# end: gentle fade over the last 1.6 s, silence before the first sound
fade = np.clip((DUR - t) / 1.6, 0, 1) ** 1.2
mix *= fade[:, None]
peak = np.abs(mix).max()
mix = mix / peak * 0.89                      # ~ -1 dBFS peak
mix = np.tanh(mix * 1.1) / np.tanh(1.1)       # soft limiter
pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
with wave.open(sys.argv[1] if len(sys.argv) > 1 else 'score.wav', 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote audio', pcm.shape)
