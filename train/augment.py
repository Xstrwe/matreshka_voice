"""Augmentation and window building (numpy only)."""
import numpy as np

SR = 16000
WIN_SAMPLES = 400 + 99 * 160     # exactly 100 feature frames


def speed(x, rnd, lo=0.85, hi=1.2):
    """Resample-based speed perturbation: shifts tempo, pitch and formants together."""
    from fractions import Fraction
    from scipy.signal import resample_poly
    f = Fraction(rnd.uniform(lo, hi)).limit_denominator(40)
    if f.numerator == f.denominator:
        return x
    return resample_poly(x, f.denominator, f.numerator).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def colored_noise(n, rnd):
    kind = rnd.integers(0, 4)
    w = rnd.standard_normal(n).astype(np.float32)
    if kind == 0:
        return w
    spec = np.fft.rfft(w)
    f = np.arange(len(spec)) + 1.0
    spec = spec / (f ** (0.5 if kind == 1 else 1.0)) if kind < 3 else spec
    x = np.fft.irfft(spec, n).astype(np.float32)
    if kind == 3:  # mains hum + harmonics, motor-like whine
        t = np.arange(n) / SR
        base = rnd.choice([50.0, 60.0, rnd.uniform(80, 400)])
        x = sum(np.sin(2 * np.pi * base * h * t + rnd.uniform(0, 6)) / h for h in range(1, 6)).astype(np.float32)
        x += 0.1 * w
    return x / (rms(x) + 1e-9)


def reverb(x, rnd):
    t60 = rnd.uniform(0.15, 0.7)
    n = int(SR * t60)
    t = np.arange(n) / SR
    ir = rnd.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / t60).astype(np.float32)
    ir[0] = 1.0 / rnd.uniform(0.3, 1.0)
    y = np.convolve(x, ir)[: len(x)]
    return y * (rms(x) / (rms(y) + 1e-9))


def place(clip, total, end_min, end_max, rnd):
    """Place clip inside a zero buffer so its end lies in [end_min, end_max] * total."""
    out = np.zeros(total, dtype=np.float32)
    clip = clip[: total]
    end = int(rnd.uniform(end_min, end_max) * total)
    end = max(len(clip), min(total, end))
    out[end - len(clip): end] = clip
    return out


def trim(x, thr=0.01):
    idx = np.where(np.abs(x) > thr)[0]
    return x[idx[0]: idx[-1] + 1] if len(idx) else x


def random_eq(x, rnd):
    """Random 2nd-order peaking/shelving-like coloration to mimic different mics and rooms."""
    from scipy.signal import lfilter
    f0 = rnd.uniform(150, 5000)
    gain_db = rnd.uniform(-12, 12)
    q = rnd.uniform(0.4, 2.0)
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a]
    den = [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a]
    y = lfilter(b, den, x).astype(np.float32)
    if rnd.random() < 0.3:   # band-limit like a cheap mic
        from scipy.signal import butter
        bb, aa = butter(2, [rnd.uniform(80, 300) / (SR / 2), rnd.uniform(3500, 7500) / (SR / 2)], btype="band")
        y = lfilter(bb, aa, y).astype(np.float32)
    return y


def farfield(x, rnd):
    """Rough robot-microphone simulation: distance (more reverb, less direct sound), a small
    band-limited mic capsule, low-level motor/fan noise and a soft limiter."""
    from scipy.signal import butter, lfilter
    t60 = rnd.uniform(0.3, 0.9)
    n = int(SR * t60)
    t = np.arange(n) / SR
    ir = rnd.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / t60).astype(np.float32)
    ir[0] = rnd.uniform(0.5, 3.0)                       # weaker direct path = farther away
    y = np.convolve(x, ir)[: len(x)]
    b, a = butter(2, [rnd.uniform(150, 400) / (SR / 2), rnd.uniform(4000, 7000) / (SR / 2)], btype="band")
    y = lfilter(b, a, y).astype(np.float32)
    y = y * (rms(x) / (rms(y) + 1e-9))
    tt = np.arange(len(y)) / SR
    f = rnd.uniform(90, 300)
    motor = sum(np.sin(2 * np.pi * f * h * tt + rnd.uniform(0, 6)) / h for h in range(1, 4)).astype(np.float32)
    motor += 0.5 * colored_noise(len(y), rnd)
    y = y + motor / (rms(motor) + 1e-9) * rms(y) * 10 ** (-rnd.uniform(10, 30) / 20)
    return np.tanh(y * 2.0) / 2.0 if rnd.random() < 0.3 else y


def augment(x, rnd, noise_bank=None, p_noise=0.85, p_reverb=0.35, p_eq=0.5, p_far=0.0):
    x = x.astype(np.float32)
    if p_far and rnd.random() < p_far:
        x = farfield(x, rnd)
        p_reverb = 0.0
    if rnd.random() < p_eq:
        x = random_eq(x, rnd)
    if rnd.random() < p_reverb:
        x = reverb(x, rnd)
    x = x / (rms(x) + 1e-9) * 10 ** (rnd.uniform(-32, -12) / 20)   # speech level
    if rnd.random() < p_noise:
        if noise_bank is not None and len(noise_bank) and rnd.random() < 0.4:
            nb = noise_bank[rnd.integers(0, len(noise_bank))]
            s = rnd.integers(0, max(1, len(nb) - len(x)))
            nz = np.resize(nb[s: s + len(x)], len(x))
            nz = nz / (rms(nz) + 1e-9)
        else:
            nz = colored_noise(len(x), rnd)
        snr = rnd.uniform(0, 30)
        x = x + nz * (rms(x) * 10 ** (-snr / 20))
    return np.clip(x, -1, 1)
