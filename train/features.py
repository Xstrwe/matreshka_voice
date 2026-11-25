"""Reference log-mel front end. engine/src/mv_features.c implements the same math."""
import numpy as np

SAMPLE_RATE = 16000
FRAME_LEN = 400      # 25 ms
HOP = 160            # 10 ms
N_FFT = 512
N_MELS = 40
F_MIN = 20.0
F_MAX = 7600.0
LOG_EPS = 1e-6


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank():
    """Triangular HTK-style mel filters, shape (N_MELS, N_FFT // 2 + 1)."""
    n_bins = N_FFT // 2 + 1
    mels = np.linspace(_hz_to_mel(F_MIN), _hz_to_mel(F_MAX), N_MELS + 2)
    hz = _mel_to_hz(mels)
    fb = np.zeros((N_MELS, n_bins), dtype=np.float64)
    for m in range(N_MELS):
        lo, ce, hi = hz[m], hz[m + 1], hz[m + 2]
        for k in range(n_bins):
            f = k * SAMPLE_RATE / N_FFT
            if lo < f < hi:
                fb[m, k] = (f - lo) / (ce - lo) if f <= ce else (hi - f) / (hi - ce)
    return fb.astype(np.float32)


_WINDOW = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(FRAME_LEN) / FRAME_LEN)).astype(np.float32)
_FB = mel_filterbank()


def logmel(audio):
    """audio: float32 in [-1, 1] at 16 kHz -> (n_frames, N_MELS) float32."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < FRAME_LEN:
        return np.zeros((0, N_MELS), dtype=np.float32)
    n = 1 + (len(audio) - FRAME_LEN) // HOP
    idx = np.arange(FRAME_LEN)[None, :] + HOP * np.arange(n)[:, None]
    frames = audio[idx] * _WINDOW
    spec = np.fft.rfft(frames, n=N_FFT, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    return np.log(power @ _FB.T + LOG_EPS).astype(np.float32)


def int16_to_float(x):
    return np.asarray(x, dtype=np.float32) / 32768.0
