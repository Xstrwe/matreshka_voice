"""Эталонный расчёт признаков log-mel на numpy.

engine/src/mv_features.c делает ту же математику на C, и tests/parity.py проверяет,
что результаты совпадают. Любое изменение здесь надо повторить в C (и наоборот).

Схема: звук режется на кадры по 25 мс с шагом 10 мс, каждый кадр умножается на окно
Ханна, по нему считается спектр мощности (БПФ на 512 точек), затем энергия
собирается в 40 треугольных мел-полос и берётся натуральный логарифм.
"""
import numpy as np

SAMPLE_RATE = 16000
FRAME_LEN = 400      # 25 мс — длина кадра в отсчётах
HOP = 160            # 10 мс — шаг между кадрами
N_FFT = 512          # кадр дополняется нулями до 512 точек (степень двойки для БПФ)
N_MELS = 40          # число мел-полос, т. е. признаков на кадр
F_MIN = 20.0
F_MAX = 7600.0       # полосы покрывают 20–7600 Гц (почти до 8 кГц, предела для 16 кГц)
LOG_EPS = 1e-6       # добавка перед логарифмом, чтобы не было log(0) на тишине


# Шкала мел (вариант HTK) ближе к слуху: низкие частоты различаются тоньше высоких.
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank():
    """Треугольные мел-фильтры в стиле HTK, форма (N_MELS, N_FFT // 2 + 1).

Центры полос равномерно расставлены по шкале мел. Каждый фильтр растёт от 0 до 1
от левого соседа к своему центру и падает до 0 к правому соседу.
"""
    n_bins = N_FFT // 2 + 1
    mels = np.linspace(_hz_to_mel(F_MIN), _hz_to_mel(F_MAX), N_MELS + 2)
    hz = _mel_to_hz(mels)
    fb = np.zeros((N_MELS, n_bins), dtype=np.float64)
    for m in range(N_MELS):
        lo, ce, hi = hz[m], hz[m + 1], hz[m + 2]
        for k in range(n_bins):
            f = k * SAMPLE_RATE / N_FFT      # частота k-го бина БПФ в герцах
            if lo < f < hi:
                fb[m, k] = (f - lo) / (ce - lo) if f <= ce else (hi - f) / (hi - ce)
    return fb.astype(np.float32)


# Окно Ханна (периодический вариант) смягчает края кадра и уменьшает «утечку» спектра.
# Окно и фильтры считаются один раз при импорте модуля.
_WINDOW = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(FRAME_LEN) / FRAME_LEN)).astype(np.float32)
_FB = mel_filterbank()


def logmel(audio):
    """Звук float32 в диапазоне [-1, 1], 16 кГц -> матрица признаков (n_кадров, N_MELS) float32.

Кадров столько, сколько целиком помещается в звук; хвост короче кадра отбрасывается.
"""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < FRAME_LEN:
        return np.zeros((0, N_MELS), dtype=np.float32)
    n = 1 + (len(audio) - FRAME_LEN) // HOP
    # Матрица индексов (n, FRAME_LEN): строка i — отсчёты i-го кадра. Так все кадры
    # нарезаются одной операцией, без цикла.
    idx = np.arange(FRAME_LEN)[None, :] + HOP * np.arange(n)[:, None]
    frames = audio[idx] * _WINDOW
    spec = np.fft.rfft(frames, n=N_FFT, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    # Умножение на матрицу фильтров суммирует мощность внутри каждой мел-полосы.
    return np.log(power @ _FB.T + LOG_EPS).astype(np.float32)


# 16-битные отсчёты (-32768..32767) -> float в [-1, 1).
def int16_to_float(x):
    return np.asarray(x, dtype=np.float32) / 32768.0
