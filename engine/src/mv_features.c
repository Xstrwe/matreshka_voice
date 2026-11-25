/* matreshka_voice: вычисление признаков (log-mel спектр).
 * Математика точно та же, что в train/features.py: модель учится на признаках
 * из Python, а на устройстве получает их отсюда, поэтому они обязаны совпадать.
 * Copyright (c) 2026 Xstrwe. Лицензия MIT.
 *
 * Путь одного кадра звука (25 мс):
 *   окно Ханна -> FFT -> мощность спектра -> 40 мел-фильтров -> логарифм.
 * Мел-шкала примерно повторяет то, как ухо различает высоту: внизу полосы узкие,
 * вверху широкие. Логарифм сжимает громкость, как это делает слух. */
#include "matreshka_voice.h"
#include "mv_internal.h"

#include <math.h>     /* cos, sin, log10, pow, logf */
#include <stdlib.h>   /* calloc, free */
#include <string.h>

/* M_PI не входит в строгий стандарт C99, поэтому задаём его сами, если нет. */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* У вещественного сигнала длины N спектр симметричен, полезны только
 * первые N/2 + 1 частотных «бинов». */
#define N_BINS (MV_N_FFT / 2 + 1)

/* Всё, что можно посчитать заранее, считаем один раз при создании:
 * окно, таблицы синусов/косинусов для FFT, перестановку битов и мел-фильтры.
 * Тогда на каждый кадр остаётся только арифметика, без cos/sin. */
struct mv_features {
    float window[MV_FRAME_LEN];                    /* окно Ханна */
    float cos_t[MV_N_FFT / 2], sin_t[MV_N_FFT / 2]; /* «поворотные множители» FFT */
    int   bitrev[MV_N_FFT];                        /* номер с битами в обратном порядке */
    /* Мел-фильтры почти везде равны нулю, поэтому для каждого храним
     * первый ненулевой бин (mel_lo) и сколько их подряд (mel_n). */
    int   mel_lo[MV_N_MELS], mel_n[MV_N_MELS];
    float mel_w[MV_N_MELS][N_BINS];
    float re[MV_N_FFT], im[MV_N_FFT];              /* рабочие массивы FFT */
};

/* Перевод герц в мелы и обратно (формула HTK). static — функция видна
 * только внутри этого файла, наружу не экспортируется. */
static double hz_to_mel(double f) { return 2595.0 * log10(1.0 + f / 700.0); }
static double mel_to_hz(double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); }

mv_features *mv_features_create(void)
{
    /* calloc выделяет память и сразу заполняет её нулями. */
    mv_features *f = (mv_features *)calloc(1, sizeof(*f));
    int i, m, k, bits = 0;
    double hz[MV_N_MELS + 2], lo_m, hi_m;
    if (!f) return NULL;
    /* Окно Ханна плавно гасит края кадра до нуля, иначе резкие края
     * дают в спектре ложные высокие частоты («растекание»). */
    for (i = 0; i < MV_FRAME_LEN; i++)
        f->window[i] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * i / MV_FRAME_LEN));
    /* Таблица e^{-2πik/N}: косинус и синус со знаком минус. */
    for (i = 0; i < MV_N_FFT / 2; i++) {
        f->cos_t[i] = (float)cos(2.0 * M_PI * i / MV_N_FFT);
        f->sin_t[i] = (float)-sin(2.0 * M_PI * i / MV_N_FFT);
    }
    /* bits = log2(N). Затем для каждого индекса переворачиваем порядок
     * его битов: 001 -> 100 и т.д. Так FFT «по Кули — Тьюки» раскладывает
     * входные данные перед вычислением. */
    while ((1 << bits) < MV_N_FFT) bits++;
    for (i = 0; i < MV_N_FFT; i++) {
        int r = 0, b;
        for (b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
        f->bitrev[i] = r;
    }
    /* Мел-фильтры: n_mels+2 точки равномерно по мел-шкале от 20 до 7600 Гц.
     * Фильтр m — треугольник: растёт от точки m к m+1 и спадает к m+2. */
    lo_m = hz_to_mel(20.0); hi_m = hz_to_mel(7600.0);
    for (i = 0; i < MV_N_MELS + 2; i++)
        hz[i] = mel_to_hz(lo_m + (hi_m - lo_m) * i / (MV_N_MELS + 1));
    for (m = 0; m < MV_N_MELS; m++) {
        int first = -1, last = -1;
        for (k = 0; k < N_BINS; k++) {
            double fr = (double)k * MV_SAMPLE_RATE / MV_N_FFT, w = 0.0;  /* частота бина k */
            if (fr > hz[m] && fr < hz[m + 2])
                w = fr <= hz[m + 1] ? (fr - hz[m]) / (hz[m + 1] - hz[m])
                                    : (hz[m + 2] - fr) / (hz[m + 2] - hz[m + 1]);
            if (w > 0.0) { if (first < 0) first = k; last = k; }
            f->mel_w[m][k] = (float)w;
        }
        f->mel_lo[m] = first < 0 ? 0 : first;
        f->mel_n[m] = first < 0 ? 0 : last - first + 1;
    }
    return f;
}

void mv_features_destroy(mv_features *f) { free(f); }

/* Быстрое преобразование Фурье (FFT) по основанию 2, «на месте» (результат
 * пишется в те же массивы re/im). Вместо N² операций делает N·log2(N):
 * для N = 512 это ~4600 «бабочек» вместо ~260 тысяч умножений. */
static void fft(mv_features *f)
{
    float *re = f->re, *im = f->im;
    int i, len;
    /* Шаг 1: переставить элементы в порядке перевёрнутых битов.
     * Условие j > i, чтобы каждую пару поменять ровно один раз. */
    for (i = 0; i < MV_N_FFT; i++) {
        int j = f->bitrev[i];
        if (j > i) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    /* Шаг 2: собирать преобразования длины 2, 4, 8, … N из вдвое коротких.
     * len <<= 1 — сдвиг влево на 1 бит, то есть умножение на 2. */
    for (len = 2; len <= MV_N_FFT; len <<= 1) {
        int half = len >> 1, step = MV_N_FFT / len, s, k;
        for (s = 0; s < MV_N_FFT; s += len) {
            for (k = 0; k < half; k++) {
                /* «Бабочка»: берём пару (a, b), поворачиваем b на угол
                 * w = e^{-2πik/len} и получаем a+wb и a−wb. */
                float wr = f->cos_t[k * step], wi = f->sin_t[k * step];
                int a = s + k, b = a + half;
                float tr = re[b] * wr - im[b] * wi;   /* комплексное умножение b·w */
                float ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr;        im[a] += ti;
            }
        }
    }
}

void mv_features_frame(mv_features *f, const float *frame, float *out)
{
    float power[N_BINS];   /* массив на стеке: живёт только во время вызова */
    int i, m;
    /* Кадр умножаем на окно, мнимую часть обнуляем, хвост до 512 — нули. */
    for (i = 0; i < MV_FRAME_LEN; i++) { f->re[i] = frame[i] * f->window[i]; f->im[i] = 0.0f; }
    for (; i < MV_N_FFT; i++) { f->re[i] = 0.0f; f->im[i] = 0.0f; }
    fft(f);
    /* Мощность = квадрат модуля комплексного числа. */
    for (i = 0; i < N_BINS; i++) power[i] = f->re[i] * f->re[i] + f->im[i] * f->im[i];
    /* Каждая мел-полоса — взвешенная сумма мощности по своим бинам.
     * Указатели w и p сразу сдвинуты на первый ненулевой бин. */
    for (m = 0; m < MV_N_MELS; m++) {
        const float *w = f->mel_w[m] + f->mel_lo[m];
        const float *p = power + f->mel_lo[m];
        float acc = 0.0f;
        int n = f->mel_n[m];
        for (i = 0; i < n; i++) acc += w[i] * p[i];
        /* +1e-6, чтобы не брать логарифм от нуля в полной тишине. */
        out[m] = logf(acc + 1e-6f);
    }
}
