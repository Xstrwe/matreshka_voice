/* matreshka_voice: log-mel front end. Matches train/features.py.
 * Copyright (c) 2026 Xstrwe. MIT License. */
#include "matreshka_voice.h"
#include "mv_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N_BINS (MV_N_FFT / 2 + 1)

struct mv_features {
    float window[MV_FRAME_LEN];
    float cos_t[MV_N_FFT / 2], sin_t[MV_N_FFT / 2];
    int   bitrev[MV_N_FFT];
    /* sparse mel filters: for each mel, first bin and weights */
    int   mel_lo[MV_N_MELS], mel_n[MV_N_MELS];
    float mel_w[MV_N_MELS][N_BINS];
    float re[MV_N_FFT], im[MV_N_FFT];
};

static double hz_to_mel(double f) { return 2595.0 * log10(1.0 + f / 700.0); }
static double mel_to_hz(double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); }

mv_features *mv_features_create(void)
{
    mv_features *f = (mv_features *)calloc(1, sizeof(*f));
    int i, m, k, bits = 0;
    double hz[MV_N_MELS + 2], lo_m, hi_m;
    if (!f) return NULL;
    for (i = 0; i < MV_FRAME_LEN; i++)
        f->window[i] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * i / MV_FRAME_LEN));
    for (i = 0; i < MV_N_FFT / 2; i++) {
        f->cos_t[i] = (float)cos(2.0 * M_PI * i / MV_N_FFT);
        f->sin_t[i] = (float)-sin(2.0 * M_PI * i / MV_N_FFT);
    }
    while ((1 << bits) < MV_N_FFT) bits++;
    for (i = 0; i < MV_N_FFT; i++) {
        int r = 0, b;
        for (b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
        f->bitrev[i] = r;
    }
    lo_m = hz_to_mel(20.0); hi_m = hz_to_mel(7600.0);
    for (i = 0; i < MV_N_MELS + 2; i++)
        hz[i] = mel_to_hz(lo_m + (hi_m - lo_m) * i / (MV_N_MELS + 1));
    for (m = 0; m < MV_N_MELS; m++) {
        int first = -1, last = -1;
        for (k = 0; k < N_BINS; k++) {
            double fr = (double)k * MV_SAMPLE_RATE / MV_N_FFT, w = 0.0;
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

/* In-place iterative radix-2 complex FFT of size MV_N_FFT. */
static void fft(mv_features *f)
{
    float *re = f->re, *im = f->im;
    int i, len;
    for (i = 0; i < MV_N_FFT; i++) {
        int j = f->bitrev[i];
        if (j > i) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (len = 2; len <= MV_N_FFT; len <<= 1) {
        int half = len >> 1, step = MV_N_FFT / len, s, k;
        for (s = 0; s < MV_N_FFT; s += len) {
            for (k = 0; k < half; k++) {
                float wr = f->cos_t[k * step], wi = f->sin_t[k * step];
                int a = s + k, b = a + half;
                float tr = re[b] * wr - im[b] * wi;
                float ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr;        im[a] += ti;
            }
        }
    }
}

void mv_features_frame(mv_features *f, const float *frame, float *out)
{
    float power[N_BINS];
    int i, m;
    for (i = 0; i < MV_FRAME_LEN; i++) { f->re[i] = frame[i] * f->window[i]; f->im[i] = 0.0f; }
    for (; i < MV_N_FFT; i++) { f->re[i] = 0.0f; f->im[i] = 0.0f; }
    fft(f);
    for (i = 0; i < N_BINS; i++) power[i] = f->re[i] * f->re[i] + f->im[i] * f->im[i];
    for (m = 0; m < MV_N_MELS; m++) {
        const float *w = f->mel_w[m] + f->mel_lo[m];
        const float *p = power + f->mel_lo[m];
        float acc = 0.0f;
        int n = f->mel_n[m];
        for (i = 0; i < n; i++) acc += w[i] * p[i];
        out[m] = logf(acc + 1e-6f);
    }
}
