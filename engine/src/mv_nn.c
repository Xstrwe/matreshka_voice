/* matreshka_voice: model loading and network inference. Matches train/model.py.
 * Copyright (c) 2026 Xstrwe. MIT License. */
#include "mv_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd_u32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static float rd_f32(const unsigned char *p)
{
    uint32_t u = rd_u32(p);
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

int mv_model_load(mv_model *m, const void *blob, size_t len)
{
    const unsigned char *p = (const unsigned char *)blob;
    size_t need, i, nf;
    const float *q;
    int b, cin, cout;
    memset(m, 0, sizeof(*m));
    if (len < MV_HEADER_BYTES || memcmp(p, MV_MAGIC, 4) != 0) return -1;
    if (rd_u32(p + 4) != 1) return -2;
    m->n_mels = (int)rd_u32(p + 8);
    m->window = (int)rd_u32(p + 12);
    m->k = (int)rd_u32(p + 16);
    for (b = 0; b < 4; b++) m->c[b] = (int)rd_u32(p + 20 + 4 * b);
    for (b = 0; b < 3; b++) m->stride[b] = (int)rd_u32(p + 36 + 4 * b);
    m->default_threshold = rd_f32(p + 48);
    if (m->n_mels <= 0 || m->n_mels > 256 || m->k <= 0 || m->k > 32 || m->window < m->k ||
        m->window > 1000)
        return -3;
    for (b = 0; b < 4; b++) if (m->c[b] <= 0 || m->c[b] > 1024) return -3;
    for (b = 0; b < 3; b++) if (m->stride[b] <= 0 || m->stride[b] > 8) return -3;
    need = 2 * (size_t)m->n_mels + (size_t)m->c[0] * m->n_mels * m->k + m->c[0];
    for (b = 0; b < 3; b++) {
        cin = m->c[b]; cout = m->c[b + 1];
        need += (size_t)cin * m->k + cin + (size_t)cout * cin + cout;
    }
    need += 2 * (size_t)m->c[3] + 2;
    if (len != MV_HEADER_BYTES + need * 4) return -4;
    m->data = (float *)malloc(need * sizeof(float));
    if (!m->data) return -5;
    for (i = 0; i < need; i++) m->data[i] = rd_f32(p + MV_HEADER_BYTES + 4 * i);
    q = m->data;
    m->mean = q;    q += m->n_mels;
    m->inv_std = q; q += m->n_mels;
    m->w0 = q;      q += (size_t)m->c[0] * m->n_mels * m->k;
    m->b0 = q;      q += m->c[0];
    for (b = 0; b < 3; b++) {
        cin = m->c[b]; cout = m->c[b + 1];
        m->dw[b] = q; q += (size_t)cin * m->k;
        m->db[b] = q; q += cin;
        m->pw[b] = q; q += (size_t)cout * cin;
        m->pb[b] = q; q += cout;
    }
    m->fcw = q; q += 2 * (size_t)m->c[3];
    m->fcb = q; q += 2;
    nf = (size_t)(q - m->data);
    return nf == need ? 0 : -6;
}

void mv_model_free(mv_model *m)
{
    free(m->data);
    memset(m, 0, sizeof(*m));
}

size_t mv_model_scratch_floats(const mv_model *m)
{
    int cmax = m->c[0], b;
    for (b = 1; b < 4; b++) if (m->c[b] > cmax) cmax = m->c[b];
    /* dw output + pw output, each at most cmax x window */
    return 2 * (size_t)cmax * m->window + cmax;
}

void mv_conv0_column(const mv_model *m, const float *const *frames, float *out)
{
    int o, j, f, k = m->k, nm = m->n_mels;
    for (o = 0; o < m->c[0]; o++) {
        const float *w = m->w0 + (size_t)o * nm * k;   /* [n_mels][k] */
        float acc = m->b0[o];
        for (j = 0; j < k; j++) {
            const float *x = frames[j];
            for (f = 0; f < nm; f++) acc += w[f * k + j] * x[f];
        }
        out[o] = acc > 0.0f ? acc : 0.0f;
    }
}

float mv_net_tail(const mv_model *m, const float *x0, int t, float *scratch)
{
    /* dw output always goes to A, pw output always to B; blocks read their input before
     * overwriting B, because pw only reads A. */
    size_t half = mv_model_scratch_floats(m) / 2;
    float *A = scratch, *B = scratch + half;
    const float *x = x0;
    int b, c, o, i, j, k = m->k;
    float l0, l1;
    for (b = 0; b < 3; b++) {
        int cin = m->c[b], cout = m->c[b + 1], s = m->stride[b];
        int tout = (t - k) / s + 1;
        for (c = 0; c < cin; c++) {
            const float *w = m->dw[b] + (size_t)c * k;
            const float *xc = x + (size_t)c * t;
            float *yc = A + (size_t)c * tout;
            for (i = 0; i < tout; i++) {
                const float *xs = xc + i * s;
                float acc = m->db[b][c];
                for (j = 0; j < k; j++) acc += w[j] * xs[j];
                yc[i] = acc;
            }
        }
        for (o = 0; o < cout; o++) {
            const float *w = m->pw[b] + (size_t)o * cin;
            float *yo = B + (size_t)o * tout;
            for (i = 0; i < tout; i++) yo[i] = m->pb[b][o];
            for (c = 0; c < cin; c++) {
                const float wc = w[c];
                const float *dc = A + (size_t)c * tout;
                for (i = 0; i < tout; i++) yo[i] += wc * dc[i];
            }
            for (i = 0; i < tout; i++) if (yo[i] < 0.0f) yo[i] = 0.0f;
        }
        x = B;
        t = tout;
    }
    for (c = 0; c < m->c[3]; c++) {
        float acc = 0.0f;
        for (i = 0; i < t; i++) acc += x[(size_t)c * t + i];
        A[c] = acc / (float)t;
    }
    l0 = m->fcb[0]; l1 = m->fcb[1];
    for (c = 0; c < m->c[3]; c++) {
        l0 += m->fcw[c] * A[c];
        l1 += m->fcw[m->c[3] + c] * A[c];
    }
    return 1.0f / (1.0f + expf(l0 - l1));
}
