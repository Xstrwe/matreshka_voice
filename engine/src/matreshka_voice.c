/* matreshka_voice: streaming engine and detection logic.
 * Copyright (c) 2026 Xstrwe. MIT License. */
#include "matreshka_voice.h"
#include "mv_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MV_VERSION "0.1.0"
#define MAX_SMOOTH 16

struct mv_engine {
    mv_model     model;
    mv_features *feat;
    mv_config    cfg;
    float        threshold;
    /* audio framing */
    float        pend[MV_FRAME_LEN];
    int          npend;
    uint64_t     samples_in;      /* total samples consumed */
    uint64_t     frames;          /* feature frames produced */
    /* last k normalized frames (ring) */
    float       *fring;           /* k x n_mels */
    /* conv0 columns (ring), t0 = window - k + 1 */
    float       *c0ring;          /* t0 x c0 */
    int          t0;
    float       *x0;              /* c0 x t0, time ordered */
    float       *scratch;
    /* detection */
    float        hist[MAX_SMOOTH];
    int          nhist;
    float        last_score;
    uint64_t     refractory_until;
    mv_score_fn  score_fn;
    void        *score_user;
};

static const char *g_err = "";

const char *mv_last_error(void) { return g_err; }
const char *mv_version(void) { return MV_VERSION; }

mv_config mv_default_config(void)
{
    mv_config c;
    c.threshold = 0.0f;
    c.infer_every = 3;
    c.smooth = 2;
    c.refractory_ms = 1500;
    return c;
}

mv_engine *mv_create(const void *blob, size_t len, const mv_config *cfg)
{
    mv_engine *e = (mv_engine *)calloc(1, sizeof(*e));
    int rc;
    if (!e) { g_err = "out of memory"; return NULL; }
    rc = mv_model_load(&e->model, blob, len);
    if (rc != 0) {
        g_err = rc == -1 ? "bad magic" : rc == -2 ? "unsupported version" :
                rc == -4 ? "size mismatch" : "invalid model";
        free(e);
        return NULL;
    }
    if (e->model.n_mels != MV_N_MELS) { g_err = "model n_mels mismatch"; mv_destroy(e); return NULL; }
    e->cfg = cfg ? *cfg : mv_default_config();
    if (e->cfg.infer_every < 1) e->cfg.infer_every = 1;
    if (e->cfg.smooth < 1) e->cfg.smooth = 1;
    if (e->cfg.smooth > MAX_SMOOTH) e->cfg.smooth = MAX_SMOOTH;
    if (e->cfg.refractory_ms < 0) e->cfg.refractory_ms = 0;
    e->threshold = e->cfg.threshold > 0.0f ? e->cfg.threshold : e->model.default_threshold;
    e->t0 = e->model.window - e->model.k + 1;
    e->feat = mv_features_create();
    e->fring = (float *)calloc((size_t)e->model.k * MV_N_MELS, sizeof(float));
    e->c0ring = (float *)calloc((size_t)e->t0 * e->model.c[0], sizeof(float));
    e->x0 = (float *)calloc((size_t)e->t0 * e->model.c[0], sizeof(float));
    e->scratch = (float *)calloc(mv_model_scratch_floats(&e->model), sizeof(float));
    if (!e->feat || !e->fring || !e->c0ring || !e->x0 || !e->scratch) {
        g_err = "out of memory";
        mv_destroy(e);
        return NULL;
    }
    return e;
}

mv_engine *mv_create_from_file(const char *path, const mv_config *cfg)
{
    FILE *fp = fopen(path, "rb");
    long n;
    void *buf;
    mv_engine *e;
    if (!fp) { g_err = "cannot open model file"; return NULL; }
    fseek(fp, 0, SEEK_END);
    n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (n <= 0 || n > (64L << 20)) { fclose(fp); g_err = "bad model file size"; return NULL; }
    buf = malloc((size_t)n);
    if (!buf || fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        free(buf); fclose(fp); g_err = "cannot read model file"; return NULL;
    }
    fclose(fp);
    e = mv_create(buf, (size_t)n, cfg);
    free(buf);
    return e;
}

void mv_destroy(mv_engine *e)
{
    if (!e) return;
    mv_model_free(&e->model);
    mv_features_destroy(e->feat);
    free(e->fring); free(e->c0ring); free(e->x0); free(e->scratch);
    free(e);
}

void mv_reset(mv_engine *e)
{
    e->npend = 0;
    e->samples_in = 0;
    e->frames = 0;
    e->nhist = 0;
    e->last_score = 0.0f;
    e->refractory_until = 0;
}

void mv_set_score_callback(mv_engine *e, mv_score_fn fn, void *user) { e->score_fn = fn; e->score_user = user; }
void mv_set_threshold(mv_engine *e, float t) { e->threshold = t; }
float mv_get_threshold(const mv_engine *e) { return e->threshold; }
float mv_last_score(const mv_engine *e) { return e->last_score; }

/* Process one complete frame in e->pend. Returns 1 on detection. */
static int on_frame(mv_engine *e, mv_detection *det)
{
    const mv_model *m = &e->model;
    int k = m->k, c0 = m->c[0], i, j;
    uint64_t fi = e->frames++;
    float *row = e->fring + (size_t)(fi % (uint64_t)k) * MV_N_MELS;
    mv_features_frame(e->feat, e->pend, row);
    for (i = 0; i < MV_N_MELS; i++) row[i] = (row[i] - m->mean[i]) * m->inv_std[i];
    if (fi + 1 < (uint64_t)k) return 0;
    {
        const float *frames[32];
        for (j = 0; j < k; j++)
            frames[j] = e->fring + (size_t)((fi + 1 + (uint64_t)j) % (uint64_t)k) * MV_N_MELS;
        /* conv0 column index = fi - k + 1 */
        mv_conv0_column(m, frames, e->c0ring + (size_t)((fi - (uint64_t)k + 1) % (uint64_t)e->t0) * c0);
    }
    if (fi + 1 < (uint64_t)m->window) return 0;
    if ((fi + 1 - (uint64_t)m->window) % (uint64_t)e->cfg.infer_every != 0) return 0;
    {
        /* gather t0 columns, oldest first, into channel-major x0 */
        uint64_t first = fi - (uint64_t)k + 1 - (uint64_t)e->t0 + 1;
        float score, avg = 0.0f;
        for (j = 0; j < e->t0; j++) {
            const float *col = e->c0ring + (size_t)((first + (uint64_t)j) % (uint64_t)e->t0) * c0;
            for (i = 0; i < c0; i++) e->x0[(size_t)i * e->t0 + j] = col[i];
        }
        score = mv_net_tail(m, e->x0, e->t0, e->scratch);
        e->last_score = score;
        if (e->score_fn) e->score_fn(e->score_user, fi, score);
        memmove(e->hist + 1, e->hist, sizeof(float) * (MAX_SMOOTH - 1));
        e->hist[0] = score;
        if (e->nhist < e->cfg.smooth) e->nhist++;
        for (i = 0; i < e->nhist; i++) avg += e->hist[i];
        avg /= (float)e->cfg.smooth;
        {
            uint64_t end = fi * MV_HOP + MV_FRAME_LEN;
            if (avg >= e->threshold && end >= e->refractory_until) {
                e->refractory_until = end + (uint64_t)e->cfg.refractory_ms * (MV_SAMPLE_RATE / 1000);
                det->end_sample = end;
                det->score = avg;
                return 1;
            }
        }
    }
    return 0;
}

int mv_process(mv_engine *e, const int16_t *pcm, size_t n, mv_detection *out, int max_out)
{
    int found = 0;
    size_t i = 0;
    while (i < n) {
        size_t take = (size_t)(MV_FRAME_LEN - e->npend);
        size_t s;
        if (take > n - i) take = n - i;
        for (s = 0; s < take; s++) e->pend[e->npend + (int)s] = (float)pcm[i + s] * (1.0f / 32768.0f);
        e->npend += (int)take;
        e->samples_in += take;
        i += take;
        if (e->npend == MV_FRAME_LEN) {
            mv_detection d;
            if (on_frame(e, &d) && found < max_out && out) out[found++] = d;
            memmove(e->pend, e->pend + MV_HOP, sizeof(float) * (MV_FRAME_LEN - MV_HOP));
            e->npend = MV_FRAME_LEN - MV_HOP;
        }
    }
    return found;
}
