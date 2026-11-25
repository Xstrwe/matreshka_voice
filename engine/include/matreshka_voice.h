/*
 * matreshka_voice: a small open-source on-device wake word engine.
 * Plain C99, no dependencies beyond libc and libm.
 *
 * Copyright (c) 2026 Xstrwe. MIT License.
 *
 * Audio input: 16 kHz, mono, signed 16-bit PCM, any chunk size.
 *
 * Model blob (.mvw), little endian:
 *   char     magic[4] = "MVW1"
 *   uint32   version (1), n_mels, window, k, c0, c1, c2, c3, stride1, stride2, stride3
 *   float32  default_threshold
 *   uint32   reserved[4]
 *   float32  mean[n_mels], inv_std[n_mels]
 *            conv0_w[c0][n_mels][k], conv0_b[c0]
 *            for each block i=1..3 (cin -> cout):
 *              dw_w[cin][k], dw_b[cin], pw_w[cout][cin], pw_b[cout]
 *            fc_w[2][c3], fc_b[2]
 */
#ifndef MATRESHKA_VOICE_H
#define MATRESHKA_VOICE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MV_SAMPLE_RATE 16000
#define MV_FRAME_LEN   400   /* 25 ms */
#define MV_HOP         160   /* 10 ms */
#define MV_N_FFT       512
#define MV_N_MELS      40

typedef struct mv_engine mv_engine;

typedef struct {
    float threshold;      /* <= 0 means: use the model's default threshold */
    int   infer_every;    /* run the network every N feature frames (10 ms each), default 3 */
    int   smooth;         /* average the last N network scores, default 2 */
    int   refractory_ms;  /* ignore new detections for this long after one, default 1500 */
} mv_config;

typedef struct {
    uint64_t end_sample;  /* absolute sample index (since create/reset) where the keyword ended */
    float    score;       /* smoothed score that crossed the threshold */
} mv_detection;

/* Called after every network run (optional, for debugging and evaluation). */
typedef void (*mv_score_fn)(void *user, uint64_t frame_index, float score);

mv_config   mv_default_config(void);

/* Create from a blob in memory. The blob is copied; the caller may free it afterwards. */
mv_engine  *mv_create(const void *blob, size_t len, const mv_config *cfg);
mv_engine  *mv_create_from_file(const char *path, const mv_config *cfg);
void        mv_destroy(mv_engine *e);
void        mv_reset(mv_engine *e);

/* Feed audio. Returns the number of detections written to out (0 or more, at most max_out). */
int         mv_process(mv_engine *e, const int16_t *pcm, size_t n,
                       mv_detection *out, int max_out);

void        mv_set_score_callback(mv_engine *e, mv_score_fn fn, void *user);
void        mv_set_threshold(mv_engine *e, float threshold);
float       mv_get_threshold(const mv_engine *e);
float       mv_last_score(const mv_engine *e);
const char *mv_last_error(void);
const char *mv_version(void);

/* ---- Lower-level pieces, exposed for tests and parity checks ---- */

typedef struct mv_features mv_features;
mv_features *mv_features_create(void);
void         mv_features_destroy(mv_features *f);
/* frame: MV_FRAME_LEN samples as float in [-1, 1]; out: MV_N_MELS log-mel values */
void         mv_features_frame(mv_features *f, const float *frame, float *out);

#ifdef __cplusplus
}
#endif
#endif
