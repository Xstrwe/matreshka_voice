/* matreshka_voice internal definitions. Copyright (c) 2026 Xstrwe. MIT License. */
#ifndef MV_INTERNAL_H
#define MV_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#define MV_MAGIC "MVW1"
#define MV_HEADER_BYTES 68

typedef struct {
    int n_mels, window, k, c[4], stride[3];
    float default_threshold;
    float *data;                         /* owns all weights */
    const float *mean, *inv_std;
    const float *w0, *b0;
    const float *dw[3], *db[3], *pw[3], *pb[3];
    const float *fcw, *fcb;
} mv_model;

int   mv_model_load(mv_model *m, const void *blob, size_t len);
void  mv_model_free(mv_model *m);
size_t mv_model_scratch_floats(const mv_model *m);
/* One conv0 output column from the last k normalized frames (rows oldest..newest). */
void  mv_conv0_column(const mv_model *m, const float *const *frames, float *out_c0);
/* Run blocks 1..3 + pooling + fc on x0 laid out as [c0][t0]; returns P(keyword). */
float mv_net_tail(const mv_model *m, const float *x0, int t0, float *scratch);

#endif
