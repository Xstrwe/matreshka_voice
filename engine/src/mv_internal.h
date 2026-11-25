/* matreshka_voice: внутренние определения, общие для файлов движка.
 * Пользователю библиотеки этот заголовок не нужен.
 * Copyright (c) 2026 Xstrwe. Лицензия MIT. */
#ifndef MV_INTERNAL_H
#define MV_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#define MV_MAGIC "MVW1"        /* первые 4 байта файла модели */
#define MV_HEADER_BYTES 68     /* 4 (magic) + 11*4 (целые) + 4 (порог) + 4*4 (резерв) */

/* Загруженная модель. Все веса лежат одним блоком в data, а остальные
 * указатели смотрят внутрь этого блока — так нужен всего один malloc/free. */
typedef struct {
    int n_mels, window, k, c[4], stride[3];  /* размеры сети: каналы c0..c3, шаги блоков */
    float default_threshold;
    float *data;                         /* владеет всеми весами */
    const float *mean, *inv_std;         /* нормализация: (x - mean) * inv_std */
    const float *w0, *b0;                /* первая свёртка */
    const float *dw[3], *db[3], *pw[3], *pb[3];  /* три блока depthwise + pointwise */
    const float *fcw, *fcb;              /* последний полносвязный слой на 2 класса */
} mv_model;

/* Разобрать блоб модели. 0 — успех, отрицательное число — код ошибки. */
int   mv_model_load(mv_model *m, const void *blob, size_t len);
void  mv_model_free(mv_model *m);
/* Сколько float нужно во временном буфере для mv_net_tail. */
size_t mv_model_scratch_floats(const mv_model *m);
/* Один столбец выхода первой свёртки из последних k нормализованных кадров
 * (frames[0] — самый старый, frames[k-1] — самый новый). */
void  mv_conv0_column(const mv_model *m, const float *const *frames, float *out_c0);
/* Остальная сеть: блоки 1..3, усреднение по времени и fc.
 * x0 хранится как [c0][t0]. Возвращает вероятность ключевого слова. */
float mv_net_tail(const mv_model *m, const float *x0, int t0, float *scratch);

#endif
