/* matreshka_voice: потоковый движок и логика срабатывания.
 * Copyright (c) 2026 Xstrwe. Лицензия MIT.
 *
 * Как течёт звук:
 *   1. mv_process копит отсчёты, пока не наберётся кадр из 400 (25 мс);
 *      затем сдвигается на 160 (10 мс), так что соседние кадры перекрываются.
 *   2. Из кадра считаются 40 log-mel признаков и нормализуются.
 *   3. По последним k кадрам считается один столбец первой свёртки
 *      и кладётся в кольцевой буфер.
 *   4. Раз в infer_every кадров остаток сети смотрит на всё окно (≈1 с)
 *      и выдаёт вероятность того, что в конце окна прозвучало слово.
 *   5. Оценки сглаживаются; если среднее выше порога и мы не в периоде
 *      «тишины» после прошлого срабатывания — сообщаем о срабатывании. */
#include "matreshka_voice.h"
#include "mv_internal.h"

#include <stdio.h>    /* FILE, fopen, fread — чтение файла модели */
#include <stdlib.h>   /* malloc, calloc, free */
#include <string.h>   /* memmove */

#define MV_VERSION "0.1.0"
#define MAX_SMOOTH 16   /* максимум оценок для сглаживания (размер массива hist) */

/* Полное состояние движка. Снаружи эта структура не видна (см. заголовок).
 * Вся память выделяется в mv_create, во время работы malloc не вызывается —
 * это важно для устройств реального времени. */
struct mv_engine {
    mv_model     model;
    mv_features *feat;
    mv_config    cfg;
    float        threshold;
    /* Нарезка звука на кадры */
    float        pend[MV_FRAME_LEN];  /* накопленные, но ещё не обработанные отсчёты */
    int          npend;               /* сколько их сейчас в pend */
    uint64_t     samples_in;      /* всего принято отсчётов */
    uint64_t     frames;          /* всего получено кадров признаков */
    /* Кольцевой буфер последних k нормализованных кадров. «Кольцевой» значит:
     * кадр номер i лежит в ячейке i % k, и новый кадр затирает самый старый. */
    float       *fring;           /* k × n_mels */
    /* Кольцевой буфер столбцов первой свёртки, t0 = window - k + 1 */
    float       *c0ring;          /* t0 × c0 */
    int          t0;
    float       *x0;              /* те же столбцы, переложенные как c0 × t0 по порядку времени */
    float       *scratch;         /* рабочая память для mv_net_tail */
    /* Срабатывание */
    float        hist[MAX_SMOOTH];   /* последние оценки, hist[0] — самая свежая */
    int          nhist;
    float        last_score;
    uint64_t     refractory_until;   /* до этого отсчёта новые срабатывания игнорируются */
    mv_score_fn  score_fn;
    void        *score_user;
};

/* Текст последней ошибки. static — переменная видна только в этом файле.
 * Это строковая константа, освобождать её не нужно. */
static const char *g_err = "";

const char *mv_last_error(void) { return g_err; }
const char *mv_version(void) { return MV_VERSION; }

mv_config mv_default_config(void)
{
    /* Структуры в C можно возвращать по значению — копируется вся структура. */
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
        /* Цепочка тернарных операторов «условие ? да : нет» выбирает текст по коду. */
        g_err = rc == -1 ? "bad magic" : rc == -2 ? "unsupported version" :
                rc == -4 ? "size mismatch" : "invalid model";
        free(e);
        return NULL;
    }
    if (e->model.n_mels != MV_N_MELS) { g_err = "model n_mels mismatch"; mv_destroy(e); return NULL; }
    /* Нет настроек — берём по умолчанию; некорректные значения поправляем. */
    e->cfg = cfg ? *cfg : mv_default_config();
    if (e->cfg.infer_every < 1) e->cfg.infer_every = 1;
    if (e->cfg.smooth < 1) e->cfg.smooth = 1;
    if (e->cfg.smooth > MAX_SMOOTH) e->cfg.smooth = MAX_SMOOTH;
    if (e->cfg.refractory_ms < 0) e->cfg.refractory_ms = 0;
    e->threshold = e->cfg.threshold > 0.0f ? e->cfg.threshold : e->model.default_threshold;
    /* Свёртка с ядром k без дополнения из window кадров даёт window - k + 1 столбцов. */
    e->t0 = e->model.window - e->model.k + 1;
    e->feat = mv_features_create();
    e->fring = (float *)calloc((size_t)e->model.k * MV_N_MELS, sizeof(float));
    e->c0ring = (float *)calloc((size_t)e->t0 * e->model.c[0], sizeof(float));
    e->x0 = (float *)calloc((size_t)e->t0 * e->model.c[0], sizeof(float));
    e->scratch = (float *)calloc(mv_model_scratch_floats(&e->model), sizeof(float));
    /* Проверяем все выделения разом; mv_destroy умеет освобождать
     * частично созданный движок, потому что free(NULL) ничего не делает. */
    if (!e->feat || !e->fring || !e->c0ring || !e->x0 || !e->scratch) {
        g_err = "out of memory";
        mv_destroy(e);
        return NULL;
    }
    return e;
}

mv_engine *mv_create_from_file(const char *path, const mv_config *cfg)
{
    FILE *fp = fopen(path, "rb");   /* "rb" — читать в двоичном режиме */
    long n;
    void *buf;
    mv_engine *e;
    if (!fp) { g_err = "cannot open model file"; return NULL; }
    /* Узнаём размер файла: прыгаем в конец, спрашиваем позицию, возвращаемся. */
    fseek(fp, 0, SEEK_END);
    n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    /* 64L << 20 = 64 МБ: модель заведомо меньше, это защита от чужих файлов. */
    if (n <= 0 || n > (64L << 20)) { fclose(fp); g_err = "bad model file size"; return NULL; }
    buf = malloc((size_t)n);
    if (!buf || fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        free(buf); fclose(fp); g_err = "cannot read model file"; return NULL;
    }
    fclose(fp);
    e = mv_create(buf, (size_t)n, cfg);   /* mv_create копирует веса, буфер больше не нужен */
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
    /* Буферы не чистим: счётчики обнулены, и старые данные просто
     * не будут использованы, пока не перезапишутся новыми. */
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

/* Обработать один полный кадр из e->pend. Возвращает 1, если сработало. */
static int on_frame(mv_engine *e, mv_detection *det)
{
    const mv_model *m = &e->model;
    int k = m->k, c0 = m->c[0], i, j;
    uint64_t fi = e->frames++;   /* номер этого кадра (постфиксный ++ вернёт старое значение) */
    /* Признаки пишем сразу в нужную ячейку кольцевого буфера и нормализуем:
     * вычитаем среднее и делим на разброс, посчитанные при обучении. */
    float *row = e->fring + (size_t)(fi % (uint64_t)k) * MV_N_MELS;
    mv_features_frame(e->feat, e->pend, row);
    for (i = 0; i < MV_N_MELS; i++) row[i] = (row[i] - m->mean[i]) * m->inv_std[i];
    if (fi + 1 < (uint64_t)k) return 0;   /* пока меньше k кадров, свёртку не посчитать */
    {
        /* Собираем указатели на последние k кадров от старого к новому.
         * Самый старый лежит в ячейке (fi + 1) % k — сразу «за» самым новым. */
        const float *frames[32];
        for (j = 0; j < k; j++)
            frames[j] = e->fring + (size_t)((fi + 1 + (uint64_t)j) % (uint64_t)k) * MV_N_MELS;
        /* Номер нового столбца свёртки = fi - k + 1. */
        mv_conv0_column(m, frames, e->c0ring + (size_t)((fi - (uint64_t)k + 1) % (uint64_t)e->t0) * c0);
    }
    /* Ждём, пока наберётся полное окно, и дальше запускаем сеть
     * только раз в infer_every кадров (по умолчанию каждые 30 мс). */
    if (fi + 1 < (uint64_t)m->window) return 0;
    if ((fi + 1 - (uint64_t)m->window) % (uint64_t)e->cfg.infer_every != 0) return 0;
    {
        /* Перекладываем t0 столбцов из кольца в x0 по порядку времени
         * (самый старый первым) и транспонируем в формат [канал][время]. */
        uint64_t first = fi - (uint64_t)k + 1 - (uint64_t)e->t0 + 1;
        float score, avg = 0.0f;
        for (j = 0; j < e->t0; j++) {
            const float *col = e->c0ring + (size_t)((first + (uint64_t)j) % (uint64_t)e->t0) * c0;
            for (i = 0; i < c0; i++) e->x0[(size_t)i * e->t0 + j] = col[i];
        }
        score = mv_net_tail(m, e->x0, e->t0, e->scratch);
        e->last_score = score;
        if (e->score_fn) e->score_fn(e->score_user, fi, score);
        /* Сдвигаем историю на одну позицию (memmove корректно работает,
         * когда области памяти перекрываются) и кладём новую оценку в начало. */
        memmove(e->hist + 1, e->hist, sizeof(float) * (MAX_SMOOTH - 1));
        e->hist[0] = score;
        if (e->nhist < e->cfg.smooth) e->nhist++;
        /* Делим всегда на smooth, а не на nhist: в самом начале, пока оценок
         * мало, среднее занижено, и ложных срабатываний на старте меньше. */
        for (i = 0; i < e->nhist; i++) avg += e->hist[i];
        avg /= (float)e->cfg.smooth;
        {
            /* Номер отсчёта, на котором заканчивается текущий кадр. */
            uint64_t end = fi * MV_HOP + MV_FRAME_LEN;
            if (avg >= e->threshold && end >= e->refractory_until) {
                /* После срабатывания какое-то время молчим, иначе одно
                 * слово дало бы несколько срабатываний подряд. */
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
    /* Звук может приходить кусками любой длины (у WireOS по 512 отсчётов).
     * Доливаем pend до полного кадра, обрабатываем кадр и сдвигаем его на шаг
     * HOP, оставляя перекрытие. Поэтому результат не зависит от размера кусков. */
    while (i < n) {
        size_t take = (size_t)(MV_FRAME_LEN - e->npend);
        size_t s;
        if (take > n - i) take = n - i;
        /* int16 -> float в диапазоне [-1, 1). Умножение на константу 1/32768
         * дешевле деления. */
        for (s = 0; s < take; s++) e->pend[e->npend + (int)s] = (float)pcm[i + s] * (1.0f / 32768.0f);
        e->npend += (int)take;
        e->samples_in += take;
        i += take;
        if (e->npend == MV_FRAME_LEN) {
            mv_detection d;
            if (on_frame(e, &d) && found < max_out && out) out[found++] = d;
            /* Оставляем последние 240 отсчётов: они станут началом следующего кадра. */
            memmove(e->pend, e->pend + MV_HOP, sizeof(float) * (MV_FRAME_LEN - MV_HOP));
            e->npend = MV_FRAME_LEN - MV_HOP;
        }
    }
    return found;
}
