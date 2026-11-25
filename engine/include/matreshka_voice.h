/*
 * matreshka_voice — небольшой открытый движок распознавания ключевого слова
 * («wake word»), который работает прямо на устройстве.
 *
 * Это публичный заголовок: всё, что нужно программе, чтобы пользоваться
 * движком. Написан на чистом C99, зависит только от libc и libm.
 *
 * Copyright (c) 2026 Xstrwe. Лицензия MIT.
 *
 * Входной звук: 16 кГц, моно, 16-битные целые со знаком (PCM), куски любой длины.
 *
 * Формат файла модели (.mvw), все числа little endian (младший байт первым):
 *   char     magic[4] = "MVW1"          — «подпись», по ней узнаём свой формат
 *   uint32   version (1), n_mels, window, k, c0, c1, c2, c3, stride1, stride2, stride3
 *   float32  default_threshold          — порог срабатывания по умолчанию
 *   uint32   reserved[4]                — запас на будущее
 *   float32  mean[n_mels], inv_std[n_mels]   — нормализация признаков
 *            conv0_w[c0][n_mels][k], conv0_b[c0]
 *            для каждого блока i=1..3 (cin -> cout):
 *              dw_w[cin][k], dw_b[cin], pw_w[cout][cin], pw_b[cout]
 *            fc_w[2][c3], fc_b[2]
 */

/* «Защита от повторного включения»: если заголовок подключат дважды,
 * второй раз препроцессор пропустит всё до #endif. */
#ifndef MATRESHKA_VOICE_H
#define MATRESHKA_VOICE_H

#include <stddef.h>   /* size_t — беззнаковый тип для размеров */
#include <stdint.h>   /* int16_t, uint64_t — целые точно заданной ширины */

/* Если заголовок подключают из C++, говорим компилятору, что функции
 * написаны на C (без «декорирования» имён), иначе компоновщик их не найдёт. */
#ifdef __cplusplus
extern "C" {
#endif

/* Параметры звука и признаков. #define — это просто подстановка текста
 * препроцессором, поэтому такие константы ничего не стоят во время работы. */
#define MV_SAMPLE_RATE 16000
#define MV_FRAME_LEN   400   /* окно анализа: 400 отсчётов = 25 мс */
#define MV_HOP         160   /* шаг между окнами: 160 отсчётов = 10 мс */
#define MV_N_FFT       512   /* длина FFT (степень двойки, окно дополняется нулями) */
#define MV_N_MELS      40    /* число мел-полос (признаков на один кадр) */

/* «Непрозрачный» тип: снаружи видно только имя структуры, а её поля
 * описаны в .c файле. Так пользователь не может испортить внутреннее
 * состояние и работает с движком только через функции ниже. */
typedef struct mv_engine mv_engine;

/* Настройки, которые можно менять без переобучения модели. */
typedef struct {
    float threshold;      /* <= 0 значит «взять порог из файла модели» */
    int   infer_every;    /* запускать сеть раз в N кадров (кадр = 10 мс), по умолчанию 3 */
    int   smooth;         /* усреднять N последних оценок сети, по умолчанию 2 */
    int   refractory_ms;  /* после срабатывания молчать столько мс, по умолчанию 1500 */
} mv_config;

/* Одно срабатывание. */
typedef struct {
    uint64_t end_sample;  /* номер отсчёта (с момента создания/сброса), где закончилось слово */
    float    score;       /* сглаженная оценка, которая превысила порог */
} mv_detection;

/* Указатель на функцию: движок вызывает её после каждого запуска сети.
 * Нужна для отладки и оценки качества; user — любые данные вызывающего. */
typedef void (*mv_score_fn)(void *user, uint64_t frame_index, float score);

/* Настройки по умолчанию. Обычно берут их и меняют одно-два поля. */
mv_config   mv_default_config(void);

/* Создать движок из модели в памяти. Модель копируется внутрь,
 * поэтому буфер blob потом можно освободить. При ошибке — NULL. */
mv_engine  *mv_create(const void *blob, size_t len, const mv_config *cfg);
/* То же самое, но модель читается из файла. */
mv_engine  *mv_create_from_file(const char *path, const mv_config *cfg);
/* Освободить всю память движка. NULL передавать можно. */
void        mv_destroy(mv_engine *e);
/* Забыть накопленный звук и историю, как будто движок только что создан. */
void        mv_reset(mv_engine *e);

/* Подать звук. Возвращает, сколько срабатываний записано в out
 * (0 или больше, но не больше max_out). */
int         mv_process(mv_engine *e, const int16_t *pcm, size_t n,
                       mv_detection *out, int max_out);

void        mv_set_score_callback(mv_engine *e, mv_score_fn fn, void *user);
void        mv_set_threshold(mv_engine *e, float threshold);
float       mv_get_threshold(const mv_engine *e);
float       mv_last_score(const mv_engine *e);   /* последняя «сырая» оценка сети */
const char *mv_last_error(void);                 /* текст последней ошибки создания */
const char *mv_version(void);

/* ---- Низкоуровневые части, открыты для тестов и проверки совпадения с Python ---- */

typedef struct mv_features mv_features;
mv_features *mv_features_create(void);
void         mv_features_destroy(mv_features *f);
/* frame: MV_FRAME_LEN отсчётов в виде float в [-1, 1];
 * out: MV_N_MELS значений логарифма мел-спектра */
void         mv_features_frame(mv_features *f, const float *frame, float *out);

#ifdef __cplusplus
}
#endif
#endif
