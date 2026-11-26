/* matreshka_voice: утилита командной строки для проверки движка на файлах.
 * Copyright (c) 2026 Xstrwe. Лицензия MIT.
 *
 * Запуск: mv_cli MODEL.mvw INPUT.(wav|raw) [параметры]
 *   -t ПОРОГ         порог срабатывания (по умолчанию — из модели)
 *   -e N             запускать сеть раз в N кадров (по умолчанию 3)
 *   -s N             сглаживать по N запускам сети (по умолчанию 2)
 *   -r МС            пауза после срабатывания в мс (по умолчанию 1500)
 *   -c КУСОК         подавать звук кусками по КУСОК отсчётов (по умолчанию 512)
 *   --scores ФАЙЛ    записать «номер_кадра оценка» для каждого запуска сети
 *   --features ФАЙЛ  записать признаки log-mel как float32 (кадры × 40), без нормализации
 *   --quiet          печатать только итоговую строку
 * Вход: WAV 16 кГц, моно, 16 бит PCM, или «сырые» отсчёты s16le без заголовка.
 *
 * Утилита нужна для тестов: по --scores и --features Python-скрипты
 * проверяют, что движок на C считает то же самое, что эталон на numpy.
 */
#include "matreshka_voice.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Прочитать WAV или «сырой» файл целиком в память. Возвращает массив
 * отсчётов (освобождать через free) и пишет их число в *n_out.
 * WAV устроен как цепочка «чанков»: 4 байта имени, 4 байта длины, данные.
 * Нам нужны чанк "fmt " (формат звука) и "data" (сами отсчёты). */
static int16_t *read_audio(const char *path, size_t *n_out)
{
    FILE *fp = fopen(path, "rb");
    unsigned char h[12];
    int16_t *pcm = NULL;
    long size;
    size_t n = 0;
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    /* Размер файла понадобится, если это не WAV, а «сырые» данные. */
    fseek(fp, 0, SEEK_END); size = ftell(fp); fseek(fp, 0, SEEK_SET);
    /* Заголовок WAV: "RIFF", размер, "WAVE". memcmp возвращает 0 при совпадении. */
    if (size >= 12 && fread(h, 1, 12, fp) == 12 && !memcmp(h, "RIFF", 4) && !memcmp(h + 8, "WAVE", 4)) {
        unsigned char ch[8];
        int fmt_ok = 0;
        while (fread(ch, 1, 8, fp) == 8) {
            /* Длина чанка — 32-битное little endian число из байтов 4..7. */
            unsigned long len = ch[4] | (ch[5] << 8) | ((unsigned long)ch[6] << 16) | ((unsigned long)ch[7] << 24);
            if (!memcmp(ch, "fmt ", 4)) {
                unsigned char f[16];
                if (len < 16 || fread(f, 1, 16, fp) != 16) break;
                {
                    int fmt = f[0] | (f[1] << 8), chans = f[2] | (f[3] << 8);
                    long rate = f[4] | (f[5] << 8) | ((long)f[6] << 16) | ((long)f[7] << 24);
                    int bits = f[14] | (f[15] << 8);
                    /* 1 — обычный PCM, 0xFFFE — «расширенный» формат; принимаем только 16 кГц моно 16 бит. */
                    if ((fmt != 1 && fmt != 0xFFFE) || chans != 1 || rate != 16000 || bits != 16) {
                        fprintf(stderr, "%s: need 16 kHz mono 16-bit PCM (got fmt %d, %d ch, %ld Hz, %d bit)\n",
                                path, fmt, chans, rate, bits);
                        fclose(fp); return NULL;
                    }
                    fmt_ok = 1;
                }
                /* Пропускаем остаток чанка; чанки выровнены по чётной длине, отсюда (len & 1). */
                fseek(fp, (long)(len - 16 + (len & 1)), SEEK_CUR);
            } else if (!memcmp(ch, "data", 4)) {
                if (!fmt_ok) break;
                /* Отсчёт занимает 2 байта; +2 байта в malloc — запас, чтобы не получить malloc(0). */
                n = len / 2;
                pcm = (int16_t *)malloc(n * 2 + 2);
                if (!pcm) break;
                n = fread(pcm, 2, n, fp);
                break;
            } else {
                fseek(fp, (long)(len + (len & 1)), SEEK_CUR);
            }
        }
        if (!pcm) { fprintf(stderr, "%s: no usable data chunk\n", path); fclose(fp); return NULL; }
    } else {
        fseek(fp, 0, SEEK_SET);
        n = (size_t)size / 2;
        pcm = (int16_t *)malloc(n * 2 + 2);
        if (pcm) n = fread(pcm, 2, n, fp);
    }
    fclose(fp);
    /* WAV и raw хранятся little endian. Проверяем порядок байтов процессора:
     * если первый байт числа 1 равен 0, процессор big endian — меняем байты местами. */
    {
        const uint16_t one = 1;
        if (*(const unsigned char *)&one == 0) {
            size_t i;
            for (i = 0; i < n; i++) { uint16_t v = (uint16_t)pcm[i]; pcm[i] = (int16_t)((v >> 8) | (v << 8)); }
        }
    }
    *n_out = n;
    return pcm;
}

/* Колбэк для --scores: user — это открытый FILE*, переданный в движок. */
static void on_score(void *user, uint64_t frame, float score)
{
    fprintf((FILE *)user, "%llu %.6f\n", (unsigned long long)frame, score);
}

int main(int argc, char **argv)
{
    mv_config cfg = mv_default_config();
    const char *scores_path = NULL, *feat_path = NULL;
    int chunk = 512, quiet = 0, i, ndet = 0;
    size_t n = 0, pos;
    int16_t *pcm;
    mv_engine *e;
    FILE *sf = NULL;
    clock_t t0, t1;
    if (argc < 3) {
        fprintf(stderr, "matreshka_voice %s\nusage: %s MODEL.mvw INPUT.wav [-t thr] [-e every] [-s smooth] "
                        "[-r ms] [-c chunk] [--scores f] [--features f] [--quiet]\n", mv_version(), argv[0]);
        return 2;
    }
    /* Разбор параметров: argv[1] — модель, argv[2] — звук, дальше опции.
     * argv[++i] берёт значение опции и сразу перескакивает через него. */
    for (i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) cfg.threshold = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-e") && i + 1 < argc) cfg.infer_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) cfg.smooth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) cfg.refractory_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--scores") && i + 1 < argc) scores_path = argv[++i];
        else if (!strcmp(argv[i], "--features") && i + 1 < argc) feat_path = argv[++i];
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (chunk < 1) chunk = 512;
    e = mv_create_from_file(argv[1], &cfg);
    if (!e) { fprintf(stderr, "model: %s\n", mv_last_error()); return 1; }
    pcm = read_audio(argv[2], &n);
    if (!pcm) { mv_destroy(e); return 1; }

    /* Признаки считаем отдельно по всему файлу, кадр за кадром — для
     * сравнения с Python (tests/parity.py). */
    if (feat_path) {
        FILE *ff = fopen(feat_path, "wb");
        mv_features *f = mv_features_create();
        float frame[MV_FRAME_LEN], out[MV_N_MELS];
        size_t s;
        int j;
        for (s = 0; ff && f && s + MV_FRAME_LEN <= n; s += MV_HOP) {
            for (j = 0; j < MV_FRAME_LEN; j++) frame[j] = pcm[s + j] / 32768.0f;
            mv_features_frame(f, frame, out);
            fwrite(out, sizeof(float), MV_N_MELS, ff);
        }
        mv_features_destroy(f);
        if (ff) fclose(ff);
    }
    if (scores_path) {
        sf = fopen(scores_path, "w");
        if (sf) mv_set_score_callback(e, on_score, sf);
    }
    /* clock() меряет процессорное время; так получаем «коэффициент реального
     * времени»: какая доля секунды CPU уходит на секунду звука. */
    t0 = clock();
    /* Подаём звук кусками, как это делал бы настоящий микрофонный поток. */
    for (pos = 0; pos < n; pos += (size_t)chunk) {
        mv_detection d[4];
        size_t len = n - pos < (size_t)chunk ? n - pos : (size_t)chunk;
        int k, got = mv_process(e, pcm + pos, len, d, 4);
        for (k = 0; k < got; k++) {
            ndet++;
            if (!quiet) printf("detection %.3f s score %.3f\n", d[k].end_sample / 16000.0, d[k].score);
        }
    }
    t1 = clock();
    if (sf) fclose(sf);
    {
        double audio_s = n / 16000.0, cpu_s = (double)(t1 - t0) / CLOCKS_PER_SEC;
        printf("summary: audio %.1f s, detections %d, threshold %.3f, cpu %.3f s, realtime factor %.4f\n",
               audio_s, ndet, mv_get_threshold(e), cpu_s, audio_s > 0 ? cpu_s / audio_s : 0.0);
    }
    free(pcm);
    mv_destroy(e);
    return 0;
}
