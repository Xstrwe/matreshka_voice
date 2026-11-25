"""Синтез положительных и отрицательных клипов голосами Piper TTS (WAV 16 кГц, моно, int16).

Запуск: python train/gen_data.py --voices-dir voices --out data

Результат: data/<train|test>/pos/*.wav — слово «Вектор»; data/<split>/neg/*_negh_*.wav —
похожие слова; data/<split>/neg/*_negs_*.wav — обычные предложения. Голос ruslan и часть
дикторов libritts не участвуют в обучении: на них проверяем, как сеть понимает новые голоса.
"""
import argparse
import os
import random
from multiprocessing import Pool

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import texts

# имя -> (имя файла модели без .onnx, номер диктора). Украинские голоса добавляют разнообразия;
# они читают буквы, а не фонемы, поэтому русский текст подгоняется под их алфавит (adapt_text).
VOICES = {
    "denis": ("ru_RU-denis-medium", None), "dmitri": ("ru_RU-dmitri-medium", None),
    "irina": ("ru_RU-irina-medium", None), "ruslan": ("ru_RU-ruslan-medium", None),
    "uklada": ("uk_UA-ukrainian_tts-medium", 0), "ukmykyta": ("uk_UA-ukrainian_tts-medium", 1),
    "uktetiana": ("uk_UA-ukrainian_tts-medium", 2), "ladaxl": ("uk_UA-lada-x_low", None),
}
# Многодикторный английский голос, которому подаём русские фонемы (espeak "ru", полученные
# через русский голос Piper). Его набор фонем включает русские, поэтому он «говорит» по-русски
# сотнями разных голосов (с акцентом). Дикторы с номером >= XFER_TEST_FROM — только для теста.
XFER = {"libritts": ("en_US-libritts_r-medium", "ru_RU-dmitri-medium", 904)}
XFER_TEST_FROM = 850
TRAIN_VOICES = ["denis", "dmitri", "irina", "uklada", "ukmykyta", "uktetiana", "ladaxl"]
TEST_VOICES = ["ruslan"]

# Замены букв: украинские «і», «и», «е» звучат как русские «и», «ы», «э».
_RU2UK = {"и": "і", "ы": "и", "э": "е", "ё": "ьо", "ъ": "'"}


def adapt_text(text, voice):
    """Русская запись -> приближение украинскими буквами для голосов с текстовым входом.
    В слове «вектор» ставится знак ударения (U+0301) на первый слог."""
    if voice is None or getattr(voice.config, "phoneme_type", None) != "text":
        return text
    t = text.lower().replace("вектор", "ве\u0301ктор")
    return "".join(_RU2UK.get(ch, ch) for ch in t)

# Глобальные переменные процесса-воркера: каждый процесс из Pool загружает голос
# один раз в _init и дальше только синтезирует.
_voice = None
_speaker = None


def _init(model_path, speaker):
    global _voice, _speaker
    from piper import PiperVoice
    _voice = PiperVoice.load(model_path)
    _speaker = speaker


_phonemizer = None


def _init_xfer(model_path, phonemizer_path):
    global _voice, _phonemizer
    from piper import PiperVoice
    _voice = PiperVoice.load(model_path)
    _phonemizer = PiperVoice.load(phonemizer_path)


def _synth_xfer(job):
    """Синтез через «чужой» голос: текст -> русские фонемы -> номера фонем -> звук.
    Диктор, темп и «живость» речи выбираются случайно, но повторяемо (seed)."""
    text, out_path, seed, spk_lo, spk_hi = job
    from piper import SynthesisConfig
    rnd = random.Random(seed)
    cfg = SynthesisConfig(speaker_id=rnd.randrange(spk_lo, spk_hi), length_scale=rnd.uniform(0.7, 1.3),
                          noise_scale=rnd.uniform(0.4, 0.9), noise_w_scale=rnd.uniform(0.5, 1.1))
    parts = []
    for sent in _phonemizer.phonemize(text):
        a = _voice.phoneme_ids_to_audio(_voice.phonemes_to_ids(sent), cfg)
        parts.append(a.astype(np.float32) / (32768.0 if a.dtype == np.int16 else 1.0))
    audio = np.concatenate(parts)
    audio = audio / (np.abs(audio).max() + 1e-9) * 0.9     # пик на 90% шкалы
    # Передискретизация к 16 кГц: делим частоты на НОД, чтобы получить целые up/down.
    g = np.gcd(16000, _voice.config.sample_rate)
    audio = resample_poly(audio, 16000 // g, _voice.config.sample_rate // g)
    sf.write(out_path, np.clip(audio, -1, 1), 16000, subtype="PCM_16")


def _synth(job):
    """Синтез обычным голосом: length_scale меняет темп, noise_* — интонацию и ритм."""
    text, out_path, seed = job
    from piper import SynthesisConfig
    rnd = random.Random(seed)
    cfg = SynthesisConfig(speaker_id=_speaker, length_scale=rnd.uniform(0.7, 1.35), noise_scale=rnd.uniform(0.4, 0.9),
                          noise_w_scale=rnd.uniform(0.5, 1.1), normalize_audio=True)
    audio = np.concatenate([c.audio_float_array for c in _voice.synthesize(adapt_text(text, _voice), syn_config=cfg)])
    sr = _voice.config.sample_rate
    if sr != 16000:
        g = np.gcd(16000, sr)
        audio = resample_poly(audio, 16000 // g, sr // g)
    sf.write(out_path, np.clip(audio, -1, 1), 16000, subtype="PCM_16")


def jobs_for(voice, split, out, n_pos, n_neg_hard, n_neg_sent, seed):
    """Список заданий (текст, путь к файлу, seed) для одного голоса и одной части данных."""
    rnd = random.Random(seed)
    jobs = []
    for kind, n, pool in [("pos", n_pos, texts.POSITIVE), ("neg", n_neg_hard, texts.HARD_NEGATIVE)]:
        d = os.path.join(out, split, kind)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            jobs.append((rnd.choice(pool), os.path.join(d, f"{voice}_{kind}{'h' if kind == 'neg' else ''}_{i:05d}.wav"), rnd.random()))
    d = os.path.join(out, split, "neg")
    sents = texts.SENTENCES * 3 + texts.generated_sentences(n_neg_sent, seed=seed)
    rnd.shuffle(sents)
    for i, s in enumerate(sents[:n_neg_sent]):
        jobs.append((s, os.path.join(d, f"{voice}_negs_{i:05d}.wav"), rnd.random()))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices-dir", default="voices")
    ap.add_argument("--out", default="data")
    ap.add_argument("--pos", type=int, default=2500)
    ap.add_argument("--neg-hard", type=int, default=1500)
    ap.add_argument("--neg-sent", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", help="comma separated voice names (default: all)")
    ap.add_argument("--xfer-scale", type=float, default=2.4, help="size of the multi-speaker set vs one voice")
    a = ap.parse_args()
    for vi, voice in enumerate(TRAIN_VOICES + TEST_VOICES):
        if a.only and voice not in a.only.split(","):
            continue
        split = "train" if voice in TRAIN_VOICES else "test"
        scale = 1.0 if split == "train" else 0.2      # тестовый набор в 5 раз меньше
        jobs = jobs_for(voice, split, a.out, int(a.pos * scale), int(a.neg_hard * scale), int(a.neg_sent * scale), 1000 + vi)
        stem, speaker = VOICES[voice]
        model = os.path.join(a.voices_dir, stem + ".onnx")
        # Синтез идёт параллельно в a.workers процессах.
        with Pool(a.workers, initializer=_init, initargs=(model, speaker)) as p:
            for k, _ in enumerate(p.imap_unordered(_synth, jobs, chunksize=16)):
                if k % 1000 == 0:
                    print(voice, split, k, "/", len(jobs), flush=True)
    # Многодикторный голос: дикторы 0..849 — обучение, 850..903 — тест.
    for name, (stem, phon, n_spk) in XFER.items():
        if a.only and name not in a.only.split(","):
            continue
        for split, scale, lo, hi in [("train", a.xfer_scale, 0, XFER_TEST_FROM), ("test", 0.2, XFER_TEST_FROM, n_spk)]:
            jobs = [(t, p, sd, lo, hi)
                    for (t, p, sd) in jobs_for(name if split == "train" else name + "test", split, a.out,
                                               int(a.pos * scale), int(a.neg_hard * scale), int(a.neg_sent * scale),
                                               2000 + (split == "test"))]
            init = (os.path.join(a.voices_dir, stem + ".onnx"), os.path.join(a.voices_dir, phon + ".onnx"))
            with Pool(a.workers, initializer=_init_xfer, initargs=init) as p:
                for k, _ in enumerate(p.imap_unordered(_synth_xfer, jobs, chunksize=16)):
                    if k % 1000 == 0:
                        print(name, split, k, "/", len(jobs), flush=True)
    print("done")


if __name__ == "__main__":
    main()
