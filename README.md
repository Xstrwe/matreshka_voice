# matreshka_voice

matreshka_voice is a small, open-source wake word engine that runs on the device.

- **Engine:** plain C99 with no dependencies beyond libc and libm. It includes the log-mel front end, a small streaming CNN and the detection logic.
- **Training:** a Python pipeline that trains a model from synthetic speech alone.
- **First model:** the Russian word «Вектор» (ВЕ́К-тор).
- **Compatibility:** works with Vector robots running WireOS. Integration notes are in [`integration/wireos`](integration/wireos).

Author: Xstrwe. License: MIT.

## Layout

```
engine/             C99 library, CLI, tests, Makefile and CMakeLists.txt
  include/matreshka_voice.h   public API and model blob format
  src/                        features + FFT, network, streaming detector
  cli/mv_cli.c                test tool for WAV/raw files
  tests/                      parity test (C vs numpy) and chunking test
train/              Python training pipeline
  texts.py                    positive, hard negative and sentence texts
  gen_data.py                 Piper TTS data generation
  gen_say_macos.sh            independent test set from macOS «Milena»
  features.py                 reference log-mel (same math as the C code)
  augment.py                  reverb, noise, EQ, speed, placement
  build_features.py           augmented 1 s windows -> features
  model.py / train.py         model and training loop (PyTorch)
  export.py                   .mvw blob + C header export (BatchNorm folded)
  mvw_ref.py                  numpy reference for a .mvw model
  eval_stream.py              streaming evaluation through the C engine
models/vektor_ru.mvw        first model (80 KB), vektor_ru.h is the same blob as a C array
integration/wireos/         vic-anim recognizer wrapper, patch, CMake snippet, deploy notes
docs/                       metrics and training logs
PLAN.md                     research summary and plan
```

## Engine

```c
#include "matreshka_voice.h"

mv_config cfg = mv_default_config();          /* threshold from the model, 30 ms hop, 1.5 s refractory */
mv_engine *e = mv_create_from_file("vektor_ru.mvw", &cfg);
mv_detection d[4];
int n = mv_process(e, pcm, n_samples, d, 4);  /* 16 kHz mono int16, any chunk size */
for (int i = 0; i < n; i++) printf("wake word at sample %llu, score %.2f\n",
                                   (unsigned long long)d[i].end_sample, d[i].score);
mv_destroy(e);
```

How it works:
- **Features:** 40 log-mel bins, 25 ms window, 10 ms hop, 512-point FFT.
- **Network:** Conv1d 40→48 (k=5), then three depthwise-separable blocks (48→48 /2, 48→64 /2, 64→64), global average pooling and a linear layer to 2 outputs. That is 20.5 k parameters over a 1 s window.
- **Streaming:** the first convolution runs once per 10 ms frame and is cached. The rest of the network runs every 30 ms.
- **Detection:** the mean of the last 2 scores must reach the threshold, followed by a 1.5 s refractory period.

Memory is allocated once, in `mv_create`, at about 220 KB including the weights.

```sh
cd engine
make                      # host build: libmatreshka_voice.a + mv_cli
make test                 # CLI smoke test, C/numpy parity, chunk-size invariance
make armv7                # static Cortex-A7/NEON build (arm-linux-gnueabi-gcc, softfp)
./mv_cli ../models/vektor_ru.mvw input.wav [-t 0.9] [--scores s.txt] [--features f.bin]
```

## Training

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install torch numpy scipy soundfile piper-tts
# Piper voices from https://huggingface.co/rhasspy/piper-voices into ./voices:
#   ru_RU-{denis,dmitri,irina,ruslan}-medium, uk_UA-ukrainian_tts-medium, uk_UA-lada-x_low,
#   en_US-libritts_r-medium (904 speakers, driven with Russian espeak phonemes)
cd train
python gen_data.py --voices-dir ../voices --out ../data
sh gen_say_macos.sh ../data/say                     # macOS only, independent test set
python build_features.py --data ../data --split test --out ../feats/test.npz --pos-aug 2 --neg-aug 2 --noise 1000 --seed 7
python build_features.py --data ../data --split train --out ../feats/train.npz --pos-aug 3 --neg-aug 3 --noise 10000
python train.py --train ../feats/train.npz --val ../feats/test.npz --out ../runs/vektor --epochs 30 --neg-weight 1.5
cd .. && python train/export.py --run runs/vektor --out models/vektor_ru.mvw --header models/vektor_ru.h --threshold 0.9
python train/eval_stream.py --model models/vektor_ru.mvw --cli engine/mv_cli --data data --out eval
```

On an Apple-silicon Mac mini, the full run takes about 40 minutes: data generation ~32 min, features ~4 min and training ~2.5 min on MPS.

The data has no recordings:
- **Positives:** «Вектор», «Вектор!», «Вектор?»
- **Hard negatives:** Виктор, сектор, лектор, доктор, Гектор, word prefixes and similar.
- **Other negatives:** Russian sentences and synthetic noise.

These come from 7 Piper voices (Russian and Ukrainian), plus 850 speakers of a multi-speaker voice driven with Russian phonemes. Augmentation adds reverb, noise and hum at 0–30 dB SNR, random EQ and band limits, speed changes of 0.85–1.2×, random level and placement, and SpecAugment masks.

## Results of the first model (`models/vektor_ru.mvw`)

All streaming numbers come from the C engine (`mv_cli`) running over long generated recordings ([`train/eval_stream.py`](train/eval_stream.py), raw data in [`docs/metrics_v3.json`](docs/metrics_v3.json)):

- **Recall:** each wake word sits between 1–2 s gaps, and half of them have speech before them. Recall counts a wake word as found when a detection lands within -0.4/+1.0 s of its end.
- **False accepts:** measured on a 2-hour stream of held-out negative speech with noise. The stream includes many phonetically close words, so it is harder than everyday audio.

| threshold | held-out speakers, clean | held-out speakers, noisy | macOS Milena (independent TTS), clean | Milena, noisy | false accepts / hour |
|---|---|---|---|---|---|
| 0.5 | 93.4 % | 93.2 % | 56.8 % | 19.8 % | 295 |
| 0.8 | 93.8 % | 88.2 % | 19.8 % | 4.7 % | 87 |
| **0.9 (default)** | **89.0 %** | **77.2 %** | **12.0 %** | **1.0 %** | **37** |
| 0.95 | 80.4 % | 58.8 % | 4.7 % | 0.5 % | 13.5 |
| 0.98 | 56.6 % | 30.8 % | 2.6 % | 0.5 % | 2.5 |

Held-out speakers means 54 speakers of the multi-speaker voice that were never used in training, 500 words per condition. The Milena set is 24 clips reused 8× with new placement and noise, so 192 words per condition. In the noisy condition, noise sits roughly 5–20 dB below the speech level.

What these numbers mean:
- **Held-out speakers:** the model works well on unseen speakers from voices close to the training data.
- **Different engine:** it generalises poorly to a different TTS engine (Milena).
- **False accepts:** the rate is far too high for always-on use.
- **Real speech:** the first model is a working end-to-end baseline, not yet a usable wake word for real speech. The main next step is real recordings (see PLAN.md, risks).

Other checks:
- **C vs Python parity:** feature difference ≤ 4e-5, score difference ≤ 1e-6 (`make test`).
- **Exported blob vs PyTorch:** score difference ≤ 9e-7.
- **armv7:** the static `make armv7` build runs under `qemu-arm -cpu cortex-a7` with identical detections; score difference vs x86-64 ≤ 1e-6.
- **Speed:** realtime factor 0.004 on one Apple-silicon core and 0.06 under qemu emulation. No measurement on a real Cortex-A7 yet. The estimate is about 9 M MAC/s.
- **Earlier runs:**
  - Three Russian voices only: the model reached under 10 % recall on an unseen voice. Adding the multi-speaker data fixed that for held-out speakers.
  - A wider model (64/64/96/96) trained with the extra Russian voice scored lower false accepts (23/h at 0.9) but did worse on Milena; see [`docs/metrics_v4_experiment.json`](docs/metrics_v4_experiment.json).

## Next steps

1. Record 100–300 real «Вектор» samples and an hour of room and TV audio through Vector's microphones. Fine-tune on them and set the threshold from real false accepts.
2. Add more native Russian speakers, such as more TTS engines or multi-speaker Russian voices.
3. Measure CPU on the robot, then build vic-anim with the WireOS patch (deployment is a separate, manual step; see `integration/wireos/README.md`).

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Xstrwe.
