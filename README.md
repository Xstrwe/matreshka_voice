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
  real_takes.py               cut a recording of repeated wake words into takes, train/eval split
  eval_real.py                recall on held-out real takes + false accepts, through the C engine
models/vektor_ru.mvw        current model (80 KB), vektor_ru.h is the same blob as a C array
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

## Fine-tuning on real recordings

Real takes stay local: `.gitignore` excludes all audio, feature files and data folders, and only
model weights and aggregate metrics are committed.

```sh
python train/real_takes.py --wav rec_16k.wav --segments segments.json --out data/real/takes   # 70/30 split by region
python train/build_features.py --real data/real/takes/train --real-aug 150 --real-only --out data/real/feats_real.npz
python train/train.py --init runs/vektor --train feats/train.npz $(printf 'data/real/feats_real.npz %.0s' 1 2 3 4 5 6 7 8) \
    --val feats/test.npz --out runs/vektor_ft --epochs 12 --lr 1e-3 --neg-weight 4.0
python train/export.py --run runs/vektor_ft --out models/vektor_ru.mvw --header models/vektor_ru.h --threshold 0.7
python train/eval_real.py --model models/vektor_ru.mvw --cli engine/mv_cli --takes data/real/takes/eval \
    --neg-stream eval/neg_stream.wav --out eval_real
```

How the fine-tuning works:
- **Real windows:** each real take becomes 150 augmented windows (speed/pitch, gain, noise, reverb, EQ, and a far-field robot-mic simulation for half of them). It also yields truncated-prefix negatives.
- **Synthetic data:** the full synthetic set, including all hard negatives, stays in the mix. The real windows are oversampled 6–8×.
- **Real background:** windows cut from real room and TV audio with no wake word (random windows, hard negatives mined where the previous model fired, and synthetic words mixed over the room audio) teach the model what a home sounds like.
- **Starting point:** training starts from the synthetic model and keeps its normalisation.

## Results (`models/vektor_ru.mvw`)

The current model is the synthetic model fine-tuned on real recordings of one speaker: two phone recordings of «Вектор» (59 training takes: near and far, TV loud and quiet, spoken and whispered) and two home room recordings with no wake word (TV and conversation, 97 min used for training). The default threshold is **0.7**, chosen from real false triggers on held-out room audio. All numbers come from the C engine (`mv_cli`); details in [`docs/metrics_room.json`](docs/metrics_room.json) and [`docs/metrics_rec2.json`](docs/metrics_rec2.json). "Previous" is the model before the room recordings, at its default 0.85.

**Real false triggers, held-out room audio** (the last 30 % of each room recording: 18.1 min recorded near, 23.3 min far).

| model | threshold | near, per hour | far, per hour |
|---|---|---|---|
| previous | 0.85 | 255 | 64 |
| **current** | **0.7** | **3.3** (1 trigger) | **0** |

The current model keeps one trigger in the near stream for every threshold from 0.5 to 0.97.

**Second «Вектор» recording, 18 held-out takes in the real room** (real TV in the background). *Weak* takes are less than 6 dB above the background: far, whispered or masked by the TV.

| model | threshold | all | TV quiet | TV loud | strong | mid | weak |
|---|---|---|---|---|---|---|---|
| previous | 0.85 | 15/18 | 13/14 | 2/4 | 6/6 | 5/7 | 4/5 |
| **current** | **0.7** | **16/18** | **14/14** | **2/4** | **6/6** | **5/7** | **5/5** |

The same takes through the clean / noisy / far simulation: 141/198 (previous: 101/198).

**First «Вектор» recording, held-out takes** (7 spoken and 3 whispered, held out by region; noisy and far use 5 random variants per take). *Clean* keeps a quiet room floor 40 dB below the speech, *noisy* adds noise at 5–20 dB SNR, *far* runs the robot-mic simulation and adds noise at 15–30 dB. The last column is the 2-hour synthetic stream full of phonetically close words (Виктор, сектор, …).

| model | threshold | spoken clean | spoken noisy | spoken far | whispered clean | whispered noisy | whispered far | synthetic false accepts / hour |
|---|---|---|---|---|---|---|---|---|
| previous | 0.85 | 2/7 | 9/35 | 12/35 | 2/3 | 10/15 | 5/15 | 16 |
| **current** | **0.7** | **3/7** | **10/35** | **16/35** | **2/3** | **10/15** | **10/15** | **63** |
| current | 0.85 | 0/7 | 7/35 | 11/35 | 1/3 | 10/15 | 8/15 | 17 |

**Short TV clip.** On 12.6 s of held-out news audio from the second recording the current model fires 3 times at 0.7 (previous: 2 at 0.85).

**Synthetic test sets** at 0.7: held-out speakers 91.6 % clean, 85.0 % noisy; macOS Milena 29.2 % clean, 8.3 % noisy; 63 false accepts per hour on the synthetic stream. Earlier models: [`docs/metrics_ft2.json`](docs/metrics_ft2.json), [`docs/metrics_rec2.json`](docs/metrics_rec2.json).

Where this leaves the model:
- **Real rooms:** on held-out room audio from the same home, false triggers dropped from hundreds per hour to about 3 per hour near the microphone and none far away.
- **Limits:** all data comes from one speaker, one phone and one home; the held-out room audio was recorded on the same day as the training part. Phonetically close words still trigger more often than in everyday speech (63 per hour on the hard synthetic stream).
- **Next:** a supervised trial on the robot's own microphones, then recordings from other speakers.

Earlier checks, still valid for the engine:
- **C vs Python parity:** feature difference ≤ 4e-5, score difference ≤ 1e-6 (`make test`).
- **armv7:** the build runs under `qemu-arm -cpu cortex-a7` with identical detections.
- **Speed:** realtime factor 0.004 on one Apple-silicon core.
- **Synthetic-only model:** [`docs/metrics_v3.json`](docs/metrics_v3.json).
- **Wider-model experiment:** [`docs/metrics_v4_experiment.json`](docs/metrics_v4_experiment.json).

## Next steps

1. Record more real «Вектор» takes (several speakers, 1–3 m away, with TV or music on, ideally through Vector's own microphones), and 30–60 min of room/TV audio with no wake word. Fine-tune on them and set the threshold from real false accepts.
2. Add more native Russian speakers, such as more TTS engines or multi-speaker Russian voices.
3. Measure CPU on the robot, then build vic-anim with the WireOS patch (deployment is a separate, manual step; see `integration/wireos/README.md`).

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Xstrwe.
