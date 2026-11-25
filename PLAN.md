# matreshka_voice: plan

Goal: a license-free, on-device wake word engine for the Russian word «Вектор»
(stress on the first syllable, ВЕ́К-тор) that runs on Vector robots with WireOS,
plus a training pipeline that can produce models for other words.

## 1. Research summary

| Option | What it is | Fit for this project |
|---|---|---|
| WireOS stock trigger | vic-anim feeds 16 kHz mono int16 audio in 512-sample frames to a closed wake word library through `SpeechRecognizerPicovoice` ([source](https://github.com/os-vector/wire-os-victor/blob/main/animProcess/src/cozmoAnim/speechRecognizer/speechRecognizerPicovoice.cpp)). Custom models live in `/data/data/com.anki.victor/persistent/picovoice/`. | Closed binary format, English-only generator, words must be in its dictionary. Russian «Вектор» cannot be produced. |
| wire-pod custom wake words | [wire-pod#543](https://github.com/kercre123/wire-pod/issues/543): custom words are handled on the WireOS side (port 8080), not in wire-pod. | Same English-only generator. |
| [openWakeWord](https://github.com/dscripka/openWakeWord) | Apache-2.0, ONNX/TFLite models on top of a shared speech embedding model. | English pretrained models only; needs an ONNX or TFLite runtime plus ~1-2 MB of front-end models. A Russian TTS-trained setup exists ([interkelstar/openwakeword-trainer](https://github.com/interkelstar/openwakeword-trainer)), reported around 50 % real-voice recall. |
| [microWakeWord](https://github.com/kahrendt/microWakeWord) | Apache-2.0 streaming CNN for microcontrollers. | Good architecture, but needs TensorFlow for training and TFLite Micro on device. Russian community work: [interkelstar/microwakeword-trainer](https://github.com/interkelstar/microwakeword-trainer) (reported 94 % real recall for «Джарвис», 82.5 KB model) and [droman42/microwakeword-irina-ru](https://huggingface.co/droman42/microwakeword-irina-ru). |
| [Snowboy](https://github.com/Kitt-AI/snowboy) | Classic hotword engine. | Archived, model training service gone. |
| Vosk keyword grammar | Full ASR with a restricted grammar. | 50+ MB model and far more CPU than a wake word needs on a Cortex-A7. |

No existing replacement of the stock trigger inside WireOS was found. The useful lessons from
the projects above: TTS-only training works if speaker and acoustic variety is large, small
streaming CNNs on log-mel features are enough, and the hard part is false accepts.

Target: Vector's Qualcomm APQ8009 (4x Cortex-A7, NEON), WireOS userland is
`armv7a-linux-gnueabi` with `-mfloat-abi=softfp`. vic-anim is C++ and links static libraries,
so a plain C99 static library is the least invasive dependency.

## 2. Plan

1. **Feature front end (shared spec).** 16 kHz, 25 ms Hann window, 10 ms hop, 512-point FFT,
   40 triangular mel filters 20-7600 Hz, natural log with 1e-6 floor. Implemented once in numpy
   (`train/features.py`) and once in C (`engine/src/mv_features.c`), checked against each other.
2. **Model.** 1 s window (100 frames). Conv1d 40→48 (k=5) + three depthwise-separable blocks
   (48→48 stride 2, 48→64 stride 2, 64→64), global average pooling, linear 64→2. About 20 k
   parameters (80 KB as float32). BatchNorm is folded into the convolutions at export.
3. **Streaming inference.** The first convolution is computed once per new 10 ms frame and kept
   in a ring buffer; the rest of the network runs every 30 ms on the cached columns. Cost is
   about 9 k MAC per frame plus about 260 k MAC per network run, under 10 M MAC/s.
4. **Detection logic.** Average of the last 2 network scores ≥ threshold, then a 1.5 s
   refractory period. Threshold is stored in the model and can be overridden at run time.
5. **Training data (no recordings needed for a first model).**
   - Positives: «Вектор», «Вектор!», «Вектор?» … from Piper TTS voices (Russian: denis, dmitri,
     irina; Ukrainian voices lada, mykyta, tetiana, lada x_low for extra speaker variety),
     random length/noise scales, speed perturbation 0.85-1.2x, random placement in the window,
     30 % with speech before the word.
   - Negatives: phonetically close words (Виктор, сектор, лектор, доктор, Гектор, …), word
     prefixes («Век…»), generated and hand-written Russian sentences, synthetic noise.
   - Augmentation: reverb (synthetic RIRs), coloured noise and hum, babble from negative
     speech, SNR 0-30 dB, level -32 to -12 dBFS, SpecAugment-style masks.
6. **Evaluation.** Speaker held out from training (Piper ruslan) and an independent TTS engine
   (macOS «Milena») for recall; a 2-hour stream of held-out negative speech and noise for
   false accepts per hour. All streaming numbers come from the C engine itself.
7. **Export.** `.mvw` blob (68-byte header + float32 weights) and an optional C header with the
   blob as a byte array, so a model can be loaded from a file or compiled in.
8. **C engine.** `libmatreshka_voice.a`: features + FFT, network, detector, any input chunk
   size, no allocation after create. CLI `mv_cli` for WAV/raw files, score dumps and timing.
   Parity test against the numpy reference; chunk-size invariance test.
9. **WireOS integration.** `SpeechRecognizerMatreshka` (an `AudioUtil::SpeechRecognizer`)
   running next to the stock trigger in `SpeechRecognizerSystem::InitVector`, active only when
   `/data/data/com.anki.victor/persistent/matreshka_voice/model.mvw` exists; optional threshold
   file next to it. It reports the same `HEY_VECTOR` result, so the rest of the pipeline is
   unchanged.
10. **Build.** Engine: `make armv7` (static, Cortex-A7/NEON). vic-anim: add the engine as
    `3rd/matreshka_voice` + `cmake/matreshka_voice.cmake`, apply `integration/wireos/wireos.patch`,
    rebuild vic-anim with the normal WireOS toolchain.
11. **Deploy and roll back** (separate step, needs approval): back up `/anki/bin/vic-anim`,
    install the new binary and the model, restart `vic-anim`. Roll back by deleting the model
    file (instant, the stock trigger keeps working) or restoring the backed-up binary.
12. **Risks.**
    - TTS-only training generalises imperfectly to real voices and to Vector's own microphones
      and beamformer. Mitigation: record 50-200 real «Вектор» samples on the robot
      (`kSaveRawMicInput` path) and fine-tune; tune the threshold on real background audio.
    - False accepts on TV/speech: measure on long real recordings before raising sensitivity.
    - vic-anim CPU budget: measured cost is small (see README), but must be confirmed on device.
    - The WireOS VAD gate means the engine only sees audio while VAD is active, like the stock
      trigger; very short or quiet calls may be missed.
