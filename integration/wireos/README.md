# WireOS integration notes

matreshka_voice works with Vector robots running WireOS. The engine runs inside `vic-anim`
next to the stock wake word recognizer and reports the same `HEY_VECTOR` result, so the rest
of the voice pipeline (intent capture, wire-pod) does not change.

## Where the audio comes from

`SpeechRecognizerSystem::Update()` (`animProcess/src/cozmoAnim/speechRecognizer/speechRecognizerSystem.cpp`)
receives processed microphone audio: 16 kHz, mono, int16, with a VAD flag. The stock trigger
gets it only while VAD is active. `SpeechRecognizerMatreshka` gets exactly the same samples.

## Files

| File | Goes to (in wire-os-victor) |
|---|---|
| `speechRecognizerMatreshka.h/.cpp` | `animProcess/src/cozmoAnim/speechRecognizer/` (picked up by the `src/cozmoAnim` source glob in `animProcess/BUILD.in`) |
| `../../engine/include`, `../../engine/src` | `3rd/matreshka_voice/include`, `3rd/matreshka_voice/src` |
| `matreshka_voice.cmake` | `cmake/matreshka_voice.cmake` |
| `wireos.patch` | apply from the repo root: `patch -p1 < wireos.patch` |

The patch:
- adds a `_matreshkaTrigger` member;
- creates it in `InitVector()` only if the model file exists;
- feeds it in `Update()`;
- stops it in the destructor;
- links the `matreshka_voice` static library into `victor_anim`.

Regenerate the source lists and rebuild vic-anim with the normal WireOS build
(`./project/victor/build-victor.sh` or the Docker build used by WireOS).

The wrapper compiles cleanly against the WireOS `audioUtil/speechRecognizer.h` header
(checked with `g++ -std=c++14 -fsyntax-only`). A full vic-anim build has not been run yet.

## Files on the robot

```
/data/data/com.anki.victor/persistent/matreshka_voice/model.mvw   # models/vektor_ru.mvw
/data/data/com.anki.victor/persistent/matreshka_voice/threshold   # optional, e.g. 0.85
```

## Deploy (separate, manual step)

1. Back up `/anki/bin/vic-anim` (e.g. to `/data/vic-anim.bak`).
2. Copy the new `vic-anim`, then `mkdir -p` the folder above and copy `model.mvw`.
3. `systemctl restart vic-anim` and watch `logcat | grep SpeechRecognizerMatreshka`.

## Roll back

- Fastest: `rm /data/data/com.anki.victor/persistent/matreshka_voice/model.mvw && systemctl restart vic-anim`.
  The patched binary then behaves like the stock one.
- Full: restore the backed-up `vic-anim` binary and restart.
