# Making matreshka_voice the only wake word engine on Vector

## What runs today (WireOS 3.0.2.36d, `integration/wireos`)

| Stage | Today | Owner |
|---|---|---|
| Microphones, beamforming, noise reduction, VAD | stock `MicDataProcessor` (Signal Essence) | WireOS |
| Wake word | **matreshka_voice** (`SpeechRecognizerMatreshka`), threshold file, mute while the robot speaks and ~2 s after a listen | this repo |
| Stock English wake word | not created when `disable_english` exists and the Russian model loaded | WireOS code, unused |
| Wake reaction (earcon, get-in animation, backpack lights) | stock trigger response | WireOS |
| Streaming the request | stock `MicDataSystem` → vic-cloud → wire-pod | WireOS / wire-pod |
| Speech to text | Vosk ru-RU in wire-pod | wire-pod |
| Command matching | `intent-data/ru-RU.json`, native behaviors first, LLM only for unmatched text | wire-pod |
| Robot phrases | built-in Russian text in `libcozmo_engine.so`, voiced by `/robot_tts` (xAI voice, raised pitch, robot voice processing) | this repo / wire-pod |

So the wake word is already ours; what is still stock is the framework around it: the
`SpeechRecognizerSystem` class that owns the English recognizer, the locale-based model switching,
the Alexa recognizers, and the code path that would fall back to the English model if the Russian
model fails to load.

## Full replacement: what it means technically

1. **Remove the English engine from the build.** Delete the stock English recognizer, its model files and
   its library from `animProcess` and the CMake files. `SpeechRecognizerSystem::InitVector()` creates
   only `SpeechRecognizerMatreshka`; `UpdateTriggerForLocale()` no longer swaps models.
   The `disable_english` flag and the English fallback disappear.
2. **matreshka_voice becomes `_victorTrigger`.** One recognizer member, one callback, one place for
   the mute/refractory logic, which moves into the engine itself (`mv_set_mute`, `mv_set_refractory`)
   instead of `SpeechRecognizerSystem` statics.
3. **Feed it the same processed beam** as today, but without the stock VAD gate: the engine has its
   own energy gate, so it is not blind for the first frames of quiet speech.
4. **Robot-side safety without English:** if the model fails to load, log an error, show it on the
   face info screen and keep the backpack button working (the button path does not use the wake word).
5. **Optional, larger step:** a small on-device Russian command model in C for the most common
   commands (time, weather, volume, sleep, come here), so they work without Vosk or wire-pod. This
   needs recorded and labelled training data and its own evaluation; it is a separate project.

## Risks

- **No fallback wake word.** A broken or missing model leaves the robot deaf to voice (the button
  still works). Mitigation: verify the model checksum at start, keep the last good model next to it.
- **False wakes and misses need tuning on real rooms.** The peak-score log is there for that; the
  threshold is a file, so it can change without a rebuild.
- **Build and update cost.** A full WireOS build takes about 40 minutes on the Pi, and every WireOS
  update means re-applying the patch. Removing stock code makes the patch larger and conflicts more
  likely.
- **Shared libraries.** `vic-anim` and `libcozmo_engine.so` must come from the same WireOS revision
  as the rest of `/anki`, or the robot faults on start.
- **Restarts.** Two `anki-robot.target` restarts within about 20 s can hang vic-anim (fault 914); the
  install scripts restart once and wait.

## Status

Steps 1-3 are built (`vic-anim.full`, staged on the Pi as `install_next.sh full`): the English engine
is gone from the build, matreshka_voice is the only recognizer, the mute and the post-listen pause
live in `SpeechRecognizerMatreshka`, and the stock VAD gate is off by default (`use_stock_vad`
brings it back without a rebuild). The mute stays in the C++ recognizer rather than the C engine,
so the C API does not change. Not yet tested on the robot.

## Order

1. Install the staged v8 (`vic-anim`) and the engine library, check for a day.
2. Remove the English engine from the build (step 1-2 above), one build, one install, 90 s check.
3. Move mute/refractory into the engine and drop the VAD gate (step 2-3), tune the threshold.
4. Decide on the on-device command model separately.
