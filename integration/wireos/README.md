# WireOS integration notes

matreshka_voice is the only wake word engine in `vic-anim` on WireOS. The stock English wake word
engine is removed from the build (its sources, its library and the model switching by language).
The engine reports the same `HEY_VECTOR` result, so the rest of the voice pipeline (streaming the
request, wire-pod) does not change. The backpack button still starts listening without the wake
word, also when the model is missing or broken.

## Where the audio comes from

`SpeechRecognizerSystem::Update()` (`animProcess/src/cozmoAnim/speechRecognizer/speechRecognizerSystem.cpp`)
receives processed microphone audio: 16 kHz, mono, int16, with a VAD flag. By default all of it goes
to `SpeechRecognizerMatreshka`; with a `use_stock_vad` file next to the model only the audio while
the stock VAD is active does (lower CPU, slower on quiet speech).

## Files

| File | Goes to (in wire-os-victor) |
|---|---|
| `speechRecognizerMatreshka.h/.cpp` | `animProcess/src/cozmoAnim/speechRecognizer/` (picked up by the `src/cozmoAnim` source glob in `animProcess/BUILD.in`) |
| `../../engine/include`, `../../engine/src` | `3rd/matreshka_voice/include`, `3rd/matreshka_voice/src` |
| `matreshka_voice.cmake` | `cmake/matreshka_voice.cmake` |
| `wireos.patch` | apply from the repo root: `patch -p1 < wireos.patch` (it also deletes the stock English recognizer sources) |

The patch:
- `SpeechRecognizerSystem` owns one recognizer, `SpeechRecognizerMatreshka`. If the model is
  missing or does not load, it logs an error and there is no wake word (no fallback to another
  language); the backpack button keeps working. The language setting no longer changes the model;
- the «Alexa» wake word ran on the removed engine, so the Alexa wake word methods are empty and the
  Alexa playback recognizer is not created;
- mute inside `SpeechRecognizerMatreshka`: no audio is processed (and the engine is reset) while the
  robot speaks and 700 ms after, and for about 2 s after each listen ends, so the robot does not
  wake itself with its own voice or the end-of-listening sound. `TextToSpeechComponent` reports
  speech start in `PostAudioEvent()` (behaviors, SDK `SayText`, LLM answers) and end in
  `OnUtteranceCompleted()`/`StopActiveTTS()` (plus the keyframe TTS callbacks);
  `MicDataSystem::ClearCurrentStreamingJob()` reports the end of a listen. (The ALSA sink's
  "speaker in use" flag is not usable: it stays true all the time);
- logs the peak raw score once a second when it is at least 0.25
  (`SpeechRecognizerMatreshka.Peak`), to tune the threshold against real voices;
- adds the Russian voice for the robot's own phrases in `textToSpeechProvider_vicos.cpp`
  (enabled by a `tts_url` file, see `../wirepod/README.md`);
- in `vic-engine` (`libcozmo_engine.so`): the robot's own phrases are built in Russian
  (`engine/components/russianSpeech.h` writes numbers, time and counts as Russian words with the
  right noun forms, for example «Сейчас час ноль пять», «двадцать один градус», «два градуса»,
  «пять лет»; `LocaleComponent::GetString` returns built-in Russian text for every behavior and
  BlackJack string; `BehaviorWallTimeCoordinator` says the time in Russian; no translation step);
  the robot does not say a person's name when it recognizes their face
  (`TurnTowardsFaceAction::ShouldSayName`), only when asked («как меня зовут») or while enrolling;
- links the `matreshka_voice` static library into `victor_anim` instead of the stock one.

Regenerate the source lists and rebuild vic-anim with the normal WireOS build
(`./project/victor/build-victor.sh` or the Docker build used by WireOS).

Tested build: wire-os-victor `ebf38c28` (WireOS 3.0.2.36d), bare metal on an arm64 Raspberry Pi
(Debian 13), `./project/victor/scripts/victor_build_release.sh`, about 40 minutes with
`CMAKE_BUILD_PARALLEL_LEVEL=3`. The patch applies cleanly and the result is
`_build/vicos/Release/bin/vic-anim` (stripped, about 80 KB smaller than the earlier build that carried both engines). Build at the same
commit as the robot's `/anki/etc/revision` so the binary matches the shared libraries in `/anki/lib`.

## Files on the robot

```
/data/data/com.anki.victor/persistent/matreshka_voice/model.mvw   # models/vektor_ru.mvw
/data/data/com.anki.victor/persistent/matreshka_voice/threshold   # optional, e.g. 0.7
/data/data/com.anki.victor/persistent/matreshka_voice/use_stock_vad     # optional, empty file
```

Only «Вектор» wakes the robot; there is no other wake word in the build. Without a loadable
`model.mvw` there is no wake word at all (the log says `SpeechRecognizerSystem.InitVector.NoModel` or
`.ModelNotLoaded`); the backpack button still starts listening. A `disable_english` file from older
builds is ignored.

## Deploy (separate, manual step)

1. Back up `/anki/bin/vic-anim` (e.g. to `/data/vic-anim.bak`) and check its sha256.
2. `mkdir -p` the folder above (owner `anki:anki`, mode 770) and copy `model.mvw` and `threshold`.
3. `mount -o rw,remount /`, copy the new `vic-anim` (owner `engine:anki`, mode 500).
4. `systemctl restart anki-robot.target`. `vic-anim.service` has `Restart=no` and a fault code on
   exit, so restart the whole target rather than only vic-anim.
5. Remount `/` read-only only after the restart: while the old process still runs from the
   replaced file, `mount -o ro,remount /` fails with "mount point is busy".
6. The Russian phrases live in `/anki/lib/libcozmo_engine.so` (`_build/vicos/Release/lib/`, built
   with `ninja cozmo_engine`). Back it up, copy it the same way (owner `anki:anki`, mode 440) and
   restart the target once. Do not restart the target twice within about 20 s: vic-anim can hang
   on shutdown during start-up and the robot shows fault 914.

## Logs

Logs go to `journalctl`. Release builds disable the `SpeechRecognizer` log channel, so the init and
detection lines are hidden by default. Enable it at runtime (it resets on the next restart):

```
curl -s "http://127.0.0.1:8889/consolevarset?key=SpeechRecognizer&value=1"
journalctl -f | grep SpeechRecognizerMatreshka
```

A missing or broken model is logged as a warning (`SpeechRecognizerMatreshka.Init`).

## CPU

Measure CPU in the same robot state before and after. Right after a restart the robot is awake and
vic-anim uses about twice as much CPU as when it is idle on the charger, even with the stock
binary. On WireOS 3.0.2.36d, measured over 60 s windows from `/proc/<pid>/stat` after a restart:
stock vic-anim 49-54% of one core, with matreshka_voice 54-60% (about +6 points).

## Roll back

Restore the backed-up `vic-anim` and/or `libcozmo_engine.so` (same owner and mode), then restart
`anki-robot.target` once. Keep a matreshka_voice build as the `vic-anim` backup if the robot
should stay Russian-only: the stock binary brings back the English wake word.
