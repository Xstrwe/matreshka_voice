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
- mutes matreshka_voice while the robot speaks through its TTS and for 700 ms after it (and resets
  the engine), so the robot does not wake itself with its own voice. `TextToSpeechComponent`
  reports speech start and end through `SpeechRecognizerSystem::SetRobotSpeaking()`: start in
  `PostAudioEvent()` (normal speech: behaviors, SDK `SayText`, LLM answers), end in
  `OnUtteranceCompleted()` and `StopActiveTTS()`, plus the keyframe TTS callbacks
  `OnAudioPlaying()`/`OnAudioComplete()`. The keyframe callbacks alone are not enough: they do not
  fire for normal speech, and the robot then wakes itself and answers its own words. (The ALSA
  sink's "speaker in use" flag is not usable for this: it stays true all the time, and gating on it
  makes the wake word deaf.)
- turns off the stock English wake word engine when `disable_english` exists next to the model and
  the model loaded: the English recognizer is not created at all, so it uses no CPU or memory;
- adds an optional Russian voice for the robot's own phrases in `textToSpeechProvider_vicos.cpp`
  (enabled by a `tts_url` file, see `../wirepod/README.md`);
- after each listen (end of the cloud stream) keeps matreshka_voice muted for about 2 s more
  (`SpeechRecognizerSystem::NoteListenEnded()` from `MicDataSystem::ClearCurrentStreamingJob()`), so
  the end-of-listening sound and echo do not wake the robot again;
- logs the peak raw score once a second when it is at least 0.25
  (`SpeechRecognizerMatreshka.Peak`), to tune the threshold against real voices;
- the robot does not say a person's name when it recognizes their face
  (`TurnTowardsFaceAction::ShouldSayName` returns false); names are still said when asked
  («как меня зовут») and during face enrollment;
- makes the robot's own phrases Russian where they are built, in `vic-engine`
  (`libcozmo_engine.so`): `engine/components/russianSpeech.h` writes numbers, time and counts as
  Russian words with the right noun forms (for example «Сейчас час ноль пять», «двадцать один
  градус», «два градуса», «пять лет»); `LocaleComponent::GetString` returns built-in Russian text
  for every behavior and BlackJack string; `BehaviorWallTimeCoordinator` says the time in Russian.
  No translation step is involved;
- creates it in `InitVector()` only if the model file exists;
- feeds it in `Update()`;
- stops it in the destructor;
- links the `matreshka_voice` static library into `victor_anim`.

Regenerate the source lists and rebuild vic-anim with the normal WireOS build
(`./project/victor/build-victor.sh` or the Docker build used by WireOS).

Tested build: wire-os-victor `ebf38c28` (WireOS 3.0.2.36d), bare metal on an arm64 Raspberry Pi
(Debian 13), `./project/victor/scripts/victor_build_release.sh`, about 40 minutes with
`CMAKE_BUILD_PARALLEL_LEVEL=3`. The patch applies cleanly and the result is
`_build/vicos/Release/bin/vic-anim` (stripped, about 12 KB larger than stock). Build at the same
commit as the robot's `/anki/etc/revision` so the binary matches the shared libraries in `/anki/lib`.

## Files on the robot

```
/data/data/com.anki.victor/persistent/matreshka_voice/model.mvw   # models/vektor_ru.mvw
/data/data/com.anki.victor/persistent/matreshka_voice/threshold   # optional, e.g. 0.7
/data/data/com.anki.victor/persistent/matreshka_voice/disable_english   # optional, empty file
```

With `disable_english` present only «Вектор» wakes the robot; the English engine is not loaded.
The flag is ignored if the model fails to load, so the robot never ends up with no wake word.

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
