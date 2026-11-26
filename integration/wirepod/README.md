# wire-pod side: Russian voice for the robot

These files go into a wire-pod checkout (`chipper/pkg/wirepod/ttr/`) that already has the xAI TTS
provider (`tts_xai.go`).

| File | What it does |
|---|---|
| `robot_tts_ru.go` | Adds `GET /robot_tts?text=...` on the wire-pod web port (8080). The text is cleaned of Acapela tags, voiced with xAI TTS as is and returned as raw 16 kHz mono 16-bit PCM. There is no translation: the robot already builds its phrases in Russian (see `../wireos`). |
| `russian_strings.patch` | Native Russian text for the phrases wire-pod itself speaks (LLM error, weather not configured, face enrolment by voice, knowledge graph errors), so the robot never says English. Apply in the wire-pod repo root: `git apply russian_strings.patch`. |
| `followup_once.patch` | Follow-up listening after an LLM answer listens once: silence closes the window instead of pressing the virtual button again every 1.5 s for 20 s. |
| `ru_intents.patch` | More Russian keyphrases for Vector's own intents (time, weather, age, names, greetings, sleep, explore, come here, photo), so these requests run the robot's native behaviors and only unmatched speech goes to the LLM. |
| `tts_xai.patch` | One hook in `DoSayText_XAI`: if `chipper/robot_voice_ru` exists, LLM answers are spoken with the robot's own `SayText` (Vector voice) instead of being streamed as raw audio. |

## How the pieces connect

1. vic-anim (patched, see `../wireos`) reads `/data/data/com.anki.victor/persistent/matreshka_voice/tts_url`
   (for example `http://192.168.8.127:8080/robot_tts`). If the file exists, every phrase the robot
   would speak with its Acapela voice (clock, weather, names, counting, SDK `SayText`) is fetched
   from wire-pod instead. The text is already Russian, built in `libcozmo_engine.so`. On any error or timeout it falls back to the stock voice.
2. The fetched audio goes through the stock TTS chain, so the usual Wwise processing
   (`Robot_Vic_External_Processing = Default_Processed`) makes it sound robotic, like the original voice.
3. With `robot_voice_ru` present, LLM answers take the same path and get the same effect.

## Switches

| Switch | On | Off |
|---|---|---|
| Raised voice pitch | `echo 1.19 > /opt/wire-pod/chipper/robot_voice_pitch` (ratio, 1.19 is about +3 semitones; needs `ffmpeg` with `rubberband`) | `rm /opt/wire-pod/chipper/robot_voice_pitch` (no restart needed) |
| Russian robot phrases | `echo http://<wire-pod-ip>:8080/robot_tts > .../matreshka_voice/tts_url` | `rm .../matreshka_voice/tts_url` (no restart needed) |
| LLM answers through the robot voice | `touch /opt/wire-pod/chipper/robot_voice_ru` | `rm /opt/wire-pod/chipper/robot_voice_ru` (no restart needed) |

## Notes

- The endpoint has no authentication and uses the wire-pod TTS key, so keep wire-pod on a
  trusted network.
- Phrases (including names the robot says) are sent to the configured TTS service.
- A phrase takes about 0.7-1 s to voice.

## Routing

wire-pod matches the transcript against `intent-data/ru-RU.json` first (exact match, then substring
match in file order); only text that matches nothing goes to the LLM (`knowledge.intentgraph`).
Native intents run Vector's own behaviors and speak the built-in Russian phrases.

The LLM prompt (`knowledge.openai_prompt` in `apiConfig.json`, not in this repo) asks for Russian,
short spoken answers and tells the model never to address the user by name.
