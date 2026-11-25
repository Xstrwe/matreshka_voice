# wire-pod side: Russian voice for the robot

These files go into a wire-pod checkout (`chipper/pkg/wirepod/ttr/`) that already has the xAI TTS
provider (`tts_xai.go`).

| File | What it does |
|---|---|
| `robot_tts_ru.go` | Adds `GET /robot_tts?text=...` on the wire-pod web port (8080). The text is cleaned of Acapela tags, translated to Russian with the configured LLM (skipped if it already has Cyrillic; results are cached), voiced with xAI TTS and returned as raw 16 kHz mono 16-bit PCM. |
| `tts_xai.patch` | One hook in `DoSayText_XAI`: if `chipper/robot_voice_ru` exists, LLM answers are spoken with the robot's own `SayText` (Vector voice) instead of being streamed as raw audio. |

## How the pieces connect

1. vic-anim (patched, see `../wireos`) reads `/data/data/com.anki.victor/persistent/matreshka_voice/tts_url`
   (for example `http://192.168.8.127:8080/robot_tts`). If the file exists, every phrase the robot
   would speak with its English Acapela voice (clock, weather, names, counting, SDK `SayText`) is
   fetched from wire-pod instead. On any error or timeout it falls back to the stock voice.
2. The fetched audio goes through the stock TTS chain, so the usual Wwise processing
   (`Robot_Vic_External_Processing = Default_Processed`) makes it sound robotic, like the original voice.
3. With `robot_voice_ru` present, LLM answers take the same path and get the same effect.

## Switches

| Switch | On | Off |
|---|---|---|
| Russian robot phrases | `echo http://<wire-pod-ip>:8080/robot_tts > .../matreshka_voice/tts_url` | `rm .../matreshka_voice/tts_url` (no restart needed) |
| LLM answers through the robot voice | `touch /opt/wire-pod/chipper/robot_voice_ru` | `rm /opt/wire-pod/chipper/robot_voice_ru` (no restart needed) |

## Notes

- The endpoint has no authentication and uses the wire-pod LLM and TTS keys, so keep wire-pod on a
  trusted network.
- Phrases (including names the robot says) are sent to the configured LLM and TTS services.
- A new phrase takes about 1-2 s (translation and TTS); repeated phrases skip the translation.
