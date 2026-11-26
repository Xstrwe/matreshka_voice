package wirepod_ttr

// Russian voice for the robot's own utterances (clock, weather, names, counting).
// The phrases are already built in Russian on the robot (libcozmo_engine, russianSpeech.h);
// vic-anim sends the text to GET /robot_tts?text=... and this handler only voices it with xAI TTS.
// No translation happens here. The reply is raw 16 kHz mono 16-bit PCM. Any error returns
// HTTP 500 and the robot falls back to its built-in voice.

import (
	"bytes"
	"context"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/fforchino/vector-go-sdk/pkg/vector"
	"github.com/fforchino/vector-go-sdk/pkg/vectorpb"
	"github.com/kercre123/wire-pod/chipper/pkg/logger"
)

// Acapela control tags such as \\pau=10\\ that the robot appends to its phrases.
var acapelaTag = regexp.MustCompile(`\\[a-zA-Z]+(=[^\\]*)?\\`)

func init() {
	http.HandleFunc("/robot_tts", robotTTSHandler)
}

func robotTTSHandler(w http.ResponseWriter, r *http.Request) {
	text := strings.TrimSpace(acapelaTag.ReplaceAllString(r.URL.Query().Get("text"), ""))
	if text == "" || len(text) > 6000 {
		http.Error(w, "bad text", http.StatusBadRequest)
		return
	}
	start := time.Now()
	pcm, err := xaiTTS(text)
	if err != nil || len(pcm) < 2 {
		logger.Println("robot_tts: tts failed")
		http.Error(w, "tts failed", http.StatusInternalServerError)
		return
	}
	pcm = gainSoftLimit(pcm[:len(pcm)-len(pcm)%2], xaiTTSGain())
	pcm = robotVoicePitch(pcm)
	logger.Println("robot_tts: \"" + text + "\" (" + time.Since(start).Round(time.Millisecond).String() + ")")
	w.Header().Set("Content-Type", "audio/L16;rate=16000;channels=1")
	w.Write(pcm)
}

// robotVoicePitch: if ./robot_voice_pitch exists (in the chipper folder) and holds a ratio such as
// 1.19 (about +3 semitones), the voice is pitched up with ffmpeg rubberband, keeping its tempo.
// The robot's own voice processing is applied on top afterwards. Any error keeps the original audio.
func robotVoicePitch(pcm []byte) []byte {
	b, err := os.ReadFile("./robot_voice_pitch")
	if err != nil {
		return pcm
	}
	ratio, err := strconv.ParseFloat(strings.TrimSpace(string(b)), 64)
	if err != nil || ratio < 0.5 || ratio > 2.0 || ratio == 1.0 {
		return pcm
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "ffmpeg", "-hide_banner", "-loglevel", "error",
		"-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
		"-af", "rubberband=pitch="+strconv.FormatFloat(ratio, 'f', 3, 64),
		"-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1")
	cmd.Stdin = bytes.NewReader(pcm)
	out, err := cmd.Output()
	if err != nil || len(out) < 2 {
		logger.Println("robot_tts: pitch shift failed, original voice")
		return pcm
	}
	return out[:len(out)-len(out)%2]
}

// sayViaRobotVoice: if ./robot_voice_ru exists (in the chipper folder), LLM answers are spoken
// through the robot's own TTS (SayText with the Vector voice). The robot then fetches the audio
// from /robot_tts and plays it through its stock voice processing, so the Russian voice sounds
// robotic like the original. Delete the file to go back to direct audio streaming.
func sayViaRobotVoice(robot *vector.Vector, input string) bool {
	if _, err := os.Stat("./robot_voice_ru"); err != nil {
		return false
	}
	_, err := robot.Conn.SayText(context.Background(), &vectorpb.SayTextRequest{
		Text:           input,
		UseVectorVoice: true,
		DurationScalar: 1.0,
	})
	if err != nil {
		logger.Println("robot_voice_ru: SayText failed, streaming instead: " + err.Error())
		return false
	}
	return true
}
