package wirepod_ttr

// Russian voice for the robot's own utterances (clock, weather, names, counting).
// The phrases are already built in Russian on the robot (libcozmo_engine, russianSpeech.h);
// vic-anim sends the text to GET /robot_tts?text=... and this handler only voices it with xAI TTS.
// No translation happens here. The reply is raw 16 kHz mono 16-bit PCM. Any error returns
// HTTP 500 and the robot falls back to its built-in voice.

import (
	"context"
	"net/http"
	"os"
	"regexp"
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
	logger.Println("robot_tts: \"" + text + "\" (" + time.Since(start).Round(time.Millisecond).String() + ")")
	w.Header().Set("Content-Type", "audio/L16;rate=16000;channels=1")
	w.Write(pcm)
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
