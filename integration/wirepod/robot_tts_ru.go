package wirepod_ttr

// Russian voice for the robot's own utterances (clock, weather, names, counting).
// vic-anim sends the English text it would speak with Acapela to GET /robot_tts?text=...;
// the phrase is translated to Russian with the configured LLM (cached) and voiced with xAI TTS.
// The reply is raw 16 kHz mono 16-bit PCM. Any error returns HTTP 500 and the robot falls back
// to its stock English voice.

import (
	"context"
	"os"
	"regexp"
	"net/http"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/fforchino/vector-go-sdk/pkg/vector"
	"github.com/fforchino/vector-go-sdk/pkg/vectorpb"
	"github.com/kercre123/wire-pod/chipper/pkg/logger"
	"github.com/kercre123/wire-pod/chipper/pkg/vars"
	"github.com/sashabaranov/go-openai"
)

// Acapela control tags such as \\pau=10\\ that the robot appends to its phrases.
var acapelaTag = regexp.MustCompile(`\\[a-zA-Z]+(=[^\\]*)?\\`)

var (
	robotTTSCacheMu sync.Mutex
	robotTTSCache   = map[string]string{}
)

func init() {
	http.HandleFunc("/robot_tts", robotTTSHandler)
}

func hasCyrillic(s string) bool {
	for _, r := range s {
		if unicode.Is(unicode.Cyrillic, r) {
			return true
		}
	}
	return false
}

func hasLetters(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) {
			return true
		}
	}
	return false
}

// translateRobotPhrase turns a short English robot phrase into Russian.
func translateRobotPhrase(text string) (string, error) {
	if hasCyrillic(text) || !hasLetters(text) {
		return text, nil
	}
	robotTTSCacheMu.Lock()
	if t, ok := robotTTSCache[text]; ok {
		robotTTSCacheMu.Unlock()
		return t, nil
	}
	robotTTSCacheMu.Unlock()

	conf := openai.DefaultConfig(vars.APIConfig.Knowledge.Key)
	conf.BaseURL = vars.APIConfig.Knowledge.Endpoint
	c := openai.NewClientWithConfig(conf)
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	resp, err := c.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:       vars.APIConfig.Knowledge.Model,
		Temperature: 0,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: "Переведи на русский короткую фразу, которую вслух произносит маленький робот. " +
				"Слова о погоде переводи по смыслу (Sunny -> Солнечно, Cloudy -> Облачно, Rain -> Дождь). " +
				"Время, числа, градусы и единицы пиши словами так, как их произносят по-русски (12:45 -> двенадцать сорок пять). " +
				"Если вся фраза - только имя человека (например Aletta! или Bob?), запиши это имя кириллицей с тем же знаком. " +
				"Ответь только переводом, без кавычек и пояснений."},
			{Role: openai.ChatMessageRoleUser, Content: text},
		},
	})
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", context.DeadlineExceeded
	}
	out := strings.TrimSpace(resp.Choices[0].Message.Content)
	if out == "" {
		return "", context.DeadlineExceeded
	}
	robotTTSCacheMu.Lock()
	if len(robotTTSCache) > 500 {
		robotTTSCache = map[string]string{}
	}
	robotTTSCache[text] = out
	robotTTSCacheMu.Unlock()
	return out, nil
}

func robotTTSHandler(w http.ResponseWriter, r *http.Request) {
	text := strings.TrimSpace(acapelaTag.ReplaceAllString(r.URL.Query().Get("text"), ""))
	if text == "" || len(text) > 6000 {
		http.Error(w, "bad text", http.StatusBadRequest)
		return
	}
	start := time.Now()
	ru, err := translateRobotPhrase(text)
	if err != nil {
		logger.Println("robot_tts: translate failed: " + err.Error())
		http.Error(w, "translate failed", http.StatusInternalServerError)
		return
	}
	pcm, err := xaiTTS(ru)
	if err != nil || len(pcm) < 2 {
		logger.Println("robot_tts: tts failed")
		http.Error(w, "tts failed", http.StatusInternalServerError)
		return
	}
	pcm = gainSoftLimit(pcm[:len(pcm)-len(pcm)%2], xaiTTSGain())
	logger.Println("robot_tts: \"" + text + "\" -> \"" + ru + "\" (" + time.Since(start).Round(time.Millisecond).String() + ")")
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
