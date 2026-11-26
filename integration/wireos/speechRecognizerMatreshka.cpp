// Распознаватель слова активации matreshka_voice для WireOS (процесс vic-anim).
// Copyright (c) 2026 Xstrwe. Лицензия MIT.
#include "speechRecognizerMatreshka.h"

#include "util/logging/logging.h"

// Движок написан на C. extern "C" говорит компилятору C++ не «украшать» имена
// функций, иначе компоновщик не найдёт mv_create, mv_process и т. д.
extern "C" {
#include "matreshka_voice.h"
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <sstream>
#include <sys/stat.h>

namespace Anki {
namespace Vector {

#define LOG_CHANNEL "SpeechRecognizer"

// Длина слова, о которой сообщаем WireOS: 800 мс звука до момента срабатывания
// (16 отсчётов на миллисекунду при 16 кГц).
static const uint64_t kKeywordLengthSamples = 16 * 800;

// Глушение: общее для всего процесса, меняется из потоков TTS и потока слушания.
namespace {
  // Сколько ещё молчать после того, как робот договорил (хвост звука и задержка обработки).
  constexpr uint64_t kSpeakerHoldMs = 700;
  // После конца слушания запроса: вместе с kSpeakerHoldMs около 2 с.
  constexpr uint64_t kListenEndExtraMs = 1300;
  // Защита: если «конец речи» не пришёл (фраза отменена), через столько мс слушаем снова.
  constexpr uint64_t kMaxSpeakMs = 60000;
  std::atomic<bool>     sRobotSpeaking{false};
  std::atomic<uint64_t> sSpeakStart_ms{0};
  std::atomic<uint64_t> sQuietUntil_ms{0};   // до этого момента звук не обрабатываем

  uint64_t NowMs()
  {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void QuietAtLeastUntil(uint64_t until)
  {
    uint64_t cur = sQuietUntil_ms;
    while (cur < until && !sQuietUntil_ms.compare_exchange_weak(cur, until)) {}
  }
}

void SpeechRecognizerMatreshka::SetRobotSpeaking(bool speaking)
{
  const uint64_t now = NowMs();
  if (speaking) {
    sSpeakStart_ms = now;
  } else {
    QuietAtLeastUntil(now + kSpeakerHoldMs);
  }
  sRobotSpeaking = speaking;
}

void SpeechRecognizerMatreshka::NoteListenEnded()
{
  QuietAtLeastUntil(NowMs() + kSpeakerHoldMs + kListenEndExtraMs);
}

// Внутреннее состояние. Мьютекс нужен, потому что Update вызывается из потока
// звука, а Reset и GetSampleCount — из других потоков. Он рекурсивный, чтобы
// один и тот же поток мог захватить его повторно.
struct SpeechRecognizerMatreshka::Impl
{
  mv_engine* engine = nullptr;
  uint64_t sampleCount = 0;   // отсчётов с момента Init(), как у штатного распознавателя
  uint64_t engineBase = 0;    // значение sampleCount при последнем сбросе движка
  bool disabled = true;       // true — звук не обрабатывается
  bool reset = false;         // запрошен сброс; выполняется в начале следующего Update
  bool muted = false;         // сейчас заглушен (робот говорит или только что слушал)
  bool useStockVad = false;   // подавать звук только при активном штатном VAD
  std::recursive_mutex mutex;
};

SpeechRecognizerMatreshka::SpeechRecognizerMatreshka() : _impl(new Impl()) {}

SpeechRecognizerMatreshka::~SpeechRecognizerMatreshka()
{
  // Движок создан функцией на C, поэтому и освобождаем его функцией на C.
  if (_impl && _impl->engine) {
    mv_destroy(_impl->engine);
    _impl->engine = nullptr;
  }
}

bool SpeechRecognizerMatreshka::ModelInstalled()
{
  struct stat st{};
  return stat(kModelPath, &st) == 0 && st.st_size > 0;
}

bool SpeechRecognizerMatreshka::Init()
{
  std::lock_guard<std::recursive_mutex> lock(_impl->mutex);
  if (_impl->engine) {
    mv_destroy(_impl->engine);
    _impl->engine = nullptr;
  }
  mv_config cfg = mv_default_config();
  // Порог можно поменять без пересборки: первая строка файла kThresholdPath.
  // Неверные значения (не число, не от 0 до 1) просто игнорируются.
  std::ifstream f(kThresholdPath);
  if (f.is_open()) {
    std::string line;
    float val = 0.0f;
    if (std::getline(f, line) && (std::istringstream(line) >> val) && val > 0.0f && val < 1.0f) {
      cfg.threshold = val;
    }
  }
  // При ошибке движок возвращает NULL, а причину — через mv_last_error().
  _impl->engine = mv_create_from_file(kModelPath, &cfg);
  if (!_impl->engine) {
    LOG_WARNING("SpeechRecognizerMatreshka.Init", "model not loaded: %s", mv_last_error());
    _impl->disabled = true;
    return false;
  }
  _impl->sampleCount = 0;
  _impl->engineBase = 0;
  _impl->disabled = false;
  {
    struct stat st{};
    _impl->useStockVad = (stat(kUseStockVadPath, &st) == 0);
  }
  LOG_INFO("SpeechRecognizerMatreshka.Init", "matreshka_voice %s loaded, threshold %.3f",
           mv_version(), mv_get_threshold(_impl->engine));
  return true;
}

void SpeechRecognizerMatreshka::Update(const AudioUtil::AudioSample* audioData, unsigned int audioDataLen)
{
  std::lock_guard<std::recursive_mutex> lock(_impl->mutex);
  if (_impl->disabled || !_impl->engine) {
    return;
  }
  // Глушение: пока робот говорит или только что закончил слушать, звук не обрабатываем,
  // а движок начинает с чистого листа, чтобы обрывок своей речи не сработал после паузы.
  {
    const uint64_t now = NowMs();
    const bool speaking = sRobotSpeaking && (now < sSpeakStart_ms + kMaxSpeakMs);
    const bool quiet = speaking || (now < sQuietUntil_ms);
    if (quiet) {
      if (!_impl->muted) {
        _impl->muted = true;
        _impl->reset = true;
        LOG_INFO("SpeechRecognizerMatreshka.Gate", "muted");
      }
      _impl->sampleCount += audioDataLen;   // счёт отсчётов идёт и в паузе
      return;
    }
    if (_impl->muted) {
      _impl->muted = false;
      LOG_INFO("SpeechRecognizerMatreshka.Gate", "listening again");
    }
  }
  // Сброс делаем здесь, в потоке звука, чтобы не трогать движок посреди обработки.
  // Движок после сброса считает отсчёты с нуля, поэтому запоминаем смещение.
  if (_impl->reset) {
    mv_reset(_impl->engine);
    _impl->engineBase = _impl->sampleCount;
    _impl->reset = false;
  }
  // AudioSample — 16-битные отсчёты, тот же формат, что ждёт mv_process.
  // Куски от WireOS короткие, а после срабатывания идёт пауза 1,5 с, так что двух мест хватает.
  mv_detection det[2];
  const int n = mv_process(_impl->engine, reinterpret_cast<const int16_t*>(audioData), audioDataLen, det, 2);
  _impl->sampleCount += audioDataLen;
  // Диагностика: раз в секунду пишем пиковую «сырую» оценку, если она заметная.
  // По этим строкам видно, насколько голос не дотягивает до порога.
  {
    static float sPeak = 0.f;
    static uint64_t sPeakFrom = 0;
    const float sc = mv_last_score(_impl->engine);
    if (sc > sPeak) {
      sPeak = sc;
    }
    if (_impl->sampleCount - sPeakFrom >= 16000) {
      if (sPeak >= 0.25f) {
        LOG_INFO("SpeechRecognizerMatreshka.Peak", "peak score %.3f", sPeak);
      }
      sPeak = 0.f;
      sPeakFrom = _impl->sampleCount;
    }
  }
  for (int i = 0; i < n; i++) {
    // Номер отсчёта конца слова в общем счёте с Init(); начало — на 800 мс раньше
    // (min не даёт уйти в отрицательные значения в первые миллисекунды).
    const uint64_t endIdx = _impl->engineBase + det[i].end_sample;
    const uint64_t beginIdx = endIdx - std::min(endIdx, kKeywordLengthSamples);
    AudioUtil::SpeechRecognizerCallbackInfo info{
      .result = kKeyword,
      .startTime_ms = static_cast<int>(beginIdx / 16),
      .endTime_ms = static_cast<int>(endIdx / 16),
      .startSampleIndex = beginIdx,
      .endSampleIndex = endIdx,
      .score = det[i].score
    };
    LOG_INFO("SpeechRecognizerMatreshka.Update", "wake word, score %.3f", det[i].score);
    DoCallback(info);   // сообщаем WireOS о слове активации
  }
}

void SpeechRecognizerMatreshka::Reset()
{
  std::lock_guard<std::recursive_mutex> lock(_impl->mutex);
  _impl->reset = true;
}

bool SpeechRecognizerMatreshka::UseStockVad() const
{
  std::lock_guard<std::recursive_mutex> lock(_impl->mutex);
  return _impl->useStockVad;
}

uint64_t SpeechRecognizerMatreshka::GetSampleCount() const
{
  std::lock_guard<std::recursive_mutex> lock(_impl->mutex);
  return _impl->sampleCount;
}

void SpeechRecognizerMatreshka::StartInternal() { _impl->disabled = (_impl->engine == nullptr); }
void SpeechRecognizerMatreshka::StopInternal() { _impl->disabled = true; }

} // namespace Vector
} // namespace Anki
