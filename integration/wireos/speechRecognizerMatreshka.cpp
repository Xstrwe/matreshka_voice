// matreshka_voice wake word recognizer for WireOS (vic-anim).
// Copyright (c) 2026 Xstrwe. MIT License.
#include "speechRecognizerMatreshka.h"

#include "util/logging/logging.h"

extern "C" {
#include "matreshka_voice.h"
}

#include <algorithm>
#include <fstream>
#include <mutex>
#include <sstream>
#include <sys/stat.h>

namespace Anki {
namespace Vector {

#define LOG_CHANNEL "SpeechRecognizer"

// Keyword length reported to the rest of the pipeline (ms of audio before the detection).
static const uint64_t kKeywordLengthSamples = 16 * 800;

struct SpeechRecognizerMatreshka::Impl
{
  mv_engine* engine = nullptr;
  uint64_t sampleCount = 0;   // samples since Init(), same convention as the Picovoice wrapper
  uint64_t engineBase = 0;    // sampleCount when the engine was last reset
  bool disabled = true;
  bool reset = false;
  std::recursive_mutex mutex;
};

SpeechRecognizerMatreshka::SpeechRecognizerMatreshka() : _impl(new Impl()) {}

SpeechRecognizerMatreshka::~SpeechRecognizerMatreshka()
{
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
  std::ifstream f(kThresholdPath);
  if (f.is_open()) {
    std::string line;
    float val = 0.0f;
    if (std::getline(f, line) && (std::istringstream(line) >> val) && val > 0.0f && val < 1.0f) {
      cfg.threshold = val;
    }
  }
  _impl->engine = mv_create_from_file(kModelPath, &cfg);
  if (!_impl->engine) {
    LOG_WARNING("SpeechRecognizerMatreshka.Init", "model not loaded: %s", mv_last_error());
    _impl->disabled = true;
    return false;
  }
  _impl->sampleCount = 0;
  _impl->engineBase = 0;
  _impl->disabled = false;
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
  if (_impl->reset) {
    mv_reset(_impl->engine);
    _impl->engineBase = _impl->sampleCount;
    _impl->reset = false;
  }
  mv_detection det[2];
  const int n = mv_process(_impl->engine, reinterpret_cast<const int16_t*>(audioData), audioDataLen, det, 2);
  _impl->sampleCount += audioDataLen;
  for (int i = 0; i < n; i++) {
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
    DoCallback(info);
  }
}

void SpeechRecognizerMatreshka::Reset()
{
  std::lock_guard<std::recursive_mutex> lock(_impl->mutex);
  _impl->reset = true;
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
