// matreshka_voice wake word recognizer for WireOS (vic-anim).
// Copyright (c) 2026 Xstrwe. MIT License.
//
// Drop-in AudioUtil::SpeechRecognizer that runs the matreshka_voice C engine.
// It is meant to run next to SpeechRecognizerPicovoice; see README.md in this folder.
#ifndef __Vector_SpeechRecognizerMatreshka_H_
#define __Vector_SpeechRecognizerMatreshka_H_

#include "audioUtil/speechRecognizer.h"

#include <memory>
#include <string>

namespace Anki {
namespace Vector {

class SpeechRecognizerMatreshka : public AudioUtil::SpeechRecognizer
{
public:
  SpeechRecognizerMatreshka();
  virtual ~SpeechRecognizerMatreshka();
  SpeechRecognizerMatreshka(const SpeechRecognizerMatreshka&) = delete;
  SpeechRecognizerMatreshka& operator=(const SpeechRecognizerMatreshka&) = delete;

  // Reported in SpeechRecognizerCallbackInfo::result, same as the stock trigger so the
  // rest of the pipeline treats it as a normal wake word.
  static constexpr const char* kKeyword = "HEY_VECTOR";

  static constexpr const char* kModelPath =
    "/data/data/com.anki.victor/persistent/matreshka_voice/model.mvw";
  static constexpr const char* kThresholdPath =
    "/data/data/com.anki.victor/persistent/matreshka_voice/threshold";

  // True if a model file is installed. Init() returns false (and the recognizer stays
  // disabled) when the file is missing or invalid.
  static bool ModelInstalled();
  bool Init();

  virtual void Update(const AudioUtil::AudioSample* audioData, unsigned int audioDataLen) override;
  void Reset();
  uint64_t GetSampleCount() const;

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;

  virtual void StartInternal() override;
  virtual void StopInternal() override;
};

} // namespace Vector
} // namespace Anki

#endif
