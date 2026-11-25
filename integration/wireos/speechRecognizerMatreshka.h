// Распознаватель слова активации matreshka_voice для WireOS (процесс vic-anim).
// Copyright (c) 2026 Xstrwe. Лицензия MIT.
//
// Готовая замена AudioUtil::SpeechRecognizer, внутри которой работает движок на C.
// Рассчитан на работу рядом со штатным распознавателем WireOS; подробности — в README.md.
//
// Это C++-обёртка: WireOS написан на C++, а движок — на C. Обёртка переводит вызовы
// WireOS (Update, Reset, Start/Stop) в функции mv_* и сообщает о срабатываниях через
// обычный колбэк распознавателя.
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
  // Копировать нельзя: объект владеет движком, и две копии освободили бы его дважды.
  SpeechRecognizerMatreshka(const SpeechRecognizerMatreshka&) = delete;
  SpeechRecognizerMatreshka& operator=(const SpeechRecognizerMatreshka&) = delete;

  // Значение SpeechRecognizerCallbackInfo::result. Совпадает со штатным словом, поэтому
  // остальная часть WireOS обрабатывает срабатывание как обычное «Hey Vector».
  static constexpr const char* kKeyword = "HEY_VECTOR";

  // Файл модели и необязательный файл с порогом (одно число от 0 до 1) на роботе.
  static constexpr const char* kModelPath =
    "/data/data/com.anki.victor/persistent/matreshka_voice/model.mvw";
  static constexpr const char* kThresholdPath =
    "/data/data/com.anki.victor/persistent/matreshka_voice/threshold";
  // Если этот файл есть (содержимое не важно), штатное английское «Hey Vector» не слушается.
  // Включить обратно: удалить файл и перезапустить anki-robot.target.
  static constexpr const char* kDisableEnglishPath =
    "/data/data/com.anki.victor/persistent/matreshka_voice/disable_english";

  // true, если файл модели есть и не пустой. Init() возвращает false (и распознаватель
  // остаётся выключенным), если файла нет или он испорчен.
  static bool ModelInstalled();
  bool Init();

  // Update получает очередной кусок звука с микрофона (16 кГц, 16 бит).
  // Reset просит начать распознавание с чистого листа (выполняется в следующем Update).
  // GetSampleCount — сколько отсчётов обработано с момента Init().
  virtual void Update(const AudioUtil::AudioSample* audioData, unsigned int audioDataLen) override;
  void Reset();
  uint64_t GetSampleCount() const;

private:
  // Приём «pimpl»: детали спрятаны в .cpp, поэтому заголовок не тянет за собой
  // matreshka_voice.h и не меняется при правках реализации.
  struct Impl;
  std::unique_ptr<Impl> _impl;

  // Вызываются базовым классом при Start()/Stop(): включить или выключить распознавание.
  virtual void StartInternal() override;
  virtual void StopInternal() override;
};

} // namespace Vector
} // namespace Anki

#endif
