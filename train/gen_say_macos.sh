#!/bin/sh
# Дополнительный тестовый набор от независимого синтезатора (голос macOS «Milena»). Только для macOS.
# Запуск: sh train/gen_say_macos.sh [ПАПКА]   (по умолчанию data/say)
# set -e: остановиться на первой же ошибке.
set -e
OUT=${1:-data/say}
mkdir -p "$OUT/pos" "$OUT/neg"
# Слово с разной скоростью речи (-r — слов в минуту) и интонацией.
# --data-format=LEI16@16000: 16 бит little endian, 16 кГц — формат, который ждёт движок.
i=0
for rate in 140 160 180 200 220 240 260 280; do
  for t in "Вектор" "Вектор!" "Вектор?"; do
    say -v Milena -r $rate -o "$OUT/pos/say_$i.wav" --data-format=LEI16@16000 "$t"; i=$((i+1))
  done
done
# Негативы: каждая строка say_negatives.txt двумя скоростями.
i=0
while IFS= read -r line; do
  for rate in 170 230; do
    say -v Milena -r $rate -o "$OUT/neg/say_$i.wav" --data-format=LEI16@16000 "$line"; i=$((i+1))
  done
done < "$(dirname "$0")/say_negatives.txt"
echo "pos: $(ls "$OUT/pos" | wc -l)  neg: $(ls "$OUT/neg" | wc -l)"
