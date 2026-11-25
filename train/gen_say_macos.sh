#!/bin/sh
# Extra test set from an independent TTS engine (macOS «Milena»). macOS only.
set -e
OUT=${1:-data/say}
mkdir -p "$OUT/pos" "$OUT/neg"
i=0
for rate in 140 160 180 200 220 240 260 280; do
  for t in "Вектор" "Вектор!" "Вектор?"; do
    say -v Milena -r $rate -o "$OUT/pos/say_$i.wav" --data-format=LEI16@16000 "$t"; i=$((i+1))
  done
done
i=0
while IFS= read -r line; do
  for rate in 170 230; do
    say -v Milena -r $rate -o "$OUT/neg/say_$i.wav" --data-format=LEI16@16000 "$line"; i=$((i+1))
  done
done < "$(dirname "$0")/say_negatives.txt"
echo "pos: $(ls "$OUT/pos" | wc -l)  neg: $(ls "$OUT/neg" | wc -l)"
