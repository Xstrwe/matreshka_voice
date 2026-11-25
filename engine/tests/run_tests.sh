#!/bin/sh
# Смоук-тест и проверка совпадения движка на C с эталоном на Python.
# Запуск из папки engine/: make test
set -e
PY=${PYTHON:-python3}
MODEL=${MODEL:-../models/vektor_ru.mvw}
WAV=${WAV:-tests/data/sample.wav}
./mv_cli "$MODEL" "$WAV"
$PY tests/parity.py "$MODEL" "$WAV" ./mv_cli
# Размер кусков, которыми подаётся звук, не должен менять результат:
# 512 — как в WireOS, 37 — нарочно «неудобное» число.
a=$(./mv_cli "$MODEL" "$WAV" -c 512 | grep -v summary || true)
b=$(./mv_cli "$MODEL" "$WAV" -c 37 | grep -v summary || true)
[ "$a" = "$b" ] && echo "CHUNKING OK" || { echo "CHUNKING MISMATCH"; exit 1; }
