#!/bin/sh
# Smoke + parity tests for the C engine. Run from engine/: make test
set -e
PY=${PYTHON:-python3}
MODEL=${MODEL:-../models/vektor_ru.mvw}
WAV=${WAV:-tests/data/sample.wav}
./mv_cli "$MODEL" "$WAV"
$PY tests/parity.py "$MODEL" "$WAV" ./mv_cli
# chunk size must not change results
a=$(./mv_cli "$MODEL" "$WAV" -c 512 | grep -v summary || true)
b=$(./mv_cli "$MODEL" "$WAV" -c 37 | grep -v summary || true)
[ "$a" = "$b" ] && echo "CHUNKING OK" || { echo "CHUNKING MISMATCH"; exit 1; }
