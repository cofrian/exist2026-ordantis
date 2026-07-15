#!/usr/bin/env bash
# Pipeline EXIST 2026 — entrenamiento en DOS FASES (head warm-up + full fine-tune).
# Por ahora ejecuta Task 2.1 (run_full.py). La Vista E de M3 usa las descripciones de Gemini
# (precompute_gemini.py); si no hay GEMINI_API_KEY en ../.env, M3 cae a 4 vistas (A-D).
set -e
cd "$(dirname "$0")"
source ../.venv/bin/activate

export TRITON_CACHE_DIR=/tmp/triton_cache_$USER
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRE=exist2026_Ordantis/preprocessed
rm -f "$PRE/gemini_predictions.DONE"      # marcador limpio para esta corrida

echo "==== $(date) :: precompute_emotions ===="
python precompute_emotions.py || echo "⚠️ emotions falló (se usarán ceros)"

# Gemini en SEGUNDO PLANO (solo red/CPU, no usa GPU) en paralelo al entrenamiento de M1/M2.
# run_full.py espera al marcador gemini_predictions.DONE antes de entrenar M3 (Vista E).
echo "==== $(date) :: precompute_gemini en background (async, conc=${GEMINI_CONCURRENCY:-15}) ===="
nohup python precompute_gemini.py > gemini.log 2>&1 &
GEMINI_PID=$!
echo "$GEMINI_PID" > "$PRE/gemini.pid"
echo "  PID Gemini: $GEMINI_PID  (log: gemini.log)"

echo "==== $(date) :: run_full.py (Task 2.1, dos fases) — M1/M2 mientras Gemini corre ===="
python run_full.py

# por si Gemini sigue vivo (no debería) al acabar todo:
wait "$GEMINI_PID" 2>/dev/null || true

echo "==== $(date) :: FIN ===="
ls -lh exist2026_Ordantis.zip 2>/dev/null || true
