#!/bin/bash
set -euo pipefail

# Conda wie in deinen Download-Skripten
source ~/miniconda3/etc/profile.d/conda.sh
conda activate presence

# Arbeitsverzeichnis wie bei den Downloads
cd ~/scratch/presence_prediction

# Modelle & Kontexte (1:1 zuordnen)
versions=("0j5y1a5u" "gg9008fi" "g0la7ajz" "78s3mxnx")
declare -a context_lengths
context_lengths[0]="2,5,10,20,40,80"
context_lengths[1]="2,5,10,20,40,80"
context_lengths[2]="2,5,10,20,40,80,120,175,270"
context_lengths[3]="2,5,10,20,40,80,120,175,270,360"

flash_attention=(false false true true)
axial_attention=(false false true true)

# Nur die letzten zwei Versionen evaluieren (Index 2 und 3)
for i in {2..3}; do
  version="${versions[$i]}"
  context="${context_lengths[$i]}"
  flash="${flash_attention[$i]}"
  axial="${axial_attention[$i]}"

  echo "[INFO] Evaluiere ${version} mit Kontext(en): ${context} (flash=${flash}, axial=${axial})"
  python -m weather_prediction.homebrew_weather \
    --evaluate_random 100 \
    --version "${version}" \
    --context_days_list "${context}" \
    --prediction_days 1 \
    --use_flash_attention "${flash}" \
    --use_axial_attention "${axial}" \
    --use_wandb \
    --wandb_project "Fixed sampling issue. Evaluate all 4 models for 100 dates"
  echo "[INFO] Fertig: ${version}"
done

echo "[INFO] Alle Evaluierungen abgeschlossen."
