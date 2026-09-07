#!/bin/bash

# Jahre und Monate
years=(2024)
months=(01 02 03 04 05 06 07 08 09 10 11 12)

# Aktiviere Conda-Umgebung
source ~/miniconda3/etc/profile.d/conda.sh
conda activate presence

# Wechsle ins Skriptverzeichnis
cd ~/scratch/presence_prediction/tud_presence_prediction

# Starte Download fuer jeden Monat
for year in "${years[@]}"; do
    for month in "${months[@]}"; do
        echo "⏳ Lade $year-$month herunter ..."
        python download_era5_single_monthly.py $year $month
        echo "✅ Fertig mit $year-$month"
    done
done