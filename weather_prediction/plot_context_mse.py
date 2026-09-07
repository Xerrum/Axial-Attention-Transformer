#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Liest:
    ~/scratch/presence_prediction/weather_prediction/forecast_results/random_eval_2024_n100_v{version}_c*.json

Speichert Plots nach:
    ~/scratch/presence_prediction/weather_prediction/forecast_results/version_{version}/

Erzeugt:
1) mse_bar_plot_{modeltype}.png
   - Balkendiagramm: avg_mse_mean vs. Kontext (Tage)
   - y-Achse fix: 0..1
   - Farblogik: Long-Context (beide Varianten) = orange, sonst = blau

2) mse_per_channel_barplot_{modeltype}.png
   - Gruppierte Balken je Kontext: t2m, d2m, msl×40
   - y-Achse fix: 0..0.35
   - Farben bleiben fest: t2m=grün, d2m=violett, msl=sandgelb (unabhängig vom Modell)
"""

import os
import re
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
# optional broken y-axis support
try:
    from brokenaxes import brokenaxes
    _HAVE_BROKENAXES = True
except Exception:
    _HAVE_BROKENAXES = False


# -------------------------
# Globale Konfiguration
# -------------------------
BASE_DIR = Path(r"C:\Users\organ\PycharmProjects\presence_prediction\weather_prediction\forecast_results\random_evaluation_N100").expanduser()
OUT_DIR = Path(r"C:\Users\organ\PycharmProjects\presence_prediction\weather_prediction\forecast_results").expanduser()
# Versionen, die in --auto geplottet werden
VERSIONS = ["0j5y1a5u", "78s3mxnx", "g0la7ajz", "gg9008fi"]

# Mapping Version -> Modeltitel (nur Wert wird im Titel genutzt)
MODEL_TITLES = {
    "0j5y1a5u": "Regular Model",
    "gg9008fi": "Regular Model fine-tuned",
    "g0la7ajz": "Long-Context Transformer",
    "78s3mxnx": "Long Context Transformer fine-tuned",
}

# Farben für OVERALL-MSE (nicht kanal-weise)
COLOR_REGULAR = "#1F77B4"  # blau
COLOR_LONGCTX = "#FF7F0E"  # orange

# Farben für PER-CHANNEL (fest, modellunabhängig)
COLOR_T2M = "#2ca02c"      # grün (t2m)
COLOR_D2M = "#9467bd"      # violett (d2m)
COLOR_MSL = "#ffbf00"      # sandgelb (msl ×10)

# --- Font sizes (global) ---
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 15
TICK_FONTSIZE  = 14
LEGEND_FONTSIZE = 12

plt.rcParams.update({
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": LABEL_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
})

def _is_long_context(modeltype: str) -> bool:
    """Erkennt Long-Context-Varianten robust (Groß/Kleinschreibung, Bindestriche egal)."""
    m = (modeltype or "").lower()
    # alles, was nach Long-Context-Transformer klingt, zählt als "long"
    return ("long" in m) and ("transformer" in m)


def plot_avg_mse_mean(output_dir: Path, x_sorted, y_sorted, modeltype: str) -> Path:
    """Einfacher Balkenplot: avg_mse_mean vs. Kontext (Tage), y in [0, 1]."""
    plt.figure(figsize=(8, 5))
    idx = np.arange(len(x_sorted))

    # Farbwahl: Long-Context = orange, sonst blau
    color = COLOR_LONGCTX if _is_long_context(modeltype) else COLOR_REGULAR
    plt.bar(idx, y_sorted, width=0.9, color=color)

    plt.xticks(idx, [str(x) for x in x_sorted], fontsize=TICK_FONTSIZE)
    plt.xlabel("Context Days", fontsize=LABEL_FONTSIZE)
    plt.ylabel("MSE", fontsize=LABEL_FONTSIZE)
    # plt.title(f"MSE at different Context-Lengths ({modeltype})", fontsize=TITLE_FONTSIZE)
    plt.ylim(0.0, 1)
    plt.yticks(np.arange(0, 1.0, 0.1), fontsize=TICK_FONTSIZE)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    safe_modeltype = modeltype.replace(" ", "_")
    out_path = output_dir / f"mse_bar_plot_{safe_modeltype}.png"
    plt.savefig(out_path, dpi=150)
    print(f"[OK] Saved: {out_path}")
    return out_path


def plot_per_channel_grouped(output_dir: Path, ctx_list, t2m_list, d2m_list, msl_list_scaled, modeltype: str) -> Path:
    """
    Gruppierter Balkenplot je Kontext:
    - drei Bars nebeneinander (t2m, d2m, msl×40)
    - y-Achse: Regular = 0..0.25, Long-Context = 0..0.35
    - Farben bleiben fest (modellunabhängig)
    """
    n = len(ctx_list)
    idx = np.arange(n)
    width = 0.9 / 3.0  # drei Balken füllen die Einheit

    # y-Limit je nach Modellfamilie
    y_max = 0.35 if _is_long_context(modeltype) else 0.15

    plt.figure(figsize=(9, 5.2))
    plt.bar(idx - width, t2m_list,        width=width, label="t2m",     color=COLOR_T2M)
    plt.bar(idx,        d2m_list,         width=width, label="d2m",     color=COLOR_D2M)
    plt.bar(idx + width, msl_list_scaled, width=width, label="msl ×40", color=COLOR_MSL)

    plt.xticks(idx, [str(c) for c in ctx_list], fontsize=TICK_FONTSIZE)
    plt.xlabel("Context Days", fontsize=LABEL_FONTSIZE)
    plt.ylabel("MSE", fontsize=LABEL_FONTSIZE)
    # plt.title(f"Per-Channel MSE vs Context Days ({modeltype})", fontsize=TITLE_FONTSIZE)
    plt.ylim(0.0, y_max)
    
    # Immer 0.05er Schritte für alle Plots verwenden
    step = 0.05
    plt.yticks(np.arange(0.0, y_max + 1e-9, step), fontsize=TICK_FONTSIZE)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()

    safe_modeltype = modeltype.replace(" ", "_")
    out_path = output_dir / f"mse_per_channel_barplot_{safe_modeltype}.png"
    plt.savefig(out_path, dpi=150)
    print(f"[OK] Saved: {out_path}")
    return out_path



def run_for_version(version: str, modeltype_label: str, show: bool = False) -> None:
    """
    Lädt alle passenden JSONs für eine Version, erstellt beide Plots,
    speichert nach version_{version} und gibt nur Save-Pfade aus.
    """
    results_dir = BASE_DIR
    output_dir = OUT_DIR / f"version_{version}"
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(
        rf"^random_eval_2024_n100_v{re.escape(version)}_c(\d+)\.json$",
        re.IGNORECASE,
    )

    x_values, y_values = [], []
    ctx_for_channels, t2m_vals, d2m_vals, msl_vals_scaled = [], [], [], []

    for filename in os.listdir(results_dir):
        m = pattern.match(filename)
        if not m:
            continue

        c_value = int(m.group(1))
        filepath = results_dir / filename

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if "avg_mse_mean" in data and data["avg_mse_mean"] is not None:
            try:
                y = float(data["avg_mse_mean"])
                x_values.append(c_value)
                y_values.append(y)
            except (TypeError, ValueError):
                pass

        ch = data.get("avg_mse_per_channel", {})
        if all(k in ch for k in ("t2m", "d2m", "msl")):
            try:
                t2 = float(ch["t2m"])
                d2 = float(ch["d2m"])
                msl_scaled = float(ch["msl"]) * 40.0
            except (TypeError, ValueError):
                continue

            ctx_for_channels.append(c_value)
            t2m_vals.append(t2)
            d2m_vals.append(d2)
            msl_vals_scaled.append(msl_scaled)

    if not x_values:
        return

    x_sorted, y_sorted = zip(*sorted(zip(x_values, y_values), key=lambda p: p[0]))
    plot_avg_mse_mean(output_dir, list(x_sorted), list(y_sorted), modeltype_label)

    if ctx_for_channels:
        order = np.argsort(ctx_for_channels)
        ctx_sorted = np.array(ctx_for_channels)[order].tolist()
        t2_sorted = np.array(t2m_vals)[order].tolist()
        d2_sorted = np.array(d2m_vals)[order].tolist()
        msl_sorted = np.array(msl_vals_scaled)[order].tolist()

        plot_per_channel_grouped(output_dir, ctx_sorted, t2_sorted, d2_sorted, msl_sorted, modeltype_label)

    # --- Report all MSE values for overall plot ---
    print(f"\n[INFO] Overall MSE values for '{modeltype_label}':")
    for ctx, mse in zip(x_sorted, y_sorted):
        print(f"  Context {ctx} days: MSE = {mse:.6f}")

    # Find and print the minimum overall MSE
    if y_sorted:
        min_idx_overall = int(np.argmin(y_sorted))
        min_ctx_overall = x_sorted[min_idx_overall]
        min_val_overall = y_sorted[min_idx_overall]
        print(f"[INFO] For '{modeltype_label}', lowest overall MSE = {min_val_overall:.6f} at context {min_ctx_overall} days")

    # --- Report minimum values for all variables ---
    if ctx_for_channels:
        # For t2m
        if t2m_vals:
            min_idx_t2m = int(np.argmin(t2m_vals))
            min_ctx_t2m = ctx_for_channels[min_idx_t2m]
            min_val_t2m = t2m_vals[min_idx_t2m]
            print(f"[INFO] For '{modeltype_label}', lowest t2m = {min_val_t2m:.6f} at context {min_ctx_t2m} days")
        
        # For d2m
        if d2m_vals:
            min_idx_d2m = int(np.argmin(d2m_vals))
            min_ctx_d2m = ctx_for_channels[min_idx_d2m]
            min_val_d2m = d2m_vals[min_idx_d2m]
            print(f"[INFO] For '{modeltype_label}', lowest d2m = {min_val_d2m:.6f} at context {min_ctx_d2m} days")
        
        # For msl (scaled)
        if msl_vals_scaled:
            min_idx_msl = int(np.argmin(msl_vals_scaled))
            min_ctx_msl = ctx_for_channels[min_idx_msl]
            min_val_msl = msl_vals_scaled[min_idx_msl]
            # Zurückrechnen auf den ursprünglichen MSL-Wert
            original_msl = min_val_msl / 40.0
            print(f"[INFO] For '{modeltype_label}', lowest msl = {original_msl:.6f} (scaled: {min_val_msl:.6f}) at context {min_ctx_msl} days")

    if show:
        plt.show()
    else:
        plt.close('all')


def main():
    parser = argparse.ArgumentParser(
        description="Plot MSE vs. context für eine oder mehrere Versionen (overall und per-channel)."
    )
    parser.add_argument("--version_number", type=str, help="Version ID, z. B. 0j5y1a5u")
    parser.add_argument("--modeltype", type=str,
                        help="Manuelles Label für den Plot-Titel/Dateinamen. Wenn nicht gesetzt, wird MODEL_TITLES genutzt.")
    parser.add_argument("--auto", action="store_true",
                        help="Wenn gesetzt, laufe über alle VERSIONS und nutze MODEL_TITLES.")
    parser.add_argument("--show", action="store_true",
                        help="Plots am Ende anzeigen (nur sinnvoll für Einzelläufe).")
    args = parser.parse_args()

    print("Running")

    if args.auto:
        for v in VERSIONS:
            label = MODEL_TITLES.get(v, args.modeltype or v)
            run_for_version(v, label, show=False)
        return

    if not args.version_number:
        parser.error("Entweder --auto oder --version_number angeben.")

    label = args.modeltype or MODEL_TITLES.get(args.version_number, args.version_number)
    run_for_version(args.version_number, label, show=args.show)


if __name__ == "__main__":
    main()
