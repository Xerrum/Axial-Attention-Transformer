# Master Thesis Code

This repository contains two related projects used in the thesis:

- **Presence Prediction** (`tud_presence_prediction/`), a sequence model for customer presence forecasting, with training, evaluation, and plotting utilities.
- **Weather Forecast** (`weather_prediction/`), a Long-Context Transformer pipeline for ERA5-based short-range weather prediction, including training, evaluation, and figure scripts.

The structure and commands below are based on the inspected source files in this archive.

---

## Project Layout (evidence)

- `tud_presence_prediction/homebrew_presence.py` — main CLI for training, prediction, evaluation, and plotting of the presence model.
- `tud_presence_prediction/models/` — model definitions, including axial/FlashAttention variants (e.g. `LearnTransformer.py`, 'LearnTransformerAxialSelf.py') and utilities under `models/internal/`.
- `tud_presence_prediction/plot_*` — plotting utilities for loss curves and evaluation figures (e.g., `plot_eval_results_fixed.py`, `plot_train.py`).
- `tud_presence_prediction/download_era5_*` — helper scripts to fetch ERA5 data if needed (e.g., `download_era5_pressure_monthly.py`, `download_era5_single_monthly.py`).

- `weather_prediction/homebrew_weather.py` — main CLI for training and evaluation on ERA5; supports context/prediction lengths and saving forecasts.
- `weather_prediction/weather_transformer.py` and `weather_axialtransformer.py` — model definitions for the weather study.
- `weather_prediction/plot_context_mse.py`, `plot_forecast_csv.py`, `plot_loss.py` — plotting and analysis scripts.
- `weather_prediction/forecast_results/` and `train_results/` — default output folders for forecasts and training logs/checkpoints.
- Example data files present (e.g., `era5_single_2021_01.netcdf`), useful to verify I/O.

These paths and filenames are taken directly from the codebase you provided.

---

## Requirements

Python packages inferred from imports across both projects are listed in `requirements.txt`. Key dependencies include:
- **Core**: `torch`, `pytorch-lightning`, `torchmetrics`, `numpy`, `pandas`, `scikit-learn`, `tqdm`, `matplotlib`, `pillow`, `colorama`, `holidays`, `wandb`
- **Weather/IO**: `netCDF4`, `cdsapi`
- **Attention/Plotting**: `flash-attn` (for GPU-accelerated attention), `brokenaxes` (for figure formatting)

> ⚠️ **FlashAttention**: Installing `flash-attn` requires a CUDA-capable GPU and a compatible PyTorch/CUDA toolchain. If you do not plan to use the FlashAttention variants, you can comment it out in `requirements.txt` and avoid those model files.

Install with:
```bash
pip install -r requirements.txt
```

---

## Quickstart

### 1) Presence Prediction

Train / evaluate / plot are handled by `tud_presence_prediction/homebrew_presence.py` (see its argparse flags in the file). Typical usage:
```bash
# Train
python tud_presence_prediction/homebrew_presence.py --train --model_file LearnTransformer --num_input_days 7 --num_days 1

# Evaluate (examples only; see file for full set of flags)
python tud_presence_prediction/homebrew_presence.py --evaluate --version <run_id_or_version>

# Plot training curves
python tud_presence_prediction/homebrew_presence.py --plot_loss --version <run_id_or_version>
```

Relevant sources: `tud_presence_prediction/homebrew_presence.py`, models under `tud_presence_prediction/models/`, and plots `plot_train.py`, `plot_eval_results_fixed.py`.

### 2) Weather Forecast

The main entry point is `weather_prediction/homebrew_weather.py`. Typical usage:
```bash
# Train
python weather_prediction/homebrew_weather.py --train \
    --data_root ~/scratch/era5_data/past \    --batch_size 4 --context_days 7 --prediction_days 1 --hidden_dim 64

# Evaluate from a specific date (YYYY-MM-DD)
python weather_prediction/homebrew_weather.py --evaluate_from_date 2024-03-15 \    --context_days 7 --prediction_days 1 --version <run_id_or_version>
```

To use the Long-Context / axial variants, see `weather_prediction/weather_transformer.py` and `weather_prediction/weather_axialtransformer.py`. Plots for context sweeps and forecast maps are in `plot_context_mse.py` and `plot_forecast_csv.py`.

---

## Data

- **ERA5**: The weather pipeline expects ERA5 single-level and pressure-level data in NetCDF format. The default `--data_root` is `~/scratch/era5_data/past`. You can adjust via CLI.
- Helper download scripts for ERA5 are provided (e.g., `tud_presence_prediction/download_era5_pressure_monthly.py`) and rely on `cdsapi`.

---

## Logging & Checkpoints

- Both projects use **PyTorch Lightning**. Checkpoints and logs are saved under `train_results/` and versioned subfolders.
- **Weights & Biases** logging is enabled where configured (`wandb`). Set `WANDB_API_KEY` in your environment to activate, or disable in code if not needed.

---

## GPU/FlashAttention Notes

- FlashAttention-based models import `flash_attn` (see `tud_presence_prediction/models/AxialDecoderOnly.py` and `weather_prediction/weather_transformer.py`).
- Ensure your CUDA and PyTorch versions are compatible with the `flash-attn` wheel you install. Otherwise, prefer the non-FlashAttention model files.

---

## Reproducibility

- Scripts write evaluation CSV/JSON and figures under `forecast_results/`, `train_results/`, and the presence/evaluation folders.
- Plotting utilities reside in `tud_presence_prediction/plot_*.py` and `weather_prediction/plot_*.py`. Re-run these to regenerate figures from saved logs.

---

## Troubleshooting

- **ImportError: flash_attn**: Remove or comment the FlashAttention lines, or install a compatible `flash-attn` build.
- **NetCDF errors**: Verify your ERA5 NetCDF files and paths. Example loaders are in `weather_prediction/data_load_test.py`.
- **CDS API**: Configure your `~/.cdsapirc` before running the ERA5 download scripts.

---
