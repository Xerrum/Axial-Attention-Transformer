# Long-Context Transformers with Axial Attention

Code for my Master's thesis at TU Darmstadt on transformer models for long
sequence forecasting. The core idea is to arrange the input sequence on a grid
and apply attention along its axes separately, which reduces the attention cost
from O(n²) to roughly O(n·√n) and makes much longer context windows tractable.

The repository contains two studies:

||Data|Reproducible|
|-|-|-|
|**`weather\_prediction/`** — short-range weather forecasting on ERA5|Public (Copernicus CDS)|**Yes**, end to end|
|**`tud\_presence\_prediction/`** — customer presence forecasting|Internal TU Darmstadt data|No — code shown for reference only|

If you want to run something, use **`weather\_prediction/`**. It is the
self-contained study and the one the thesis results are built on.

\---

## 1\. Weather forecasting on ERA5 (reproducible)

Short-range forecasting from ERA5 reanalysis fields. ERA5 is openly available
from the Copernicus Climate Data Store, so every step below can be reproduced
from scratch.

### Setup

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `pytorch-lightning`, `torchmetrics`, `numpy`,
`pandas`, `matplotlib`, `netCDF4`, `cdsapi`, `wandb`.

`flash-attn` is optional and needs a CUDA-capable GPU with a matching
PyTorch/CUDA toolchain. Without it, use the non-FlashAttention model files —
comment the dependency out in `requirements.txt`.

### Getting the data

ERA5 access is free but requires a Copernicus CDS account. Put your credentials
in `\~/.cdsapirc`, then use the download helpers:

```bash
python download\_era5\_single\_monthly.py
python download\_era5\_pressure\_monthly.py
```

The pipeline expects single-level and pressure-level NetCDF files under
`--data\_root` (default `\~/scratch/era5\_data/past`). A sample file
(`era5\_single\_2021\_01.netcdf`) is included so you can verify the I/O path
before downloading anything.

### Training and evaluation

```bash
# Train
python weather\_prediction/homebrew\_weather.py --train \\
    --data\_root \~/scratch/era5\_data/past \\
    --batch\_size 4 --context\_days 7 --prediction\_days 1 --hidden\_dim 64

# Evaluate a trained run from a given start date
python weather\_prediction/homebrew\_weather.py \\
    --evaluate\_from\_date 2024-03-15 \\
    --context\_days 7 --prediction\_days 1 --version <run\_id>
```

Model definitions: `weather\_transformer.py` (baseline) and
`weather\_axialtransformer.py` (axial attention variant). Run
`homebrew\_weather.py --help` for the full flag set.

### Figures

```bash
python weather\_prediction/plot\_loss.py           # training curves
python weather\_prediction/plot\_context\_mse.py    # error vs. context length
python weather\_prediction/plot\_forecast\_csv.py   # forecast maps
```

Checkpoints and logs land in `train\_results/`, forecasts and metrics in
`forecast\_results/`. Weights \& Biases logging activates if `WANDB\_API\_KEY` is
set in the environment; otherwise disable it in the config.

### Results

The axial attention variant more than doubles the usable context window. Under
the same memory budget, the standard attention baseline saturates at **80 days**
of context; the long-context model trains on **more than 160 days**.

This was run as a proof of concept to probe where the limit moves, not as an
accuracy benchmark — the result above is a capability gain, not an error
reduction. Whether the longer context also improves forecast quality is a
separate question; see `plot\_context\_mse.py` for the error-vs-context sweep.

## 2\. Presence prediction (reference only, not runnable)

A sequence model for forecasting customer presence, developed on an internal
TU Darmstadt dataset. **The data is not public and is not part of this
repository**, and neither are the trained checkpoints, logs or evaluation
outputs derived from it. The code will not run without it, and there is no
substitute dataset — please do not open issues asking for the data.

It is included because it is the second study of the thesis and because the
axial attention implementation is shared between both. Read it, don't run it.

* `homebrew\_presence.py` — CLI for training, prediction, evaluation and plotting
* `models/` — model definitions, including the axial and FlashAttention variants
(`LearnTransformer.py`, `LearnTransformerAxialSelf.py`, `AxialDecoderOnly.py`)
and shared components under `models/internal/`
* `plot\_train.py`, `plot\_eval\_results\_fixed.py` — figure scripts

The interesting part for a reader is `models/` — the attention implementation
there is the same one used in the weather study, where you can actually execute
it.

\---

## Troubleshooting

* **`ImportError: flash\_attn`** — install a `flash-attn` build matching your
CUDA/PyTorch versions, or switch to the non-FlashAttention model files.
* **NetCDF read errors** — check your ERA5 file paths and variable names;
`weather\_prediction/data\_load\_test.py` is a minimal loader to test against.
* **CDS API errors** — `\~/.cdsapirc` missing or malformed, or the CDS request
queue is still processing your job.

\---

## 

