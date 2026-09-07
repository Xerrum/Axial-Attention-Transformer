# -*- coding: utf-8 -*-
"""
plot_forecast_csv_auto.py

Reads forecast CSV files and plots selected variables.

Key features:
- AUTO mode to loop over all files for a given set of version codes.
- Global VERSIONS list and VARIABLES_TO_PLOT list for easy configuration.
- Variables in CSV end with "pred", "truth", or "true"; separator can be one or more underscores.
- Ground truth: ORANGE with 'x' markers.
- Prediction: BLUE with 'o' markers.
- X-axis: timestamps; Y-axis: values.
- Title includes model variant inferred from the filename suffix code.
- Filename pattern: forecast_center_20240315_v{version}.csv
- Consistent figure size and unified y-limits per variable across all versions.

Fixes:
- Normalize base variable keys across files.
- Coerce to numeric at plot time.
- Sort by timestamp before plotting to keep truth lines identical across versions.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# GLOBAL CONFIG (edit here)
# -------------------------
BASE_DIR: str = r"C:\Users\organ\PycharmProjects\presence_prediction\weather_prediction\forecast_results\forecast_15_03"

# Which files to pick in AUTO mode (by version code in the filename).
VERSIONS: List[str] = [
    "0j5y1a5u",
    "gg9008fi",
    "g0la7ajz",
    "78s3mxnx",
]

# Which variables to plot by base name (without the suffix).
# Leave empty ([]) to auto-discover all pairs present.
VARIABLES_TO_PLOT: List[str] = ["t2m", "u10", "v10", "msl"]

# Map version code -> readable title
MODEL_TITLES: Dict[str, str] = {
    "0j5y1a5u": "Regular Model",
    "gg9008fi": "Regular Model finetuned",
    "g0la7ajz": "Long-Context Transformer",
    "78s3mxnx": "Long Context Transformer Finetuned",
}

# Candidate timestamp column names (case-insensitive)
TIMESTAMP_CANDIDATES = ["timestamp", "time", "date", "datetime", "valid_time", "init_time"]

# --- Font sizes (global) ---
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 15
TICK_FONTSIZE  = 14
LEGEND_FONTSIZE = 12

# Figure size & y-limit padding
FIGSIZE = (8, 5)
Y_PAD_FRAC = 0.02  # 2% padding


def _find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in TIMESTAMP_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    # Fallback: first column parseable as datetime
    first = df.columns[0]
    try:
        pd.to_datetime(df[first])
        return first
    except Exception:
        return None


def _infer_model_title_from_path(path: str) -> str:
    # look for ..._v[8-chars].csv
    m = re.search(r"_v([a-z0-9]{8})\.csv$", os.path.basename(path))
    if m:
        code = m.group(1)
        return MODEL_TITLES.get(code, code)
    return os.path.basename(path)


# Regex to capture base variable and suffix; allows one or more underscores before suffix
_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_+(?P<sfx>pred|truth|true)$", re.IGNORECASE)


def _normalize_base(base: str) -> str:
    """
    Normalize a base variable name for consistent matching across files.
    - lowercase
    - collapse internal whitespace
    - (optional) aliasing can be enabled if your exports vary (e.g., msl×10 -> msl)
    """
    b = " ".join(str(base).strip().lower().split())
    # Example aliases if you ever need them:
    # if b in ("msl×10", "msl_x10", "msl*10", "msl times 10"):
    #     b = "msl"
    return b


def discover_pairs_pred_truth(df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """
    Discover (truth, pred) pairs using suffixes '_pred', '_true' or '_truth'
    with one or more underscores. Returns a dict keyed by a normalized
    base name (lowercase), with values = (truth_col_name, pred_col_name).
    """
    base_to_truth: Dict[str, str] = {}
    base_to_pred: Dict[str, str] = {}

    for c in df.columns:
        m = _SUFFIX_RE.match(c.strip())
        if not m:
            continue
        raw_base = m.group("base")
        sfx = m.group("sfx").lower()
        norm_base = _normalize_base(raw_base)

        if sfx == "pred":
            if norm_base not in base_to_pred:
                base_to_pred[norm_base] = c
        elif sfx in ("truth", "true"):
            if norm_base not in base_to_truth:
                base_to_truth[norm_base] = c

    pairs: Dict[str, Tuple[str, str]] = {}
    for norm_base in sorted(set(base_to_truth) | set(base_to_pred)):
        tcol = base_to_truth.get(norm_base)
        pcol = base_to_pred.get(norm_base)
        if tcol is not None and pcol is not None:
            pairs[norm_base] = (tcol, pcol)
    return pairs


def plot_variable(
    df: pd.DataFrame,
    ts_col: str,
    truth_col: str,
    pred_col: str,
    title_prefix: str,
    var_display_name: Optional[str],
    out_path: Optional[str] = None,
    show: bool = False,
    dpi: int = 160,
    ylim: Optional[Tuple[float, float]] = None,
):
    # Robust parse and consistent preprocessing at plot time
    x = pd.to_datetime(df[ts_col], errors="coerce")
    y_true = pd.to_numeric(df[truth_col], errors="coerce")
    y_pred = pd.to_numeric(df[pred_col], errors="coerce")

    # Drop NaN timestamps and sort by time for consistent line shapes
    plot_df = pd.DataFrame({"x": x, "y_true": y_true, "y_pred": y_pred}).dropna(subset=["x"])
    plot_df = plot_df.sort_values("x", kind="mergesort")

    plt.figure(figsize=FIGSIZE)
    plt.plot(plot_df["x"], plot_df["y_true"].values, marker="x", linestyle="-", color="#FF7F0E", label="Ground Truth")
    plt.plot(plot_df["x"], plot_df["y_pred"].values, marker="o", linestyle="-", color="#1F77B4", label="Prediction")

    # Format x-axis to show only time (HH:MM)
    from matplotlib.dates import DateFormatter, HourLocator
    plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(HourLocator(byhour=range(0, 24, 3)))  # Alle 3 Stunden

    plt.xlabel("Timestamp", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Value", fontsize=LABEL_FONTSIZE)
    plt.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

    disp_name = var_display_name or truth_col
    plt.title(f"{title_prefix} - {disp_name}", fontsize=TITLE_FONTSIZE)
    if ylim is not None and np.isfinite(ylim[0]) and np.isfinite(ylim[1]):
        plt.ylim(ylim)
    leg = plt.legend()
    if leg:
        for text in leg.get_texts():
            text.set_fontsize(LEGEND_FONTSIZE)

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close()


def _collect_auto_csvs(base_dir: str, versions: List[str]) -> List[str]:
    paths = []
    for v in versions:
        fname = f"forecast_center_20240315_v{v}.csv"
        p = Path(base_dir) / fname
        paths.append(str(p))
    return paths


def _compute_var_ranges(csv_list: List[str], variables_selector: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    First pass: compute global y-limits per variable across all CSVs.
    variables_selector: if empty, auto-discover per file and union.
    Returns dict: var(normed) -> (ymin, ymax) with small padding.
    """
    var_minmax: Dict[str, Tuple[float, float]] = {}

    for csv_path in csv_list:
        if not os.path.isfile(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        pairs = discover_pairs_pred_truth(df)
        if not pairs:
            continue

        # decide variables for this file (normalize selector)
        if variables_selector:
            selector_norm = [_normalize_base(v) for v in variables_selector]
            vars_for_file = [v for v in selector_norm if v in pairs]
        else:
            vars_for_file = list(pairs.keys())

        for v in vars_for_file:
            tcol, pcol = pairs[v]
            y = pd.concat(
                [
                    pd.to_numeric(df[tcol], errors="coerce"),
                    pd.to_numeric(df[pcol], errors="coerce"),
                ],
                axis=0,
            )
            y = y.replace([np.inf, -np.inf], np.nan).dropna()
            if y.empty:
                continue
            ymin, ymax = float(y.min()), float(y.max())
            if v in var_minmax:
                cur_min, cur_max = var_minmax[v]
                var_minmax[v] = (min(cur_min, ymin), max(cur_max, ymax))
            else:
                var_minmax[v] = (ymin, ymax)

    # add padding
    padded: Dict[str, Tuple[float, float]] = {}
    for v, (mn, mx) in var_minmax.items():
        if not np.isfinite(mn) or not np.isfinite(mx):
            padded[v] = (mn, mx)
            continue
        if mx == mn:
            pad = max(1e-6, abs(mx) * 0.01)
        else:
            pad = (mx - mn) * Y_PAD_FRAC
        padded[v] = (mn - pad, mx + pad)
    return padded


def main():
    parser = argparse.ArgumentParser(description="Plot selected variables from forecast CSVs (with AUTO mode).")
    parser.add_argument("--auto", action="store_true", help="Enable AUTO mode using global or provided versions list.")
    parser.add_argument("--base_dir", default=BASE_DIR, help="Base directory for AUTO mode.")
    parser.add_argument("--versions", nargs="*", default=None, help="Override VERSIONS in AUTO mode.")
    parser.add_argument("--csv", nargs="+", help="Path(s) to CSV file(s). Ignored if --auto is used.")
    parser.add_argument("--variables", nargs="*", default=None, help="Variables to plot (base names without suffix).")
    parser.add_argument("--outdir", default=None, help="Directory to save plots. Defaults to <base_dir>/plots if not set.")
    parser.add_argument("--show", action="store_true", help="Show plots on screen.")
    parser.add_argument("--dpi", type=int, default=160, help="Output figure DPI.")
    args = parser.parse_args()

    # Determine CSV list
    if args.auto:
        versions = args.versions if args.versions is not None else VERSIONS
        csv_list = _collect_auto_csvs(args.base_dir, versions)
        if args.outdir is None:
            args.outdir = str(Path(args.base_dir) / "plots")
    else:
        if not args.csv:
            parser.error("Either --auto or --csv must be provided.")
        csv_list = args.csv
        if args.outdir is None:
            # Default to 'plots' next to the first CSV provided
            args.outdir = str(Path(csv_list[0]).parent / "plots")

    # Determine variables to plot
    if args.variables is not None and len(args.variables) > 0:
        variables = args.variables
    else:
        variables = VARIABLES_TO_PLOT  # may be empty -> auto-discover per file

    # First pass: compute unified y-limits per variable across all files
    var_ranges = _compute_var_ranges(csv_list, variables)

    # Process each CSV - second pass: plot with the computed y-limits
    for csv_path in csv_list:
        if not os.path.isfile(csv_path):
            print(f"[WARN] File not found: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[ERROR] Could not read {csv_path}: {e}")
            continue

        ts_col = _find_timestamp_column(df)
        if ts_col is None:
            print(f"[ERROR] No timestamp-like column found in {csv_path}. Columns: {list(df.columns)}")
            continue

        pairs = discover_pairs_pred_truth(df)  # keys normalized

        # Decide which variables to plot for this file (normalize selection)
        if variables:
            want_norm = [_normalize_base(v) for v in variables]
            vars_for_file = [vn for vn in want_norm if vn in pairs]
            missing = [variables[i] for i, vn in enumerate(want_norm) if vn not in pairs]
            for m in missing:
                print(f"[WARN] Variable '{m}' not found (truth/pred pair missing) in {csv_path}.")
        else:
            vars_for_file = list(pairs.keys())

        if not vars_for_file:
            print("[WARN] No variables to plot for this file. Skipping.")
            continue

        title_prefix = _infer_model_title_from_path(csv_path)

        for v_norm in vars_for_file:
            truth_col, pred_col = pairs[v_norm]

            # Use the original requested label if available, else the normalized key
            label_for_title = next((orig for orig in (variables or []) if _normalize_base(orig) == v_norm), v_norm)

            out_path = None
            if args.outdir:
                base = os.path.splitext(os.path.basename(csv_path))[0]
                out_path = os.path.join(args.outdir, f"{base}__{v_norm}.png")

            ylim = var_ranges.get(v_norm, None)

            plot_variable(
                df=df,
                ts_col=ts_col,
                truth_col=truth_col,
                pred_col=pred_col,
                title_prefix=title_prefix,
                var_display_name=label_for_title,
                out_path=out_path,
                show=args.show,
                dpi=args.dpi,
                ylim=ylim,
            )


if __name__ == "__main__":
    main()
