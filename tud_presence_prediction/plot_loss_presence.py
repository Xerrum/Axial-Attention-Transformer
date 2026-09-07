# plotTrain_Val_loss_cli.py
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = r"C:\Users\organ\PycharmProjects\presence_prediction\tud_presence_prediction\training_results"

# Lists are still used for optional multi-mode, but you can ignore that.
VERSIONS = ["27dtj5h4", "kkldscdq"]
TITLES   = ["Non-Dynamic Model", "Dynamic Model"]
COLORS   = ["#1F77B4", "#FF7F0E"]

# Titles off by default
SHOW_TITLES = False

# Explicit version → color mapping (robust)
VERSION2COLOR = {
    "27dtj5h4": "#1F77B4",  # Non-Dynamic → blue
    "kkldscdq": "#FF7F0E",  # Dynamic → orange
}

# -----------------------------
# Helpers
# -----------------------------
def read_xy(fp):
    """Read x (steps) and y (metric) from a comma-separated CSV with quotes."""
    import csv
    x, y = [], []
    if not os.path.exists(fp):
        print(f"[WARN] File not found: {fp}")
        return x, y

    try:
        with open(fp, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header if present
            for row in reader:
                # Expect at least 5 columns (0=step, 4=value)
                if len(row) >= 5:
                    try:
                        x.append(int(row[0]))
                        y.append(float(row[4]))
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"[ERROR] Failed to read file {fp}: {e}")

    print(f"[INFO] Extracted {len(x)} data points from {os.path.basename(fp)}")
    return x, y


def metric_label(name: str) -> str:
    """Map filename metric → label for y-axis/title (handles common typos)."""
    n = (name or "").strip().lower()
    mapping = {
        "loss": "Loss",
        "recall": "Recall",
        "precision": "Precision",
        "presicion": "Precision",
        "accuracy": "Accuracy",
        "accurracy": "Accuracy",
        "f1": "F1 Score",
        "f1score": "F1 Score",
        "f1_score": "F1 Score",
    }
    return mapping.get(n, name.capitalize())


def parse_filename(fp):
    """
    Robustly parse '<kind>_<metric>_<version>[...].csv'.

    - 'kind' is the first token (train/val)
    - 'version' is any token that matches one of VERSIONS
    - 'metric' is everything between kind and version, excluding 'training'
    """
    base = os.path.splitext(os.path.basename(fp))[0]
    parts = base.split("_")
    if len(parts) < 3:
        return None, None, None

    kind = parts[0]

    # find version token among parts using the known VERSIONS list
    version = None
    for tok in parts[1:]:
        if tok in VERSIONS:
            version = tok
            break

    # fallback: last token if nothing matched explicitly
    if version is None:
        version = parts[-1]

    # metric tokens are all between kind and version, excluding 'training'
    metric_tokens = []
    for tok in parts[1:]:
        if tok == version:
            break
        if tok.lower() == "training":
            continue
        metric_tokens.append(tok)

    metric = "_".join(metric_tokens) if metric_tokens else ""
    return kind, metric, version


def auto_glob(version=None, include_training=True):
    """
    Find all train/val metric CSVs for the given version or all versions if none specified.
    Returns a list of (kind, metric, version, training_suffix) tuples.
    """
    if version:
        patt_train = os.path.join(BASE_DIR, f"train_*_{version}*.csv")
        patt_val   = os.path.join(BASE_DIR, f"val_*_{version}*.csv")
    else:
        patt_train = os.path.join(BASE_DIR, "train_*.csv")
        patt_val   = os.path.join(BASE_DIR, "val_*.csv")

    files = sorted(glob.glob(patt_train)) + sorted(glob.glob(patt_val))
    parsed = []
    for fp in files:
        base = os.path.splitext(os.path.basename(fp))[0]
        is_training = base.endswith("_training")

        if is_training and not include_training:
            continue

        if is_training:
            # strip the '_training' to parse kind/metric/version
            base_wo_training = base[:-9]  # len("_training") == 9
            k, m, v = parse_filename(base_wo_training + ".csv")
            if k and (m is not None) and (v or version):
                v_final = version or v      # trust the filter when provided
                parsed.append((k, m, v_final, "_training"))
        else:
            k, m, v = parse_filename(fp)
            if k and (m is not None) and (v or version):
                v_final = version or v      # trust the filter when provided
                parsed.append((k, m, v_final, ""))

    return parsed

# -----------------------------
# Plotting
# -----------------------------
def plot_one(kind: str, metric: str, version: str, modelname: str,
             color=None, grid=True, suffix="", save=False, outdir=None):
    infile = os.path.join(BASE_DIR, f"{kind}_{metric}_{version}{suffix}.csv")
    x, y = read_xy(infile)
    if not x:
        print(f"[ERROR] No data in {infile}. Skipping.")
        return

    plt.figure(figsize=(7, 4))

    # color bound to version, independent of titles
    color_to_use = color or VERSION2COLOR.get(version)
    if color_to_use:
        plt.plot(x, y, linewidth=2, color=color_to_use)
    else:
        plt.plot(x, y, linewidth=2)

    plt.xlabel("Steps")
    plt.ylabel(metric_label(metric))

    if SHOW_TITLES:
        title_suffix = " (Training)" if suffix == "_training" else ""
        plt.title(f"{kind.capitalize()} {metric_label(metric)}{title_suffix} for {modelname}")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

    # --- Set y-axis limits depending on metric ---
    if metric.lower() == "loss":
        ax.set_ylim(0, 2)
    else:
        ax.set_ylim(0, 1)

    # --- Enable both horizontal and vertical grid lines ---
    if grid:
        ax.grid(True, which="both", axis="both", linestyle="--", linewidth=0.6, alpha=0.5)


    plt.tight_layout()

    if save:
        # save next to the CSV file (default), unless outdir is provided
        target_dir = outdir or os.path.dirname(infile)
        os.makedirs(target_dir, exist_ok=True)
        fname = f"{kind}_{metric}_{version}{suffix}.png"
        fpath = os.path.join(target_dir, fname)
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved plot -> {fpath}")
        plt.close()
    else:
        plt.show()

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot train/val metrics from training logs.")
    parser.add_argument("--version", type=str, help="Optional version filter for --auto")
    parser.add_argument("--modelname", type=str, help="Model name for titles and file names")
    parser.add_argument("--color", type=str, help="Override line color (e.g. '#FF0000')")
    parser.add_argument("--grid", action="store_true", help="Add grid lines to the plot")
    parser.add_argument("--save", action="store_true", help="Save plots instead of only showing them")
    parser.add_argument("--outdir", type=str, help="Override output directory; defaults to CSV folder")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--auto", action="store_true", help="Auto-detect and plot all metrics in the folder")
    group.add_argument("--multi", action="store_true", help="Use predefined VERSIONS and TITLES for plotting")
    group.add_argument("--single", nargs=2, metavar=("KIND", "METRIC"),
                       help="Manual: KIND in {train,val}, METRIC in {loss,accuracy,precision,recall,f1,...}")

    args = parser.parse_args()

    if args.auto:
        # If a version is provided, auto-glob only that version; otherwise all
        items = auto_glob(args.version)
        if not items:
            print(f("[WARN] No CSV files found in {BASE_DIR}"))
            return

        # group by version for better organization
        by_version = {}
        for kind, metric, version, suffix in items:
            by_version.setdefault(version, []).append((kind, metric, version, suffix))

        print(f"[INFO] Found {len(items)} metric files across {len(by_version)} versions.")
        for version, files in by_version.items():
            modelname = args.modelname or f"Version_{version}"
            print(f"[INFO] Plotting {len(files)} metrics for version {version}...")
            for kind, metric, v, suffix in files:
                color = args.color or VERSION2COLOR.get(v)
                plot_one(kind, metric, v, modelname, color, args.grid, suffix,
                         save=args.save, outdir=args.outdir)

    elif args.multi:
        # Optional mode, not needed if you run --auto
        if len(VERSIONS) != len(TITLES) or len(VERSIONS) != len(COLORS):
            print(f"[ERROR] VERSIONS, TITLES and COLORS must have same length. "
                  f"{len(VERSIONS)} vs {len(TITLES)} vs {len(COLORS)}")
        for version, title, _unused in zip(VERSIONS, TITLES, COLORS):
            items = auto_glob(version)
            if not items:
                print(f"[WARN] No matching CSV files found for version '{version}' in {BASE_DIR}")
                continue
            for kind, metric, v, suffix in items:
                color = args.color or VERSION2COLOR.get(v)
                plot_one(kind, metric, v, title, color, args.grid, suffix,
                         save=args.save, outdir=args.outdir)

    elif args.single:
        if not args.version:
            print("[ERROR] --version is required with --single")
            return
        kind, metric = args.single
        modelname = args.modelname or f"Version_{args.version}"
        color = args.color or VERSION2COLOR.get(args.version)
        plot_one(kind, metric, args.version, modelname, color, args.grid,
                 save=args.save, outdir=args.outdir)

    else:
        print("[ERROR] Please specify an action: --auto, --multi, or --single")

if __name__ == "__main__":
    main()
