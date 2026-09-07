#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json
from typing import Dict, Tuple, List, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from io import BytesIO
from PIL import Image

# optional sklearn import for your preferred confusion-matrix API
try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

VERSIONS = ["27dtj5h4", "kkldscdq"]
TITLES = ["Non-Dynamic", "Dynamic"]
CONTEXTS = [1,7,14,28]

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _detect_schema(obj: Any) -> str:
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
        keys = set(obj[0].keys())
        if {"user_id", "day", "slot", "y_true"} <= keys:
            return "flat"
    if isinstance(obj, dict) and "users" in obj and isinstance(obj["users"], list):
        return "users"
    return "unknown"

def load_from_json(json_path: str,
                   slots_per_day_cli: int = None,
                   threshold: float = None,
                   apply_sigmoid: bool = False) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    with open(json_path, "r") as f:
        data = json.load(f)

    schema = _detect_schema(data)
    grouped: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {}

    if schema == "flat":
        recs = data
        pred_key = None
        sample = recs[0]
        if "y_pred" in sample: pred_key = "y_pred"
        elif "y_score" in sample: pred_key = "y_score"
        elif "logits" in sample: pred_key = "logits"
        else:
            raise ValueError("No 'y_pred'/'y_score'/'logits' field found in JSON.")

        by_user: Dict[str, List[dict]] = {}
        for r in recs:
            by_user.setdefault(str(r["user_id"]), []).append(r)

        for uid, rows in by_user.items():
            days_sorted = sorted({int(r["day"]) for r in rows})
            day_to_idx = {d: i for i, d in enumerate(days_sorted)}
            D = len(days_sorted)

            S = slots_per_day_cli
            if S is None:
                S = max(int(r["slot"]) for r in rows) + 1
            y_true = np.full((D, S), np.nan, dtype=float)
            y_hat  = np.full((D, S), np.nan, dtype=float)

            for r in rows:
                di = day_to_idx[int(r["day"])]
                si = int(r["slot"])
                if si >= S: continue
                y_true[di, si] = float(r["y_true"])
                y_hat[di, si]  = float(r[pred_key])

            vals = y_hat.copy()
            if pred_key == "logits" or apply_sigmoid:
                vals = _sigmoid(vals)
            if (pred_key in ["y_score", "logits"]) or apply_sigmoid:
                thr = 0.5 if threshold is None else float(threshold)
                y_pred = (vals >= thr).astype(float)
            else:
                y_pred = vals

            # Prefer actual prediction dates from JSON if available
            dates_obj = u.get("dates", {}) if isinstance(u, dict) else {}
            pred_dates = dates_obj.get("prediction_dates", None)
            day_labels = [f"Day {i+1}" for i in range(D)]
            grouped[uid] = (y_true, y_pred, day_labels)

    elif schema == "users":
        users = data["users"]
        S_meta = data.get("slots_per_day")

        for u in users:
            uid = str(u.get("user_id", "unknown"))
            y_true = np.asarray(u.get("targets"), dtype=float)
            if y_true.ndim != 2:
                raise ValueError(f"User {uid}: 'targets' must be 2D (pred_days, slots).")
            D, S_true = y_true.shape

            if   "probs"   in u: pred_key = "probs"
            elif "y_pred"  in u: pred_key = "y_pred"
            elif "y_score" in u: pred_key = "y_score"
            elif "logits"  in u: pred_key = "logits"
            else:
                raise ValueError(f"User {uid}: need one of 'probs'/'y_pred'/'y_score'/'logits'.")

            scores = np.asarray(u[pred_key], dtype=float)
            if scores.shape != y_true.shape:
                raise ValueError(f"User {uid}: shape mismatch targets{y_true.shape} vs {pred_key}{scores.shape}.")

            if S_meta is not None and int(S_meta) != S_true:
                print(f"[WARN] User {uid}: slots_per_day meta={S_meta} != targets.shape[1]={S_true}")

            if pred_key == "logits" or apply_sigmoid:
                scores = _sigmoid(scores)
            if pred_key in ["probs", "y_score", "logits"] or apply_sigmoid:
                thr = 0.5 if threshold is None else float(threshold)
                y_pred = (scores >= thr).astype(float)
            else:
                y_pred = scores.astype(float)

            dates_obj = u.get("dates", {}) or {}
            pred_dates = dates_obj.get("prediction_dates", None)
            if isinstance(pred_dates, list) and len(pred_dates) == D:
                day_labels = [str(d) for d in pred_dates]
            else:
                day_labels = [f"Day {i+1}" for i in range(D)]
            grouped[uid] = (y_true, y_pred, day_labels)

    else:
        raise ValueError("Unknown JSON format.")

    return grouped

def _fig_size(D: int, S: int, cell_w: float = None, cell_h: float = None) -> Tuple[float, float]:
    w = max(8.0, min((cell_w or 0.22) * S, 30.0))
    h = max(1.6, min((cell_h or 0.55) * D, 20.0))
    return w, h

def plot_correctness_grid(y_true: np.ndarray, y_pred: np.ndarray, day_labels: List[str],
                          title: str, outfile: str, dpi: int = 300, cell_lw: float = 0.6,
                          color_correct: str = "#2e7d32", color_incorrect: str = "#c62828"):
    D, S = y_true.shape
    correct = (y_true == y_pred) & ~np.isnan(y_true) & ~np.isnan(y_pred)
    vals = np.full((D, S), -1, dtype=int)  # -1: missing, 0: incorrect, 1: correct
    vals[(~np.isnan(y_true) & ~np.isnan(y_pred)) & (~correct)] = 0
    vals[correct] = 1

    cmap = ListedColormap(["#e0e0e0", color_incorrect, color_correct])  # missing, incorrect, correct
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    fig_w, fig_h = _fig_size(D, S)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(vals, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

    # grid lines
    for x in range(S + 1):
        ax.axvline(x - 0.5, color="black", linewidth=cell_lw)
    for y in range(D + 1):
        ax.axhline(y - 0.5, color="black", linewidth=cell_lw)

    # y-axis (days)
    ax.set_yticks(np.arange(D))
    ax.set_yticklabels(day_labels)

    # x-axis: ticks at 0, 3, 8, 13, ..., 48, S-1 (for S=52), label each with actual time.
    # Assumptions:
    #   - Slot 0 corresponds to 8:00 AM
    #   - Each slot advances by 15 minutes
    def _format_time_from_index(idx: int, start_hour: int = 8, start_minute: int = 0, step_min: int = 15) -> str:
        total_min = start_hour * 60 + start_minute + idx * step_min
        hour_24 = (total_min // 60) % 24
        minute = total_min % 60
        ampm = "AM" if hour_24 < 12 else "PM"
        hour_12 = hour_24 % 12
        if hour_12 == 0:
            hour_12 = 12
        return f"{hour_12}:{minute:02d} {ampm}"

    if S >= 1:
        # build center ticks: start at 3, step 5, stop before (S-3)
        center_step = 5
        center_start = 3
        center_stop = max(center_start, S - 3)  # exclusive in range()
        centers = list(range(center_start, center_stop, center_step))

        tick_positions = [0] + centers + ([S - 1] if S > 1 else [])
        tick_labels = [_format_time_from_index(i) for i in tick_positions]

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        # Optional: shrink labels a bit so they fit
        ax.tick_params(axis='x', labelsize=8)
    else:
        ax.set_xticks([])


    ax.set_xlabel("Time slots")  # keep your existing label
    # No title / legend here
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()



def plot_true_pred_side_by_side(y_true: np.ndarray, y_pred: np.ndarray, day_labels: List[str],
                                title: str, outfile: str, dpi: int = 300, cell_lw: float = 0.6,
                                color_one: str = "#2e7d32", color_zero: str = "#c62828"):
    """
    Displays Ground truth and Prediction stacked vertically,
    with each grid at half the original height.
    """
    D, S = y_true.shape

    # keep width, reduce height by half
    fig_w, fig_h = _fig_size(D, S)
    fig_h *= 1.2  # previously 1.6

    fig, axes = plt.subplots(
        2, 1,
        figsize=(fig_w, fig_h),
        dpi=dpi,
        gridspec_kw={"height_ratios": [0.7, 0.7]}  # equal half height per grid
    )

    def draw(ax, mat, head):
        vals = np.full(mat.shape, -1, dtype=int)  # -1 missing, 0 absent, 1 present
        m = ~np.isnan(mat)
        vals[m & (mat == 0)] = 0
        vals[m & (mat == 1)] = 1
        cmap = ListedColormap(["#e0e0e0", color_zero, color_one])  # missing, 0, 1
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

        ax.imshow(vals, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

        # cell borders
        for x in range(S + 1):
            ax.axvline(x - 0.5, color="black", linewidth=cell_lw)
        for y in range(D + 1):
            ax.axhline(y - 0.5, color="black", linewidth=cell_lw)

        ax.set_yticks(np.arange(D))
        ax.set_yticklabels(day_labels)
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_title(head, pad=6)

    # exact headings
    draw(axes[0], y_true, "Ground truth")
    draw(axes[1], y_pred, "Prediction")

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight", dpi=dpi)
    plt.close()



def plot_context_grid(ctx: np.ndarray, day_labels: List[str],
                      title: str, outfile: str, dpi: int = 300, cell_lw: float = 0.6,
                      color_one: str = "#BEF28A", color_zero: str = "#FFC9C8"):
    # Same rendering as before, only the writing changed per request
    D, S = ctx.shape
    fig_w, fig_h = _fig_size(D, S)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    vals = np.full(ctx.shape, -1, dtype=int)   # -1 missing, 0 absent, 1 present
    m = ~np.isnan(ctx)
    vals[m & (ctx == 0)] = 0
    vals[m & (ctx == 1)] = 1

    cmap = ListedColormap(["#e0e0e0", color_zero, color_one])  # missing, absent, present
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    ax.imshow(vals, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

    for x in range(S + 1):
        ax.axvline(x - 0.5, color="black", linewidth=cell_lw)
    for y in range(D + 1):
        ax.axhline(y - 0.5, color="black", linewidth=cell_lw)

    ax.set_yticks(np.arange(D)); ax.set_yticklabels(day_labels)
    # no x ticks, no xlabel here
    ax.set_xticks([]); ax.set_xlabel("")
    # enforce exact title wording
    ax.set_title("Context", pad=8)

    plt.tight_layout(); plt.savefig(outfile, bbox_inches="tight"); plt.close()


def save_presence_legend(
    outfile: str,
    color_one: str = "#BEF28A",      # Present (context/GT/Pred)
    color_zero: str = "#FFC9C8",     # Absent  (context/GT/Pred)
    color_correct: str = "#2e7d32",  # Correct
    color_incorrect: str = "#c62828",# Incorrect
    dpi: int = 300
):
    """
    Saves a clean, vertical legend image:
    ┌──────────────┐
    │ ■ Present    │
    │ ■ Absent     │
    │ ■ Correct    │
    │ ■ Incorrect  │
    └──────────────┘
    """

    # Make the figure tall enough so the labels are not clipped
    fig_height = 1.2 * 4   # 1.2 inches per row
    fig = plt.figure(figsize=(2.4, fig_height), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis("off")

    handles = [
        Patch(facecolor=color_one, edgecolor="black", label="Present"),
        Patch(facecolor=color_zero, edgecolor="black", label="Absent"),
        Patch(facecolor=color_correct, edgecolor="black", label="Correct"),
        Patch(facecolor=color_incorrect, edgecolor="black", label="Incorrect"),
    ]

    legend = ax.legend(
        handles=handles,
        ncol=1,                  # one per row
        loc="center",
        frameon=True,
        fontsize=11,
        handlelength=1.4,
        handleheight=1.4,
        borderpad=0.8,
        labelspacing=1.0
    )

    # Adjust layout to prevent cut-off
    plt.subplots_adjust(left=0.15, right=0.85, top=0.98, bottom=0.02)
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = y_true[mask].astype(int).flatten()
    yp = y_pred[mask].astype(int).flatten()
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    return cm

def plot_confusion_counts_total(
    y_true=None, y_pred=None,
    title: str = "", outfile: str = "", dpi: int = 300,
    totals: np.ndarray = None,
    figsize: Tuple[float, float] = (6.0, 6.0),
):
    """
    Confusion matrix mit TP oben links (Present/Present) und ohne abgeschnittene Labels.

    Konvention:
      - Positive class = 1 ("Present")
      - Negative class = 0 ("Absent")
      - Rows = True label, Cols = Predicted label
      - Oben links = TP
    """
    import numpy as _np
    import matplotlib.pyplot as _plt

    # Figure: constrained layout verhindert abgeschnittene Achsen-/Figuren-Texte
    try:
        fig = _plt.figure(figsize=figsize, dpi=dpi, layout='constrained')  # Matplotlib ≥3.4
    except TypeError:
        fig = _plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)

    # === Matrix berechnen in [1,0]-Reihenfolge (Zeile 0 = true=1; Spalte 0 = pred=1) ===
    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        if totals is not None:
            # totals in sklearn-Standard [[TN, FP],[FN, TP]] -> in TP-first-Ansicht umordnen
            cm_std = _np.asarray(totals, dtype=int)
            cm_tp_first = _np.array([
                [cm_std[1, 1], cm_std[1, 0]],  # [TP, FN]
                [cm_std[0, 1], cm_std[0, 0]]   # [FP, TN]
            ], dtype=int)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_tp_first,
                                          display_labels=["Present", "Absent"])
            disp.plot(cmap="Blues", values_format='d', ax=ax, colorbar=False)
            for txt in np.ravel(disp.text_):   # ConfusionMatrixDisplay exposes the text artists
                txt.set_fontsize(20)           # increase size
                txt.set_fontweight('bold')     # make them bold
        else:
            # labels=[1,0] erzwingt oben links = TP
            cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=["Present", "Absent"])
            disp.plot(cmap="Blues", values_format='d', ax=ax, colorbar=False)

        # Feinschliff
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)

    except Exception:
        # Fallback ohne sklearn, erwartet totals im Standard
        if totals is None:
            raise ValueError("Manual fallback requires `totals`.")
        cm_std = _np.asarray(totals, dtype=int)
        cm_tp_first = _np.array([
            [cm_std[1, 1], cm_std[1, 0]],
            [cm_std[0, 1], cm_std[0, 0]]
        ], dtype=int)
        im = ax.imshow(cm_tp_first, cmap=_plt.cm.Blues, aspect='equal', interpolation='nearest')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Present", "Absent"])
        ax.set_yticklabels(["Present", "Absent"])
        maxv = cm_tp_first.max() if cm_tp_first.size > 0 else 0
        for i in range(2):
            for j in range(2):
                val = int(cm_tp_first[i, j])
                color = "white" if maxv > 0 and val > maxv * 0.5 else "black"
                ax.text(j, i, f"{val}", ha="center", va="center", fontsize=25, color=color)

    # Achsentitel & Suptitle, ohne das Layout zu stören
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    try:
        fig.suptitle(title, y=0.99, fontsize=12)  # suptitle sauber positionieren
    except Exception:
        fig.suptitle(title, fontsize=12)

    # Wenn 'layout' nicht verfügbar war, sorge manuell für Ränder
    if not hasattr(fig, "get_constrained_layout") and not getattr(fig, "_layoutbox", None):
        # konservative Ränder, damit Y-Label nicht abgeschnitten wird
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.14, top=0.90)

    # Wichtig: kein bbox_inches='tight' -> das beschneidet oft Achsenlabels
    fig.savefig(outfile, dpi=dpi)
    _plt.close(fig)

def plot_metrics_by_context(versions=None, base_dir="evaluations", out_dir_base=None, threshold=0.5, apply_sigmoid=False):
    """
    Creates bar charts for accuracy, precision and recall by context length for each version.
    Calculates metrics based on raw predictions with the given threshold.

    Adds: visible tiny bars for true zeros + an explicit "0" label above them.
    """
    if versions is None:
        versions = VERSIONS

    if out_dir_base is None:
        out_dir_base = os.path.join(base_dir, "plots")

    # Metrics we want to calculate and plot
    metrics_to_plot = ["accuracy", "precision", "recall"]
    metric_colors = ["#1F77B4", "#FF7F0E", "#2CA02C"]  # Blue, Orange, Green

    # ---- visibility parameters for true zeros ----
    MIN_ZERO_BAR = 0.010   # ~0.4% of full y-range (0..1). Big enough to render, still "tiny".
    ZERO_LABEL_PAD = 0.003 # distance above the tiny bar for the "0" text

    def calculate_metrics(y_true, y_pred):
        # Flatten and filter out NaN values
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        yt = y_true[mask].flatten()
        yp = y_pred[mask].flatten()

        if len(yt) == 0:  # No valid data
            return {metric: 0 for metric in metrics_to_plot}

        # Confusion matrix counts
        tp = np.sum((yt == 1) & (yp == 1))
        tn = np.sum((yt == 0) & (yp == 0))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))

        # Metrics
        metrics = {}
        metrics["accuracy"]  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics["recall"]    = tp / (tp + fn) if (tp + fn) > 0 else 0
        return metrics

    # Process each version
    for version_idx, version in enumerate(versions):
        out_dir = os.path.join(out_dir_base, version)
        os.makedirs(out_dir, exist_ok=True)

        title = TITLES[version_idx] if version_idx < len(TITLES) else f"Version {version}"

        metrics_data = {metric: [] for metric in metrics_to_plot}
        context_values = []

        for ctx in CONTEXTS:
            json_path = os.path.join(base_dir, f"eval_v2_ctx{ctx}_{version}.json")
            if not os.path.exists(json_path):
                print(f"[WARN] File not found: {json_path}")
                continue

            try:
                grouped = load_from_json(
                    json_path,
                    threshold=threshold,
                    apply_sigmoid=apply_sigmoid
                )

                all_true, all_pred = [], []
                for _, (y_true, y_pred, _) in grouped.items():
                    all_true.append(y_true)
                    all_pred.append(y_pred)

                if all_true and all_pred:
                    all_true_combined = np.concatenate([arr.flatten() for arr in all_true])
                    all_pred_combined = np.concatenate([arr.flatten() for arr in all_pred])
                    metrics = calculate_metrics(all_true_combined, all_pred_combined)

                    context_values.append(ctx)
                    for metric in metrics_to_plot:
                        metrics_data[metric].append(metrics[metric])

            except Exception as e:
                print(f"[ERROR] Error processing {json_path}: {e}")

        if not context_values:
            print(f"[WARN] No data found for version {version}")
            continue

        plt.figure(figsize=(9, 5.2))
        bar_width = 0.9 / len(metrics_to_plot)
        idx = np.arange(len(context_values))

        for i, (metric, color) in enumerate(zip(metrics_to_plot, metric_colors)):
            offset = (i - len(metrics_to_plot) / 2 + 0.5) * bar_width

            # draw bars normally first
            bars = plt.bar(idx + offset, metrics_data[metric], width=bar_width, color=color, label=metric.capitalize())

            # Now post-process zeros: enforce a minimum visible height + outline + "0" label
            for j, (bar, val) in enumerate(zip(bars, metrics_data[metric])):
                if val == 0:
                    # 1) force a tiny height so it's visible
                    bar.set_height(MIN_ZERO_BAR)
                    # 2) move the bar's bottom to zero (should already be)
                    bar.set_y(0.0)
                    # 3) label "0" just above that tiny bar
                    x_center = bar.get_x() + bar.get_width() / 2.0
                    plt.text(x_center, MIN_ZERO_BAR + ZERO_LABEL_PAD, "0",
                             ha="center", va="bottom", fontsize=10, color="black", zorder=4)

        plt.xlabel("Context Days")
        plt.ylabel("Value")
        # plt.title(f"Metrics by Context Length – {title} Model")
        plt.xticks(idx, [str(c) for c in context_values])
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1.0, 0.1))
        plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)  # Add grid for better visibility
        plt.legend()

        out_path = os.path.join(out_dir, f"metrics_by_context_{version}_{threshold}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[OK] Metrics plot saved: {out_path}")



def main():
    ap = argparse.ArgumentParser(description="Create per-user grids and a global confusion matrix from JSON.")
    version_group = ap.add_mutually_exclusive_group(required=True)
    version_group.add_argument("--version", type=str, help="Version of model used for evaluation")
    version_group.add_argument("--auto", action="store_true", help="Process all versions in VERSIONS list")
    ap.add_argument("--slots-per-day", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--apply-sigmoid", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--cell-lw", type=float, default=0.6)
    ap.add_argument("--color-correct", type=str, default="#2e7d32")
    ap.add_argument("--color-incorrect", type=str, default="#c62828")
    ap.add_argument("--color-one", type=str, default="#BEF28A")
    ap.add_argument("--color-zero", type=str, default="#FFC9C8")
    ap.add_argument("--title_", type=str, default="Dynamic")
    ap.add_argument("--plot-metrics", action="store_true", help="Plot metrics by context length")
    args = ap.parse_args()

    # Which versions to process
    versions_to_process = VERSIONS if args.auto else [args.version]

    # Optional "metrics by context" mode
    if args.plot_metrics:
        plot_metrics_by_context(versions=versions_to_process, threshold=args.threshold)
        return

    # ---- Standard plotting flow (per version, per user) ----
    # ---- Standard plotting flow (per version, per context, per user) ----
    for i, version in enumerate(versions_to_process):
        print(f"Processing version: {version}")

        # Select a title for this version (fall back to args.title_)
        if args.auto:
            title = TITLES[i] if i < len(TITLES) else f"Version {version}"
        else:
            try:
                title_index = VERSIONS.index(version)
                title = TITLES[title_index]
            except (ValueError, IndexError):
                title = args.title_

        # iterate over all configured contexts
        for ctx in CONTEXTS:
            in_path = f"evaluations/eval_v2_ctx{ctx}_{version}.json"
            out_dir = f"evaluations/plots/{version}/ctx{ctx}"

            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(os.path.dirname(in_path), exist_ok=True)

            if not os.path.exists(in_path):
                print(f"[WARN] Missing file for context={ctx}: {in_path}")
                continue

            # Load grouped data for this context
            grouped = load_from_json(
                in_path,
                slots_per_day_cli=args.slots_per_day,
                threshold=args.threshold,
                apply_sigmoid=args.apply_sigmoid
            )

            # Pull context arrays + labels (if present) from JSON for per-user context grids
            with open(in_path, "r") as _f:
                _raw = json.load(_f)
            _ctx_map = {
                str(u.get("user_id", "unknown")): np.asarray(u["context_targets"], dtype=float)
                for u in _raw.get("users", [])
                if "context_targets" in u
            }
            _ctx_labels_map = {
                str(u.get("user_id", "unknown")): (u.get("dates", {}) or {}).get("context_dates", None)
                for u in _raw.get("users", [])
            }

            # === aggregate confusion totals over all users for this context ===
            totals = np.zeros((2, 2), dtype=int)  # sklearn order [[TN,FP],[FN,TP]]

            for uid, (y_true, y_pred, day_labels) in grouped.items():
                user_dir = os.path.join(out_dir, f"user_{uid}")
                os.makedirs(user_dir, exist_ok=True)

                # --- Context grid (exact titles/labels behavior preserved) ---
                ctx_arr = _ctx_map.get(uid, None)
                raw_labels = _ctx_labels_map.get(uid, None)
                if (ctx_arr is not None) and isinstance(raw_labels, list) and len(raw_labels) == (ctx_arr.shape[0] if hasattr(ctx_arr, "shape") else 0):
                    ctx_day_labels = [str(d) for d in raw_labels]
                else:
                    ctx_day_labels = [f"Day {i+1}" for i in range(ctx_arr.shape[0])] if isinstance(ctx_arr, np.ndarray) else day_labels

                # if isinstance(ctx_arr, np.ndarray):
                    # plot_context_grid(
                    #     ctx=ctx_arr,
                    #     day_labels=ctx_day_labels,
                    #     title="Context",
                    #     outfile=os.path.join(user_dir, f"context_{version}_{args.threshold}.png"),
                    #     dpi=args.dpi,
                    #     cell_lw=args.cell_lw,
                    #     color_one=args.color_one,
                    #     color_zero=args.color_zero
                    # )

                # # --- Ground truth / Prediction grids ---
                # plot_true_pred_side_by_side(
                #     y_true=y_true,
                #     y_pred=y_pred,
                #     day_labels=day_labels,
                #     title="",
                #     outfile=os.path.join(user_dir, f"gt_pred_{version}_{args.threshold}.png"),
                #     dpi=args.dpi,
                #     cell_lw=args.cell_lw,
                #     color_one=args.color_one,
                #     color_zero=args.color_zero
                # )

                # # --- Correctness grid ---
                # plot_correctness_grid(
                #     y_true=y_true,
                #     y_pred=y_pred,
                #     day_labels=day_labels,
                #     title="",
                #     outfile=os.path.join(user_dir, f"correctness_{version}_{args.threshold}.png"),
                #     dpi=args.dpi,
                #     cell_lw=args.cell_lw,
                #     color_correct=args.color_correct,
                #     color_incorrect=args.color_incorrect
                # )

            #     # --- accumulate confusion counts for this user into totals ---
            #     # Mask NaNs then compute sklearn-style confusion matrix
            #     mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
            #     yt = y_true[mask].astype(int).ravel()
            #     yp = y_pred[mask].astype(int).ravel()
            #     if yt.size > 0:
            #         if _HAVE_SK:
            #             cm_user = confusion_matrix(yt, yp, labels=[0, 1])  # [[TN,FP],[FN,TP]]
            #         else:
            #             # manual fallback
            #             tn = int(((yt == 0) & (yp == 0)).sum())
            #             fp = int(((yt == 0) & (yp == 1)).sum())
            #             fn = int(((yt == 1) & (yp == 0)).sum())
            #             tp = int(((yt == 1) & (yp == 1)).sum())
            #             cm_user = np.array([[tn, fp], [fn, tp]], dtype=int)
            #         totals += cm_user

            # # Save a legend (one per context for tidy folders)
            # legend_path = os.path.join(out_dir, f"legend_presence_{version}_ctx{ctx}.png")
            # save_presence_legend(
            #     outfile=legend_path,
            #     color_one=args.color_one,
            #     color_zero=args.color_zero,
            #     color_correct=args.color_correct,
            #     color_incorrect=args.color_incorrect,
            #     dpi=args.dpi
            # )
            # print(f"[OK] Legend saved: {legend_path}")

            # Global confusion matrix for this context (TP-first layout internally handled)
            # cm_outfile = os.path.join(out_dir, f"confusion_counts_total_{version}_ctx{ctx}_{args.threshold}.png")
            # plot_confusion_counts_total(
            #     y_true=None, y_pred=None,
            #     title=f"Confusion Matrix — {title} Presence Prediction (ctx={ctx} days)",
            #     outfile=cm_outfile,
            #     dpi=args.dpi,
            #     totals=totals
            # )
            # print(f"[OK] Total confusion matrix saved: {cm_outfile}")



        # # Global confusion matrix across users (keeps your TP-first styling)
        # plot_confusion_counts_total(
        #     y_true=None, y_pred=None,
        #     title=f"Confusion Matrix — {title} Presence Prediction",
        #     outfile=os.path.join(out_dir, f"confusion_counts_total_{version}_{args.threshold}.png"),
        #     dpi=args.dpi,
        #     totals=totals
        # )
        # print(f"[OK] Total confusion matrix saved at {os.path.join(out_dir, f'confusion_counts_total_{version}_{args.threshold}.png')}")

if __name__ == "__main__":
    main()
