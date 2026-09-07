#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_user_reports_json.py
Erzeugt pro User:
  1) Rot/Grün-Grid (Correct vs Incorrect)
  2) Confusion-Matrix (Counts + Normalized)
  3) Appendix: Ground Truth vs Prediction nebeneinander

Eingabe: JSON. Unterstützte Schemata:
A) "flat":  [{"user_id": "...", "day": int, "slot": int, "y_true": 0/1, "y_pred"| "y_score"| "logits": ...}, ...]
B) "users": {"users": [{"user_id": "...", "y_true": [[...],[...]], "y_pred"| "y_score"| "logits": [[...],[...]], "slots_per_day": S (optional)}]}

Beispiele:
  python make_user_reports_json.py --json eval_slots.json --slots-per-day 48 --outdir reports/
  python make_user_reports_json.py --json eval_slots.json --slots-per-day 48 --threshold 0.4 --outdir reports/
  python make_user_reports_json.py --json eval_slots_users.json --outdir reports/ --apply-sigmoid
"""

import argparse, os, json, math
from typing import Dict, Tuple, List, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ===== Plain ASCII console preview (pair mode only) =====
def _bin_str_line(arr, one='1', zero='0', miss='.', cast_int=True):
    # turn a row of {0,1,NaN} into e.g. "1010.0110"
    out = []
    for v in arr:
        if np.isnan(v):
            out.append(miss)
        else:
            out.append(one if (int(v) if cast_int else v >= 0.5) == 1 else zero)
    return "".join(out)

def _console_preview_user_pair_plain(user_id, day_labels, targets_2d, probs_2d, threshold=0.5):
    """
    Prints, for each prediction day:
        Day X  GT  <binary-string>
                PR  <binary-string at threshold>
    Example (48 slots): "GT 1010..."; "PR 1001..."
    """
    print(f"\n=== User {user_id} ===", flush=True)
    print("Legend: 1=at home, 0=not at home, .=missing", flush=True)
    for di, day_name in enumerate(day_labels):
        gt = targets_2d[di].astype(float)
        pr = probs_2d[di].astype(float)
        pb = np.where(np.isnan(pr), np.nan, (pr >= float(threshold)).astype(float))

        gt_line = _bin_str_line(gt, one='1', zero='0', miss='.')
        pr_line = _bin_str_line(pb, one='1', zero='0', miss='.')

        print(f"{day_name:<8} GT {gt_line}", flush=True)
        print(f"{'':8} PR {pr_line}  @thr={threshold:.2f}", flush=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

# ------------------------------ JSON Loader ------------------------------

def _detect_schema(obj: Any) -> str:
    """Rückgabe: 'flat', 'users', oder 'unknown'."""
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
    """
    Gibt Dict[user_id] -> (y_true[D,S], y_pred[D,S], day_labels) zurück.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    schema = _detect_schema(data)
    grouped: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {}

    if schema == "flat":
        # Erwartet Liste von Records
        recs = data
        # ggf. Feldnamen ermitteln
        pred_key = None
        sample = recs[0]
        if "y_pred" in sample: pred_key = "y_pred"
        elif "y_score" in sample: pred_key = "y_score"
        elif "logits" in sample: pred_key = "logits"
        else:
            raise ValueError("Kein 'y_pred'/'y_score'/'logits' Feld in JSON gefunden.")

        # Gruppieren nach User
        by_user: Dict[str, List[dict]] = {}
        for r in recs:
            by_user.setdefault(str(r["user_id"]), []).append(r)

        for uid, rows in by_user.items():
            # Tage/Slots
            days_sorted = sorted({int(r["day"]) for r in rows})
            day_to_idx = {d: i for i, d in enumerate(days_sorted)}
            D = len(days_sorted)

            S = slots_per_day_cli
            if S is None:
                S = max(int(r["slot"]) for r in rows) + 1  # inferieren
            y_true = np.full((D, S), np.nan, dtype=float)
            y_hat  = np.full((D, S), np.nan, dtype=float)

            for r in rows:
                di = day_to_idx[int(r["day"])]
                si = int(r["slot"])
                if si >= S:
                    continue
                y_true[di, si] = float(r["y_true"])
                y_hat[di, si]  = float(r[pred_key])

            # Scores/Logits → binär
            # Reihenfolge: logits→sigmoid; y_score: ggf. --apply-sigmoid; dann threshold
            vals = y_hat.copy()
            if pred_key == "logits" or apply_sigmoid:
                vals = _sigmoid(vals)
            if (pred_key in ["y_score", "logits"]) or apply_sigmoid:
                thr = 0.5 if threshold is None else float(threshold)
                y_pred = (vals >= thr).astype(float)
            else:
                y_pred = vals  # bereits binär

            day_labels = [f"Day {i+1}" for i in range(D)]
            grouped[uid] = (y_true, y_pred, day_labels)

    elif schema == "users":
        # Erwartet {"users": [ { "user_id": ..., "targets": [[...]],
        #                       "probs" | "y_pred" | "y_score" | "logits": [[...]] } ],
        #           "slots_per_day": S (optional) }
        users = data["users"]
        S_meta = data.get("slots_per_day")  # optional meta

        for u in users:
            uid = str(u.get("user_id", "unknown"))

            y_true = np.asarray(u.get("targets"))
            if y_true.ndim != 2:
                raise ValueError(f"User {uid}: 'targets' must be 2D (pred_days, slots).")
            D, S_true = y_true.shape

            # akzeptiere auch 'probs'
            if   "probs"   in u: pred_key = "probs"
            elif "y_pred"  in u: pred_key = "y_pred"
            elif "y_score" in u: pred_key = "y_score"
            elif "logits"  in u: pred_key = "logits"
            else:
                raise ValueError(f"User {uid}: need one of 'probs'/'y_pred'/'y_score'/'logits'.")

            scores = np.asarray(u[pred_key])
            if scores.shape != y_true.shape:
                raise ValueError(f"User {uid}: shape mismatch targets{y_true.shape} vs {pred_key}{scores.shape}.")

            # optional Konsistenzcheck zu S_meta
            if S_meta is not None and int(S_meta) != S_true:
                print(f"[WARN] User {uid}: slots_per_day meta={S_meta} != targets.shape[1]={S_true}")

            # -> Wahrscheinlichkeiten herstellen
            # 'logits' brauchen Sigmoid; 'probs' / 'y_score' sind [0..1]; 'y_pred' ist schon binär
            if pred_key == "logits" or apply_sigmoid:
                scores = 1.0 / (1.0 + np.exp(-scores))

            # binarisieren für die Grids/CM (Schwelle sweepbar)
            if pred_key in ["probs", "y_score", "logits"] or apply_sigmoid:
                thr = 0.5 if threshold is None else float(threshold)
                y_pred = (scores >= thr).astype(float)
            else:
                # 'y_pred' ist bereits 0/1, behalte gleichzeitig eine "Pseudo-Prob"
                y_pred = scores.astype(float)
                scores = scores.astype(float)

            day_labels = [f"Day {i+1}" for i in range(D)]
            # Falls du die probs im Renderer brauchen willst, kannst du sie zusätzlich mit zurückgeben.
            grouped[uid] = (y_true.astype(float), y_pred.astype(float), day_labels)

    else:
        raise ValueError(
            "Unbekanntes JSON-Format. Erwartet entweder eine Liste von Records "
            "mit Feldern ['user_id','day','slot','y_true', ...] oder ein Dict "
            "mit {'users':[{'user_id', 'y_true', 'y_pred'...}]}"
        )
    return grouped

# ------------------------------ Plot Helpers ------------------------------

def _fig_size(D: int, S: int, cell_w: float = None, cell_h: float = None) -> Tuple[float, float]:
    w = max(8.0, min((cell_w or 0.22) * S, 30.0))
    h = max(1.6, min((cell_h or 0.55) * D, 20.0))
    return w, h

def plot_correctness_grid(y_true: np.ndarray, y_pred: np.ndarray, day_labels: List[str],
                          title: str, outfile: str, dpi: int = 300, cell_lw: float = 0.6,
                          color_correct: str = "#2e7d32", color_incorrect: str = "#c62828"):
    D, S = y_true.shape
    correct = (y_true == y_pred) & ~np.isnan(y_true) & ~np.isnan(y_pred)
    grid_vals = correct.astype(int)

    x = np.arange(S + 1)
    y = np.arange(D + 1)
    fig_w, fig_h = _fig_size(D, S)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    facecolors = np.where(grid_vals == 1, color_correct, color_incorrect).astype(object)
    missing = np.isnan(y_true) | np.isnan(y_pred)
    facecolors[missing] = "#e0e0e0"

    ax.pcolormesh(x, y, np.zeros_like(grid_vals, dtype=float),
                  edgecolors="black", linewidth=cell_lw, facecolors=facecolors, shading="flat")
    ax.set_yticks(np.arange(D) + 0.5); ax.set_yticklabels(day_labels)
    ax.set_xticks([]); ax.set_xlabel("Time slots"); ax.set_title(title, pad=10); ax.invert_yaxis()
    ax.legend(handles=[
        Patch(facecolor=color_correct, edgecolor="black", label="Correct"),
        Patch(facecolor=color_incorrect, edgecolor="black", label="Incorrect"),
        Patch(facecolor="#e0e0e0", edgecolor="black", label="Missing")
    ], loc="upper right", frameon=True)
    plt.tight_layout(); plt.savefig(outfile, bbox_inches="tight"); plt.close()

def plot_true_pred_side_by_side(y_true: np.ndarray, y_pred: np.ndarray, day_labels: List[str],
                                title: str, outfile: str, dpi: int = 300, cell_lw: float = 0.6,
                                color_one: str = "#1b5e20", color_zero: str = "#37474f"):
    D, S = y_true.shape
    x = np.arange(S + 1); y = np.arange(D + 1)
    fig_w, fig_h = _fig_size(D, S); fig_w *= 1.6
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=dpi)

    for ax, mat, head in zip(axes, [y_true, y_pred], ["Ground Truth", "Prediction"]):
        vals = np.full_like(mat, -1, dtype=int)
        missing = np.isnan(mat)
        vals[~missing & (mat == 0)] = 0
        vals[~missing & (mat == 1)] = 1

        face = np.empty(vals.shape, dtype=object); face[:] = "#e0e0e0"
        face[vals == 0] = color_zero; face[vals == 1] = color_one

        ax.pcolormesh(x, y, np.zeros_like(vals, dtype=float),
                      edgecolors="black", linewidth=cell_lw, facecolors=face, shading="flat")
        ax.set_yticks(np.arange(D) + 0.5); ax.set_yticklabels(day_labels)
        ax.set_xticks([]); ax.set_xlabel("Time slots"); ax.set_title(head, pad=8); ax.invert_yaxis()

    fig.suptitle(title, y=1.02, fontsize=13)
    plt.tight_layout(); plt.savefig(outfile, bbox_inches="tight"); plt.close()

def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, title_prefix: str,
                            out_counts: str, out_norm: str, dpi: int = 300):
    from sklearn.metrics import confusion_matrix
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = y_true[mask].astype(int).flatten()
    yp = y_pred[mask].astype(int).flatten()

    labels = [0, 1]
    cm = confusion_matrix(yt, yp, labels=labels)
    cm_norm = confusion_matrix(yt, yp, labels=labels, normalize="true")

    def _draw(mat: np.ndarray, title: str, outfile: str):
        fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=dpi)
        im = ax.imshow(mat, interpolation="nearest")
        ax.set_title(title, pad=10)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["True 0", "True 1"])
        ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                text = f"{mat[i, j]:.2f}" if mat.dtype.kind == "f" else f"{int(mat[i, j])}"
                ax.text(j, i, text, ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout(); plt.savefig(outfile, bbox_inches="tight"); plt.close()

    _draw(cm, f"{title_prefix} – Confusion (Counts)", out_counts)
    _draw(cm_norm, f"{title_prefix} – Confusion (Normalized)", out_norm)

# ------------------------------ CLI ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Erzeuge pro User Grid- & Confusion-Reports aus JSON.")
    ap.add_argument("--json", type=str, required=True, help="Pfad zur JSON-Datei")
    ap.add_argument("--slots-per-day", type=int, default=None, help="Nur für 'flat'-Schema nötig, falls nicht ableitbar")
    ap.add_argument("--outdir", type=str, default="reports", help="Ausgabeverzeichnis")
    ap.add_argument("--threshold", type=float, default=None, help="Falls 'y_score' oder 'logits' genutzt werden")
    ap.add_argument("--apply-sigmoid", action="store_true",
                    help="Auf Werte Sigmoid anwenden (für Logits oder unkalibrierte Scores)")
    # Optik
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--cell-lw", type=float, default=0.6)
    ap.add_argument("--color-correct", type=str, default="#2e7d32")
    ap.add_argument("--color-incorrect", type=str, default="#c62828")
    ap.add_argument("--color-one", type=str, default="#1b5e20")
    ap.add_argument("--color-zero", type=str, default="#37474f")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    grouped = load_from_json(args.json,
                             slots_per_day_cli=args.slots_per_day,
                             threshold=args.threshold,
                             apply_sigmoid=args.apply_sigmoid)

    for uid, (y_true, y_pred, day_labels) in grouped.items():
        user_dir = os.path.join(args.outdir, f"user_{uid}")
        os.makedirs(user_dir, exist_ok=True)

        plot_correctness_grid(
            y_true, y_pred, day_labels,
            title=f"User {uid} – Correct vs Incorrect",
            outfile=os.path.join(user_dir, "grid_correctness.png"),
            dpi=args.dpi, cell_lw=args.cell_lw,
            color_correct=args.color_correct, color_incorrect=args.color-incorrect if False else args.color_incorrect
        )
        plot_confusion_matrices(
            y_true, y_pred, title_prefix=f"User {uid}",
            out_counts=os.path.join(user_dir, "confusion_counts.png"),
            out_norm=os.path.join(user_dir, "confusion_normalized.png"),
            dpi=args.dpi
        )
        plot_true_pred_side_by_side(
            y_true, y_pred, day_labels,
            title=f"User {uid} – Ground Truth vs Prediction",
            outfile=os.path.join(user_dir, "appendix_true_vs_pred.png"),
            dpi=args.dpi, cell_lw=args.cell_lw,
            color_one=args.color_one, color_zero=args.color_zero
        )
        print(f"[OK] User {uid}: Reports gespeichert unter {user_dir}")

if __name__ == "__main__":
    main()
