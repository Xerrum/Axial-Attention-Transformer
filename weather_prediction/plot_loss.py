# plotTrain_Val_loss_cli.py
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE_DIR = r"C:\Users\organ\PycharmProjects\presence_prediction\weather_prediction\train_results"

VERSIONS = ["0j5y1a5u", "gg9008fi", "g0la7ajz", "78s3mxnx"]
TITLES   = ["Regular Model", "Fine-Tuned Regular Model", "Long-Context Transformer", "Finetuned Long-Context Transformer"]

# Colors by MODEL FAMILY (not by finetune state)
COLOR_REGULAR = "#1F77B4"  # blue
COLOR_LONGCTX = "#FF7F0E"  # orange

# Desired limits:
TRAIN_LIMIT_REGULAR = 2.0
VAL_LIMIT_REGULAR   = 0.5
TRAIN_LIMIT_FT      = 0.5
VAL_LIMIT_FT        = 0.2

def is_finetuned(title: str) -> bool:
    return "fine" in title.lower()

def is_long_context(title: str) -> bool:
    return "long" in title.lower()

def choose_limit(kind: str, title: str, override: float | None) -> float:
    if override is not None:
        return override
    ft = is_finetuned(title)
    if kind == "train":
        return TRAIN_LIMIT_FT if ft else TRAIN_LIMIT_REGULAR
    else:
        return VAL_LIMIT_FT if ft else VAL_LIMIT_REGULAR

def choose_color(title: str) -> str:
    return COLOR_LONGCTX if is_long_context(title) else COLOR_REGULAR

def read_xy(fp):
    x, y = [], []
    with open(fp, 'r') as f:
        for line in f:
            values = line.strip().split(';')
            if len(values) < 2:
                continue
            x.append(int(values[0]))
            y.append(float(values[1]))
    return x, y

def plot_one(kind: str, version: str, title: str, color=None, ylimit=None):
    infile = os.path.join(BASE_DIR, f"{kind}_{version}.csv")
    x, y = read_xy(infile)

    plt.figure()
    plt.plot(x, y, color=color)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    # plt.title(f"{'Train' if kind == 'train' else 'Val'}-Loss for {title}")
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    plt.grid(True)

    if ylimit is not None:
        plt.ylim(0, ylimit)

    output_dir = os.path.join(BASE_DIR, version)
    os.makedirs(output_dir, exist_ok=True)
    safe_title = title.replace(" ", "_")
    outfile = os.path.join(output_dir, f"{kind}_{version}_{safe_title}.png")
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {outfile}")

# NEW: one-split, two-curves (Regular vs Long-Context) figure
def plot_pair(kind: str,
              reg_version: str, reg_title: str,
              long_version: str, long_title: str,
              label_prefix: str,
              ylimit_override: float | None):
    assert kind in ("train", "val")

    reg_fp  = os.path.join(BASE_DIR, f"{kind}_{reg_version}.csv")
    long_fp = os.path.join(BASE_DIR, f"{kind}_{long_version}.csv")
    x_r, y_r = read_xy(reg_fp)
    x_l, y_l = read_xy(long_fp)

    c_reg  = choose_color(reg_title)   # blue
    c_long = choose_color(long_title)  # orange

    # y-limit: override or max of auto limits for the two titles
    if ylimit_override is not None:
        ylimit = ylimit_override
    else:
        ylimit = max(choose_limit(kind, reg_title, None),
                     choose_limit(kind, long_title, None))

    plt.figure(figsize=(7.5, 4.5))
    linestyle = '-' if kind == 'train' else '--'

    plt.plot(x_r, y_r, linestyle=linestyle, linewidth=2.0, color=c_reg,
             label=f"{label_prefix} {reg_title}")
    plt.plot(x_l, y_l, linestyle=linestyle, linewidth=2.0, color=c_long,
             label=f"{label_prefix} {long_title}")

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    nice_kind = "Train" if kind == "train" else "Validation"
    plt.title(f"{nice_kind} Loss — {reg_title} vs {long_title}")
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    plt.grid(True)
    plt.legend(loc="best", frameon=True)
    plt.ylim(0, ylimit)

    out_dir = os.path.join(BASE_DIR, "duo_figures")
    os.makedirs(out_dir, exist_ok=True)
    safe = f"{reg_title}__vs__{long_title}".replace(" ", "_")
    outfile = os.path.join(out_dir, f"duo_{label_prefix.strip().lower()}_{kind}_{safe}.png")
    plt.savefig(outfile, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved duo plot: {outfile}")

def main():
    parser = argparse.ArgumentParser(description="Plot train/val loss from Regular/Long-Context model logs.")
    # Existing modes
    parser.add_argument("--auto", action="store_true", help="Plot all versions defined in the script (separate charts).")
    parser.add_argument("--version", type=str, help="Version number (used in filenames train_{version}.csv / val_{version}.csv).")
    parser.add_argument("--title", type=str, help="Model name for plot title and output filename.")
    parser.add_argument("--ylimit", type=float, help="Upper limit for y-axis (loss value). Overrides auto logic if set.")

    # NEW: duo mode — FOUR separate plots
    parser.add_argument("--duo", action="store_true",
                        help="Create FOUR plots: base train, base val, finetuned train, finetuned val — each overlays Regular vs Long-Context.")

    # Indices into VERSIONS/TITLES for convenience (match your defaults)
    parser.add_argument("--reg_idx", type=int, default=0, help="Index for Regular Model (default: 0).")
    parser.add_argument("--reg_ft_idx", type=int, default=1, help="Index for Fine-Tuned Regular Model (default: 1).")
    parser.add_argument("--long_idx", type=int, default=2, help="Index for Long-Context Transformer (default: 2).")
    parser.add_argument("--long_ft_idx", type=int, default=3, help="Index for Finetuned Long-Context Transformer (default: 3).")

    args = parser.parse_args()

    if args.duo:
        try:
            reg_ver, reg_title       = VERSIONS[args.reg_idx],     TITLES[args.reg_idx]
            long_ver, long_title     = VERSIONS[args.long_idx],    TITLES[args.long_idx]
            reg_ft_ver, reg_ft_title = VERSIONS[args.reg_ft_idx],  TITLES[args.reg_ft_idx]
            long_ft_ver,long_ft_title= VERSIONS[args.long_ft_idx], TITLES[args.long_ft_idx]
        except IndexError:
            raise SystemExit("[ERROR] One of the provided *_idx values is out of range.")

        # 1) Base TRAIN
        plot_pair("train", reg_ver, reg_title, long_ver, long_title,
                  label_prefix="Base", ylimit_override=args.ylimit)
        # 2) Base VAL
        plot_pair("val",   reg_ver, reg_title, long_ver, long_title,
                  label_prefix="Base", ylimit_override=args.ylimit)
        # 3) Finetuned TRAIN
        plot_pair("train", reg_ft_ver, reg_ft_title, long_ft_ver, long_ft_title,
                  label_prefix="Finetuned", ylimit_override=args.ylimit)
        # 4) Finetuned VAL
        plot_pair("val",   reg_ft_ver, reg_ft_title, long_ft_ver, long_ft_title,
                  label_prefix="Finetuned", ylimit_override=args.ylimit)
        return

    # Prior behaviors stay intact
    if args.auto:
        for version, title in zip(VERSIONS, TITLES):
            color = choose_color(title)
            for kind in ("train", "val"):
                ylimit = choose_limit(kind, title, args.ylimit)
                plot_one(kind, version, title, color=color, ylimit=ylimit)
        return

    if args.version and args.title:
        color = choose_color(args.title)
        for kind in ("train", "val"):
            ylimit = choose_limit(kind, args.title, args.ylimit)
            plot_one(kind, args.version, args.title, color=color, ylimit=ylimit)
        return

    parser.print_help()

if __name__ == "__main__":
    main()
