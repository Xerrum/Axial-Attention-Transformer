import argparse
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(date: str, version: str, title: str):
    """
    Load evaluation JSON by date and version, then plot the confusion matrix.
    Example filename: eval_20251015_211258.json
    """
    base_path = r"C:\Users\organ\PycharmProjects\presence_prediction\tud_presence_prediction\evaluations"
    file_name = f"eval_{date}_{version}.json"
    file_path = os.path.join(base_path, file_name)

    # --- Load data ---
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return

    # --- Extract predictions and targets ---
    y_true = data.get("targets")
    y_pred = data.get("predictions")

    if y_true is None or y_pred is None:
        print("❌ Error: JSON file must contain both 'targets' and 'predictions' keys.")
        return

    # --- Compute confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)

    # --- Plot confusion matrix ---
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Absent", "Present"])
    disp.plot(cmap="Blues", values_format='d', ax=ax, colorbar=False)  # remove legend (colorbar)

    ax.set_title(f"Confusion Matrix — {title}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()

    # --- Save plot ---
    plots_dir = os.path.join(base_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_file = os.path.join(plots_dir, f"confusion_matrix_{date}_{version}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {plot_file}")

    # --- Show plot ---
    plt.show()

    # --- Print raw matrix values ---
    print("✅ Confusion matrix:\n", cm)


if __name__ == "__main__":
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(description="Plot confusion matrix from eval JSON file.")
    parser.add_argument("date", type=str, help="Date of evaluation (format: yyyymmdd)")
    parser.add_argument("version", type=str, help="Version identifier (e.g., 211258)")
    parser.add_argument("plot_title", type=str, help="Title for the confusion matrix plot")
    args = parser.parse_args()

    plot_confusion_matrix(args.date, args.version, args.plot_title)
