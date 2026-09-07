import os
import sys
import joblib
import numpy as np

def print_scalar():
    scaler_path = r"C:\Users\organ\Downloads\scaler.joblib"  # Raw-String
    SINGLE_VARS = ["t2m", "d2m", "u10", "v10", "msl"] 
    PRESSURE_VARS = ["t", "r", "u", "v"]
    PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 300]  # hPa
    PREDICTION_YEAR = 2024
    YEARS_FOR_TRAINING = [2023]


    """Druckt den Inhalt eines joblib-Scalers in lesbarem Format"""
    if not os.path.exists(scaler_path):
        print(f"Fehler: Datei nicht gefunden: {scaler_path}")
    
    print(f"Lade Scaler aus: {scaler_path}")
    scaler = joblib.load(scaler_path)
        
    if not isinstance(scaler, dict) or "mean" not in scaler or "std" not in scaler:
        print("Warnung: Die geladene Datei hat nicht das erwartete Format")
        print(f"Inhalt: {scaler}")
        
    channel_names = []
    channel_names.extend(SINGLE_VARS)
    for level in PRESSURE_LEVELS:
        for var in PRESSURE_VARS:
            channel_names.append(f"{var}_{level}hPa")
                
    print(f"\n{'=' * 60}")
    print(f"SCALER STATISTIKEN")
    print(f"{'=' * 60}")
    print(f"Kanal".ljust(15) + "| " + "Mittelwert".ljust(15) + "| " + "Standardabweichung".ljust(20))
    print(f"{'-' * 60}")
        
    for i, channel in enumerate(channel_names):
        if i < len(scaler["mean"]):
            print(f"{channel[:14].ljust(15)}| {scaler['mean'][i]:<15.5f}| {scaler['std'][i]:<20.5f}")
        
    print(f"\n{'-' * 60}")
    print(f"Zusammenfassung:")
    print(f"  Mean Min: {scaler['mean'].min():.5f}, Max: {scaler['mean'].max():.5f}, Avg: {scaler['mean'].mean():.5f}")
    print(f"  Std  Min: {scaler['std'].min():.5f}, Max: {scaler['std'].max():.5f}, Avg: {scaler['std'].mean():.5f}")
    print(f"{'=' * 60}")
        
    # Am Ende des Skripts hinzufuegen
    input("\nDruecken Sie Enter, um zu beenden...")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

plt.style.use("seaborn-v0_8-whitegrid")

save_dir = r"C:\Users\organ\Downloads"
os.makedirs(save_dir, exist_ok=True)

# ---------- Plot 1: March 15–17 (with Time2Vec) ----------
# You have 48 data points → 48 timestamps.
# Create exactly 48 equally spaced 1-hour steps (or 30-min if you prefer the look)
hours_march = pd.date_range("2024-03-15 00:00", periods=48, freq="1h")

# visually digitized data (48 points each)
gt_march = np.array([
    280.00, 279.97, 272.18, 272.69, 273.38, 273.88, 274.27, 274.60, 274.95, 275.33, 275.73, 276.06,
    276.32, 276.49, 276.64, 276.79, 276.92, 277.00, 277.05, 277.07, 277.12, 277.27, 277.52, 277.78,
    277.98, 278.06, 278.08, 278.16, 278.27, 278.38, 278.49, 278.59, 278.69, 278.84, 278.99, 279.09,
    279.14, 279.17, 279.24, 279.39, 279.57, 279.70, 279.77, 279.82, 279.87, 279.92, 279.92, 279.82
])
pred_march = np.array([
    280.00, 280.00, 274.50, 274.55, 274.73, 274.87, 274.87, 274.79, 274.71, 274.38, 274.28, 274.57,
    274.92, 275.25, 275.43, 274.83, 274.86, 274.69, 274.19, 274.18, 274.54, 274.50, 274.14, 273.74,
    273.64, 273.37, 273.09, 272.88, 272.74, 272.74, 272.89, 273.11, 273.27, 273.45, 273.72, 273.91,
    274.09, 274.28, 274.50, 274.46, 273.73, 273.25, 273.04, 272.90, 272.55, 272.27, 272.17, 272.00
])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(hours_march, pred_march, "o-", color="tab:blue", label="Prediction @center")
ax.plot(hours_march, gt_march, "x-", color="tab:orange", label="Ground Truth @center")
ax.set_title("Center Pixel Time Series – t2m", fontsize=12)
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "t2m_center_time2vec.svg"), format="svg")
plt.close()

# ---------- Plot 2: January 15–17 (without Time2Vec) ----------
hours_jan = pd.date_range("2024-01-15 00:00", periods=48, freq="1h")

pred_jan = np.array([
    283.37, 281.32, 280.85, 280.90, 280.90, 280.59, 280.18, 279.78, 279.60, 279.51, 279.42, 279.20,
    278.75, 278.30, 278.03, 277.94, 278.17, 278.48, 278.75, 278.79, 278.70, 278.61, 278.52, 278.34,
    277.45, 277.36, 277.49, 277.63, 277.72, 278.03, 278.52, 279.06, 279.33, 279.38, 279.33, 279.24,
    279.06, 278.66, 278.24, 277.63, 277.18, 277.00, 277.01, 277.30, 277.64, 278.07, 278.38, 278.57
])
gt_jan = np.array([
    285.90, 285.66, 285.32, 284.99, 284.84, 284.85, 284.74, 285.00, 285.36, 285.61, 286.87, 287.52,
    288.00, 287.81, 287.44, 287.68, 286.23, 285.71, 285.27, 284.94, 284.60, 284.40, 284.27, 284.18,
    284.21, 284.08, 283.70, 283.63, 283.22, 282.36, 281.40, 280.78, 280.74, 281.17, 281.66, 282.28,
    282.68, 282.86, 283.06, 282.97, 282.63, 281.86, 281.04, 280.23, 278.99, 277.52, 275.73, 275.00
])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(hours_jan, pred_jan, "o-", color="tab:blue", label="Prediction @center")
ax.plot(hours_jan, gt_jan, "x-", color="tab:orange", label="Ground Truth @center")
ax.set_title("Center Pixel Time Series – t2m", fontsize=12)
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "t2m_center_no_time2vec.svg"), format="svg")
plt.close()

print("✅ Saved two SVG files to:", save_dir)
