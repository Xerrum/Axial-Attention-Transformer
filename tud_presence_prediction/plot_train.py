import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
file_path = r"C:\Users\organ\Downloads\metrics.csv"

# Load the metrics
df = pd.read_csv(file_path)

# Create masks for the three series
mask_step  = df['train_loss_step'].notna()
mask_epoch = df['train_loss_epoch'].notna()
mask_val   = df['val_loss'].notna()

plt.figure(figsize=(12, 6))

# 1) Train loss per step
plt.plot(
    df.loc[mask_step, 'step'],
    df.loc[mask_step, 'train_loss_step'],
    label='Train Loss (step)',
    alpha=0.7
)

# 2) Train loss per epoch (dashed + square markers)
plt.plot(
    df.loc[mask_epoch, 'step'],
    df.loc[mask_epoch, 'train_loss_epoch'],
    label='Train Loss (epoch)'
)

# 3) Validation loss (dash-dot + circle markers)
plt.plot(
    df.loc[mask_val, 'step'],
    df.loc[mask_val, 'val_loss'],
    label='Validation Loss',
    color = "red"
)

# Formatting
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
