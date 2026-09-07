import json
import pandas as pd
import numpy as np
import torch
import os
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from tud_presence_prediction.models.LearnTransformerAxialSelf import LearnTransformer
from torch.utils.data import TensorDataset, DataLoader, random_split

# Konfiguration
BATCH_SIZE = 8
EPOCHS = 2
DAYS = 7
TIMESLOTS = 24
LOCATION_DIM = 8
CALENDAR_DIM = 3
HIDDEN_DIM = 8

# Synthetische Daten generieren
def generate_synthetic_data(num_samples=100):
    x_location = torch.randn(num_samples, DAYS, TIMESLOTS, LOCATION_DIM)
    x_calendar = torch.randn(num_samples, DAYS, TIMESLOTS, CALENDAR_DIM)
    
    # Zielwerte: Person ist tagsueber anwesend (8-18 Uhr) an Wochentagen
    target = torch.zeros(num_samples, DAYS, TIMESLOTS)
    for i in range(num_samples):
        for d in range(DAYS):
            # Montag-Freitag
            if d < 5:
                target[i, d, 8:18] = 1.0
    
    # Zufaellige Variationen
    noise = torch.rand(num_samples, DAYS, TIMESLOTS) < 0.1
    target = torch.where(noise, 1 - target, target)
    
    return x_location, x_calendar, target

def main():
    # Daten generieren
    print("Generiere Trainingsdaten...")
    x_location, x_calendar, target = generate_synthetic_data(200)
    
    # Train-Test-Split
    train_size = int(0.8 * len(x_location))
    train_dataset = TensorDataset(
        x_location[:train_size], 
        x_calendar[:train_size], 
        target[:train_size]
    )
    val_dataset = TensorDataset(
        x_location[train_size:], 
        x_calendar[train_size:], 
        target[train_size:]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Modell initialisieren
    print("Initialisiere Modell...")
    model = LearnTransformer(
        location_dim=LOCATION_DIM,
        calendar_dim=CALENDAR_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=4,
        dropout=0.1,
        num_of_layers=2,
        use_flash_attention=True,
        step_size=1
    )
    
    # Callbacks und Logger
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="transformer-{epoch}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss"
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )
    logger = TensorBoardLogger("tb_logs", name="transformer_model")
    
    # Training
    print("Starte Training...")
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10
    )
    trainer.fit(model, train_loader, val_loader)
    print("Training abgeschlossen!")
    
    # Bestes Modell laden
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Lade bestes Modell: {best_model_path}")
        model = LearnTransformer.load_from_checkpoint(best_model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
    # Forecasting
    print("Fuehre Forecast durch...")
    model.set_model_prediction_length(DAYS)
    model.eval()
    
    # Test mit einem Beispiel
    with torch.no_grad():
        device = next(model.parameters()).device  # Das aktuelle Geräts des Modells ermitteln
        test_input_loc = x_location[0:1].to(device)
        test_input_cal = x_calendar[0:1].to(device)
        print(f"shape input:\n Location: {test_input_loc.shape}")
        print(f"shape input:\n Calendar: {test_input_cal.shape}")
        predictions = model(test_input_loc, test_input_cal)
        predictions = torch.sigmoid(predictions)
    
    # Visualisierung
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.imshow(target[0].numpy(), cmap='Blues', aspect='auto')
    plt.colorbar(label='Tatsaechliche Anwesenheit')
    plt.title('Tatsaechliche Anwesenheit')
    plt.xlabel('Zeitslot')
    plt.ylabel('Tag')
    
    plt.subplot(2, 1, 2)
    plt.imshow(predictions[0].numpy(), cmap='Blues', aspect='auto')
    plt.colorbar(label='Vorhergesagte Anwesenheit')
    plt.title('Vorhergesagte Anwesenheit')
    plt.xlabel('Zeitslot')
    plt.ylabel('Tag')
    
    plt.tight_layout()
    plt.savefig('forecast_results.png')
    print("Ergebnisse wurden in 'forecast_results.png' gespeichert.")

if __name__ == "__main__":
    main()