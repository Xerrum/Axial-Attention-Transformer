import os
import glob
import time
import argparse
import json
import pandas as pd
from typing import Optional, Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime, timedelta
import wandb
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
from pathlib import Path
import tempfile

from sklearn.preprocessing import StandardScaler
import joblib

# NetCDF4-based loader
import netCDF4 as nc
from netCDF4 import num2date

# Import the WeatherTransformer model
from weather_prediction.weather_transformer import WeatherTransformer
from weather_prediction.weather_axialtransformer import WeatherAxialTransformer

# Wettervariablen-Definition:
# SINGLE_VARS: Meteorologische Variablen auf Einzelniveau (nahe Erdoberfläche)
#   - t2m: 2-Meter-Temperatur [K]
#   - d2m: 2-Meter-Taupunkttemperatur [K]
#   - u10: 10-Meter-Windgeschwindigkeit (Ost-West Komponente) [m/s]
#   - v10: 10-Meter-Windgeschwindigkeit (Nord-Süd Komponente) [m/s]
#   - msl: Luftdruck auf Meereshöhe [Pa]
# 
# PRESSURE_VARS: Meteorologische Variablen auf verschiedenen Druckniveaus
#   - t: Temperatur [K]
#   - r: Relative Luftfeuchtigkeit [%]
#   - u: Windgeschwindigkeit (Ost-West Komponente) [m/s]
#   - v: Windgeschwindigkeit (Nord-Süd Komponente) [m/s]

# SINGLE_VARS = ["t2m", "d2m", "u10", "v10", "msl"] 
# PRESSURE_VARS = ["t", "r", "u", "v"]
# PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 300]

SINGLE_VARS = ["t2m", "d2m", "u10", "v10", "msl"] 
PRESSURE_VARS = ["t", "r", "u", "v"]
PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 300]  # hPa
PREDICTION_YEAR = 2024
YEARS_FOR_TRAINING = [2023]
HOURS_IN_DAY = 24


def load_contiguous_years(
    year: int,
    include_prev: bool = True,
    include_next: bool = False,
    normalize: bool = True,
    scaler=None,
):
    """
    Load and stitch multiple years contiguously into a single [T,H,W,C] series.
    Default: previous year + evaluation year. Optionally include next year.

    Returns:
        times_sorted: np.ndarray(datetime64[h]) of shape [T]
        lat, lon: 1D coords
        X_sorted_norm: np.ndarray [T,H,W,C] (normalized if normalize=True)
    """
    years = []
    if include_prev:
        years.append(year - 1)
    years.append(year)
    if include_next:
        years.append(year + 1)

    times_list, X_list = [], []
    lat_ref = lon_ref = None

    for y in years:
        if y < PREDICTION_YEAR:
            data_dir = os.path.join(os.path.expanduser("~/scratch/era5_data"), "past")
        else:
            data_dir = os.path.expanduser("~/scratch/era5_data")

        single_paths = sorted(glob.glob(os.path.join(data_dir, f"era5_single_{y}_*.netcdf")))
        pressure_paths = sorted(glob.glob(os.path.join(data_dir, f"era5_pressure_{y}_*.netcdf")))
        if not single_paths or not pressure_paths:
            print(f"[WARNING] No data files for year {y} in {data_dir}, skipping.")
            continue

        t, lat, lon, X = build_stacked_tensor_single_plus_pressure(single_paths, pressure_paths)

        # ensure same grid across years
        if lat_ref is None:
            lat_ref, lon_ref = lat, lon
        else:
            # reuse your grid check for consistency
            _ensure_same_grid(lat_ref, lon_ref, lat, lon)

        times_list.append(t)
        X_list.append(X)

    if not times_list:
        raise FileNotFoundError("No yearly data could be loaded to form a contiguous series.")

    times = np.concatenate(times_list, axis=0)
    X = np.concatenate(X_list, axis=0)

    # sort by time (robustness)
    order = np.argsort(times)
    times_sorted = times[order]
    X_sorted = X[order]

    X_sorted = X_sorted.astype(np.float32, copy=False)
    if normalize:
        if scaler is None:
            print("[WARNING] load_contiguous_years: no scaler provided; building a fresh one (not recommended).")
            stats = compute_channel_stats(X_sorted)
            X_sorted_norm = apply_channel_scaling(X_sorted, stats, inplace=True)
        else:
            X_sorted_norm = apply_channel_scaling(X_sorted, scaler, inplace=True)
    else:
        X_sorted_norm = X_sorted

    return times_sorted, lat_ref, lon_ref, X_sorted_norm


def _read_time_from_var(ds: nc.Dataset, var_name: str) -> np.ndarray:
    """Return time as numpy datetime64[h] using the first dimension of the given variable."""
    v = ds.variables[var_name]
    time_dim_name = v.dimensions[0]  # assume [time, lat, lon]
    time_var = ds.variables[time_dim_name]
    dts = num2date(time_var[:], time_var.units)
    return np.array(dts, dtype="datetime64[h]")


def _ensure_same_grid(lat_s, lon_s, lat_p, lon_p):
    if lat_s.shape != lat_p.shape or lon_s.shape != lon_p.shape:
        raise ValueError("Lat/Lon grids differ between single-level and pressure-level files.")
    if not (np.allclose(lat_s, lat_p) and np.allclose(lon_s, lon_p)):
        raise ValueError("Lat/Lon coordinates mismatch between datasets.")


def load_single_level_nc(paths: list):
    """Load single-level arrays with exact variable names.
    Returns (times, lat, lon, transposed) with array shaped [T, H, W, C].
    """
    all_data = []
    times_all = []
    lat_ref = lon_ref = None

    for fp in sorted(paths):
        with nc.Dataset(fp) as ds:
            # read reference time from t2m
            t_arr = _read_time_from_var(ds, "t2m")
            
            # get coordinates
            lat = np.asarray(ds.variables["latitude"][:])
            lon = np.asarray(ds.variables["longitude"][:])

            # store reference coordinates from first file
            if lat_ref is None:
                lat_ref, lon_ref = lat, lon
            
            # collect all variables for this file
            file_vars = []
            for var_name in SINGLE_VARS:
                if var_name in ds.variables:
                    data = ds.variables[var_name][:]
                    file_vars.append(data)
            
            # Stack variables for this file
            file_stacked = np.stack(file_vars, axis=-1)  # [T, H, W, n_vars]
            print(f"Aus dem Pfad {fp} ist entstanden: {file_stacked.shape}")
            all_data.append(file_stacked)
            times_all.append(t_arr)
    
    # Concatenate all files along time dimension
    transposed = np.concatenate(all_data, axis=0)
    times = np.concatenate(times_all, axis=0)
    
    print(f"[INFO] Single level data loaded: shape={transposed.shape}, time range={times.shape}")
    
    return times, lat_ref, lon_ref, transposed


def load_pressure_level_nc(paths: list):
    """Load pressure-level arrays with exact variable names.
    Returns (times, lat, lon, data_array) with array shaped [T, H, W, L*C].
    where L is the number of pressure levels.
    """
    all_data = []
    times_all = []
    lat_ref = lon_ref = None

    for fp in sorted(paths):
        with nc.Dataset(fp) as ds:
            # Get pressure levels
            levels = np.asarray(ds.variables["pressure_level"][:])
            
            # Convert to hPa if needed
            levels_hpa = levels if levels.max() <= 1100 else levels / 100.0
            
            # Read time and coordinates
            t_arr = _read_time_from_var(ds, "t")
            
            lat = np.asarray(ds.variables["latitude"][:])
            lon = np.asarray(ds.variables["longitude"][:])
            
            if lat_ref is None:
                lat_ref, lon_ref = lat, lon
            
            # Prepare data for all pressure levels and variables
            time_steps = t_arr.shape[0]
            combined_data = np.zeros((time_steps, lat.shape[0], lon.shape[0], 
                                     len(PRESSURE_LEVELS) * len(PRESSURE_VARS)))
            
            # Extract data for each level and variable
            for level_idx, level in enumerate(PRESSURE_LEVELS):
                # Find closest available level
                closest_level_idx = np.abs(levels_hpa - level).argmin()
                
                for var_idx, var_name in enumerate(PRESSURE_VARS):
                    if var_name in ds.variables:
                        # Extract data for this level
                        data = ds.variables[var_name][:, closest_level_idx, :, :]
                        # Place in combined array
                        channel_idx = level_idx * len(PRESSURE_VARS) + var_idx
                        combined_data[:, :, :, channel_idx] = data 

            all_data.append(combined_data) # combined data has shape [T, H, W, L*C]
            times_all.append(t_arr)

            print(f"Aus dem Pfad {fp} ist entstanden: {combined_data.shape}")

    # Concatenate all files along time dimension
    transposed = np.concatenate(all_data, axis=0)
    times = np.concatenate(times_all, axis=0)
    
    print(f"[INFO] Pressure level data loaded: shape={transposed.shape}, time range={times.shape}")
    
    return times, lat_ref, lon_ref, transposed


def compute_channel_stats(X):  # X: [B,T,C,H,W] od. [N,C] od. [T,H,W,C]
    if X.ndim == 5:
        B,T,C,H,W = X.shape
        Xr = X.reshape(-1, C)                # [B*T*H*W, C]
    elif X.ndim == 2:
        _, C = X.shape
        Xr = X
    elif X.ndim == 4:  # für [T,H,W,C]
        T,H,W,C = X.shape
        Xr = X.reshape(-1, C)                # [T*H*W, C]
    else:
        raise ValueError(f"compute_channel_stats: unerwartete Shape {X.shape} mit {X.ndim} Dimensionen")
    mu  = Xr.mean(axis=0)                    # [C]
    std = Xr.std(axis=0, ddof=0)            # [C]
    std[std < 1e-6] = 1e-6                  # Stabilität
    return {"mean": mu.astype(np.float32), "std": std.astype(np.float32)}

def apply_channel_scaling(X, stats, *, inplace=False, eps=1e-6):  # X: [B,T,C,H,W] or [T,H,W,C]
    # ensure float32 once
    X = X.astype(np.float32, copy=not inplace)

    if X.ndim == 5:  # [B,T,C,H,W]
        mu  = stats["mean"][None, None, :, None, None].astype(np.float32, copy=False)
        std = stats["std"][None, None, :, None, None].astype(np.float32, copy=False)
    elif X.ndim == 4:  # [T,H,W,C]
        mu  = stats["mean"][None, None, None, :].astype(np.float32, copy=False)
        std = stats["std"][None, None, None, :].astype(np.float32, copy=False)
    else:
        raise ValueError(f"apply_channel_scaling: unexpected shape {X.shape} with {X.ndim} dims")

    # in-place: X = (X - mu) / (std + eps)
    np.subtract(X, mu, out=X, casting='unsafe')
    np.divide(X, std + eps, out=X, casting='unsafe')
    return X


def invert_channel_scaling(Xn, stats):  # Xn: [B,T,C,H,W] od. [T,H,W,C]
    if Xn.ndim == 5:  # [B,T,C,H,W]
        mu  = stats["mean"][None,None,:,None,None]
        std = stats["std"][None,None,:,None,None]
    elif Xn.ndim == 4:  # [T,H,W,C]
        mu  = stats["mean"][None,None,None,:]
        std = stats["std"][None,None,None,:]
    else:
        raise ValueError(f"invert_channel_scaling: unerwartete Shape {Xn.shape} mit {Xn.ndim} Dimensionen")
    return Xn * (std + 1e-6) + mu


def build_stacked_tensor_single_plus_pressure(single_paths: list, pressure_paths: list) -> tuple:
    """Build combined tensor from single-level and pressure-level data.
    Data has shape [T,H,W,C]
    Returns (times, lat, lon, X) where X has shape [T, H, W, C].
    """
    # Load data from both file types
    t_s, lat_s, lon_s, data_s = load_single_level_nc(single_paths)
    t_p, lat_p, lon_p, data_p = load_pressure_level_nc(pressure_paths)
    _ensure_same_grid(lat_s, lon_s, lat_p, lon_p)

    # Find common timestamps between the datasets
    common_t = np.intersect1d(t_s, t_p)
    if len(common_t) == 0:
        raise ValueError("No overlapping time stamps between datasets")
    
    print(f"[INFO] Found {len(common_t)} common timestamps between datasets")
    
    # Get indices for the common timestamps
    idx_s = np.nonzero(np.in1d(t_s, common_t))[0]
    idx_p = np.nonzero(np.in1d(t_p, common_t))[0]
    
    # Combine data along the channel dimension
    X = np.concatenate([data_s[idx_s], data_p[idx_p]], axis=-1)

    print(f"[INFO] Created Return Tensor X with all Channels. Shape of X: {X.shape}") # [T,H,W,C]



    return common_t, lat_s, lon_s, X


def build_time_scalars(times_array) -> np.ndarray:
    """
    times_array: np.ndarray von numpy.datetime64 (Stundenauflösung)
    Ausgabe: [T,2]  (hour_frac, doy_frac) in [0,1]
    """
    import pandas as pd
    T = len(times_array)
    out = np.zeros((T, 2), dtype=np.float32)
    for i in range(T):
        ts = pd.Timestamp(times_array[i])
        out[i, 0] = (ts.hour + ts.minute / 60.0) / 24.0   # hour_frac
        out[i, 1] = ts.dayofyear / 365.25                 # doy_frac
    return out


class ERA5SpatiotemporalDataset(Dataset):
    """Dataset fuer ERA5-Wetterdaten, das Sequenzen fuer Training und Prediction bereitstellt."""
    def __init__(
        self,
        data_root: str = None,
        context_len: int = 12,
        prediction_steps: int = 24,
        normalize: bool = True,
        area_subset: Optional[Tuple[float, float, float, float]] = None,  # optionaler Geo-Ausschnitt
    ):
        super().__init__()

        # Standardmaessig das Verzeichnis auf ~/scratch/era5_data/past setzen
        if data_root is None:
            data_root = os.path.join(os.path.expanduser("~/scratch/era5_data"), "past")
        
        # Pfade fuer die NetCDF-Dateien
        single_paths = sorted(glob.glob(os.path.join(os.path.expanduser(data_root), "era5_single_*.netcdf")))
        pressure_paths = sorted(glob.glob(os.path.join(os.path.expanduser(data_root), "era5_pressure_*.netcdf")))
        
        # Filterung nach Jahren fuer das Training
        print(f"[INFO] Filtering training data for years: {YEARS_FOR_TRAINING}")
        single_paths = [p for p in single_paths if any(f"_{year}_" in p for year in YEARS_FOR_TRAINING)]
        pressure_paths = [p for p in pressure_paths if any(f"_{year}_" in p for year in YEARS_FOR_TRAINING)]
        
        if len(single_paths) == 0 or len(pressure_paths) == 0:
            raise FileNotFoundError(f"No NetCDF files found for specified years {YEARS_FOR_TRAINING} in {data_root}.")
        
        print(f"[INFO] Found {len(single_paths)} single-level files and {len(pressure_paths)} pressure-level files for years {YEARS_FOR_TRAINING}")

        # Rest der Methode bleibt unveraendert
        times, lat, lon, X = build_stacked_tensor_single_plus_pressure(single_paths, pressure_paths) # X has shape [T,H,W,C]

        # optional geographic subset (simple index-based subset)
        if area_subset is not None:
            lat_min, lat_max, lon_min, lon_max = area_subset
            lat_idx = np.where((np.minimum(lat_min, lat_max) <= lat) & (lat <= np.maximum(lat_min, lat_max)))[0]
            lon_idx = np.where((np.minimum(lon_min, lon_max) <= lon) & (lon <= np.maximum(lon_min, lon_max)))[0]
            X = X[:, :, lat_idx[:, None], lon_idx]
            self.lat = lat[lat_idx]
            self.lon = lon[lon_idx]
        else:
            self.lat = lat
            self.lon = lon

        self.time = times
        self.X = X.astype(np.float32)  # [T, H, W, C]
        self.context_len = int(context_len)
        self.prediction_steps = int(prediction_steps)
        self.normalize = normalize

        print("[INFO] Starting Normalization")
        if self.normalize:
            # Originale Form merken
            original_shape = self.X.shape
            T, H, W, C = original_shape
            
            stats = compute_channel_stats(self.X)  # mean/std pro Kanal
            self.X = apply_channel_scaling(self.X, stats)
            self.scaler = stats

        else:
            self.scaler = None

        self.T_total, self.H, self.W, self.C = self.X.shape
        print(f"[INFO] Dataset loaded with shape: [T={self.T_total}, H={self.H}, W={self.W}, C={self.C}]")
        print(f"[INFO] Time range: {np.datetime_as_string(times[0])} to {np.datetime_as_string(times[-1])}")

    def __len__(self):
        # Anzahl der moeglichen Startindizes fuer Sequenzen
        return self.T_total - (self.context_len + self.prediction_steps) + 1

    def __getitem__(self, idx):
        # Hol dir Sequenz aus Vergangenheit + Zukunft
        T_seq = self.context_len + self.prediction_steps
        seq_hw_c = self.X[idx:idx + T_seq]                 # [T_seq, H, W, C]
        times_slice = self.time[idx:idx + T_seq]           # [T_seq]
        time_feats  = build_time_scalars(times_slice)      # [T_seq, 2]

        # Channel-Dimension nach vorne
        seq = torch.from_numpy(seq_hw_c).permute(0, 3, 1, 2)       # [T_seq, C, H, W]
        time_feats = torch.from_numpy(time_feats)                  # [T_seq, 2]
        return seq, time_feats



def create_weather_dataloaders(
    data_root: str = None,
    batch_size: int = 1,
    context_len: int = 12,
    prediction_steps: int = 24,
    num_workers: int = 4,
    block_size: int = 3,  # Groesse der Bloecke fuer das Shuffling
    val_fraction: float = 0.15,  # Anteil fuer Validierung
):
    """
    Erstellt Trainings- und Validierungs-DataLoaders fuer Wetterdaten.
    
    Folgt der Logik von homebrew_presence.py:
    1. Daten werden in zusammenhaengende Bloecke aufgeteilt
    2. Bloecke werden gemischt (nicht die Daten innerhalb eines Blocks)
    3. Bloecke werden in Train und Validation aufgeteilt
    
    Args:
        data_root: Pfad zum Verzeichnis mit den Trainingsdaten
        batch_size: Batch-Groesse fuer DataLoader
        context_len: Anzahl der Kontext-Zeitschritte fuer jedes Sample
        prediction_steps: Anzahl der Prediction-Zeitschritte fuer jedes Sample
        num_workers: Anzahl der Worker-Prozesse fuer DataLoader
        block_size: Anzahl der zusammenhaengenden Samples pro Block
        val_fraction: Anteil der Daten fuer Validierung (0-1)
    
    Returns:
        tuple: (train_loader, val_loader, in_channels, scaler)
            - train_loader: DataLoader fuer Trainingsdaten
            - val_loader: DataLoader fuer Validierungsdaten
            - in_channels: Anzahl der Input-Kanaele
            - scaler: Trainierter StandardScaler für Normalisierung
    """
    # Standardmaessig den past-Ordner fuer Trainingsdaten verwenden
    if data_root is None:
        data_root = os.path.join(os.path.expanduser("~/scratch/era5_data"), "past")
    
    print(f"[INFO] Loading training data from {data_root}")
    
    # Dataset laden
    ds = ERA5SpatiotemporalDataset(
        data_root=data_root,
        context_len=context_len,
        prediction_steps=prediction_steps,
        normalize=True,
        area_subset=None,
    )
    
    # Bestimme die Gesamtanzahl von Samples und Zeitstempel
    N = len(ds)
    times = ds.time
    
    # Erstelle zusammenhaengende Bloecke
    all_blocks = []
    for i in range(0, N, block_size):
        # Sammle Indizes fuer diesen Block
        block_indices = []
        for j in range(block_size):
            if i + j < N:
                block_indices.append(i + j)
        
        # Block hinzufuegen, wenn nicht leer
        if block_indices:
            # Berechne Start- und Endzeit fuer den Block (fuer Logging)
            start_time = times[block_indices[0]]
            end_idx = block_indices[-1] + (context_len + prediction_steps - 1)
            if end_idx < len(times):
                end_time = times[end_idx]
            else:
                end_time = times[-1]
            all_blocks.append((block_indices, start_time, end_time))
    
    # Mische die Bloecke (mit festem Seed fuer Reproduzierbarkeit)
    np.random.seed(42)
    np.random.shuffle(all_blocks)
    
    # Aufteilung in Train und Val
    n_blocks = len(all_blocks)
    n_val = max(1, int(round(n_blocks * val_fraction)))
    n_train = n_blocks - n_val
    
    train_blocks = all_blocks[:n_train]
    val_blocks = all_blocks[n_train:]
    
    # Extrahiere Indizes fuer jeden Split
    train_indices = [idx for block, _, _ in train_blocks for idx in block]
    val_indices = [idx for block, _, _ in val_blocks for idx in block]
    
    # Zeitbereiche ausgeben
    if train_blocks:
        earliest_train = min([start for _, start, _ in train_blocks])
        latest_train = max([end for _, _, end in train_blocks])
        print(f"[INFO] Training data time range: {np.datetime_as_string(earliest_train)} to {np.datetime_as_string(latest_train)}")
    
    if val_blocks:
        earliest_val = min([start for _, start, _ in val_blocks])
        latest_val = max([end for _, _, end in val_blocks])
        print(f"[INFO] Validation data time range: {np.datetime_as_string(earliest_val)} to {np.datetime_as_string(latest_val)}")
    
    print(f"[INFO] Dataset split: Train={len(train_indices)}, Val={len(val_indices)} samples")
    
    def collate(batch):
        # batch: Liste von (seq, time_feats)
        seqs, times = zip(*batch)
        return torch.stack(seqs, dim=0), torch.stack(times, dim=0)   # [B,T,C,H,W], [B,T,2]


    # DataLoader erstellen
    train_loader = DataLoader(
        Subset(ds, train_indices), 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True, 
        collate_fn=collate
    )
    
    val_loader = DataLoader(
        Subset(ds, val_indices), 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True, 
        collate_fn=collate
    )

    in_channels = ds.C
              
    return train_loader, val_loader, in_channels, ds.scaler


def load_data_for_date_range(start_date, end_date, context_len, prediction_steps, normalize=True, scaler=None):
    """
    Laedt Daten fuer einen bestimmten Zeitbereich, jahresuebergreifend.
    
    Args:
        start_date: Startdatum als 'YYYY-MM-DD' oder datetime
        end_date: Enddatum als 'YYYY-MM-DD' oder datetime
        context_len: Anzahl der Kontext-Zeitschritte
        prediction_steps: Anzahl der Prediction-Zeitschritte
        normalize: Ob die Daten normalisiert werden sollen
        scaler: Optional, ein vortrainierter StandardScaler
    
    Returns:
        dict: Mit Zeitreihen, Daten und Metadaten
    """
    # Konvertiere Strings zu datetime-Objekten falls noetig
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Erstelle Liste der Jahre, die wir laden muessen
    years_to_load = list(range(start_date.year, end_date.year + 1))
    print(f"[INFO] Loading data for years: {years_to_load}")
    
    # Lade Daten aus jedem Jahr
    data_by_year = {}
    for year in years_to_load:
        # Suche nach Daten im Jahresverzeichnis (oder Hauptverzeichnis fuer aktuelles Jahr)
        if year < PREDICTION_YEAR:
            # Vergangene Jahre im past-Verzeichnis
            data_dir = os.path.join(os.path.expanduser("~/scratch/era5_data"), "past")
        else:
            # Aktuelles Jahr im Hauptverzeichnis
            data_dir = os.path.expanduser("~/scratch/era5_data")
        
        print(f"[INFO] Searching for {year} data in {data_dir}")
        
        # Filtere Dateien fuer das gewuenschte Jahr
        single_paths = sorted(glob.glob(os.path.join(data_dir, f"era5_single_{year}_*.netcdf")))
        pressure_paths = sorted(glob.glob(os.path.join(data_dir, f"era5_pressure_{year}_*.netcdf")))
        
        if not single_paths or not pressure_paths:
            print(f"[WARNING] No data files found for year {year}")
            continue
            
        print(f"[INFO] Found {len(single_paths)} single-level and {len(pressure_paths)} pressure-level files for {year}")
        
        # Lade Daten fuer dieses Jahr
        times, lat, lon, X = build_stacked_tensor_single_plus_pressure(single_paths, pressure_paths)
        
        data_by_year[year] = {
            'times': times,
            'lat': lat,
            'lon': lon,
            'X': X
        }
    
    if not data_by_year:
        raise ValueError(f"No data found for the specified date range: {start_date} to {end_date}")
    
    # Konvertiere Datumsangaben zu numpy.datetime64 fuer einfache Filterung
    start_np = np.datetime64(start_date)
    end_np = np.datetime64(end_date)
    
    # Kombiniere Daten aus allen Jahren
    all_times = []
    all_X = []
    
    for year, data in data_by_year.items():
        times = data['times']
        # Finde Indizes im gewuenschten Datumsbereich
        mask = (times >= start_np) & (times <= end_np)
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) > 0:
            all_times.append(times[valid_indices])
            all_X.append(data['X'][valid_indices])
    
    if not all_times:
        raise ValueError(f"No data points found in the specified date range: {start_date} to {end_date}")
    
    # Kombiniere zu einem Array
    combined_times = np.concatenate(all_times)
    combined_X = np.concatenate(all_X)
    
    # Sortiere nach Zeit
    sort_idx = np.argsort(combined_times)
    sorted_times = combined_times[sort_idx]
    sorted_X = combined_X[sort_idx]

    # Normalisiere falls gewünscht
    if normalize:
        if scaler is not None:
            print("[INFO] Normalisiere Daten mit bereitgestelltem Scaler")
            sorted_X = apply_channel_scaling(sorted_X, scaler)
        else:
            print("[WARNING] Kein Scaler verfügbar gewesen, erstelle einen Neuen (unerwünscht!!!)")
            stats = compute_channel_stats(sorted_X)
            sorted_X = apply_channel_scaling(sorted_X, stats)
    
    # Nehme die Lat/Lon von der ersten Quelle
    first_year = list(data_by_year.keys())[0]
    lat = data_by_year[first_year]['lat']
    lon = data_by_year[first_year]['lon']
    
    print(f"[INFO] Combined data: {len(sorted_times)} timesteps from {np.datetime_as_string(sorted_times[0])} to {np.datetime_as_string(sorted_times[-1])}")
    
    return {
        'times': sorted_times,
        'X': sorted_X.astype(np.float32),
        'lat': lat,
        'lon': lon,
        'scaler': scaler
    }

def visualize_predictions_for_wandb(
    predictions, targets, lat, lon, target_times, logger=None, scaler=None
):
    """
    Visualisierung + Logging für WandB (ohne BytesIO):
      - Heatmaps NUR für Temperatur-Kanäle (t2m + t_<level>hPa), Colormaps: coolwarm / RdBu_r
      - Timeseries pro Kanal am Zentralpixel (H//2, W//2)
      - Kanalweiser MSE auf NORMALISIERTER Skala (vergleichbar über Kanäle) + globaler normierter MSE

    Args:
        predictions: torch.Tensor [B, T, C, H, W]
        targets:     torch.Tensor [B, T, C, H, W]
        lat, lon:    1D/2D Koordinaten (für axes extent)
        target_times: iterable von Zeitstempeln
        logger:      Lightning/W&B-Logger (mit .experiment)
        scaler:      sklearn.preprocessing.StandardScaler (erforderlich für Denorm-Plots)
    """

    if logger is None or not hasattr(logger, "experiment"):
        print("[WARNING] WandB logger not provided, skipping visualizations")
        return

    # ---- Tensor -> NumPy (Batch=1 annehmen)
    predictions = predictions.squeeze(0).detach().cpu().numpy()  # [T, C, H, W]
    targets     = targets.squeeze(0).detach().cpu().numpy()      # [T, C, H, W]
    T, C, H, W  = predictions.shape
    mid_t       = T // 2

    # ---- Kanalnamen
    channel_names = []
    channel_names.extend(SINGLE_VARS)  # ["t2m","d2m","u10","v10","msl"]
    for level in PRESSURE_LEVELS:      # [1000,925,850,700,500,300]
        for var in PRESSURE_VARS:      # ["t","r","u","v"]
            channel_names.append(f"{var}_{level}hPa")
    assert len(channel_names) == C, f"[visualize] Erwartete {C} Channels, habe {len(channel_names)}"

    # ---- Zeitachse
    dt_times = [pd.Timestamp(t).to_pydatetime() for t in target_times]

    # ---- Normalisierte Kopien für MSE (nicht denormalisieren!)
    pred_norm = predictions.copy()  # [T, C, H, W]
    targ_norm = targets.copy()

    # ---- Für Plots physikalisch denormalisieren via scaler
    pred_plot = predictions.copy()
    targ_plot = targets.copy()
    if scaler is not None:
        print("[INFO] Denormalisiere Daten für die Visualisierung")
        # Umordnen für invert_channel_scaling
        pred_plot_hwc = pred_plot.transpose(0, 2, 3, 1)  # [T, H, W, C]
        targ_plot_hwc = targ_plot.transpose(0, 2, 3, 1)  # [T, H, W, C]
        
        # Eigene Skalierungsfunktion verwenden statt inverse_transform
        pred_plot_hwc = invert_channel_scaling(pred_plot_hwc, scaler)
        targ_plot_hwc = invert_channel_scaling(targ_plot_hwc, scaler)
        
        # Zurück zu [T, C, H, W] konvertieren für weitere Verarbeitung
        pred_plot = pred_plot_hwc.transpose(0, 3, 1, 2)
        targ_plot = targ_plot_hwc.transpose(0, 3, 1, 2)
    else:
        print("[WARNING] Kein Scaler übergeben – Visualisierung bleibt in Norm-Skala")

    # ---- msl (Pa) nur für PLOTS nach hPa umrechnen (normierte MSEs bleiben z-Skala)
    try:
        msl_idx = channel_names.index("msl")
        pred_plot[:, msl_idx] = pred_plot[:, msl_idx] / 100.0  # Pa -> hPa
        targ_plot[:, msl_idx] = targ_plot[:, msl_idx] / 100.0
    except ValueError:
        msl_idx = None

    # ---- Normalisierte MSEs: zeitlicher Verlauf je Kanal [T, C] & Mittel über Zeit [C]
    mse_per_channel_time_norm = ((pred_norm - targ_norm) ** 2).mean(axis=(2, 3))  # [T, C]
    mse_norm_per_channel      = mse_per_channel_time_norm.mean(axis=0)            # [C]
    mse_norm_global           = float(((pred_norm - targ_norm) ** 2).mean())      # skalar

    # ---- Figuren sammeln und auf einmal loggen (mit Matplotlib-Figuren)
    wandb_images = {}

    for c, ch in enumerate(channel_names):
        # ---------- Timeseries @ Center pixel ----------
        ci, cj   = H // 2, W // 2
        ts_pred  = pred_plot[:, c, ci, cj]
        ts_true  = targ_plot[:, c, ci, cj]

        fig_ts, ax_ts = plt.subplots(1, 1, figsize=(10, 6))
        ax_ts.plot(dt_times, ts_pred, "-", marker="o", label="Prediction @center")
        ax_ts.plot(dt_times, ts_true, "-", marker="x", label="Ground Truth @center")
        ax_ts.set_title(f"Time Series Center pixel – {ch}")
        ax_ts.set_xlabel("Time")
        ax_ts.set_ylabel("Value")
        ax_ts.grid(True)
        ax_ts.legend()
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
        plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()

        wandb_images[f"timeseries_center/{ch}"] = wandb.Image(fig_ts)
        plt.close(fig_ts)

        # ---------- Heatmaps ONLY for Temperature Channels ----------
        is_temperature = (ch == "t2m") or ch.startswith("t_")
        if not is_temperature:
            continue

        vmin = min(pred_plot[:, c].min(), targ_plot[:, c].min())
        vmax = max(pred_plot[:, c].max(), targ_plot[:, c].max())

        # t = 0
        fig0, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        im1 = ax1.imshow(
            pred_plot[0, c],
            extent=[float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))],
            aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax
        )
        ax1.set_title(f"Prediction {ch} (t=0)")
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(
            targ_plot[0, c],
            extent=[float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))],
            aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax
        )
        ax2.set_title(f"Ground Truth {ch} (t=0)")
        plt.colorbar(im2, ax=ax2)

        diff0 = pred_plot[0, c] - targ_plot[0, c]
        vmaxd = max(abs(diff0.min()), abs(diff0.max()))
        im3 = ax3.imshow(
            diff0,
            extent=[float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))],
            aspect="auto", cmap="RdBu_r", vmin=-vmaxd, vmax=vmaxd
        )
        ax3.set_title("Differenz (t=0)")
        plt.colorbar(im3, ax=ax3)
        plt.tight_layout()

        wandb_images[f"heatmap_t0/{ch}"] = wandb.Image(fig0)
        plt.close(fig0)

        # t = mid
        figm, (axm1, axm2, axm3) = plt.subplots(1, 3, figsize=(18, 5))

        im_m1 = axm1.imshow(
            pred_plot[mid_t, c],
            extent=[float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))],
            aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax
        )
        axm1.set_title(f"Prediction {ch} (t={mid_t})")
        plt.colorbar(im_m1, ax=axm1)

        im_m2 = axm2.imshow(
            targ_plot[mid_t, c],
            extent=[float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))],
            aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax
        )
        axm2.set_title(f"Ground Truth {ch} (t={mid_t})")
        plt.colorbar(im_m2, ax=axm2)

        diffm  = pred_plot[mid_t, c] - targ_plot[mid_t, c]
        vmaxdm = max(abs(diffm.min()), abs(diffm.max()))
        im_m3 = axm3.imshow(
            diffm,
            extent=[float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))],
            aspect="auto", cmap="RdBu_r", vmin=-vmaxdm, vmax=vmaxdm
        )
        axm3.set_title("Differenz (t=mid)")
        plt.colorbar(im_m3, ax=axm3)
        plt.tight_layout()

        wandb_images[f"heatmap_tmid/{ch}"] = wandb.Image(figm)
        plt.close(figm)

    # ---- Upload der Figuren in einem Rutsch
    if wandb_images:
        logger.experiment.log(wandb_images)

    # ---- Zusätzlich: MSE (NORMALISIERT) pro Kanal + global loggen
    mse_metrics = {f"mse_norm/{ch}": float(mse_norm_per_channel[i]) for i, ch in enumerate(channel_names)}
    mse_metrics["mse_norm/global"] = mse_norm_global
    logger.experiment.log(mse_metrics)

    print("[INFO] Visualizations + normalized MSE uploaded to WandB")


def evaluate_from_date(
    model,
    start_date,
    context_len=12,
    prediction_steps=24,
    logger=None,
    visualize=True,
    save_csv=True,
    csv_dir=os.path.join("weather_prediction", "forecast_results"),
    csv_filename=None,
    version=None,
):

    """
    Evaluiert ModellPredictionn ab einem bestimmten Datum mit jahresübergreifendem Kontext.
    
    Args:
        model: Trainiertes WeatherTransformer-Modell
        start_date: Startdatum für die Prediction im Format 'YYYY-MM-DD'
        context_len: Anzahl der Kontext-Zeitschritte
        prediction_steps: Anzahl der vorherzusagenden Zeitschritte
        logger: Optional, WandB-Logger
        visualize: Ob detaillierte Visualisierungen erstellt werden sollen
    
    Returns:
        dict: Evaluierungsmetriken und Predictiondaten
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Logge die verwendeten Wettervariablen zu wandb
    if logger:
        logger.experiment.config.update({
            "weather_vars/single_level": SINGLE_VARS,
            "weather_vars/pressure_vars": PRESSURE_VARS,
            "weather_vars/pressure_levels": PRESSURE_LEVELS,
            "weather_vars/total_channels": len(SINGLE_VARS) + len(PRESSURE_VARS) * len(PRESSURE_LEVELS)
        })

    # Konvertiere Startdatum zu datetime
    if isinstance(start_date, str):
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start_dt = start_date
    
    # Berechne Kontextstart (kontext_len Stunden vor dem Startdatum)
    context_start = start_dt - timedelta(hours=context_len)
    
    # Predictionende (prediction_steps Stunden nach dem Startdatum)
    forecast_end = start_dt + timedelta(hours=prediction_steps)
    
    print(f"[INFO] Context period: {context_start} to {start_dt}")
    print(f"[INFO] Forecast period: {start_dt} to {forecast_end}")
    
    # Hole den Scaler vom Modell
    if hasattr(model, "scaler") and model.scaler is not None:
        scaler = model.scaler
        print("[INFO] Scaler vom Modell-Objekt geladen")
    else:
        raise RuntimeError("Modell hat keinen Scaler. Stelle sicher, dass der Scaler geladen wurde.")
    
    # Lade Daten für den gesamten Zeitbereich (Kontext + Prediction)
    data = load_data_for_date_range(
        start_date=context_start,
        end_date=forecast_end,
        context_len=context_len,
        prediction_steps=prediction_steps,
        normalize=True,
        scaler=scaler
    )
    
    times = data['times']
    X = data['X']
    lat = data['lat']
    lon = data['lon']
    
    # Finde Indizes fuer Kontext- und Predictionzeitraum
    context_start_np = np.datetime64(context_start)
    start_np = np.datetime64(start_dt)
    
    # Finde naechstliegenden Zeitpunkt zum gewuenschten Start
    context_start_idx = np.argmin(np.abs(times - context_start_np))
    forecast_start_idx = np.argmin(np.abs(times - start_np))
    
    # Stelle sicher, dass wir genuegend Kontextdaten haben
    if forecast_start_idx - context_start_idx < context_len:
        missing = context_len - (forecast_start_idx - context_start_idx)
        print(f"[WARNING] Not enough context data. Missing {missing} timesteps. Using available data.")
        context_start_idx = max(0, forecast_start_idx - context_len)
    
    # Stelle sicher, dass wir genuegend Predictiondaten haben
    if len(times) - forecast_start_idx < prediction_steps:
        available = len(times) - forecast_start_idx
        print(f"[WARNING] Not enough forecast data. Only {available} timesteps available instead of {prediction_steps}.")
        prediction_steps = available
    
    # Extrahiere Kontext- und Zieldaten
    context_data = X[context_start_idx:forecast_start_idx]
    target_data = X[forecast_start_idx:forecast_start_idx + prediction_steps]
    
    context_times = times[context_start_idx:forecast_start_idx]
    target_times = times[forecast_start_idx:forecast_start_idx + prediction_steps]
    
    print(f"[INFO] Using {len(context_times)} context timesteps: {np.datetime_as_string(context_times[0])} to {np.datetime_as_string(context_times[-1])}")
    print(f"[INFO] Predicting {len(target_times)} timesteps: {np.datetime_as_string(target_times[0])} to {np.datetime_as_string(target_times[-1])}")
    
    # Vorbereite Daten fuer das Modell
    x_input  = torch.from_numpy(context_data).permute(0, 3, 1, 2).unsqueeze(0).to(device)  # [1,T_ctx,C,H,W]
    x_target = torch.from_numpy(target_data).permute(0, 3, 1, 2).unsqueeze(0).to(device)   # [1,T_fut,C,H,W]

    # Zeit-Features
    t_ctx_np = build_time_scalars(context_times)   # [T_ctx,2]
    t_fut_np = build_time_scalars(target_times)    # [T_fut,2]
    t_ctx = torch.from_numpy(t_ctx_np).unsqueeze(0).to(device)  # [1,T_ctx,2]
    t_fut = torch.from_numpy(t_fut_np).unsqueeze(0).to(device)  # [1,T_fut,2]

    # Prediction
    with torch.no_grad():
        start_time = time.time()
        model.set_model_prediction_length(prediction_steps)
        predictions = model(x_input, target=None, time_context=t_ctx, time_future=t_fut)  # [1,T_fut,C,H,W]
        end_time = time.time()
    evaluation_time = end_time - start_time
 
    # =========================
    # CSV EXPORT (center pixel)
    # =========================
    if save_csv:
        # ---- Kanalnamen in gleicher Reihenfolge wie überall im File
        channel_names = []
        channel_names.extend(SINGLE_VARS)  # ["t2m","d2m","u10","v10","msl"]
        for level in PRESSURE_LEVELS:      # [1000,925,850,700,500,300]
            for var in PRESSURE_VARS:      # ["t","r","u","v"]
                channel_names.append(f"{var}_{level}hPa")

        # ---- Tensor -> NumPy, Batch=1 entfernen
        preds_np = predictions.squeeze(0).detach().cpu().numpy()  # [T, C, H, W]
        trues_np = x_target.squeeze(0).detach().cpu().numpy()     # [T, C, H, W]

        # ---- Denormalisieren (deine invert_channel_scaling, erwartet [T,H,W,C])
        preds_hwc = preds_np.transpose(0, 2, 3, 1)  # [T,H,W,C]
        trues_hwc = trues_np.transpose(0, 2, 3, 1)

        if hasattr(model, "scaler") and model.scaler is not None:
            preds_hwc = invert_channel_scaling(preds_hwc, model.scaler)
            trues_hwc = invert_channel_scaling(trues_hwc, model.scaler)
        else:
            print("[WARNING] No scaler on model; CSV will be written in normalized units.")

        # ---- msl von Pa nach hPa
        try:
            msl_idx = channel_names.index("msl")
            preds_hwc[:, :, :, msl_idx] = preds_hwc[:, :, :, msl_idx] / 100.0
            trues_hwc[:, :, :, msl_idx] = trues_hwc[:, :, :, msl_idx] / 100.0
        except ValueError:
            pass

        # ---- Center-Pixel (i,j)
        T, H, W, C = preds_hwc.shape
        ci, cj = H // 2, W // 2
        pred_ctr = preds_hwc[:, ci, cj, :]   # [T, C]
        true_ctr = trues_hwc[:, ci, cj, :]   # [T, C]

        # ---- Zeitstempel als ISO-Strings
        ts_strings = [np.datetime_as_string(t) for t in target_times]

        # ---- Tidy Rows: timestamp + je Kanal 2 Spalten (__pred, __true)
        rows = []
        for t_idx, ts in enumerate(ts_strings):
            row = {"timestamp": ts}
            for c_idx, ch in enumerate(channel_names):
                row[f"{ch}__pred"] = float(pred_ctr[t_idx, c_idx])
                row[f"{ch}__true"] = float(true_ctr[t_idx, c_idx])
            rows.append(row)

        df = pd.DataFrame(rows)

        # ---- Pfad & Dateiname
        os.makedirs(csv_dir, exist_ok=True)
        if csv_filename is None:
            date_str = pd.Timestamp(target_times[0]).strftime("%Y%m%d")
            csv_filename = f"forecast_center_{date_str}_v{version}.csv"
        csv_path = os.path.join(csv_dir, csv_filename)

        # ---- Speichern
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Center-pixel CSV saved to: {csv_path}")
    # =========================

    # Visualisierungen
    if visualize and logger:
        print("[INFO] Creating detailed visualizations for WandB...")
        visualize_predictions_for_wandb(predictions, x_target, lat, lon, target_times, logger, scaler=scaler)

    evaluation_time = end_time - start_time
    print(f"[INFO] Prediction took {evaluation_time:.2f} seconds")
    
    se = (predictions - x_target) ** 2                 # [B, T, C, H, W]
    mse_map = se.mean(dim=1)                           # über Zeit -> [B, C, H, W]
    avg_mse = mse_map.mean().item()                    # globaler Mittelwert

    # ===== MSE-BERECHNUNG (pixelweise über die Zeit) =====
    # 1) Quadratische Fehler
    se = (predictions - x_target) ** 2  # [1,T,C,H,W]

    # 2) Pixelweiser MSE über die Zeit: erst über Zeit mitteln -> [1,C,H,W]
    mse_map = se.mean(dim=1)

    # 3) Globaler Mittelwert (alle Kanäle & Pixel)
    avg_mse = mse_map.mean().item()

    # 4) Optional weiterhin: MSE je Zeitschritt (für Verlauf)
    mse_per_step = [
        F.mse_loss(predictions[:, t], x_target[:, t]).item()
        for t in range(prediction_steps)
    ]

    print(f"[INFO] Average MSE (pixelwise-over-time): {avg_mse:.6f}")
    print(f"[INFO] MSE per step: min={min(mse_per_step):.6f}, max={max(mse_per_step):.6f}")

    # 5) KANALWEISER MSE auf NORMALISIERTER SKALA (Zeit & Raum gemittelt)
    #    -> [C]
    mse_per_channel_norm_torch = se.mean(dim=(0, 1, 3, 4)).squeeze(0)  # [C]
    mse_per_channel_norm = mse_per_channel_norm_torch.detach().cpu().numpy()

    # 6) Kanalnamen erzeugen wie im Rest (SINGLE_VARS + t/r/u/v @ PRESSURE_LEVELS)
    channel_names = []
    channel_names.extend(SINGLE_VARS)
    for level in PRESSURE_LEVELS:
        for var in PRESSURE_VARS:
            channel_names.append(f"{var}_{level}hPa")

    # 7) Logging zu W&B
    if logger:
        metrics = {
            "evaluation/avg_mse_pixelwise_over_time": avg_mse,
            "evaluation/min_mse_per_step": min(mse_per_step),
            "evaluation/max_mse_per_step": max(mse_per_step),
            "evaluation/prediction_time": evaluation_time,
        }
        # kanalweise normalisierte MSEs
        for i, ch in enumerate(channel_names):
            metrics[f"evaluation/mse_norm/{ch}"] = float(mse_per_channel_norm[i])

        logger.experiment.log(metrics)

        # zusätzlich: Schrittverlauf als Linie
        data_table = [[i, mse, np.datetime_as_string(target_times[i])] for i, mse in enumerate(mse_per_step)]
        table = wandb.Table(data=data_table, columns=["step", "mse", "time"])
        logger.experiment.log({
            "evaluation/mse_per_step": wandb.plot.line(table, "step", "mse", title="MSE per Prediction Step")
        })

    # MSE-Zeitreihe mit Zeitstempeln erstellen
    mse_time_series = [
        {
            "step": i,
            "mse": mse,
            "timestamp": np.datetime_as_string(target_times[i]),
            "hours_from_start": i  # Stunden seit Predictionbeginn
        }
        for i, mse in enumerate(mse_per_step)
    ]
    
    # Trend des MSE berechnen (steigend/fallend)
    if len(mse_per_step) > 1:
        mse_trend = "steigend" if mse_per_step[-1] > mse_per_step[0] else "fallend"
        mse_trend_ratio = mse_per_step[-1] / mse_per_step[0] if mse_per_step[0] > 0 else float('inf')
    else:
        mse_trend = "unbekannt"
        mse_trend_ratio = 1.0
    
    # Erstelle Ergebnisdict
    results = {
        "avg_mse": avg_mse,
        "mse_per_step": mse_per_step,
        "mse_time_series": mse_time_series,
        "mse_trend": mse_trend,
        "mse_end_to_start_ratio": mse_trend_ratio,
        "context_start": np.datetime_as_string(context_times[0]),
        "context_end": np.datetime_as_string(context_times[-1]),
        "forecast_start": np.datetime_as_string(target_times[0]),
        "forecast_end": np.datetime_as_string(target_times[-1]),
        "evaluation_time": evaluation_time,
    }
    
    return results


def evaluate_random_dates(
    model,
    num_samples=10,
    year=PREDICTION_YEAR,
    context_len=24,
    prediction_steps=24,
    logger=None,
    times=None,
    X_normalized=None,
    lat=None,
    lon=None,
    visualize=False
):
    """
    Evaluate on multiple random timestamps from `year`. Context can come from the previous year
    (stitched timeline). Forecast is kept inside `year` by construction.
    """
    model.eval()
    device = next(model.parameters()).device

    # PE extension stays unchanged
    desired = max(16384, int(context_len))
    if hasattr(model, "encoder_posenc") and hasattr(model.encoder_posenc, "pe"):
        current = model.encoder_posenc.pe.size(0)
        if desired > current:
            extend_time_posenc_(model.encoder_posenc, desired)
            print(f"[INFO] Extended encoder positional encoding from {current} to {desired}.")
    if hasattr(model, "decoder_posenc") and hasattr(model.decoder_posenc, "pe"):
        cur_dec = model.decoder_posenc.pe.size(0)
        if desired > cur_dec:
            extend_time_posenc_(model.decoder_posenc, desired)
            print(f"[INFO] Extended decoder positional encoding from {cur_dec} to {desired}.")

    # scaler must exist (as before)
    if not hasattr(model, "scaler") or model.scaler is None:
        raise RuntimeError("Modell hat keinen Scaler. Stelle sicher, dass der Scaler geladen wurde.")
    scaler = model.scaler

    # If nothing preloaded, load previous year + evaluation year contiguously
    if times is None or X_normalized is None or lat is None or lon is None:
        print(f"[INFO] Loading contiguous data for years [{year-1}, {year}]...")
        # include_next=False keeps forecasts within the year by mask below
        times, lat, lon, X_normalized = load_contiguous_years(
            year=year, include_prev=True, include_next=False, normalize=True, scaler=scaler
        )
        print(f"[INFO] Contiguous series loaded: {np.datetime_as_string(times[0])} → {np.datetime_as_string(times[-1])}")
    else:
        print(f"[INFO] Using preloaded stitched data with {len(times)} timesteps.")

    # Build candidate indices that:
    #  1) start inside `year`
    #  2) have >= context_len samples before
    #  3) have >= prediction_steps samples after
    #  4) (optional) keep forecast fully inside `year`
    year_start = np.datetime64(datetime(year, 1, 1))
    year_end   = np.datetime64(datetime(year, 12, 31, 23))

    idx = np.arange(len(times))
    # step 1: start inside year
    mask_year = (times >= year_start) & (times <= year_end)

    # step 2+3: enough context before and horizon after
    mask_context = (idx - context_len) >= 0
    mask_horizon = (idx + prediction_steps) <= len(times)

    # step 4: keep the forecast entirely inside `year`
    end_idx_for_sample = idx + (prediction_steps - 1)
    # For indices where horizon condition fails, end_idx_for_sample may exceed len(times)-1; protect with mask_horizon
    end_idx_for_sample = np.minimum(end_idx_for_sample, len(times) - 1)
    mask_forecast_in_year = (times[end_idx_for_sample] <= year_end)

    valid_mask = mask_year & mask_context & mask_horizon & mask_forecast_in_year
    valid_indices = idx[valid_mask]

    if len(valid_indices) == 0:
        raise ValueError(
            f"No valid evaluation indices for year={year}, context_len={context_len}, prediction_steps={prediction_steps}. "
            f"Check data coverage."
        )

    # sample without replacement
    np.random.seed(42)
    chosen = np.random.choice(valid_indices, size=min(num_samples, len(valid_indices)), replace=False)
    chosen.sort()
    selected_eval_indices = [int(i) for i in chosen.tolist()]
    selected_eval_times = [np.datetime_as_string(times[i]) for i in selected_eval_indices]

    # prepare names and accumulators (unchanged)
    channel_names = []
    channel_names.extend(SINGLE_VARS)
    for level in PRESSURE_LEVELS:
        for var in PRESSURE_VARS:
            channel_names.append(f"{var}_{level}hPa")
    num_channels = len(channel_names)

    all_results = []
    avg_mse_values = []
    evaluation_times = []
    all_predictions, all_targets, all_target_times = [], [], []
    all_mse_per_channel = np.zeros((0, num_channels))

    # fix prediction horizon in the model
    model.set_model_prediction_length(prediction_steps)

    # loop
    for i, eval_idx in enumerate(chosen):
        eval_time = times[eval_idx]
        eval_datetime = pd.Timestamp(eval_time).to_pydatetime()
        print(f"\n[INFO] Evaluation {i+1}/{len(chosen)} @ {eval_datetime}")

        context_start_idx = eval_idx - context_len
        context_end_idx   = eval_idx
        target_end_idx    = eval_idx + prediction_steps

        context_data  = X_normalized[context_start_idx:context_end_idx]
        target_data   = X_normalized[context_end_idx:target_end_idx]
        context_times = times[context_start_idx:context_end_idx]
        target_times  = times[context_end_idx:target_end_idx]

        x_input  = torch.from_numpy(context_data).float().permute(0, 3, 1, 2).unsqueeze(0).to(device)
        x_target = torch.from_numpy(target_data).float().permute(0, 3, 1, 2).unsqueeze(0).to(device)

        t_ctx_np = build_time_scalars(context_times)
        t_fut_np = build_time_scalars(target_times)
        t_ctx = torch.from_numpy(t_ctx_np).float().unsqueeze(0).to(device)
        t_fut = torch.from_numpy(t_fut_np).float().unsqueeze(0).to(device)

        with torch.no_grad():
            start_time = time.time()
            if torch.cuda.is_available():
                amp_dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
                autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()
            with autocast_ctx:
                predictions = model(x_input, target=None, time_context=t_ctx, time_future=t_fut)
            end_time = time.time()
        evaluation_time = end_time - start_time
        print(f"[INFO] Forecast in {evaluation_time:.2f}s")

        # metrics on CPU (unchanged)
        predictions_cpu = predictions.detach().to("cpu", non_blocking=True)
        x_target_cpu    = x_target.detach().to("cpu", non_blocking=True)

        se = (predictions_cpu - x_target_cpu) ** 2
        mse_map = se.mean(dim=1)
        avg_mse = mse_map.mean().item()
        mse_per_step = [F.mse_loss(predictions_cpu[:, t], x_target_cpu[:, t]).item() for t in range(len(target_times))]

        mse_per_channel_norm = se.mean(dim=(0, 1, 3, 4)).numpy()

        all_predictions.append(predictions_cpu)
        all_targets.append(x_target_cpu)
        all_target_times.append(target_times)
        all_mse_per_channel = np.vstack([all_mse_per_channel, mse_per_channel_norm])

        print(f"[INFO] MSE @ {eval_datetime}: {avg_mse:.6f} (step min={min(mse_per_step):.6f}, max={max(mse_per_step):.6f})")

        mse_time_series = [
            {"step": j, "mse": m, "timestamp": np.datetime_as_string(target_times[j]), "hours_from_start": j}
            for j, m in enumerate(mse_per_step)
        ]
        result = {
            "avg_mse": avg_mse,
            "mse_per_step": mse_per_step,
            "mse_time_series": mse_time_series,
            "context_start": np.datetime_as_string(context_times[0]),
            "context_end": np.datetime_as_string(context_times[-1]),
            "forecast_start": np.datetime_as_string(target_times[0]),
            "forecast_end": np.datetime_as_string(target_times[-1]),
            "evaluation_time": evaluation_time,
            "eval_datetime": eval_datetime.isoformat(),
        }
        all_results.append(result)
        avg_mse_values.append(avg_mse)
        evaluation_times.append(evaluation_time)

        # free GPU
        del predictions, x_target, x_input, t_ctx, t_fut
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc; gc.collect()

    # aggregate (unchanged summary logic)
    if not all_results:
        print("[WARNING] No successful evaluations")
        return {"error": "No successful evaluations", "num_samples": 0}

    avg_mse_mean = float(np.mean(avg_mse_values))
    avg_mse_std  = float(np.std(avg_mse_values))
    avg_mse_min  = float(np.min(avg_mse_values))
    avg_mse_max  = float(np.max(avg_mse_values))
    avg_eval_time = float(np.mean(evaluation_times))

    best_idx  = int(np.argmin(avg_mse_values))
    worst_idx = int(np.argmax(avg_mse_values))

    avg_mse_per_channel = np.mean(all_mse_per_channel, axis=0)

    if logger:
        metrics = {
            "multi_eval/avg_mse_mean": avg_mse_mean,
            "multi_eval/avg_mse_std": avg_mse_std,
            "multi_eval/avg_mse_min": avg_mse_min,
            "multi_eval/avg_mse_max": avg_mse_max,
            "multi_eval/avg_evaluation_time": avg_eval_time,
            "multi_eval/num_samples": len(all_results),
        }
        for i, ch in enumerate(channel_names):
            metrics[f"multi_eval/mse_norm/{ch}"] = float(avg_mse_per_channel[i])
        logger.experiment.log(metrics)

    print(f"\n[INFO] Summary over {len(all_results)} timestamps")
    print(f"  avg MSE: {avg_mse_mean:.6f} ± {avg_mse_std:.6f}  range[{avg_mse_min:.6f}, {avg_mse_max:.6f}]")
    print(f"  avg eval time: {avg_eval_time:.2f}s")

    mse_channel_pairs = [(ch, float(avg_mse_per_channel[i])) for i, ch in enumerate(channel_names)]
    mse_channel_pairs.sort(key=lambda x: x[1], reverse=True)
    print("\n[INFO] Top-3 channels by normalized MSE:")
    for ch, mse in mse_channel_pairs[:3]:
        print(f"  {ch}: {mse:.6f}")

    return {
        "avg_mse_mean": avg_mse_mean,
        "avg_mse_std": avg_mse_std,
        "avg_mse_min": avg_mse_min,
        "avg_mse_max": avg_mse_max,
        "avg_evaluation_time": avg_eval_time,
        "num_samples": len(all_results),
        "context_len": context_len,
        "prediction_steps": prediction_steps,
        "year": year,
        "individual_results": all_results,
        "best_case_idx": best_idx,
        "worst_case_idx": worst_idx,
        "avg_mse_per_channel": {ch: float(avg_mse_per_channel[i]) for i, ch in enumerate(channel_names)},
        "selected_eval_indices": selected_eval_indices,
        "selected_eval_times": selected_eval_times,
    }



def save_model_info(model_dir, version, model_params, training_params, metrics=None, norm_stats=None):
    """Speichert umfassende Modell-Metadaten"""
    info = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "model_params": {
            "hidden_dim": model_params["hidden_dim"],
            "num_heads": model_params["num_heads"],
            "num_layers": model_params["num_of_layers"],
            "dropout": model_params["dropout"],
            "flash_attention": model_params["use_flash_attention"],
            "stride": model_params.get("stride", 2),
        },
        "training_params": {
            "batch_size": training_params["batch_size"],
            "context_len": training_params["context_len"],
            "prediction_steps": training_params["prediction_steps"],
            "max_epochs": training_params["max_epochs"],
            "training_time_seconds": training_params["training_time"]
        },
        "metrics": metrics or {},
    }

    version_dir = os.path.join(model_dir, f"version_{version}")
    os.makedirs(version_dir, exist_ok=True)
    info_file = os.path.join(version_dir, "model_info.json")
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=4)

    print(f"[INFO] Modell-Dokumentation gespeichert unter: {info_file}")


class LossHistoryLogger(pl.Callback):
    """Callback to collect train/val losses from Lightning metrics for plotting"""
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.train_losses.append(train_loss.item())
            print(f"Training epoch ended with avg loss: {train_loss.item():.4f}")

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.val_losses.append(val_loss.item())
            print(f"Validation epoch ended with avg loss: {val_loss.item():.4f}")


def train_model(train_loader, val_loader, in_channels, hidden_dim=64, num_heads=4, 
                dropout=0.1, num_of_layers=4, max_epochs=10, context_len=24, prediction_steps=24,
                stride=2, model_name="WeatherTransformer", fixed_learning_rate=3e-4,
                weight_decay=1e-2, use_flash_attention=True, plot_loss=False, logger=None, scaler=None, use_axial_attention: bool = False):
    """Train the WeatherTransformer model with PyTorch Lightning"""
    # Erstelle Standardverzeichnisstruktur
    ModelClass = WeatherAxialTransformer if use_axial_attention else WeatherTransformer
    model_name = "WeatherAxialTransformer" if use_axial_attention else "WeatherTransformer"

    base_dir = os.path.join(os.path.expanduser("~/scratch/presence_prediction"),
                            "weather_prediction", "training_results", model_name)
    os.makedirs(base_dir, exist_ok=True)


    print(f"[INFO] Initialisiere Modell")
    # Initialisiere Modell mit LR Scheduler
    model = ModelClass(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_of_layers=num_of_layers,
        dropout=dropout,
        stride=stride,
        use_flash_attention=use_flash_attention,
        fixed_learning_rate=fixed_learning_rate,
        weight_decay=weight_decay,
        use_LR_scheduler=True,
        context_len=context_len,
    )

    # Logge die verwendeten Wettervariablen zu wandb
    if logger:
        logger.experiment.config.update({
            "weather_vars/single_level": SINGLE_VARS,
            "weather_vars/pressure_vars": PRESSURE_VARS,
            "weather_vars/pressure_levels": PRESSURE_LEVELS,
            "weather_vars/total_channels": len(SINGLE_VARS) + len(PRESSURE_VARS) * len(PRESSURE_LEVELS)
        })

    print(f"[INFO] Setze Predictionhorizont auf {prediction_steps} Zeitschritte")
    model.set_model_prediction_length(prediction_steps)
    
    version_str = str(logger.version) if logger else datetime.now().strftime("%Y%m%d-%H%M%S")

    # Checkpoint-Verzeichnis für diese Version
    version_dir = os.path.join(base_dir, f"version_{version_str}")
    ckpt_dir = os.path.join(version_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Erstelle Callbacks für Training
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:07d}-{val_loss:.2f}",
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
        verbose=True,
        min_delta=0.005,
    )
    
    loss_logger = LossHistoryLogger()
    
    callbacks = [
        loss_logger,
        checkpoint_callback,
        early_stop_callback,
    ]
    
    # Initialisiere Trainer mit Logger wenn vorhanden
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator='auto',
        devices='auto',
        precision='bf16-mixed',
        log_every_n_steps=10,
        enable_progress_bar=True,
        logger=logger
    )
    
    # Starte Training
    print(f"[INFO] Starte Training mit {'FlashAttention' if use_flash_attention else 'Standard Attention'} für {max_epochs} Epochen")
    
    # Messe Trainingszeit
    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"[INFO] Training abgeschlossen in {training_time:.2f} Sekunden")
    
    # Logge Trainingszeit zu wandb wenn Logger vorhanden
    if logger:
        logger.experiment.config.update({"training_time": training_time}, allow_val_change=True)
        logger.experiment.summary["training_time_seconds"] = training_time
    
    # Sammle Modellparameter für Dokumentation
    model_params = {
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_of_layers": num_of_layers,
        "dropout": dropout,
        "use_flash_attention": use_flash_attention,
        "stride": stride
    }
    
    training_params = {
        "batch_size": getattr(train_loader, 'batch_size', 1),
        "context_len": context_len,
        "prediction_steps": prediction_steps,
        "max_epochs": max_epochs,
        "training_time": training_time
    }
    
    # Scaler speichern
    if scaler is not None:
        scaler_path = os.path.join(version_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        print(f"[INFO] Scaler gespeichert unter: {os.path.abspath(scaler_path)}")
        model.scaler = scaler

    save_model_info(
        model_dir=os.path.join(os.path.expanduser("~/scratch/presence_prediction"), "weather_prediction", "training_results", model_name),
        version=version_str,
        model_params=model_params,
        training_params=training_params
        )

    print(f"[INFO] Bestes Modell gespeichert unter: {checkpoint_callback.best_model_path}")

    # Optional: Loss-Historie plotten
    if plot_loss and hasattr(loss_logger, 'train_losses') and hasattr(loss_logger, 'val_losses'):
        import matplotlib.pyplot as plt
    
        epochs = range(1, len(loss_logger.train_losses) + 1)
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, loss_logger.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, loss_logger.val_losses, 'r-', label='Validation Loss')
        plt.title('Loss während des Trainings')
        plt.xlabel('Epoche')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
        # Speichere für lokale Kopie
        plots_dir = os.path.join(version_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, "loss_history.png")
        plt.savefig(plot_path)
        plt.close()
    
        # Log zu Weights & Biases
        if logger:
            logger.experiment.log({
                "loss_history": wandb.Image(plot_path),
                "loss/train": loss_logger.train_losses,
                "loss/val": loss_logger.val_losses
            })
    
        print(f"[INFO] Loss-Historie gespeichert unter {plot_path}")

    return model, version_str, training_time


def load_model(model_path=None, version=None, model_name=None, use_axial_attention: bool = False, disable_time2vec: bool = False, use_flash_attention: bool = False):
    """
    Lädt ein trainiertes WeatherTransformer-Modell und den zugehörigen Scaler.
    
    Args:
        model_path: Direkter Pfad zur Modelldatei
        version: Versionsnummer für Standard-Verzeichnisstruktur
        model_name: Name des Modells (Standard: "WeatherTransformer")
        disable_time2vec: bool, ob die Verwendung von Time2Vec deaktiviert werden soll
    
    Returns:
        Geladenes WeatherTransformer-Modell
    """
    # Effektiven Model-Namen & Klasse bestimmen (axial vs. non-axial)
    if model_name is None:
        model_name = "WeatherAxialTransformer" if use_axial_attention else "WeatherTransformer"
    ModelClass = WeatherAxialTransformer if use_axial_attention else WeatherTransformer

    # Option 1: Direkter Pfad
    if model_path:
        checkpoint_path = model_path
    # Option 2: Pfad anhand von Version bestimmen
    elif version is not None:
        base_dir = os.path.join(
            os.path.expanduser("~/scratch/presence_prediction"),
            "weather_prediction", "training_results", model_name
        )
        checkpoint_path = os.path.join(base_dir, f"version_{version}", "checkpoints", "last.ckpt")
    else:
        raise ValueError("Bitte entweder model_path oder version angeben.")

    print(f"[INFO] Lade Modell aus {checkpoint_path}")

    # --- Sanitize checkpoint: remove PE keys to avoid size mismatch on longer contexts ---

    try:
        _ckpt = torch.load(checkpoint_path, map_location="cpu")
        _path_to_load = checkpoint_path
        if isinstance(_ckpt, dict) and "state_dict" in _ckpt:
            sd = _ckpt["state_dict"]
            removed = []
            for k in ["encoder_posenc.pe", "decoder_posenc.pe"]:
                if k in sd:
                    sd.pop(k, None)
                    removed.append(k)
            if removed:
                print(f"[INFO] Removed PE keys from checkpoint to avoid size mismatch: {removed}")
                tmp_dir = tempfile.mkdtemp(prefix="san_ckpt_")
                sanitized_path = os.path.join(tmp_dir, "sanitized.ckpt")
                torch.save(_ckpt, sanitized_path)
                _path_to_load = sanitized_path
    except Exception as e:
        print(f"[WARN] Could not sanitize checkpoint ({e}), loading original file.")
        _path_to_load = checkpoint_path

    # Load with Lightning (PE keys now absent, no shape clash)
    model = ModelClass.load_from_checkpoint(
        _path_to_load,
        strict=False,
        use_time2vec=False if disable_time2vec else True,
        use_flash_attention=use_flash_attention 
    )
    model.eval()

    try:
        base_dir = os.path.join(
            os.path.expanduser("~/scratch/presence_prediction"),
            "weather_prediction", "training_results", model_name
        )
        scaler_path = os.path.join(base_dir, f"version_{version}", "scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            model.scaler = scaler
            print(f"[INFO] Scaler erfolgreich geladen aus {scaler_path}")
        else:
            print("[WARN] Kein Scaler gefunden, Modell wird ohne Scaler geladen.")
    except Exception as e:
        print(f"[WARN] Fehler beim Laden des Scalers: {e}")

    return model


def extend_time_posenc_(pos_enc_module, new_max_len: int):
    import torch, math
    with torch.no_grad():
        pe = pos_enc_module.pe
        old_len, d_model = pe.size()
        if new_max_len <= old_len:
            return
        device = pe.device
        position = torch.arange(old_len, new_max_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
                             * (-math.log(10000.0) / d_model))
        extra = torch.zeros(new_max_len - old_len, d_model, device=device, dtype=torch.float32)
        extra[:, 0::2] = torch.sin(position * div_term)
        extra[:, 1::2] = torch.cos(position * div_term)
        new_pe = torch.cat([pe, extra.to(pe.dtype)], dim=0)
        pos_enc_module.register_buffer('pe', new_pe, persistent=True)


def fine_tune_model_from_checkpoint(
    # --- checkpoint selection ---
    model_path: str = None,
    version: str = None,
    use_axial_attention: bool = False,

    # --- new data/hparam overrides for FT ---
    data_root: str = None,
    batch_size: int = 1,
    context_len: int = 24,          # in hours
    prediction_steps: int = 24,     # in hours (horizon)

    # training overrides
    fixed_learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    max_epochs: int = 10,
    num_workers: int = 4,
    block_size: int = 3,
    use_flash_attention: bool = True,

    # logging
    use_wandb: bool = False,
    wandb_project: str = "Weather Prediction Working",
):
    """
    Fine-tune: lädt einen bestehenden Checkpoint und trainiert mit neuem Kontext/Horizont
    (stride bleibt UNVERÄNDERT wie im Checkpoint). Scheduled Sampling wird für FT deaktiviert.

    Returns:
        (model, version_str, training_time_seconds)
    """
    # ----------------- Build loaders für NEUEN Kontext/Horizon -----------------
    train_loader, val_loader, in_channels, scaler = create_weather_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        context_len=context_len,
        prediction_steps=prediction_steps,
        num_workers=num_workers,
        block_size=block_size,
    )

    # ----------------- Logger -----------------
    logger = None
    if use_wandb:
        logger = WandbLogger(project=wandb_project)
        logger.experiment.config.update({
            "finetune/context_len": context_len,
            "finetune/prediction_steps": prediction_steps,
            "finetune/lr": fixed_learning_rate,
            "finetune/weight_decay": weight_decay,
            "finetune/max_epochs": max_epochs,
            "finetune/use_flash_attention": use_flash_attention,
            "finetune/use_axial_attention": use_axial_attention,
        })

    # ----------------- Checkpoint auflösen -----------------
    ModelClass = WeatherAxialTransformer if use_axial_attention else WeatherTransformer
    model_name  = "WeatherAxialTransformer" if use_axial_attention else "WeatherTransformer"

    if model_path:
        checkpoint_path = model_path
    elif version is not None:
        base_dir = os.path.join(
            os.path.expanduser("~/scratch/presence_prediction"),
            "weather_prediction", "training_results", model_name
        )
        checkpoint_path = os.path.join(base_dir, f"version_{version}", "checkpoints", "last.ckpt")
    else:
        raise ValueError("Provide either model_path or version for fine-tuning.")

    print(f"[INFO] Fine-tuning from checkpoint: {checkpoint_path}")

    # ----------------- Laden mit OVERRIDES (stride unverändert) -----------------
    # Wichtig:
    #  - context_len im Modell überschreiben (training_step sliced daran)
    #  - prediction_steps via set_model_prediction_length setzen
    #  - SS im FT ausschalten (ss_mode='off')
    model = ModelClass.load_from_checkpoint(
        checkpoint_path,
        strict=True,                      # stride & Parameter aus dem CKPT unverändert
        in_channels=in_channels,          # Schutz, falls Kanäle geprüft werden
        context_len=context_len,
        use_flash_attention=use_flash_attention,
        fixed_learning_rate=fixed_learning_rate,
        weight_decay=weight_decay,
        ss_mode="off",                    # <—— SS im FT deaktivieren
        ss_p_start=0.0, ss_p_end=0.0,
        ss_warmup_epochs=0,
    )

    extend_time_posenc_(model.encoder_posenc, 16384)

    # scaler für spätere inverse-Transform beilegen
    model.scaler = scaler

    # neuen Horizon (in Zeitschritten) setzen
    model.set_model_prediction_length(prediction_steps)

    # ----------------- Trainer & Callbacks -----------------
    base_dir = os.path.join(os.path.expanduser("~/scratch/presence_prediction"),
                            "weather_prediction", "training_results", model_name)
    version_str = str(logger.version) if logger else datetime.now().strftime("%Y%m%d-%H%M%S")
    version_dir = os.path.join(base_dir, f"version_{version_str}")
    ckpt_dir = os.path.join(version_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:07d}-{val_loss:.2f}",
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=True
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
        verbose=True,
        min_delta=0.005,
    )
    loss_logger = LossHistoryLogger()

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[loss_logger, checkpoint_callback, early_stop_callback],
        accelerator='auto',
        devices='auto',
        precision='bf16-mixed',
        log_every_n_steps=10,
        enable_progress_bar=True,
        logger=logger
    )

    # ----------------- Fine-tuning starten -----------------
    start = time.time()
    print(f"[INFO] Start FT: context_len={context_len}, horizon={prediction_steps}")
    trainer.fit(model, train_loader, val_loader)
    end = time.time()
    training_time = end - start
    print(f"[INFO] Fine-tuning finished in {training_time:.2f}s")

    # ----------------- Metadaten speichern (wie train_model) -----------------
    model_params = {
        "hidden_dim": model.hparams.hidden_dim,
        "num_heads": model.hparams.num_heads,
        "num_of_layers": model.hparams.num_of_layers,
        "dropout": model.hparams.dropout,
        "use_flash_attention": bool(model.use_flash_attention),
        "stride": model.hparams.stride,   # unverändert aus CKPT
    }
    training_params = {
        "batch_size": getattr(train_loader, 'batch_size', 1),
        "context_len": context_len,
        "prediction_steps": prediction_steps,
        "max_epochs": max_epochs,
        "training_time": training_time
    }

    scaler_path = os.path.join(version_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Scaler gespeichert unter: {os.path.abspath(scaler_path)}")
    model.scaler = scaler

    save_model_info(
        model_dir=os.path.join(os.path.expanduser("~/scratch/presence_prediction"), "weather_prediction", "training_results", model_name),
        version=version_str,
        model_params=model_params,
        training_params=training_params
    )
    print(f"[INFO] Best checkpoint: {checkpoint_callback.best_model_path}")

    if use_wandb:
        logger.experiment.config.update({"training_time": training_time}, allow_val_change=True)
        logger.experiment.summary["training_time_seconds"] = training_time
        wandb.finish()

    return model, version_str, training_time


def main():
    """Main function to parse arguments and run appropriate actions"""
    parser = argparse.ArgumentParser(description="Weather Prediction with Encoder-Decoder Transformer")
    parser.add_argument('--data_root', type=str, default=os.path.join(os.path.expanduser("~/scratch/era5_data"), "past"), 
                      help='Path to training data directory (default: ~/scratch/era5_data/past)')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--evaluate_from_date', type=str, help='Evaluate forecast from a specific date (format: YYYY-MM-DD)')
    parser.add_argument('--model_path', type=str, help='Path to saved model checkpoint')
    parser.add_argument('--version', type=str, help='Version number of saved model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--context_days', type=int, default=1, help='Number of context days')
    parser.add_argument('--prediction_days', type=int, default=1, help='Number of prediction days')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=1, help='Maximum number of training epochs')
    parser.add_argument('--use_flash_attention', type=lambda x: x.lower()=="true", default=False, help='Use Flash Attention if available')
    parser.add_argument('--plot_loss', action='store_true', help='Plot training and validation loss')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--stride', type=int, default=4, help='Stride for convolutional downsampling')
    parser.add_argument('--block_size', type=int, default=3, help='Size of blocks for training data partitioning')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='Weather Prediction Working', help='WandB project name')
    parser.add_argument('--evaluate_random', type=int, help='Evaluate on N random dates from specified year')
    parser.add_argument('--context_days_list', type=str, 
                   help='Kommagetrennte Liste von Kontexttagen für die Evaluierung (z.B. "1,2,5")')
    parser.add_argument("--use_axial_attention", type=lambda x: x.lower()=="true", default=False, help="Use axial self-attention variant of the weather transformer.")
    ### Fine-tuning 
    parser.add_argument('--finetune', action='store_true', help='Fine-tune from a checkpoint with new context/horizon')
    parser.add_argument('--ft_model_path', type=str, help='Path to base checkpoint (alt. zu --version)')
    parser.add_argument('--ft_version', type=str, help='Version of base checkpoint for fine-tuning')
    parser.add_argument('--ft_context_days', type=int, default=None, help='Context (days) for FT')
    parser.add_argument('--ft_prediction_days', type=int, default=None, help='Prediction horizon (days) for FT')
    parser.add_argument('--ft_max_epochs', type=int, default=None, help='Max epochs for FT')
    parser.add_argument('--ft_lr', type=float, default=2e-4, help='Learning rate for FT')
    parser.add_argument('--ft_use_wandb', action='store_true', default=False, help='Enable WandB for FT')


    # Beispiel Befehl: python -m weather_prediction.homebrew_weather --train --use_wandb --max_epochs 20 --context_len 48 --prediction_steps 12 --block_size 3 --use_flash_attention True
    # Beispiel Befehl: python -m weather_prediction.homebrew_weather --finetune --ft_version 20240930-101500 --ft_context_days 2 --ft_prediction_days 1 --ft_max_epochs 8 --ft_use_wandb
    args = parser.parse_args()
    
    context_len = int(args.context_days * HOURS_IN_DAY)
    prediction_steps = int(args.prediction_days * HOURS_IN_DAY)

    # Initialize wandb logger if requested
    logger = None
    if args.use_wandb:
        logger = WandbLogger(project=args.wandb_project)
        # Log all command line arguments
        logger.experiment.config.update(vars(args))
        # Define summary metrics
        logger.experiment.define_metric("training_time_seconds", summary="last")
    
    # Perform requested operations
    if args.train:
        # Build dataloaders
        train_loader, val_loader, in_channels, scaler = create_weather_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            context_len=context_len,
            prediction_steps=prediction_steps,
            num_workers=args.num_workers,
            block_size=args.block_size
        )
        
        # Train model
        print(f"[INFO] Starting training with {args.num_layers} layers, {args.hidden_dim} hidden dim, {args.num_heads} heads")
        model, version, training_time = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            in_channels=in_channels,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            num_of_layers=args.num_layers,
            max_epochs=args.max_epochs,
            context_len=context_len,
            prediction_steps=prediction_steps,
            stride=args.stride,
            fixed_learning_rate=args.lr,
            use_flash_attention=args.use_flash_attention,
            plot_loss=args.plot_loss,
            logger=logger,
            scaler=scaler,
            use_axial_attention=args.use_axial_attention
        )

    
    # Evaluierung ab einem bestimmten Datum
    if args.evaluate_from_date:
        eval_model_name = "WeatherAxialTransformer" if args.use_axial_attention else "WeatherTransformer"
        model = load_model(
            model_path=args.model_path,
            version=args.version,
            model_name=eval_model_name,
            use_axial_attention=args.use_axial_attention,
            use_flash_attention=args.use_flash_attention
        )

        print(f"[INFO] Evaluating forecast from date: {args.evaluate_from_date}")
        results = evaluate_from_date(
            model=model,
            start_date=args.evaluate_from_date,
            context_len=context_len,
            prediction_steps=prediction_steps,
            logger=logger,
            version=args.version,
        )

        
        # Speichere die Ergebnisse
        version = args.version or "custom"
        date_str = args.evaluate_from_date.replace("-", "")
        base_dir = os.path.join("weather_prediction", "forecast_results")
        os.makedirs(base_dir, exist_ok=True)
        
        results_file = os.path.join(base_dir, f"forecast_{date_str}_v{version}.json")
        with open(results_file, 'w') as f:
            # Sicherstellen dass die Ergebnisse JSON-serialisierbar sind
            json.dump(results, f, indent=4)
            
        print(f"[INFO] Forecast results saved to: {results_file}")

    if args.evaluate_random:
        eval_model_name = "WeatherAxialTransformer" if args.use_axial_attention else "WeatherTransformer"
        model = load_model(
            model_path=args.model_path,
            version=args.version,
            model_name=eval_model_name,
            use_axial_attention=args.use_axial_attention,
            use_flash_attention=args.use_flash_attention
        )

        # determine context list
        if args.context_days_list:
            context_days_list = [float(x) for x in args.context_days_list.split(',')]
            print(f"[INFO] Evaluating with context_days_list={context_days_list}")
        else:
            context_days_list = [args.context_days]

        # --- NEW: preload previous year + evaluation year stitched once ---
        year = PREDICTION_YEAR
        if not hasattr(model, "scaler") or model.scaler is None:
            raise RuntimeError("Modell hat keinen Scaler. Stelle sicher, dass der Scaler geladen wurde.")

        print(f"[INFO] Preloading contiguous data for random eval: [{year-1}, {year}]")
        times, lat, lon, X_normalized = load_contiguous_years(
            year=year, include_prev=True, include_next=False, normalize=True, scaler=model.scaler
        )
        print(f"[INFO] Preloaded stitched timeline: {np.datetime_as_string(times[0])} → {np.datetime_as_string(times[-1])}")

        # loop over contexts
        for context_days in context_days_list:
            current_context_len = int(context_days * HOURS_IN_DAY)
            print(f"\n[INFO] Random eval with context_days={context_days} (context_len={current_context_len})")

            results = evaluate_random_dates(
                model=model,
                num_samples=args.evaluate_random,
                year=year,
                context_len=current_context_len,
                prediction_steps=int(args.prediction_days * HOURS_IN_DAY),
                logger=logger,
                times=times,
                X_normalized=X_normalized,
                lat=lat,
                lon=lon,
                visualize=False
            )

            version = args.version or "custom"
            base_dir = os.path.join("weather_prediction", "forecast_results")
            os.makedirs(base_dir, exist_ok=True)
            results_file = os.path.join(
                base_dir,
                f"random_eval_{year}_n{args.evaluate_random}_v{version}_c{context_days}.json"
            )
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"[INFO] Random evaluation results saved to: {results_file}")

            import gc, torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            gc.collect()


        if not args.context_days_list:
            # Backwards-compatible: ein einzelner Kontext (wie vorher)
            print(f"[INFO] Evaluating on {args.evaluate_random} random dates from {PREDICTION_YEAR}")
            context_len = int(args.context_days * HOURS_IN_DAY)
            prediction_steps = int(args.prediction_days * HOURS_IN_DAY)

            results = evaluate_random_dates(
                model=model,
                num_samples=args.evaluate_random,
                context_len=context_len,
                prediction_steps=prediction_steps,
                logger=logger
            )

            version = args.version or "custom"
            base_dir = os.path.join("weather_prediction", "forecast_results")
            os.makedirs(base_dir, exist_ok=True)
            results_file = os.path.join(base_dir, f"random_eval_{PREDICTION_YEAR}_n{args.evaluate_random}_v{version}_c{args.context_days}.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"[INFO] Random evaluation results saved to: {results_file}")


    if args.finetune:
        ft_context_len = int((args.ft_context_days if args.ft_context_days is not None else args.context_days) * HOURS_IN_DAY)
        ft_prediction_steps = int((args.ft_prediction_days if args.ft_prediction_days is not None else args.prediction_days) * HOURS_IN_DAY)
        ft_max_epochs = args.ft_max_epochs if args.ft_max_epochs is not None else args.max_epochs
        ft_lr = args.ft_lr if args.ft_lr is not None else args.lr

        model, version, training_time = fine_tune_model_from_checkpoint(
            model_path=args.ft_model_path,
            version=args.ft_version,
            use_axial_attention=args.use_axial_attention,
            data_root=args.data_root,
            batch_size=args.batch_size,
            context_len=ft_context_len,
            prediction_steps=ft_prediction_steps,
            fixed_learning_rate=ft_lr,
            weight_decay=1e-2,
            max_epochs=ft_max_epochs,
            num_workers=args.num_workers,
            block_size=args.block_size,
            use_flash_attention=args.use_flash_attention,
            use_wandb=args.ft_use_wandb,
            wandb_project=args.wandb_project,
        )


    if not (args.train or args.evaluate_from_date or args.evaluate_random or args.finetune):
        parser.print_help()            
    
    # Finalize wandb run
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()