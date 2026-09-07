import json
import pandas as pd
import numpy as np
import math
import torch
import holidays
import os
import importlib
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import TensorBoardLogger
import wandb
from pytorch_lightning.loggers import WandbLogger
from tud_presence_prediction.models.LearnTransformer import LearnTransformer
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from tud_presence_prediction.models.internal.model_util import model_util
import time
from datetime import datetime, timedelta

try:
    import colorama
    colorama.just_fix_windows_console()  # enables VT sequences in legacy hosts
except Exception:
    pass

CONSOLE_THRESHOLD= 0.5

_NORMALIZATION_STATS = None  # filled by create_multiuser_dataloaders

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points in kilometers (haversine)."""
    R = 6371.0
    dlat = math.radians(float(lat2) - float(lat1))
    dlon = math.radians(float(lon2) - float(lon1))
    a = math.sin(dlat/2.0)**2  + math.cos(math.radians(float(lat1))) * \
        math.cos(math.radians(float(lat2))) * math.sin(dlon/2.0)**2
    return 2.0 * R * math.asin(math.sqrt(a))


def get_model_class(use_axial_attention=False):
    """Importiert die richtige Modelklasse basierend auf der Benutzerauswahl."""
    if use_axial_attention:
        module_name = "LearnTransformerAxialSelf"
    else:
        module_name = "LearnTransformer"
        
    try:
        module = importlib.import_module(f"tud_presence_prediction.models.{module_name}")
        return getattr(module, "LearnTransformer")
    except (ImportError, AttributeError) as e:
        print(f"[FEHLER] Konnte das Modell {module_name} nicht laden: {e}")
        return None


def load_and_process_multiuser(
    file_path: str,
    selected_user_idx: int = None,  # Optional für einen einzelnen User
    tz: str = 'Europe/Berlin',
    start_time: str = '08:00',
    end_time: str = '20:45',
    freq: str = '15min',
    fill_missing: bool = False,
    fill_method: str = 'ffill',
    holiday_country: str = 'DE',
    distance_threshold: float = 0.0
) -> dict:
    """
    Lädt Rohdaten für mehrere Nutzer im JSON-Format und verarbeitet sie.
    Args:
        file_path (str): Pfad zur JSON-Datei mit den Rohdaten.
        selected_user_idx (int, optional): Index des Nutzers, der verarbeitet werden soll.
        tz (str): Zeitzone für die Zeitstempel.
        start_time (str): Startzeit für die Tageszeiträume.
        end_time (str): Endzeit für die Tageszeiträume.
        freq (str): Frequenz der Zeitstempel (z.B. '15min').
        fill_missing (bool): Ob fehlende Werte aufgefüllt werden sollen.
        fill_method (str): Methode zum Auffüllen fehlender Werte ('ffill' oder 'bfill').
        holiday_country (str): Land für Feiertagsinformationen.
        distance_threshold (float): Schwellenwert für die Entfernung, um Labels zu generieren.

    Returns:
        user_data: ein Dictionary mit Nutzer-IDs als Schlüsseln und Tensoren für Location- und Kalender-Features sowie Labels.
            Die Tensoren haben die Form:   
            X_loc: [days, time_slots, 3] (Latitude, Longitude, Distance)
            X_cal: [days, time_slots, 8] (7 Wochentage als One-Hot + 1 Feiertags-Flag)
            target: [days, time_slots] (1 für zu Hause, 0 für abwesend)
        user_timestamps: ein Dictionary mit Nutzer-IDs als Schlüsseln und Listen von Zeitstempeln für jeden Nutzer.
    """
    print(f"[DEBUG] Lese Datei: {file_path}")
    
    # Die gesamte Datei als ein JSON-Array einlesen
    with open(file_path, 'r') as f:
        try:
            all_data = json.load(f)
            print(f"[DEBUG] JSON-Datei erfolgreich geladen mit {len(all_data)} Einträgen")
        except json.JSONDecodeError as e:
            print(f"[FEHLER] Die Datei enthält kein gültiges JSON: {e}")
            return {}
    
    # Daten in User-Paare aufteilen (Home-Koordinaten + Records)
    user_data = {}
    user_count = 0
    user_timestamps = {}  # Speichert die Zeitstempel für jeden User
    
    # Durchlaufe die Daten paarweise
    for i in range(0, len(all_data), 2):
        if i + 1 >= len(all_data):
            print(f"[WARNUNG] Unvollständiges Paar am Ende der Datei")
            break
            
        # Extrahiere Home-Koordinaten und Records für diesen User
        home_coords = all_data[i] # extrahiert das erste dict objekt
        records = all_data[i+1] # extrahiert das zweite dict object
        
        # Prüfe, ob es sich um ein gültiges Paar handelt
        if 'coordinate' not in home_coords or not isinstance(records, list):
            print(f"[WARNUNG] Ungültiges Datenpaar bei Index {i}, überspringe")
            continue
            
        user_idx = user_count
        user_count += 1
        
        # Überprüfe, ob dieser User verarbeitet werden soll
        if selected_user_idx is not None and user_idx != selected_user_idx:
            continue
            
        # Home-Koordinaten extrahieren
        home_lat, home_lon = home_coords['coordinate']
        
        print(f"[DEBUG] Verarbeite User {user_idx}: Home ({home_lat}, {home_lon}), {len(records)} Records")
        
        # DataFrame erzeugen
        df = pd.DataFrame(records)  
        if df.empty:
            print(f"[WARNUNG] User {user_idx}: Keine Datensätze verfügbar")
            continue
            
        df['dt'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['dt'] = df['dt'].dt.tz_convert(tz) # ändert Zeitzone

        user_timestamps[user_idx] = df['dt'].dt.date.unique().tolist()  # Speichere die Zeitstempel für diesen User
        
        # Prüfen, ob fehlende Koordinaten vorhanden sind
        null_coords = df['coordinate'].isna()
        if null_coords.any():
            null_count = null_coords.sum()
            print(f"[WARNUNG] User {user_idx}: {null_count} Records mit fehlenden Koordinaten gefunden, {(len(df) - null_count) / len(df) * 100:.2f}% der Daten sind verfügbar")
            
            # TODO: Create better filling method like mean method
            if fill_missing:
                # Sortiere nach Zeitstempel für zeitbasierte Füllung
                df = df.sort_values('dt')
                
                # Manuelle Implementierung der Füllmethoden
                if fill_method == 'ffill':
                    # Forward Fill: Vorwärts durch die Daten iterieren
                    last_valid_coord = None
                    for idx in df.index:
                        if df.at[idx, 'coordinate'] is None:
                            if last_valid_coord is not None:
                                df.at[idx, 'coordinate'] = last_valid_coord
                            else:
                                df.at[idx, 'coordinate'] = [home_lat, home_lon] # place the user at home for the first timestep
                        else:
                            last_valid_coord = df.at[idx, 'coordinate']
                    
                elif fill_method == 'bfill':
                    # Backward Fill: Rückwärts durch die Daten iterieren
                    next_valid_coord = None
                    for idx in reversed(df.index):
                        if df.at[idx, 'coordinate'] is None:
                            if next_valid_coord is not None:
                                df.at[idx, 'coordinate'] = next_valid_coord
                            else:
                                df.at[idx, 'coordinate'] =[home_lat, home_lon]                        
                        else:
                            next_valid_coord = df.at[idx, 'coordinate']
                
                # Überprüfen, ob noch fehlende Werte existieren
                remaining_nulls = df['coordinate'].isna().sum()
                if remaining_nulls > 0:
                    print(f"[WARNUNG] User {user_idx}: {remaining_nulls} Records konnten nicht gefüllt werden und werden entfernt")
                    df = df[df['coordinate'].notna()].copy()
                else:
                    print(f"[INFO] User {user_idx}: Fehlende Koordinaten mit Methode '{fill_method}' erfolgreich gefüllt")

        # Entferne doppelte Zeitstempel vor der weiteren Verarbeitung
        duplicate_count = df['dt'].duplicated().sum()
        if duplicate_count > 0:
            print(f"[WARNUNG] User {user_idx}: {duplicate_count} duplizierte Zeitstempel gefunden")
        # Sortiere nach Zeitstempel und behalte nur den letzten Eintrag pro Zeitstempel
        df = df.sort_values('dt').drop_duplicates(subset=['dt'], keep='last')
        print(f"[INFO] User {user_idx}: Duplizierte Zeitstempel wurden bereinigt")

        # TESTED AND SAW THAT THERE ARE NO NAN VALUES HERE
        # null_coords = df['coordinate'].isna()
        # if null_coords.any():
        #     print(f"[INFO] Nach Füllen der Koordinaten gibt es immer noch NaN values")
        # null_timestamps = df['dt'].isna()
        # if null_timestamps.any():
        #     print(f"[INFO] Nach Überprüfen der Timestamps, gibt es immer noch NaN values")
        

        # # DataFrame für jeden User als separate Datei speichern - VERBESSERT
        # # Absoluter Pfad, unabhängig vom Arbeitsverzeichnis
        # save_dir = os.path.join(os.path.abspath(os.path.dirname(file_path)), "user_dataframes")
        # os.makedirs(save_dir, exist_ok=True)
        # user_df_path = os.path.join(save_dir, f"user_{user_idx}_data_{datetime.now().strftime('%Y%m%d')}.csv")
        #
        # try:
        #     df.to_csv(user_df_path)
        #     print(f"[INFO] User {user_idx}: DataFrame mit {len(df)} Zeilen in {user_df_path} gespeichert")
        #     print(f"[INFO] Absoluter Pfad: {os.path.abspath(user_df_path)}")
        # except Exception as e:
        #     print(f"[FEHLER] Beim Speichern der Datei für User {user_idx} ist ein Fehler aufgetreten: {e}")

        # Jetzt können wir sicher auf die Koordinaten zugreifen

        df['lat'] = df['coordinate'].apply(lambda c: c[0])
        df['lon'] = df['coordinate'].apply(lambda c: c[1])
        df['distance'] = df.apply(lambda r: haversine_km(r['lat'], r['lon'], home_lat, home_lon), axis=1)

        # Kalender-Features initialisieren
        cal = holidays.CountryHoliday(holiday_country)
        
        # Timeslot-Range definieren
        per_day_times = pd.date_range(start=start_time, end=end_time, freq=freq).time # array with values 0 to 51 
        
        # Entferne ersten und letzten Tag jedes Users, da diese möglicherweise unvollständig sind
        dates = sorted(df['dt'].dt.date.unique())
        if len(dates) > 2:  # Nur wenn mindestens 3 Tage verfügbar sind
            dates = dates[1:-1]
            print(f"[INFO] User {user_idx}: Erster und letzter Tag wurden entfernt, {len(dates)} Tage verbleiben")
        else:
            print(f"[WARNUNG] User {user_idx}: Nicht genügend Tage verfügbar, kann ersten und letzten Tag nicht entfernen")


        X_loc_list, X_cal_list, target_list = [], [], []
        
        # Gruppiere die Daten nach Tag
        for date in dates:
            # Filtere die Daten für den aktuellen Tag
            day_data = df[df['dt'].dt.date == date]
            
            if len(day_data) == 0:
                print(f"[WARNUNG] User {user_idx}: Keine Daten für Tag {date}")
                continue
            
            # Sortiere nach Zeit und setze den Zeitstempel als Index
            day_data = day_data.sort_values('dt').set_index('dt')
            
            # Location-Array direkt aus den vorhandenen Daten
            loc_arr = np.stack([
                day_data['lat'].values,
                day_data['lon'].values,
                day_data['distance'].values
            ], axis=-1)
            
            # Kalender-Array für die tatsächlichen Zeitstempel
            cal_feats = []
            for dt in day_data.index:
                wd = dt.weekday()
                wd_oh = np.eye(7)[wd]
                is_hol = int(dt.date() in cal)
                cal_feats.append(np.concatenate([wd_oh, [is_hol]]))
            cal_arr = np.stack(cal_feats, axis=0)
            
            # Labels mit Schwellwert
            labels = (day_data['distance'] <= distance_threshold).astype(int).values
            
            # Überprüfe auf NaNs
            if np.isnan(loc_arr).any():
                print(f"[WARNUNG] User {user_idx}, Tag {date}: NaN-Werte in loc_arr gefunden!")
            
            X_loc_list.append(loc_arr)
            X_cal_list.append(cal_arr)
            target_list.append(labels)
        
        if len(X_loc_list) > 0:
            # Alle Tage zu Tensoren konvertieren
            X_loc = torch.tensor(np.stack(X_loc_list), dtype=torch.float32)
            X_cal = torch.tensor(np.stack(X_cal_list), dtype=torch.float32)
            target = torch.tensor(np.stack(target_list), dtype=torch.float32)
            
            user_data[user_idx] = (X_loc, X_cal, target) # X_loc has shape [days, time_slots, 3] and is a tensor of floats
            print(f"[DEBUG] User {user_idx} - Tensors: X_loc {X_loc.shape}, X_cal {X_cal.shape}, target {target.shape}")

        print("NaN in X_loc:", torch.isnan(X_loc).sum().item())
        print(f"Shape X_loc: {X_loc.shape}\n")

        if torch.isnan(target).any():
            print(f"[WARNUNG] User {user_idx}: NaN-Werte im Ziel-Tensor gefunden!")

    return user_data, user_timestamps


class UserBatchSampler(torch.utils.data.Sampler):
    """
    Sampler for FIXED-length, NON-OVERLAPPING blocks at runtime.
    - days_per_block: total block length (e.g., consecutive_days + num_days)
    - stride defaults to days_per_block => non-overlapping blocks
    - Shuffles the ORDER of blocks; NEVER shuffles days within a block
    """
    def __init__(self, user_dataset_map, batch_size, days_per_block=7, stride=None, min_batch_size=1):
        self.user_dataset_map = user_dataset_map
        self.batch_size = int(batch_size)
        self.days_per_block = int(days_per_block)
        self.stride = int(stride) if stride is not None else self.days_per_block  # <-- non-overlap
        self.min_batch_size = int(min_batch_size)
        self.batches = []

        all_blocks = []
        for _u, indices in user_dataset_map.items():
            idx = sorted(indices)  # chronological per user
            n = len(idx)
            if n < self.days_per_block:
                continue
            # non-overlapping windows by default
            for i in range(0, n - self.days_per_block + 1, self.stride):
                block = idx[i:i + self.days_per_block]
                if len(block) == self.days_per_block:
                    all_blocks.append(block)

        np.random.shuffle(all_blocks)  # shuffle block order (not contents)
        for i in range(0, len(all_blocks), self.batch_size):
            batch_blocks = all_blocks[i:i + self.batch_size]
            if len(batch_blocks) >= self.min_batch_size:
                self.batches.append(batch_blocks)

    def __iter__(self):
        np.random.shuffle(self.batches)  # reshuffle batch order each epoch
        for blocks in self.batches:
            flat = []
            for b in blocks:
                flat.extend(b)
            yield flat

    def __len__(self):
        return len(self.batches)


class DynamicUserBatchSampler(torch.utils.data.Sampler):
    """
    Dynamic windows for TRAIN. Each epoch rotates block sizes from possible_block_sizes.
    - Never shuffles the order of days inside a user
    - Shuffles window order (and user order) each epoch
    """
    def __init__(self, user_dataset_map, batch_size, possible_block_sizes=None, seed: int = 1234):
        self.user_dataset_map = user_dataset_map
        self.batch_size = int(batch_size)
        self.possible_block_sizes = [int(s) for s in (possible_block_sizes or [])]
        if not self.possible_block_sizes:
            raise ValueError("DynamicUserBatchSampler: possible_block_sizes must be non-empty")
        self.current_block_size = self.possible_block_sizes[0]
        self.epoch = 0
        # cache users with chronologically sorted indices
        self._users = [(u, sorted(idx)) for u, idx in self.user_dataset_map.items()]
        self._base_seed = int(seed)

    def _windows_for(self, size):
        size = int(size)
        blocks = []
        # shuffle user order per epoch to avoid bias
        rng = np.random.default_rng(self._base_seed + self.epoch)
        user_order = list(range(len(self._users)))
        rng.shuffle(user_order)
        for k in user_order:
            user_idx, indices = self._users[k]
            n = len(indices)
            if n < size:
                continue
            # stride 1 sliding windows over chronological indices
            for i in range(0, n - size + 1):
                block = indices[i:i+size]
                blocks.append((user_idx, block))
        # shuffle window order per epoch
        rng.shuffle(blocks)
        return blocks

    def __iter__(self):
        # rotate block size deterministically by epoch
        self.current_block_size = self.possible_block_sizes[self.epoch % len(self.possible_block_sizes)]
        self.epoch += 1

        blocks = self._windows_for(self.current_block_size)
        # pack into batches of blocks
        batch, bs = [], self.batch_size
        for (u, block) in blocks:
            batch.append(block)
            if len(batch) == bs:
                # flatten block-of-days into a flat index list
                flat = [d for b in batch for d in b]
                yield flat
                batch = []
        # drop last partial batch (let collate_loose keep tail within a batch if desired)

    def __len__(self):
        # rough estimate for logging
        size = int(self.current_block_size)
        total_blocks = 0
        for _, indices in self._users:
            n = len(indices)
            if n >= size:
                total_blocks += (n - size + 1)
        return total_blocks // max(1, self.batch_size)


class MultiUserDataset(torch.utils.data.Dataset):
    """
    Dataset für multiple User, das die Zuordnung von Datenpunkten zu Usern speichert.
    """
    def __init__(self, user_data_dict):
        self.user_data = []  # Alle Daten in einer flachen Liste
        self.user_to_indices = {}  # Mapping von User-ID -> Indizes in self.user_data
        
        idx = 0
        for user_idx, (X_loc, X_cal,target) in user_data_dict.items():
            # Für jeden Tag dieses Users
            self.user_to_indices[user_idx] = []
            
            for day in range(X_loc.size(0)):
                self.user_data.append((X_loc[day], X_cal[day],target[day], user_idx))
                self.user_to_indices[user_idx].append(idx)
                idx += 1
    
    def __len__(self):
        return len(self.user_data)
    
    def __getitem__(self, idx):
        return self.user_data[idx]

def create_multiuser_dataloaders(
    user_data,
    context_list=None,           # used when dynamic=True, e.g. [1,7,14,28]
    context_days=None,           # used when dynamic=False (single fixed context)
    num_days: int = 7,           # forecast horizon
    stride: int = 1,              # added default for stride
    current_epoch: int = 0,      # pass trainer.current_epoch
    dynamic: bool = True,        # <-- from your new CLI flag
    train_ratio: float = 0.85,   # 85/15 split BY BLOCKS per user
    leave_days: int = 0,         # drop these days from the END (ignored entirely)
    batch_size: int = 4,         # batch in UNITS OF BLOCKS
    shuffle: bool = True,        # shuffle TRAIN blocks only (never days inside block)
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    normalization_stats_prev: dict | None = None  # reuse μ/σ across epochs if provided
):
    """
    Unified, block-based loaders:
      - Picks THIS-epoch block_size:
          * dynamic=True  -> context = context_list[current_epoch % len(context_list)]
          * dynamic=False -> context = context_days
        block_size = max(context + num_days, num_days+1)
      - Trims last `leave_days` per user completely.
      - Forms NON-overlapping, chronological blocks per user of length `block_size`.
      - Splits blocks 85/15 per user (by count). Shuffles TRAIN blocks if requested.
      - Per-user normalization: fit μ/σ on TRAIN blocks only (days×time → per-feature) if not provided,
        then apply to both train & val blocks for that user.
      - Returns DataLoaders where each item is already one full block (X_loc_block, X_cal_block, target_block).
    """
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader

    # --- 1) Choose context -> block_size for THIS epoch ---
    if dynamic:
        if not context_list or len(context_list) == 0:
            raise ValueError("dynamic=True requires a non-empty context_list.")
        context = int(context_list[current_epoch % len(context_list)])
    else:
        if context_days is None:
            raise ValueError("dynamic=False requires context_days to be set.")
        context = int(context_days)

    block_size = int(context) + int(num_days)
    block_size = max(block_size, int(num_days) + 1)  # ensure at least 1 history day

    # --- helpers ---
    def _make_blocks(usable_len: int, block_len: int):
        """Return time-contiguous blocks as lists of indices in 0..usable_len-1."""
        return [list(range(i, i + block_len)) for i in range(0, usable_len - block_len + 1, stride)]
               

    class _BlockDataset(Dataset):
        def __init__(self, blocks):
            # each item is tuple (X_loc_block, X_cal_block, target_block)
            self.blocks = blocks
        def __len__(self):
            return len(self.blocks)
        def __getitem__(self, idx):
            return self.blocks[idx]

    # --- accumulators ---
    train_blocks_all = []
    val_blocks_all   = []
    normalization_stats = {} if normalization_stats_prev is None else dict(normalization_stats_prev)

    # --- 2) Per-user: trim tail, form blocks, split by blocks, fit/apply normalization ---
    for user_idx, (X_loc, X_cal, target) in user_data.items():
        total_days = X_loc.size(0)
        usable = max(0, total_days - max(0, int(leave_days)))
        if usable < block_size:
            continue  # no full block for this user this epoch

        per_user_blocks = _make_blocks(usable, block_size) # creates blocks of size block_size and leaves out usable_len
        if not per_user_blocks:
            continue

        if shuffle and len(per_user_blocks) > 1:
            rng = np.random.default_rng()
            rng.shuffle(per_user_blocks)

        n = len(per_user_blocks)
        n_train = min(max(int(round(n * float(train_ratio))), 1), n)
        train_block_idxs = per_user_blocks[:n_train]
        val_block_idxs   = per_user_blocks[n_train:] if (n - n_train) > 0 else []

        # (c) slice usable prefix ONCE; indices in blocks are in 0..usable-1
        X_loc_u = X_loc[:usable]
        X_cal_u = X_cal[:usable]
        tgt_u   = target[:usable]

        # (d) fit per-user μ/σ from TRAIN blocks only (unless provided)
        key = str(user_idx)
        if key not in normalization_stats:
            train_days = sorted({d for b in train_block_idxs for d in b})
            if len(train_days) > 0:
                mu  = X_loc_u[train_days].float().mean(dim=(0, 1))                # [C]
                std = X_loc_u[train_days].float().std(dim=(0, 1)).clamp_min(1e-6) # [C]
                normalization_stats[key] = {
                    "mean": mu.detach().cpu().tolist(),
                    "std":  std.detach().cpu().tolist(),
                    "note": "Per-user Z-score, fit on TRAIN blocks only."
                }

        # (e) apply μ/σ to usable prefix
        stat = normalization_stats.get(str(user_idx))
        if stat is not None:
            mu  = torch.tensor(stat["mean"], device=X_loc_u.device, dtype=X_loc_u.dtype)
            std = torch.tensor(stat["std"],  device=X_loc_u.device, dtype=X_loc_u.dtype)
            X_loc_u = (X_loc_u - mu) / std

        # (f) materialize blocks as tensors (chronology preserved within each block)
        def _gather(block_indices):
            out = []
            for b in block_indices:
                Xl = X_loc_u[b, ...]  # [D, T, C]
                Xc = X_cal_u[b, ...]  # [D, T, C_cal]
                Y  = tgt_u[b, ...]    # [D, ...]
                out.append((Xl, Xc, Y))
            return out

        train_blocks_all.extend(_gather(train_block_idxs))
        val_blocks_all.extend(_gather(val_block_idxs))

    # --- 3) Build datasets/loaders (no sampler; each item is a full block) ---
    train_ds = _BlockDataset(train_blocks_all)
    val_ds   = _BlockDataset(val_blocks_all)

    pw = False if dynamic else bool(persistent_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=pw)

    # validation: no need for persistent workers at all
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=min(2, num_workers), pin_memory=pin_memory,
                            persistent_workers=False)

    print(f"[DEBUG] E{current_epoch} dynamic={dynamic} ctx={context} -> block_size={block_size} | "
          f"train_blocks={len(train_ds)} val_blocks={len(val_ds)}")

    return train_loader, val_loader, block_size, normalization_stats


def create_dataloaders(X_loc, X_cal, target, train_ratio=0.7, val_ratio=0.15, batch_size=4):
    """
    Erstellt DataLoader für Training, Validierung und Test.
    
    Args:
        X_loc (torch.Tensor): Location Features
        X_cal (torch.Tensor): Kalender Features
        target (torch.Tensor): Labels
        train_ratio (float): Anteil der Daten für Training
        val_ratio (float): Anteil der Daten für Validierung
        batch_size (int): Batch-Größe für DataLoader
    
    Returns:
        Tuple von DataLoadern: (train_loader, val_loader, test_loader)
    """
    
    # Datensatz erstellen
    dataset = TensorDataset(X_loc, X_cal, target)
    
    # Aufteilung in Train/Val/Test
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # DataLoader erstellen
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    print(f"[DEBUG] Datensatz aufgeteilt in: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)} Samples")
    
    return train_loader, val_loader, test_loader



def _get_user_norm(stats_dict, user_idx, device, dtype=torch.float32):
    if stats_dict is None:
        return None, None
    s = stats_dict.get(str(user_idx))
    if not s: return None, None
    mu  = torch.tensor(s["mean"], dtype=dtype, device=device)
    std = torch.tensor(s["std"],  dtype=dtype, device=device)
    return mu, std


def train_model(
    train_loader, val_loader,
    location_dim, calendar_dim,
    hidden_dim=32, num_heads=4, dropout=0.4, num_of_layers=3,
    max_epochs=20, patience=10, step_size=1, model_name="LearnTransformer",
    plot_loss=False, use_flash_attention=True, use_axial_attention=False,
    logger=None,
    dataloader_cfg: dict | None = None,
):
    """
    If `dataloader_cfg` is provided, we IGNORE the passed train/val loaders and
    build loaders inside via a DataModule that calls `create_multiuser_dataloaders`
    *each epoch*, so dynamic context works. Otherwise we use the provided loaders.

    Saves like Weather:
      ~/scratch/presence_prediction/tud_presence_prediction/training_results/<ModelName>/version_<N>/checkpoints/...
    and writes model_info.json next to the checkpoints (version_<N>/model_info.json).
    """
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

    # ---------------- paths: same pattern as Weather ----------------
    base_dir = os.path.join(
        os.path.expanduser("~/scratch/presence_prediction"),
        "tud_presence_prediction", "training_results", model_name
    )
    version_str = str(logger.version) if logger is not None else datetime.now().strftime("%Y%m%d-%H%M%S")
    version_dir = os.path.join(base_dir, f"version_{version_str}")
    ckpt_dir = os.path.join(version_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if logger is not None:
        logger.experiment.config["model_version"] = version_str

    # NOTE: do NOT rely on globals().get("args") here; take num_days from cfg or fallback
    cfg_num_days = None
    if dataloader_cfg is not None:
        cfg_num_days = int(dataloader_cfg.get("num_days", 7))

    # ---------------- model class and init ----------------
    ModelClass = get_model_class(use_axial_attention)
    if ModelClass is None:
        raise ValueError("No valid model class found.")

    # Sauberes LR-Setup für OneCycle/Cosine/etc.:
    # - max_lr moderat (z.B. 3e-4)
    # - start_lr deutlich kleiner (z.B. max_lr/25)
    base_max_lr = 3e-4
    start_lr    = base_max_lr / 25.0    # ≈ 1.2e-5
    min_lr      = base_max_lr / 100.0   # ≈ 3e-6

    model = ModelClass(
        location_dim=location_dim,
        calendar_dim=calendar_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        num_of_layers=num_of_layers,
        step_size=step_size,
        num_days=(cfg_num_days if cfg_num_days is not None else 7),
        use_LR_scheduler=True,
        fixed_learning_rate=start_lr,     # <— Start-LR (klein!)
        LR_scheduler_max=base_max_lr,     # <— Max-LR
        LR_scheduler_min=min_lr,          # <— Min-LR
        LR_increase_phase_percentage=0.30,# 30% Warmup (statt 15% aggressiv)
        gradient_clip_val=1.0,
        weight_decay=0.01,
        use_flash_attention=use_flash_attention
    )

    # ---------------- callbacks: save_last=True like Weather ----------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:07d}-val_loss_epoch={val_loss_epoch:.2f}",
        save_top_k=1,
        monitor="val_loss_epoch",
        mode="min",
        save_last=True
    )

    is_dynamic = bool(dataloader_cfg.get("dynamic", True)) if dataloader_cfg is not None else False

    early_stop_callback = EarlyStopping(
        monitor="val_loss_epoch",   # wichtig: EPOCH-aggregiert, nicht step!
        mode="min",
        min_delta=3e-4,             # vorher: 0.003  →  jetzt: 0.0003
        patience=(24 if is_dynamic else 12),  # non-dynamic gern leicht höher als 10
        check_on_train_epoch_end=False,
        strict=False,
        verbose=True,
    )

    loss_logger = LossHistoryLogger()

    # ---------------- DataModule wrapper (only if dataloader_cfg is given) ----------------
    dm = None
    normalization_stats_out = None

    if dataloader_cfg is not None:
        # EXPECTED KEYS in dataloader_cfg:
        # user_data (dict), dynamic (bool), context_list (list[int]) or context_days (int),
        # num_days (int), leave_days (int), train_ratio (float, optional),
        # batch_size (int), shuffle (bool), num_workers, pin_memory, persistent_workers
        user_data = dataloader_cfg["user_data"]
        dynamic   = bool(dataloader_cfg.get("dynamic", True))
        stride = int(dataloader_cfg.get("stride", 1))  # added default for stride
        num_days  = int(dataloader_cfg["num_days"])
        leave_days= int(dataloader_cfg.get("leave_days", 0))
        train_ratio = float(dataloader_cfg.get("train_ratio", 0.85))
        batch_size  = int(dataloader_cfg.get("batch_size", 4))
        shuffle     = bool(dataloader_cfg.get("shuffle", True))
        num_workers = int(dataloader_cfg.get("num_workers", 8))
        pin_memory_ = bool(dataloader_cfg.get("pin_memory", True))
        persistent_workers_ = bool(dataloader_cfg.get("persistent_workers", True))

        if dynamic:
            context_list = list(map(int, dataloader_cfg["context_list"]))
            context_days = None
        else:
            context_list = None
            context_days = int(dataloader_cfg["context_days"])

        class PresenceDM(pl.LightningDataModule):
            def __init__(self):
                super().__init__()
                self._epoch = 0
                self._norm_stats = None
                self._train_loader = None
                self._val_loader = None

            @property
            def current_epoch(self):
                return self._epoch

            def _rebuild(self):
                tl, vl, block_size, self._norm_stats = create_multiuser_dataloaders(
                    user_data=user_data,
                    context_list=context_list,
                    context_days=context_days,
                    num_days=num_days,
                    current_epoch=self.current_epoch,
                    dynamic=dynamic,
                    stride=stride,
                    train_ratio=train_ratio,
                    leave_days=leave_days,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=pin_memory_,
                    persistent_workers=persistent_workers_,
                    normalization_stats_prev=self._norm_stats
                )
                self._train_loader, self._val_loader = tl, vl
                if logger is not None:
                    logger.experiment.log({
                        "debug/epoch": self.current_epoch,
                        "debug/block_size": block_size,
                        "debug/train_blocks": len(tl.dataset),
                        "debug/val_blocks": len(vl.dataset) if vl is not None else 0,
                    })

            def setup(self, stage=None):
                pass

            def train_dataloader(self):
                self._rebuild()
                self._epoch += 1
                return self._train_loader

            def val_dataloader(self):
                return self._val_loader

            def get_norm_stats(self):
                return self._norm_stats or {}

        dm = PresenceDM()

    # ---------------- trainer ----------------
    trainer = pl.Trainer(
        default_root_dir=os.path.join(base_dir, "lightning_logs"),
        max_epochs=max_epochs,
        min_epochs=20,
        callbacks=[loss_logger, checkpoint_callback, early_stop_callback],
        logger=logger,
        devices="auto",
        accelerator="auto",
        precision="bf16-mixed",
        log_every_n_steps=1,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
        reload_dataloaders_every_n_epochs=1 if dm is not None else 0,
        accumulate_grad_batches=4,   # <— NEU: stabilisiert Gradienten
    )


    # ---------------- train ----------------
    print(f"[INFO] Starting training ({'FlashAttention' if use_flash_attention else 'Standard Attention'}) for {max_epochs} epochs")
    start_time = time.time()
    if dm is not None:
        trainer.fit(model, datamodule=dm)
        normalization_stats_out = dm.get_norm_stats()
    else:
        trainer.fit(model, train_loader, val_loader)
        normalization_stats_out = globals().get("_NORMALIZATION_STATS", {})  # fallback if you still set it elsewhere
    end_time = time.time()
    training_time = end_time - start_time
    if logger is not None:
        logger.experiment.config["training_time"] = training_time
        logger.experiment.summary["training_time_seconds"] = training_time

    # ---------------- metadata ----------------
    model_params = {
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_of_layers": num_of_layers,
        "dropout": dropout,
        "use_flash_attention": use_flash_attention
    }

    # What we save about data/training:
    if dataloader_cfg is not None:
        training_params = {
            "batch_size": int(dataloader_cfg.get("batch_size", 4)),
            "dynamic_sampling": bool(dataloader_cfg.get("dynamic", True)),
            "context_list": list(map(int, dataloader_cfg.get("context_list", []))) if bool(dataloader_cfg.get("dynamic", True)) else None,
            "context_days": int(dataloader_cfg.get("context_days")) if not bool(dataloader_cfg.get("dynamic", True)) else None,
            "num_days": int(dataloader_cfg.get("num_days", 7)),
            "leave_days": int(dataloader_cfg.get("leave_days", 0)),
            "train_ratio": float(dataloader_cfg.get("train_ratio", 0.85)),
            "max_epochs": int(max_epochs),
            "training_time": float(training_time),
        }
    else:
        # fall back: unknown loader config (kept minimal)
        training_params = {
            "batch_size": getattr(train_loader, "batch_size", 4),
            "dynamic_sampling": None,
            "context_list": None,
            "context_days": None,
            "num_days": (cfg_num_days if cfg_num_days is not None else 7),
            "leave_days": None,
            "train_ratio": 0.85,
            "max_epochs": int(max_epochs),
            "training_time": float(training_time),
        }

    save_model_info(
        model_dir=base_dir,
        version=version_str,
        model_params=model_params,
        training_params=training_params,
        normalization_stats=normalization_stats_out
    )  # writes model_info.json next to checkpoints. :contentReference[oaicite:3]{index=3}

    # optional: also stash raw torch states inside the run folder (as you had)
    if wandb.run is not None:
        checkpoint_dir = os.path.join(wandb.run.dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_torch_states.pth"))
        opt = model.optimizers()
        if hasattr(opt, "optimizer"):
            torch.save(opt.optimizer.state_dict(), os.path.join(checkpoint_dir, "final_torch_optimizer.pth"))
        wandb.save(os.path.join(checkpoint_dir, "*.pth"))

    print(f"[INFO] Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"[INFO] Version directory: {version_dir}")
    return model, version_str, training_time



def create_forecast_for_user(model, user_data, user_idx, user_timestamps, 
                             num_days=3, num_input_days=10):
    if user_idx not in user_data:
        raise ValueError(f"Daten für User {user_idx} nicht gefunden")
    
    model.eval()
    
    # Setze die Vorhersagelänge für das Modell
    model.set_model_prediction_length(num_days)
    
    # Bestimme das Gerät des Modells
    device = next(model.parameters()).device
    print(f"[INFO] Modell befindet sich auf Gerät: {device}")
    
    X_loc, X_cal, target = user_data[user_idx]
    total_days = X_loc.size(0)

    # Apply saved normalization for this user (if available)
    device = next(model.parameters()).device
    mu, std = _get_user_norm(globals().get("_NORMALIZATION_STATS"), user_idx, device, dtype=X_loc.dtype)
    if mu is not None and std is not None:
        X_loc = (X_loc.to(device) - mu) / std
        X_cal = X_cal.to(device)
        target = target.to(device)
    else:
        # keep original device moves later
        pass
    
    # Überprüfe, ob genügend Tage für die Eingabe vorhanden sind
    if total_days <= num_input_days:
        print(f"[WARNUNG] User {user_idx} hat nur {total_days} Tage, " 
              f"benötigt aber mindestens {num_input_days} für Input")
        return 0, None
        
    # Input-Bereich definieren
    input_start_idx = total_days - num_input_days
    input_end_idx = total_days
    
    # Eingabedaten auf das gleiche Gerät wie das Modell verschieben
    x_loc_input = X_loc[input_start_idx:input_end_idx].unsqueeze(0).to(device)
    x_cal_input = X_cal[input_start_idx:input_end_idx].unsqueeze(0).to(device)
    
    print(f"[INFO] User {user_idx}: Verwende Tage {input_start_idx}-{input_end_idx-1} als Input, "
          f"sage die nächsten {num_days} Tage vorher")
    
    # Zeit messen für die Vorhersage
    start_time = time.time()
    
    with torch.no_grad():
        # Modell-Vorhersage für die Zukunft
        pred_future = model(x_loc_input, x_cal_input)
    
    end_time = time.time()
    evaluation_time = end_time - start_time
    
    print(f"[INFO] Evaluationszeit: {evaluation_time:.2f} Sekunden")
    
    return evaluation_time, pred_future


def evaluate_forecast(model, user_data, input_days=20, prediction_days=3, input_list=None, version=None, logger=None, timestamps=None):
    """
    Rich evaluation producing per-user probability maps and extended metrics.
    Zusätzlich: Konsolen-Vorschau (plain ASCII) mit Kontext (Vergangenheit) + GT/PR je Vorhersagetag.

    JSON schema (v2):
    {
      "version": "<string>",
      "timestamp": "...",
      "input_days": <int>,
      "prediction_days": <int>,
      "slots_per_day": <int>,
      "metrics_global": {...},
      "users": [
        {
          "user_id": "<id>",
          "context": {...},
          "context_targets": [[0/1,...], ...],   # NEU: [inp_days, slots] - das, was das Modell als Kontext gesehen hat
          "targets": [[0/1,...], ...],           # [prediction_days, slots]
          "probs":   [[0.0..1.0,...], ...],
          "binary_at_0_5": [[0/1,...], ...],
          "metrics": {...}
        }, ...
      ]
    }
    """
    import numpy as np
    from datetime import datetime
    import os, json, time
    import torch
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
        average_precision_score, confusion_matrix
    )

    # --- kleine Plain-ASCII Helfer (lokal, kein ANSI, keine Abhängigkeit) ---
    def _bin_str_line(arr, one='1', zero='0', miss='.', cast_int=True):
        out = []
        for v in arr:
            if np.isnan(v):
                out.append(miss)
            else:
                out.append(one if (int(v) if cast_int else v >= 0.5) == 1 else zero)
        return "".join(out)

    model.eval()
    device = next(model.parameters()).device
    model.set_model_prediction_length(int(prediction_days))

    # Liste der zu evaluierenden Kontextlängen
    ctx_list = [int(input_days)] if not input_list else [int(x) for x in input_list]
    written_paths = []

    for inp_days in ctx_list:
        users_payload = []
        all_probs_flat, all_targets_flat = [], []
        slots_per_day = None

        with torch.no_grad():
            for user_idx, (X_loc, X_cal, target) in user_data.items():
                # optional: Normalisierung wie bei Forecast (falls Stats vorhanden)
                mu, std = _get_user_norm(globals().get("_NORMALIZATION_STATS"), user_idx, device, dtype=X_loc.dtype)
                if mu is not None and std is not None:
                    X_loc = (X_loc.to(device) - mu) / std
                    X_cal = X_cal.to(device)
                    target = target.to(device)
                else:
                    X_loc, X_cal, target = X_loc.to(device), X_cal.to(device), target.to(device)

                total_days = X_loc.size(0)
                slots_per_day = int(target.size(-1))

                # Fenster prüfen
                if total_days <= (inp_days + prediction_days):
                    print(f"[WARN] User {user_idx} has only {total_days} days; needs >= {inp_days + prediction_days}. Skipping.")
                    continue

                # Indizes
                test_start  = total_days - prediction_days
                input_start = max(0, test_start - inp_days)
                input_end   = test_start

                # Kontext-Targets (Vergangenheit) als numpy (für Vorschau + JSON)
                ctx_t_np = target[input_start:input_end].detach().cpu().numpy().astype(float)  # [inp_days, slots]

                # Tensors mit Batch-Dim
                x_loc_input   = X_loc[input_start:input_end].unsqueeze(0)
                x_cal_input   = X_cal[input_start:input_end].unsqueeze(0)
                target_future = target[test_start:].unsqueeze(0)  # [1, pred_days, slots]

                # Vorwärts (Logits -> Probs)
                logits = model(x_loc_input, x_cal_input)          # [1, pred_days, slots]
                if logits.shape[1] > prediction_days:
                    logits = logits[:, :prediction_days]
                probs = torch.sigmoid(logits).detach().cpu().numpy()[0]       # [pred_days, slots]
                t_np  = target_future.detach().cpu().numpy()[0].astype(float) # [pred_days, slots]

                # -------- TERMINAL-VORSCHAU: Kontext + GT/PR (plain ASCII) --------
                thr = CONSOLE_THRESHOLD if 'CONSOLE_THRESHOLD' in globals() else 0.5
                print(f"\n=== Ctx={inp_days} | User {user_idx} | pred_days={prediction_days} | slots={probs.shape[1]} ===", flush=True)

                # Kontext-Tage (Vergangenheit): d = -inp_days .. -1
                for k in range(ctx_t_np.shape[0]):
                    d_rel = -(ctx_t_np.shape[0] - k)
                    line  = _bin_str_line(ctx_t_np[k], one='1', zero='0', miss='.')
                    print(f"CTX {d_rel:>3}   {line}", flush=True)

                # Vorhersage-Tage (Zukunft): Day +1 .. +prediction_days
                pb = np.where(np.isnan(probs), np.nan, (probs >= float(thr)).astype(float))
                for d in range(probs.shape[0]):
                    gt_line = _bin_str_line(t_np[d], one='1', zero='0', miss='.')
                    pr_line = _bin_str_line(pb[d],   one='1', zero='0', miss='.')
                    print(f"Day {d+1:<2}  GT {gt_line}", flush=True)
                    print(f"{'':7} PR {pr_line}  @thr={thr:.2f}", flush=True)

                # ------ Metriken @0.5 (Probs bleiben für Threshold-Sweeps in JSON) ------
                p_bin = (probs >= 0.5).astype(int)
                yt = t_np.flatten(); yp = p_bin.flatten()
                valid = ~np.isnan(yt)
                yt = yt[valid].astype(int); yp = yp[valid].astype(int)
                pr = probs.flatten()[valid]

                cm = confusion_matrix(yt, yp, labels=[0,1])
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                bal_acc = balanced_accuracy_score(yt, yp) if (np.any(yt==0) and np.any(yt==1)) else float("nan")
                mcc = matthews_corrcoef(yt, yp) if (np.any(yt==0) and np.any(yt==1)) else float("nan")
                brier = float(np.mean((pr - yt)**2)) if pr.size > 0 else float("nan")
                prev = float(np.mean(yt)) if yt.size > 0 else float("nan")
                try:
                    roc = roc_auc_score(yt, pr) if (np.any(yt==0) and np.any(yt==1)) else float("nan")
                except Exception:
                    roc = float("nan")
                try:
                    ap = average_precision_score(yt, pr) if (np.any(yt==1)) else float("nan")
                except Exception:
                    ap = float("nan")

                per_user_metrics = {
                    "accuracy": float(accuracy_score(yt, yp)) if yt.size > 0 else float("nan"),
                    "precision": float(precision_score(yt, yp, zero_division=0)) if yt.size > 0 else float("nan"),
                    "recall": float(recall_score(yt, yp, zero_division=0)) if yt.size > 0 else float("nan"),
                    "f1": float(f1_score(yt, yp, zero_division=0)) if yt.size > 0 else float("nan"),
                    "specificity": float(spec),
                    "balanced_accuracy": float(bal_acc),
                    "mcc": float(mcc),
                    "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
                    "roc_auc": float(roc), "pr_auc": float(ap),
                    "brier": float(brier), "prevalence": float(prev)
                }

                # --- Dates inline aus `timestamps` ableiten, auf Länge ausrichten, und in JSON einbetten ---
                dates_payload = None
                if timestamps:
                    key = user_idx if user_idx in timestamps else (str(user_idx) if str(user_idx) in timestamps else None)
                    if key is not None:
                        user_dates_raw = timestamps[key]

                        # 1) Normalisieren zu 'YYYY-MM-DD' (datetime/date/pandas.Timestamp/np.datetime64 → String)
                        norm_sorted = []
                        for d in user_dates_raw:
                            try:
                                if hasattr(d, "strftime"):              # datetime.date / datetime.datetime / pandas.Timestamp
                                    norm_sorted.append(d.strftime("%Y-%m-%d"))
                                elif isinstance(d, np.datetime64):      # NumPy datetime64
                                    norm_sorted.append(np.datetime_as_string(d, unit="D"))
                                else:
                                    norm_sorted.append(str(d))           # Fallback (z.B. bereits String)
                            except Exception:
                                norm_sorted.append(str(d))
                        norm_sorted = sorted(norm_sorted)

                        # 2) Auf tatsächlich verwendete Tage ausrichten
                        #    (im Loader werden i.d.R. erster/letzter Tag entfernt)
                        if len(norm_sorted) == total_days:
                            aligned = norm_sorted
                        elif len(norm_sorted) >= total_days + 2:
                            mid = norm_sorted[1:-1]
                            aligned = mid[-total_days:] if len(mid) >= total_days else norm_sorted[-total_days:]
                        elif len(norm_sorted) > total_days:
                            aligned = norm_sorted[-total_days:]
                        else:
                            aligned = None

                        # 3) Kontext- und Vorhersagefenster schneiden
                        if aligned is not None and len(aligned) >= total_days:
                            context_dates    = aligned[input_start:input_end]
                            prediction_dates = aligned[test_start:total_days]
                            if (len(context_dates) == (input_end - input_start)) and (len(prediction_dates) == (total_days - test_start)):
                                dates_payload = {
                                    "context_dates": context_dates,
                                    "prediction_dates": prediction_dates
                                }

                # --- User-Payload bauen, optional dates anhängen, einmal appenden ---
                user_payload = {
                    "user_id": str(user_idx),
                    "context": {
                        "input_range_days": [int(input_start), int(input_end - 1)],
                        "prediction_range_days": [int(test_start), int(total_days - 1)],
                        "total_days_available": int(total_days)
                    },
                    "context_targets": ctx_t_np.tolist(),   # [inp_days, slots]
                    "targets": t_np.tolist(),               # [pred_days, slots]
                    "probs": probs.tolist(),                # [pred_days, slots]
                    "binary_at_0_5": p_bin.tolist(),
                    "metrics": per_user_metrics
                }
                if dates_payload is not None:
                    user_payload["dates"] = dates_payload

                users_payload.append(user_payload)

                all_probs_flat.append(pr)
                all_targets_flat.append(yt)

        # ---- Globale Metriken ----
        if len(all_targets_flat) > 0:
            Y = np.concatenate(all_targets_flat)
            P = np.concatenate(all_probs_flat)
            Yb = (P >= 0.5).astype(int)
            cmG = confusion_matrix(Y, Yb, labels=[0,1])
            tn, fp, fn, tp = cmG.ravel() if cmG.size == 4 else (0,0,0,0)
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            bal_acc = balanced_accuracy_score(Y, Yb) if (np.any(Y==0) and np.any(Y==1)) else float("nan")
            mcc = matthews_corrcoef(Y, Yb) if (np.any(Y==0) and np.any(Y==1)) else float("nan")
            brier = float(np.mean((P - Y)**2)) if P.size > 0 else float("nan")
            prev = float(np.mean(Y)) if Y.size > 0 else float("nan")
            try:
                roc = roc_auc_score(Y, P) if (np.any(Y==0) and np.any(Y==1)) else float("nan")
            except Exception:
                roc = float("nan")
            try:
                ap = average_precision_score(Y, P) if (np.any(Y==1)) else float("nan")
            except Exception:
                ap = float("nan")

            metrics_global = {
                "accuracy": float(accuracy_score(Y, Yb)) if Y.size > 0 else float("nan"),
                "precision": float(precision_score(Y, Yb, zero_division=0)) if Y.size > 0 else float("nan"),
                "recall": float(recall_score(Y, Yb, zero_division=0)) if Y.size > 0 else float("nan"),
                "f1": float(f1_score(Y, Yb, zero_division=0)) if Y.size > 0 else float("nan"),
                "specificity": float(spec),
                "balanced_accuracy": float(bal_acc),
                "mcc": float(mcc),
                "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
                "roc_auc": float(roc), "pr_auc": float(ap),
                "brier": float(brier), "prevalence": float(prev)
            }
        else:
            metrics_global = {}

        # ---- JSON schreiben (eine Datei pro Kontextlänge) ----
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(os.path.expanduser("~"), "scratch", "presence_prediction",
                                   "tud_presence_prediction", "evaluation_results")
        os.makedirs(results_dir, exist_ok=True)
        json_path = os.path.join(results_dir, f"eval_v2_ctx{inp_days}_{version or timestamp}.json")

        payload = {
            "version": version or "presence_eval_v2",
            "timestamp": datetime.now().isoformat(),
            "input_days": int(inp_days),
            "prediction_days": int(prediction_days),
            "slots_per_day": int(slots_per_day) if slots_per_day is not None else None,
            "metrics_global": metrics_global,
            "users": users_payload
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Saved rich eval JSON → {json_path}")

        if metrics_global:
            mg = metrics_global
            print(f"[SUMMARY ctx={inp_days}] acc={mg['accuracy']:.3f} f1={mg['f1']:.3f} "
                  f"rec={mg['recall']:.3f} prec={mg['precision']:.3f} bacc={mg['balanced_accuracy']:.3f} "
                  f"mcc={mg['mcc']:.3f} prAUC={mg['pr_auc']:.3f}")

        if logger is not None and metrics_global:
            ns = f"eval_v2/{inp_days}"
            logger.experiment.log({
                f"{ns}/accuracy": mg['accuracy'],
                f"{ns}/precision": mg['precision'],
                f"{ns}/recall": mg['recall'],
                f"{ns}/f1_score": mg['f1'],
                f"{ns}/balanced_accuracy": mg['balanced_accuracy'],
                f"{ns}/mcc": mg['mcc'],
                f"{ns}/pr_auc": mg['pr_auc'],
                f"{ns}/roc_auc": mg['roc_auc'],
                f"{ns}/brier": mg['brier'],
                f"{ns}/prevalence": mg['prevalence']
            })

        written_paths.append(json_path)

    return written_paths[-1] if len(written_paths) == 1 else written_paths





def load_model(model_file="LearnTransformer", version=None, production=False,
               model_path_production=None, use_flash_attention=True, use_axial_attention=False):
    """
    Load by version number (Weather-style). Prefer .../version_<N>/checkpoints/last.ckpt;
    fall back to the newest *.ckpt in that folder. Optionally allow explicit model_path_production.
    Also loads normalization stats from version_<N>/model_info.json.
    """
    # choose class
    actual_model_file = "LearnTransformerAxialSelf" if use_axial_attention else "LearnTransformer"
    actual_class_name = "LearnTransformer"

    # resolve checkpoint path
    if model_path_production:
        checkpoint_path = model_path_production
    elif version is not None:
        base_dir = os.path.join(
            os.path.expanduser("~/scratch/presence_prediction"),
            "tud_presence_prediction", "training_results", model_file
        )
        ckpt_dir = os.path.join(base_dir, f"version_{str(version)}", "checkpoints")
        last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            checkpoint_path = last_ckpt
        else:
            files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
            if not files:
                raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
            files.sort()
            checkpoint_path = os.path.join(ckpt_dir, files[-1])
    else:
        raise ValueError("Provide a version number or a model_path_production.")

    print(f"[INFO] Loading model from {checkpoint_path}")

    # load normalization stats from the same version folder (next to checkpoints)
    info_path = os.path.join(
        os.path.expanduser("~/scratch/presence_prediction"),
        "tud_presence_prediction", "training_results", model_file,
        f"version_{str(version)}", "model_info.json"
    )
    norm_stats = None
    try:
        with open(info_path, "r") as f:
            info_json = json.load(f)
            norm_stats = info_json.get("normalization_stats", None)
    except Exception as e:
        print(f"[WARNUNG] Could not load normalization_stats: {e}")
    globals()["_NORMALIZATION_STATS"] = norm_stats

    # dynamic import + load
    model_module_class = getattr(
        importlib.import_module(f".models.{actual_model_file}", package="tud_presence_prediction"),
        actual_class_name
    )
    map_loc = None if torch.cuda.is_available() else torch.device("cpu")
    model = model_module_class.load_from_checkpoint(checkpoint_path, map_location=map_loc)
    model.eval()

    print(f"[INFO] Model ({actual_model_file}) loaded")
    print(f"[INFO] Flash-Attention: {'On' if getattr(model, 'use_flash_attention', use_flash_attention) else 'Off'}")
    return model


class LossHistoryLogger(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        # just ensure the buffer exists (used only for debugging, not logging)
        if not hasattr(pl_module, 'training_step_outputs'):
            pl_module.training_step_outputs = []

    # Do NOT log val metrics here; models already log val_*_epoch in validation_step.
    def on_validation_epoch_start(self, trainer, pl_module):
        return

    def on_validation_epoch_end(self, trainer, pl_module):
        return

    # You said: only train loss per step (we already log that inside training_step).
    # So: do not log train_loss per epoch either.
    def on_train_epoch_end(self, trainer, pl_module):
        return


def plot_training_history(train_values, val_values, save_path=None, metric_name="Loss", title=None):
    """
    Plottet den Verlauf von Metriken während des Trainings.
    
    Args:
        train_values: Liste mit Trainingswerten pro Epoche
        val_values: Liste mit Validierungswerten pro Epoche
        save_path: Optional, Pfad zum Speichern des Plots
        metric_name: Name der Metrik (z.B. "Loss" oder "Accuracy")
        title: Optional, Titel des Plots
    """
    if not title:
        title = f'Verlauf des {metric_name}s während des Trainings'
    
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_values) + 1)
    plt.plot(epochs, train_values, 'b-', linewidth=2, label=f'Training {metric_name}')
    plt.plot(epochs, val_values, 'r-', linewidth=2, label=f'Validation {metric_name}')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Epoche', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Beschriftungen und Rahmen
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[INFO] {metric_name}-Plot gespeichert unter: {save_path}")
    
    plt.show()


def plot_training_history_model_util(train_values, val_values, save_path=None, metric_name="Loss", title=None):
    """
    Plottet den Verlauf von Metriken aus model_util.
    """
    if not title:
        title = f'Verlauf des {metric_name}s während des Trainings'
    
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_values) + 1)
    plt.plot(epochs, train_values, 'b-', linewidth=2, label=f'Training {metric_name}')
    plt.plot(epochs, val_values, 'r-', linewidth=2, label=f'Validation {metric_name}')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Batch', fontsize=14)  # model_util loggt pro Batch
    plt.ylabel(metric_name, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Beschriftungen und Rahmen
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[INFO] {metric_name}-Plot gespeichert unter: {save_path}")
    
    plt.show()


def save_model_info(model_dir, version, model_params, training_params, metrics=None, normalization_stats=None):
    """Speichert umfassende Modell-Metadaten"""
    info = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "model_name": "LearnTransformer",
        "model_params": {
            "hidden_dim": model_params["hidden_dim"],
            "num_heads": model_params["num_heads"],
            "num_layers": model_params["num_of_layers"],
            "dropout": model_params["dropout"],
            "flash_attention": model_params["use_flash_attention"],
        },
        "training_params": {
            "batch_size": training_params["batch_size"],
            "dynamic_sampling": training_params["dynamic_sampling"],
            "context_list": training_params.get("context_list"),
            "num_days": training_params["num_days"],
            "train_block_sizes": training_params.get("train_block_sizes"),
            "val_block_days": training_params.get("val_block_days"),
            "leave_days": training_params.get("leave_days"),
            "max_epochs": training_params["max_epochs"],
            "training_time_seconds": training_params["training_time"],
        },
        "metrics": metrics or {},
        "tensorboard_logs": f"version_{version}/lightning_logs"
    }


    if normalization_stats is not None:
        info["normalization_stats"] = normalization_stats

    
    # Speichere im Checkpoints-Verzeichnis
    os.makedirs(os.path.join(model_dir, f"version_{version}"), exist_ok=True)
    info_file = os.path.join(model_dir, f"version_{version}", "model_info.json")
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"[INFO] Modell-Dokumentation gespeichert unter: {info_file}")

if __name__ == '__main__':
    import argparse
    from datetime import datetime

    # Kommandozeilenargumente für verschiedene Anwendungsfälle
    parser = argparse.ArgumentParser(description='Homebrew Presence Prediction')
    parser.add_argument('--train', action='store_true', help='Modell trainieren')
    parser.add_argument('--predict', action='store_true', help='Vorhersage durchführen')
    parser.add_argument('--evaluate', action='store_true', help='Modell evaluieren')
    parser.add_argument('--plot_loss', action='store_true', help='Trainings- und Validierungsverlauf plotten')
    parser.add_argument('--model_file', type=str, default='LearnTransformer', help='Name des Modellfiles')
    parser.add_argument('--version', type=str, help='Version des trainierten Modells')
    parser.add_argument('--production', action='store_true', help='Modell für Produktion laden')
    parser.add_argument('--model_path_production', type=str, help='Pfad zum Produktionsmodell')
    parser.add_argument('--num_days', type=int, default=7, help='Anzahl der vorherzusagenden Tage')
    parser.add_argument('--num_input_days', type=int, default=10, help='Anzahl der Eingabetage für Vorhersage')
    parser.add_argument('--input_list', type=str, help="Kommaseparierte Liste von Eingabetagen für Evaluation, z.B. '1,3,7,14'")
    parser.add_argument('--consecutive_days', type=int, default=3, help='Anzahl der zusammenhängenden Tage für den Trainingsprozess NONE DYNAMIC SAMPLING')
    parser.add_argument("--leave_days", type=int, default=3, help="Anzahl der Tage am Ende jedes Users, die für Evaluation (Test) reserviert werden.")
    parser.add_argument("--context_list", type=str, default="7,14,21,28", help="Kommaseparierte Liste von Kontextlängen (z. B. '7,14,21,28'). Trainingsblock = context + num_days.")
    parser.add_argument('--user_idx', type=int, help='Index des Benutzers für Vorhersage oder Evaluation')
    parser.add_argument('--max_epochs', type=int, default=40, help='Maximale Anzahl der Trainingsepochen')
    parser.add_argument('--flash_attention', type=lambda x: x.lower() == 'true', default=True) # flashattention wird benutzt, wenn True in der Kommandozeile angegeben wird
    parser.add_argument('--use_axial', action='store_true', help='Verwendet das Axial-Attention-Modell')
    parser.add_argument('--wandb-project', type=str, help='Project name the logs are saved at')
    parser.add_argument("--dynamic", action="store_true", help="Enabling dynamic sampling.")
    parser.add_argument("--stride", type=int, help="Setting stride size for forming blocks.")
    parser.add_argument("--patience", type=int, help="Setting patience for training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size in Blöcken")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--persistent_workers", type=lambda x: str(x).lower() in ["1","true","yes"], default=True)



    # =========================================================================
    # VERWENDUNGSHINWEISE
    # =========================================================================
    # 1. TRAINING:
    #    python homebrew_presence.py --train
    #    - Trainiert ein neues Modell mit den Standard-Hyperparametern
    #    - Speichert das Modell in tud_presence_prediction/training_results/
    #
    # 2. TRAINING + EVALUATION:
    #    python homebrew_presence.py --train --evaluate
    #    - Trainiert ein neues Modell und evaluiert es direkt auf dem Test-Datensatz
    #    - Gibt detaillierte Metriken wie Accuracy, Precision, Recall und F1 aus
    #
    # 3. EVALUATION EINES VORHANDENEN MODELLS:
    #    python homebrew_presence.py --evaluate --version 66
    #    - Lädt ein bereits trainiertes Modell aus Version 66
    #    - Evaluiert es auf dem globalen Test-Datensatz
    #
    #    python homebrew_presence.py --evaluate --user_idx 2 --version 66
    #    - Evaluiert das Modell nur auf den Daten von User 2
    #
    # 4. VORHERSAGE FÜR EINEN BENUTZER:
    #    python homebrew_presence.py --predict --user_idx 2 --version 66 --num_days 3 --num_input_days 10
    #    - Erstellt eine Vorhersage für User 2 mit den letzten 10 Tagen als Input
    #    - Sagt die nächsten 3 Tage vorher
    #    - Misst nur die Evaluierungszeit ohne Metriken zu berechnen
    #
    # 5. PRODUKTION:
    #    python homebrew_presence.py --predict --production --model_path_production "/pfad/zum/modell.ckpt" ...
    #    - Lädt ein Modell von einem bestimmten Pfad für Produktionsumgebungen
    # =========================================================================
    
    args = parser.parse_args()
    context_list = [int(s) for s in args.context_list.split(",") if str(s).strip()]
    num_non_use_days = args.leave_days  # Anzahl der Tage, die am Ende jedes Users vorne weg gelassen wird, um daraus später zu predicten
    
    logger = WandbLogger(project=args.wandb_project)

    run = logger.experiment
    run.define_metric("training_time_seconds", summary="last")  # Aggregator für die Runs-Tabelle

    # Speichere alle Kommandozeilenargumente im Logger
    logger.experiment.config.update(vars(args))

    # Modelltyp für Log-Ausgabe definieren
    model_type = "AxialAttention" if args.use_axial else "Standard"
    print(f"[INFO] Verwende {model_type} Transformer-Modell")

    print("[INFO] Lade Daten...")
    txt_file = 'tud_presence_prediction/data/local_data/storage/dynamic_multiuser_20250512T124907646662_raw.txt'
    user_data, timestamps = load_and_process_multiuser(txt_file, fill_missing=True, distance_threshold=0.02) # distance_threshold is about 20m
 
    model = None

    if args.train:
        dataloader_cfg = {
            "user_data": user_data,
            "dynamic": bool(args.dynamic),
            "stride": int(args.stride) if args.stride else None,
            "context_list": [int(s) for s in args.context_list.split(",")] if args.dynamic else None,
            "context_days": int(args.consecutive_days) if not args.dynamic else None,
            "num_days": int(args.num_days),
            "leave_days": int(args.leave_days),
            "train_ratio": 0.85,
            "shuffle": True,
            "pin_memory": True,
            "persistent_workers": True,
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "persistent_workers": bool(args.persistent_workers),
        }


        model, version, training_time = train_model(
            train_loader=None, val_loader=None,
            location_dim=3, calendar_dim=8,
            hidden_dim=32, num_heads=4, dropout=0.2, num_of_layers=6,
            max_epochs=args.max_epochs,
            step_size=1, model_name=("LearnTransformerAxial" if args.use_axial else args.model_file),
            plot_loss=args.plot_loss, use_flash_attention=args.flash_attention, use_axial_attention=args.use_axial,
            logger=logger,
            dataloader_cfg=dataloader_cfg,
            patience=args.patience if args.patience else 10 
        )


        run.summary["training_time_seconds"] = float(training_time)   # erscheint als Spalte

        # print(f"[INFO] Modell wurde mit Version {version} gespeichert")
        # print(f"[INFO] Trainingszeit: {training_time:.2f} Sekunden")
        # print(f"[INFO] Use Flashattention: {args.flash_attention}")
        # print(f"[INFO] Die Anzahl an zusammenhängenden Tagen ist {args.consecutive_days} Tage")

    elif args.version or args.model_path_production:
        # PHASE 2: LADEN EINES BESTEHENDEN MODELLS
        # Lädt ein vortrainiertes Modell aus einem Checkpoint
        model = load_model(
            model_file=args.model_file,
            version=args.version, 
            production=args.production,
            model_path_production=args.model_path_production,
            use_flash_attention=args.flash_attention,
            use_axial_attention=args.use_axial
        )
    
    else:
        # Keine Aktion ausgewählt oder erforderliche Parameter fehlen
        print("[FEHLER] Entweder --train oder --version/--model_path_production muss angegeben werden")
        exit(1)

    # PHASE 3: VORHERSAGE FÜR EINEN BENUTZER
    if args.predict and model:
        if args.user_idx is None:
            print("[FEHLER] Für die Vorhersage muss ein Benutzerindex (--user_idx) angegeben werden")
            exit(1)
        
        print(f"[INFO] Erstelle Forecast für User {args.user_idx}")
        
        evaluation_time, predictions = create_forecast_for_user(
            model,
            user_data,
            args.user_idx,
            timestamps,
            num_days=args.num_days,
            num_input_days=args.num_input_days,
        )
        
        if predictions is not None:
            print(f"[INFO] Vorhersage erstellt in {evaluation_time:.2f} Sekunden")
            print(f"[INFO] Vorhersageform: {predictions.shape}")
            
            # Optional: Konvertiere die Vorhersagen in binäre Werte für bessere Lesbarkeit
            binary_predictions = (torch.sigmoid(predictions) > 0.5).float()
            print("[INFO] Beispielvorhersagen (1=zuhause, 0=abwesend):")
            
            # Zeige die ersten paar Zeitslots für jeden vorhergesagten Tag
            for day in range(min(args.num_days, predictions.shape[1])):
                slots_to_show = min(12, predictions.shape[2])  # Zeige max. 5 Zeitslots
                print(f"Tag {day+1}: {binary_predictions[0, day, :slots_to_show].tolist()}")
        else:
            print("[FEHLER] Keine Vorhersage erstellt")
    
    elif args.evaluate and model:
        print("[INFO] Führe Forecast-Evaluation durch")

        # input_list aus CLI parsen (falls gesetzt)
        eval_input_list = None
        if args.input_list:
            eval_input_list = [int(s) for s in args.input_list.split(",") if str(s).strip()]

        # ggf. auf einen User einschränken
        eval_data = user_data
        if args.user_idx is not None:
            if args.user_idx not in user_data:
                print(f"[FEHLER] User {args.user_idx} nicht gefunden")
                exit(1)
            eval_data = {args.user_idx: user_data[args.user_idx]}

        # führt die reiche Evaluation aus und erhält Pfad/ Pfade
        out_paths = evaluate_forecast(
            model,
            eval_data,
            input_days=args.num_input_days,
            prediction_days=args.num_days,
            input_list=eval_input_list,
            logger=logger,
            version=args.version,
            timestamps=timestamps
        )

        # Normalisiere auf Liste
        if isinstance(out_paths, str):
            out_paths = [out_paths]

        print("[INFO] Evaluation abgeschlossen. Geschriebene Dateien:")
        for p in out_paths:
            print(f"  - {p}")
