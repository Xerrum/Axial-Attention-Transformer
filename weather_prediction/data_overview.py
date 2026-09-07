#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ERA5 data overview for ~/scratch/era5_data/past

Streaming, low-memory version:
- No [T,H,W,C] tensors are materialized.
- Statistics are computed in time-batches per variable.
- Percentiles are approximated via a fixed-size reservoir per channel.

Outputs:
~/scratch/era5_data/past/era5_overview_stats_updated.csv
"""

import os, glob, time
from pathlib import Path
import numpy as np
import pandas as pd
import netCDF4 as nc
from netCDF4 import num2date

# -----------------------------
# Configuration (matches your loader)
# -----------------------------
DATA_ROOT = os.path.expanduser("~/scratch/era5_data/past")
SINGLE_VARS = ["t2m", "d2m", "u10", "v10", "msl"]           # near-surface
PRESSURE_VARS = ["t", "r", "u", "v"]                        # on pressure levels
PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 300]           # hPa
SINGLE_PATTERN = "era5_single_*.netcdf"
PRESSURE_PATTERN = "era5_pressure_*.netcdf"
OUT_CSV = "era5_overview_stats_updated.csv"

# Streaming controls
TIME_BATCH = 8                # how many timesteps to load at once from a file
RESERVOIR_SIZE = 20000        # samples per channel to estimate p5/median/p95 (constant RAM)
YEARS = [2023]

# -----------------------------
# Helpers
# -----------------------------

# --- replace your _filter_by_year with this robust version ---
def _filter_by_year(filepath, years=None):
    """
    Heuristic filename filter. Works for names like:
    era5_single_2023_01.netcdf, era5_pressure_2023_01_01-10.netcdf
    Not sufficient alone; we *also* filter by decoded time.
    """
    if years is None or years == []:
        return True
    if isinstance(years, (int, np.integer)):
        years = [int(years)]
    years = [int(y) for y in years]

    fn = os.path.basename(filepath)
    # allow typical separators around the year token
    for y in years:
        ystr = str(y)
        if (f"_{ystr}_" in fn) or (f"_{ystr}-" in fn) or (f"-{ystr}_" in fn) or (f"-{ystr}-" in fn) or (f"_{ystr}." in fn) or (f"-{ystr}." in fn):
            return True
    # fallback: plain substring (last resort)
    return any(str(y) in fn for y in years)

# --- add this helper somewhere near your other helpers ---
def _filter_times_by_year_array(times: np.ndarray, years):
    """Return only timestamps that fall within the given year(s)."""
    if years is None or years == []:
        return times
    if isinstance(years, (int, np.integer)):
        years = [int(years)]
    years = [int(y) for y in years]

    sel = np.zeros(times.shape, dtype=bool)
    for y in years:
        start = np.datetime64(f"{y}-01-01T00")
        end   = np.datetime64(f"{y}-12-31T23")
        sel |= (times >= start) & (times <= end)
    return times[sel]

def _to_float_nan(a) -> np.ndarray:
    ma = np.ma.array(a, copy=False)
    return ma.filled(np.nan).astype(np.float32, copy=False)

def _read_times(ds: nc.Dataset) -> np.ndarray:
    """
    Return the time axis as numpy datetime64[h] using the proper time coordinate.
    Robust against calendars/units; falls back to index-based hours if needed.
    """
    for cand in ("time", "valid_time", "forecast_time"):
        if cand in ds.variables:
            tv = ds.variables[cand]
            vals = tv[:]
            units = getattr(tv, "units", None)
            cal = getattr(tv, "calendar", "standard")
            try:
                dts = num2date(vals, units, calendar=cal, only_use_cftime_datetimes=False)
                return np.array(dts, dtype="datetime64[h]")
            except Exception:
                # Fallback: treat values as monotonically increasing hours
                # (safe for ERA5 monthly/decadal blocks if units metadata is broken)
                return (np.arange(vals.shape[0]).astype("timedelta64[h]") + np.datetime64("1900-01-01")).astype("datetime64[h]")
    raise KeyError("No time coordinate found (tried: 'time', 'valid_time', 'forecast_time').")

def _ensure_same_grid(lat_s, lon_s, lat_p, lon_p):
    if lat_s.shape != lat_p.shape or lon_s.shape != lon_p.shape:
        raise ValueError("Lat/Lon grids differ between single-level and pressure-level files.")
    if not (np.allclose(lat_s, lat_p) and np.allclose(lon_s, lon_p)):
        raise ValueError("Lat/Lon coordinates mismatch between datasets.")

def _scan_times_and_grid(single_files, pressure_files):
    """
    Read the unioned time axes of single-level and pressure-level files,
    check that both use the same lat/lon grid, and return the common
    intersection of timestamps plus the reference grid.

    Returns
    -------
    common_times : np.ndarray of dtype 'datetime64[h]'
        Sorted intersection of all timestamps present in both datasets.
    lat, lon : np.ndarray
        Reference latitude and longitude coordinates (shared grid).
    """
    # --- reference from the first files ---
    with nc.Dataset(single_files[0]) as ds:
        ds.set_auto_maskandscale(True)
        ts0 = _read_times(ds)  # decode proper 'time' coordinate
        lat_s = np.asarray(ds.variables["latitude"][:])
        lon_s = np.asarray(ds.variables["longitude"][:])

    with nc.Dataset(pressure_files[0]) as ds:
        ds.set_auto_maskandscale(True)
        tp0 = _read_times(ds)  # decode proper 'time' coordinate
        lat_p = np.asarray(ds.variables["latitude"][:])
        lon_p = np.asarray(ds.variables["longitude"][:])

    # --- same grid? ---
    _ensure_same_grid(lat_s, lon_s, lat_p, lon_p)

    # --- gather all times (cheap: read only time coords) ---
    single_times = set(ts0.tolist())
    for fp in single_files[1:]:
        with nc.Dataset(fp) as ds:
            ds.set_auto_maskandscale(True)
            single_times.update(_read_times(ds).tolist())

    pressure_times = set(tp0.tolist())
    for fp in pressure_files[1:]:
        with nc.Dataset(fp) as ds:
            ds.set_auto_maskandscale(True)
            pressure_times.update(_read_times(ds).tolist())

    # --- intersection as sorted numpy datetime64[h] array ---
    common = np.array(sorted(single_times.intersection(pressure_times)),
                      dtype="datetime64[h]")
    
    common = _filter_times_by_year_array(common, YEARS)
    if common.size == 0:
        raise RuntimeError(f"No timestamps found in requested year(s): {YEARS}")
    return common, lat_s, lon_s


def _scan_present_vars(single_files, pressure_files):
    """Union of variables actually present across files (to avoid phantom channels)."""
    single_present = set()
    for fp in single_files:
        with nc.Dataset(fp) as ds:
            for v in SINGLE_VARS:
                if v in ds.variables:
                    single_present.add(v)

    pressure_present = set()
    for fp in pressure_files:
        with nc.Dataset(fp) as ds:
            for v in PRESSURE_VARS:
                if v in ds.variables:
                    pressure_present.add(v)
    return sorted(list(single_present)), sorted(list(pressure_present))

def _grid_info(lat, lon):
    H, W = len(lat), len(lon)
    dlat = np.abs(np.diff(lat)).mean() if len(lat) > 1 else np.nan
    dlon = np.abs(np.diff(lon)).mean() if len(lon) > 1 else np.nan
    return dict(H=H, W=W, dlat=dlat, dlon=dlon)

class Reservoir:
    """Fixed-size per-channel reservoir for approximate quantiles."""
    def __init__(self, C, cap=20000, dtype=np.float32):
        self.cap = cap
        self.buf = np.full((C, cap), np.nan, dtype=dtype)
        self.fill = np.zeros(C, dtype=np.int64)
        # We track how many candidates we *considered* for fairness
        self.seen = np.zeros(C, dtype=np.int64)

    def update(self, c: int, data_valid: np.ndarray, subsample_cap: int = 4096):
        """Update reservoir for channel c from a 1D vector of valid (non-NaN) values."""
        n = data_valid.size
        if n == 0:
            return
        # downsample per-batch to bound Python loop cost
        k = min(n, subsample_cap)
        take = data_valid if n <= k else np.random.choice(data_valid, size=k, replace=False)
        for x in take:
            self.seen[c] += 1
            if self.fill[c] < self.cap:
                self.buf[c, self.fill[c]] = x
                self.fill[c] += 1
            else:
                j = np.random.randint(0, self.seen[c])
                if j < self.cap:
                    self.buf[c, j] = x

    def percentiles(self, c: int, qs=(5, 50, 95)):
        used = self.buf[c, :self.fill[c]]
        if used.size == 0 or np.all(np.isnan(used)):
            return tuple(np.nan for _ in qs)
        return tuple(np.percentile(used[~np.isnan(used)], qs))

def _units_map(single_files, pressure_files):
    units = {}
    with nc.Dataset(single_files[0]) as ds:
        for v in SINGLE_VARS:
            if v in ds.variables and hasattr(ds.variables[v], "units"):
                units[v] = getattr(ds.variables[v], "units")
    with nc.Dataset(pressure_files[0]) as ds:
        for v in PRESSURE_VARS:
            if v in ds.variables and hasattr(ds.variables[v], "units"):
                units[v] = getattr(ds.variables[v], "units")
    return units

def _compute_stats_streaming(single_files, pressure_files, common_times, lat, lon,
                             single_present, pressure_present, time_batch=8, reservoir_size=20000):
    """
    Streaming stats:
      - Iterate files and time batches.
      - Update per-channel accumulators.
      - Maintain per-channel reservoir for percentiles.
    """
    H, W = len(lat), len(lon)
    T_common = len(common_times)
    if T_common == 0:
        raise RuntimeError("No overlapping timestamps between single and pressure files.")

    print(f"[INFO] Processing {len(single_files)} single-level and {len(pressure_files)} pressure-level files")
    print(f"[INFO] Time batching: {time_batch} steps, reservoir size: {reservoir_size}")

    # Build channel list (only present vars)
    single_chans = list(single_present)
    pres_chans = []
    for lev in PRESSURE_LEVELS:
        for v in pressure_present:
            pres_chans.append(f"{v}_{lev}hPa")
    names = single_chans + pres_chans
    C_s, C_p, C_total = len(single_chans), len(pres_chans), len(names)

    # Accumulators (float64 for numerical safety)
    mins = np.full(C_total, np.inf, dtype=np.float64)
    maxs = np.full(C_total, -np.inf, dtype=np.float64)
    sums = np.zeros(C_total, dtype=np.float64)
    sumsq = np.zeros(C_total, dtype=np.float64)
    nan_counts = np.zeros(C_total, dtype=np.int64)

    # Reservoirs for percentiles
    res = Reservoir(C_total, cap=reservoir_size, dtype=np.float32)

    # ---------- Single-level pass ----------
    print(f"[INFO] Starting single-level file processing...")
    for file_ix, fp in enumerate(single_files):
        print(f"[INFO] Processing single-level file {file_ix+1}/{len(single_files)}: {os.path.basename(fp)}")
        with nc.Dataset(fp) as ds:
            ds.set_auto_maskandscale(True)
            t = _read_times(ds)
            if t.size == 0:
                continue
            # local indices matching intersection
            mask = np.isin(t, common_times)
            t_idx = np.nonzero(mask)[0]
            if t_idx.size == 0:
                continue

            # process time in small batches
            for i in range(0, t_idx.size, time_batch):
                if i % (time_batch * 10) == 0:
                    print(f"  - Time step {i}/{t_idx.size} for {os.path.basename(fp)}")
                sel = t_idx[i:i+time_batch]
                for c, v in enumerate(single_chans):
                    if v not in ds.variables:
                        continue
                    data = _to_float_nan(ds.variables[v][sel, :, :])  # [tb,H,W]
                    vec = data.reshape(-1)
                    valid = vec[~np.isnan(vec)]
                    # stats
                    if valid.size:
                        mn, mx = float(np.min(valid)), float(np.max(valid))
                        mins[c] = mn if mn < mins[c] else mins[c]
                        maxs[c] = mx if mx > maxs[c] else maxs[c]
                        sums[c] += np.sum(valid, dtype=np.float64)
                        sumsq[c] += np.sum(valid.astype(np.float64)**2, dtype=np.float64)
                    nan_counts[c] += (vec.size - valid.size)
                    # reservoir
                    res.update(c, valid)

    # ---------- Pressure-level pass ----------
    print(f"[INFO] Starting pressure-level file processing...")
    for file_ix, fp in enumerate(pressure_files):
        print(f"[INFO] Processing pressure-level file {file_ix+1}/{len(pressure_files)}: {os.path.basename(fp)}")
        with nc.Dataset(fp) as ds:
            ds.set_auto_maskandscale(True)
            t = _read_times(ds)
            if t.size == 0:
                continue
            mask = np.isin(t, common_times)
            t_idx = np.nonzero(mask)[0]
            if t_idx.size == 0:
                continue

            # find level variable and map to requested levels
            levels = None
            for k in ("pressure_level", "level", "isobaricInhPa"):
                if k in ds.variables:
                    levels = np.asarray(ds.variables[k][:]); break
            if levels is None:
                raise KeyError(f"No pressure level coordinate found in {fp}")
            levels_hpa = levels if levels.max() <= 1100 else levels / 100.0
            lev_to_ix = {lev: int(np.abs(levels_hpa - lev).argmin()) for lev in PRESSURE_LEVELS}

            for i in range(0, t_idx.size, time_batch):
                if i % (time_batch * 10) == 0:
                    print(f"  - Time step {i}/{t_idx.size} for {os.path.basename(fp)}")
                sel = t_idx[i:i+time_batch]
                # loop present vars and target levels
                for v in pressure_present:
                    if v not in ds.variables:
                        continue
                    for lev_pos, lev in enumerate(PRESSURE_LEVELS):
                        chan_idx = len(single_chans) + (lev_pos * len(pressure_present)) + pressure_present.index(v)
                        data = _to_float_nan(ds.variables[v][sel, lev_to_ix[lev], :, :])  # [tb,H,W]
                        vec = data.reshape(-1)
                        valid = vec[~np.isnan(vec)]
                        if valid.size:
                            mn, mx = float(np.min(valid)), float(np.max(valid))
                            mins[chan_idx] = mn if mn < mins[chan_idx] else mins[chan_idx]
                            maxs[chan_idx] = mx if mx > maxs[chan_idx] else maxs[chan_idx]
                            sums[chan_idx] += np.sum(valid, dtype=np.float64)
                            sumsq[chan_idx] += np.sum(valid.astype(np.float64)**2, dtype=np.float64)
                        nan_counts[chan_idx] += (vec.size - valid.size)
                        res.update(chan_idx, valid)

    print("[INFO] Computing final statistics...")
    # Finalize stats
    total_elements = T_common * H * W
    valids = total_elements - nan_counts
    means = np.zeros(C_total, dtype=np.float64)
    stds  = np.zeros(C_total, dtype=np.float64)
    p5 = np.full(C_total, np.nan, dtype=np.float64)
    med = np.full(C_total, np.nan, dtype=np.float64)
    p95 = np.full(C_total, np.nan, dtype=np.float64)

    for c in range(C_total):
        if valids[c] > 0:
            means[c] = sums[c] / valids[c]
            # population std (consistent with your previous code path)
            var = max(0.0, (sumsq[c] / valids[c]) - (means[c] ** 2))
            stds[c] = float(np.sqrt(var))
            q5, q50, q95 = res.percentiles(c, qs=(5, 50, 95))
            p5[c], med[c], p95[c] = q5, q50, q95

    stats = {
        "channel": names,
        "min": mins,
        "p5": p5,
        "median": med,
        "mean": means,
        "p95": p95,
        "max": maxs,
        "std": stds,
        "nan_count": nan_counts.astype(np.int64),
        "valid_frac": (valids / total_elements).astype(np.float64),
    }
    return pd.DataFrame(stats)


# -----------------------------
# Main
# -----------------------------
def main():
    # Normalize YEARS to a list (accept int, list, None)
    years = YEARS
    if isinstance(years, (int, np.integer)):
        years = [int(years)]
    elif years in (None, [], ()):
        years = None
    else:
        years = [int(y) for y in years]

    print(f"[INFO] Scanning data in: {DATA_ROOT}")

    # Find files
    all_single_files   = sorted(glob.glob(str(Path(DATA_ROOT) / SINGLE_PATTERN)))
    all_pressure_files = sorted(glob.glob(str(Path(DATA_ROOT) / PRESSURE_PATTERN)))

    # Coarse filename filter (strict year filter happens in _scan_times_and_grid via time axis)
    if years:
        single_files = [f for f in all_single_files if _filter_by_year(f, years)]
        pressure_files = [f for f in all_pressure_files if _filter_by_year(f, years)]
        print(f"[INFO] Filtered {len(single_files)}/{len(all_single_files)} single-level files (year={years})")
        print(f"[INFO] Filtered {len(pressure_files)}/{len(all_pressure_files)} pressure-level files (year={years})")
    else:
        single_files = all_single_files
        pressure_files = all_pressure_files

    if not single_files:
        raise SystemExit(f"No single-level files found matching {SINGLE_PATTERN}"
                         + (f" for year(s) {years}" if years else ""))
    if not pressure_files:
        raise SystemExit(f"No pressure-level files found matching {PRESSURE_PATTERN}"
                         + (f" for year(s) {years}" if years else ""))

    start = time.time()
    print("[INFO] Scanning time axes and grid information...")
    common_times, lat, lon = _scan_times_and_grid(single_files, pressure_files)  # <- enforces YEARS internally

    single_present, pressure_present = _scan_present_vars(single_files, pressure_files)
    gi = _grid_info(lat, lon)

    print(f"[INFO] Time span (after year filter): {np.datetime_as_string(common_times[0])} → "
          f"{np.datetime_as_string(common_times[-1])}  (T={len(common_times)} steps)")
    print(f"[INFO] Grid: H={gi['H']}, W={gi['W']}, Δlat≈{gi['dlat']:.3f}°, Δlon≈{gi['dlon']:.3f}°")
    print(f"[INFO] Single vars present: {single_present}")
    print(f"[INFO] Pressure vars present: {pressure_present}")
    print(f"[INFO] Starting statistics computation...")

    df = _compute_stats_streaming(
        single_files, pressure_files, common_times, lat, lon,
        single_present, pressure_present,
        time_batch=TIME_BATCH, reservoir_size=RESERVOIR_SIZE
    )

    # Attach units from CF metadata
    print("[INFO] Attaching units from metadata...")
    units_map = _units_map(single_files, pressure_files)
    df["unit"] = [
        (units_map.get(name.split("_")[0], "")) if ("_" in name and name.endswith("hPa"))
        else units_map.get(name, "")
        for name in df["channel"].tolist()
    ]
    df = df[["channel", "unit", "min", "p5", "median", "mean", "p95", "max", "std", "nan_count", "valid_frac"]]

    # Output filename that reflects the year selection
    if years:
        years_label = "-".join(str(y) for y in years)
        out_csv_name = f"era5_overview_stats_{years_label}.csv"
    else:
        out_csv_name = OUT_CSV

    out_path = str(Path(DATA_ROOT) / out_csv_name)
    with pd.option_context('display.max_rows', 200, 'display.width', 140,
                           'display.float_format', '{:,.3f}'.format):
        print("\n=== ERA5 Channel Overview (streaming) ===")
        print(df)
    df.to_csv(out_path, index=False)
    elapsed = time.time() - start
    print(f"\n[OK] Wrote CSV: {out_path}  |  Elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
