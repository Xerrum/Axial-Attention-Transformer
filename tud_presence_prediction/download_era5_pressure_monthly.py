import cdsapi
import os
import sys
import calendar

if len(sys.argv) != 3:
    print("Usage: python download_era5_pressure_monthly.py <year> <month>")
    sys.exit(1)

year = sys.argv[1]
month = sys.argv[2]

output_dir = os.path.expanduser('~/scratch/era5_data')
os.makedirs(output_dir, exist_ok=True)

# 10-Tage-Blöcke (max. 16 Tage pro Request); bei 31 Tagen kommt der 31. separat
def day_blocks(y: int, m: int):
    ndays = calendar.monthrange(y, m)[1]
    blocks = [(1, 10), (11, 20), (21, min(30, ndays))]
    if ndays == 31:
        blocks.append((31, 31))
    return [(a, b) for a, b in blocks if a <= b]

c = cdsapi.Client()

# Feste Parameter (wie angefordert)
variables = [
    "relative_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]
levels = ["300", "500", "700", "850", "925", "1000"]
times = [f"{h:02d}:00" for h in range(24)]
area = [60, 0, 45, 15]  # N, W, S, E

y_int = int(year)
m_int = int(month)

for d1, d2 in day_blocks(y_int, m_int):
    days = [f"{d:02d}" for d in range(d1, d2 + 1)]
    output_file = os.path.join(output_dir, f"era5_pressure_{year}_{month}_{days[0]}-{days[-1]}.netcdf")

    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [year],
        "month": [month],
        "day": days,
        "time": times,
        "pressure_level": levels,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }

    print(f"Starte Download (Pressure-Levels) für {year}-{month} [{days[0]}–{days[-1]}] ...")
    c.retrieve("reanalysis-era5-pressure-levels", request, output_file)
    print(f"✅ Download abgeschlossen: {output_file}")
