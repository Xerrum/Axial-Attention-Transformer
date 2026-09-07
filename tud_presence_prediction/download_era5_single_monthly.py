import cdsapi
import os
import sys

if len(sys.argv) != 3:
    print("Usage: python download_era5_monthly.py <year> <month>")
    sys.exit(1)

year = sys.argv[1]
month = sys.argv[2]

output_dir = os.path.expanduser('~/scratch/era5_data')
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f'era5_single_{year}_{month}.netcdf')

c = cdsapi.Client()

request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "2m_temperature",
        "mean_sea_level_pressure"
    ],
    "year": [year],
    "month": [month],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [60, 0, 45, 15]
}

print(f"Starte Download für {year}-{month} ...")
c.retrieve('reanalysis-era5-single-levels', request, output_file)
print(f"✅ Download abgeschlossen: {output_file}")
