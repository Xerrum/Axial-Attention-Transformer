import netCDF4 as nc
from netCDF4 import num2date

ds = nc.Dataset(r"C:\Users\organ\PycharmProjects\presence_prediction\weather_prediction\era5_pressure_2021_01_01-10.netcdf")
dz = nc.Dataset(r"C:\Users\organ\PycharmProjects\presence_prediction\weather_prediction\era5_single_2021_01.netcdf")

print(ds.variables.keys())
print(ds["pressure_level"])
pressure_levels = ds["pressure_level"][:]
print("Min:", pressure_levels.min())
print("Max:", pressure_levels.max())

print(dz.variables.keys())