# -------------------------------------------------------------------------------------------

# IMPORT MODULES

import sys as sys

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, "..")  # CHANGE path to location on module
from module_coupling import (
    get_coordinate_data,
    regrid_from_noresm_to_diam,
    sort_in_diam_order,
)

# -------------------------------------------------------------------------------------------

# SPECIFY

noresm2_file = "../../data/input_to_regression/onlyCO2.nc"

diam_latitudes = np.arange(-90.0, 90.0, 1)
diam_longitudes = np.arange(-180.0, 180.0, 1)
list_lats, list_longs = get_coordinate_data()

ncells = list_lats.shape[0]


# -------------------------------------------------------------------------------------------

# READ IN DATA


ncfile = Dataset(noresm2_file)
in_temperature = (
    ncfile.variables["TREFHT"][12 * 51 : 12 * 81, :, :] - 273.15
)  # years 1901-1930
noresm_latitudes = ncfile.variables["lat"][:]
noresm_longitudes = ncfile.variables["lon"][:]
ncfile.close()


nyears = in_temperature.shape[0] // 12
nlats = noresm_latitudes.shape[0]
nlons = noresm_longitudes.shape[0]


# -------------------------------------------------------------------------------------------

# CALCULATE 1901-1930 TEMPERATURE REFERENCE

month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
average_in_temp = np.average(
    in_temperature, axis=0, weights=np.tile(month_lengths, nyears)
)

# -------------------------------------------------------------------------------------------

# CONVERT REFERENCE TEMPERATURE FROM NorESM TO DIAM GRID

temp = regrid_from_noresm_to_diam(average_in_temp)
average_out_temp = sort_in_diam_order(diam_latitudes, diam_longitudes, temp)


# -------------------------------------------------------------------------------------------

# WRITE TEMPERATURES TO FILES

file = open("../../data/NorESM2_1901_1930_reference_temperatures.txt", "w")

file.writelines("# Column 1: Latitude\n")
file.writelines("# Column 2: Longitude\n")
file.writelines("# Column 3: 1901-1930 average reference temperature\n")

for icell in range(ncells):
    file.writelines(["%16.1f" % list_lats[icell]])
    file.writelines(["%16.1f" % list_longs[icell]])
    file.writelines(["%16.7f" % average_out_temp[icell]])
    file.write("\n")

file.close()

file = open("../../data/NorESM2_1901_1930_reference_temperatures_all.txt", "w")

file.writelines("# Column 1: Latitude\n")
file.writelines("# Column 2: Longitude\n")
file.writelines("# Column 3: 1901-1930 average reference temperature\n")

for ilat, lat in enumerate(diam_latitudes):
    for ilon, lon in enumerate(diam_longitudes):
        file.writelines(["%16.1f" % lat])
        file.writelines(["%16.1f" % lon])
        file.writelines(["%16.7f" % temp[ilat, ilon]])
        file.write("\n")

file.close()


# -------------------------------------------------------------------------------------------

# COMPARE WITH PI TEMPERATURES

pi_temp = np.loadtxt(
    "/home/jennybj/Documents/coupling_noresm2_diam/data/input/NorESM2_picontrol_regional_temperatures.txt",
    usecols=3,
)

temp_diff = average_out_temp - pi_temp

idx = np.argsort(temp_diff)
temp_diff = temp_diff[idx]

print(temp_diff)
print(temp_diff[:10])
print(temp_diff[-10:])
print(np.mean(temp_diff))

# -------------------------------------------------------------------------------------------
