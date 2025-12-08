# ----------------------------------------------------------------------------------------

# MODULES

import os as os
import sys as sys

import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "..")  # CHANGE path to location on module
from module_coupling import *

# ----------------------------------------------------------------------------------------

# DEFINE

case_name = "full_couple_population"  # CHANGE to wanted NorESM2 case name

# File names:
outfile = "../../data/input/input_emissions_" + case_name + ".nc"

# Constants:
earth_radius = 6.3781e6
nyears = 5
nlats = 96
nlons = 144

noresm_latitudes = np.linspace(-90, 90, nlats)
noresm_longitudes = np.linspace(0, 357.5, nlons)

# -----------------------------------------------------------------------------------------------

# INPUT DATA AND PARAMETERS

ga, beta, delta, alpha, energyshare, rss, theta, b = get_constants()

# Array of all corrdinates from DIAM:
diam_latitudes, diam_longitudes = get_coordinate_data()
pop = get_initial_population()
ai = get_initial_ai()
price = get_price()

chi = 1  # percentage of energy use that is dirty in 1990
kss = (
    ((rss + delta) / (alpha * theta))
    * (b ** (-1 / theta))
    * ((price / (1 - theta)) ** ((1 - theta) / theta))
) ** (1 / (alpha - 1))
xss = ((1 - theta) * b / price) ** (
    1 / theta
) * kss**alpha  # 0.08042639   # total energy use in 1990?

print("price = ", price)
print("kss = ", kss)
print("xss = ", xss)
print("ai = ", ai)

# ----------------------------------------------------------------------------------------

# CALCULATE REGIONAL EMISSIONS IN 1990

regional_emissions = pop * ai * xss * chi / 1000  # GtC

global_emissions = np.sum(regional_emissions, axis=0)
print("Global emissions in GtC:", global_emissions)

# -------------------------------------------------------------------------

# PUT EMISSIONS INTO GRID

# Put the DIAM emission data into an array (time, lat, lon):
gridded_latitudes = np.arange(-90, 91)
gridded_longitudes = np.arange(-180, 180)
gridded_co2 = np.zeros((len(gridded_latitudes), len(gridded_longitudes)))

for i in range(ncells):
    # Find index of coordinates in the new array:
    index_lat = np.where(diam_latitudes[i] == gridded_latitudes)
    index_lon = np.where(diam_longitudes[i] == gridded_longitudes)

    # Place emission data in correct place in array:
    gridded_co2[index_lat[0][0], index_lon[0][0]] += regional_emissions[i]

# -------------------------------------------------------------------------

# INTERPOLATE EMISSIONS FROM DIAM TO NorESM GRID

# Specify arrays:
interp_lat = gridded_latitudes
interp_lon = np.arange(0.0, 360.0, 1)  # DIAM has lon -180 to 179
noresm_co2 = np.zeros((nlats, nlons))

# Shift longitude coordinates to match NorESM:
interp_co2 = np.zeros(gridded_co2.shape)
interp_co2[:, 0:180] = gridded_co2[:, 180:]
interp_co2[:, 180:] = gridded_co2[:, 0:180]

# Smooth to improve interpolation:
smoothing = gaussian_filter(interp_co2, sigma=1)

# Interpolate:
f = interp2d(interp_lon, interp_lat, smoothing, kind="linear")
interpolated = f(noresm_longitudes, noresm_latitudes)

# Remove emissions from +/- 90 degrees latitude:
interpolated[0, :] = 0
interpolated[-1, :] = 0

# Make sure the sum of the emissions stays the same:
ratio = global_emissions / np.sum(interpolated)
noresm_co2 = interpolated * ratio

print("Check global emissions: ", np.sum(noresm_co2))

# -------------------------------------------------------------------------

# CHANGE CO2 DATA TO CORRECT FORMAT

# Convert from ktC to CO2 kg s-1:
noresm_co2 *= 3.67  # ktC to ktCO2
noresm_co2 *= 1e12  # Gt to kg
noresm_co2 /= 365 * 24 * 60 * 60  # kg to kg s-1

dlats = noresm_latitudes[1] - noresm_latitudes[0]

# Convert from kg s-1 to kg m-2 s-1:
for i in range(1, noresm_latitudes.shape[0] - 1):
    lat = noresm_latitudes[i]

    cell_area = (
        np.pi
        / 180
        * earth_radius**2
        * np.abs(
            np.sin(np.deg2rad(lat - 0.5 * dlats))
            - np.sin(np.deg2rad(lat + 0.5 * dlats))
        )
        * 2.5
    )

    noresm_co2[i, :] /= cell_area  # kg s-1 to kg m-2 s-1

# A = 2*pi*R^2 |sin(lat1)-sin(lat2)| |lon1-lon2|/360
# = (pi/180)R^2 |sin(lat1)-sin(lat2)| |lon1-lon2|

# Repeat value, as file must have monthly values and at least 1 extra year
noresm_co2 = np.tile(noresm_co2, (nyears * 12, 1, 1))

print(np.sum(noresm_co2[0, :, :]))

# Create ensemble member with small perturbation:
noresm_co2[0, 30, 0] += 1e-12

print(np.sum(noresm_co2[0, :, :]))

# -------------------------------------------------------------------------

# Extra check:

temp = noresm_co2[0, :, :] * 60 * 60 * 24 * 365 / 3.67e12  # kg CO2 m-2 s-1 to GtC m-2

for i in range(1, noresm_latitudes.shape[0] - 1):
    lat = noresm_latitudes[i]

    cell_area = (
        np.pi
        / 180
        * earth_radius**2
        * np.abs(
            np.sin(np.deg2rad(lat - 0.5 * dlats))
            - np.sin(np.deg2rad(lat + 0.5 * dlats))
        )
        * 2.5
    )

    temp[i, :] *= cell_area  # GtC m-2 to GtC

print(np.sum(temp))


# -------------------------------------------------------------------------

# MAKE RELEVANT TIME VARIABLES

# Create time and date values (middle of every month):
time_val = []
date_val = []
bound_val = []
start_day = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
start_year = np.array(
    [
        19900116,
        19900215,
        19900316,
        19900416,
        19900516,
        19900616,
        19900716,
        19900816,
        19900916,
        19901016,
        19901116,
        19901216,
    ]
)
start_bound = np.array(
    [
        [0, 31],
        [31, 59],
        [59, 90],
        [90, 120],
        [120, 151],
        [151, 181],
        [181, 212],
        [212, 243],
        [243, 273],
        [273, 304],
        [304, 334],
        [334, 365],
    ]
)

print("The file is created for ", nyears, "years from ", start_year[0])

for year in range(nyears):
    time_val += (start_day + year * 365).tolist()
    date_val += (start_year + year * 10000).tolist()
    bound_val += (start_bound + year * 365).tolist()
# -------------------------------------------------------------------------

# MAKE OUTPUT FILE

# Delete if file exists:
if os.path.exists(outfile):
    os.remove(outfile)

# Create new files open for writing:
file = Dataset(outfile, "w")

# Create the dimensions:
file.createDimension("time", None)
file.createDimension("bound", 2)
file.createDimension("lat", nlats)
file.createDimension("lon", nlons)

# Create time variable:
time = file.createVariable(varname="time", datatype="d", dimensions=("time"))
time[:] = time_val  # fill values
time.units = "days since 1990-01-01 00:00:00"  # set attributes
time.long_name = "time"
time.calendar = "noleap"
time.axis = "T"
time.bounds = "time_bnds"
time.realtopology = "linear"
time.standard_name = "time"

# Create date variable:
date = file.createVariable(varname="date", datatype="i", dimensions=("time",))
date[:] = date_val
date.long_name = "date"
date.units = "YYYYMMDD"

# Create time bounds variable:
time_bnds = file.createVariable(
    varname="time_bnds", datatype="d", dimensions=("time", "bound")
)
time_bnds[:] = bound_val

# Create the latitude variable:
lat = file.createVariable(varname="lat", datatype="d", dimensions=("lat",))
lat[:] = noresm_latitudes
lat.long_name = "latitude"
lat.units = "degrees_north"

# Create the longitude variable:
lon = file.createVariable(
    varname="lon", datatype="d", dimensions=("lon",)
)  # create dimension variable
lon[:] = noresm_longitudes
lon.long_name = "longitude"
lon.units = "degrees_east"

# Create co2 flux variable:
noresm_co2 = noresm_co2.tolist()
CO2_flux = file.createVariable(
    varname="CO2_flux", datatype="f", dimensions=("time", "lat", "lon"), fill_value=1e20
)
CO2_flux[:] = noresm_co2
CO2_flux.missing_vale = np.array(1e20, dtype=np.float32)
CO2_flux.cell_method = "time: mean"
CO2_flux.long_name = "CO2 Anthropogenic Emissions"
CO2_flux.units = "kg m-2 s-1"

file.close()

# ----------------------------------------------------------------------------------------
