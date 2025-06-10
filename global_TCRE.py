# -------------------------------------------------------------------------------------------

# IMPORT MODULES

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from netCDF4 import Dataset

from module_coupling import calculate_annual_mean

# -------------------------------------------------------------------------------------------

# SPECIFY

path = "/home/jennybj/Documents/koding/filer/scenarios/"

pi_file = path + "N1850_f19_tn14_20190730esm.nc"
histssp_file = path + "onlyCO2.nc"

hist_co2_file = "/home/jennybj/Documents/NorESM-DIAM/emissions-cmip6_CO2_anthro_surface_175001-201512_fv_1.9x2.5_c20181011.nc"
ssp_co2_file = "/home/jennybj/Documents/NorESM-DIAM/emissions-cmip6_CO2_anthro_surface_ScenarioMIP_IAMC-AIM-ssp370_201401-210112_fv_1.9x2.5_c20190207.nc"

temperature_files = [pi_file, histssp_file]
co2_files = [hist_co2_file, ssp_co2_file]

earth_radius = 6.3781e6

# -------------------------------------------------------------------------------------------

# READ IN DATA

cumulative_co2_e1 = np.loadtxt(
    "full_couple_baseline_cumulative_emissions.txt", usecols=1
)
cumulative_co2_e2 = np.loadtxt(
    "full_couple_baseline_e2_cumulative_emissions.txt", usecols=1
)

in_temperature = []
in_co2 = []

for i, temp_file in enumerate(temperature_files):
    ncfile = Dataset(temp_file)
    in_temperature.append(ncfile.variables["TREFHT"][:] - 273.15)

    if i == 0:
        latitudes = ncfile.variables["lat"][:]
        longitudes = ncfile.variables["lon"][:]

    ncfile.close()

ncfile = Dataset(hist_co2_file)
in_co2_hist = ncfile.variables["CO2_flux"][100 * 12 : -2 * 12]
ncfile.close()

ncfile = Dataset(ssp_co2_file)
in_co2_ssp = ncfile.variables["CO2_flux"][12:-12]
ncfile.close()

weights = np.cos(np.deg2rad(latitudes))

month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
in_temperature_pi = np.average(
    in_temperature[0][: 100 * 12, :, :], axis=0, weights=np.tile(month_lengths, 100)
)

in_co2 = np.concatenate((in_co2_hist, in_co2_ssp))

nyears = in_temperature[1].shape[0] // 12
nlats = latitudes.shape[0]
nlons = longitudes.shape[0]

diff_lon = longitudes[1] - longitudes[0]
diff_lat = latitudes[1] - latitudes[0]

pi_temp = np.average(np.average(in_temperature_pi, axis=1), axis=0, weights=weights)
print("PI temp: ", pi_temp)

# -------------------------------------------------------------------------------------------

# CALCULATE ANNUAL CUMULATIVE EMISSIONS IN GtC

co2 = calculate_annual_mean(in_co2)

# Convert from kg m-2 s-1 to kg s-1:
for ilat in range(nlats):
    cell_area = (
        np.pi
        / 180
        * earth_radius**2
        * np.abs(
            np.sin(np.deg2rad(latitudes[ilat] - diff_lat / 2))
            - np.sin(np.deg2rad(latitudes[ilat] + diff_lat / 2))
        )
        * diff_lon
    )

    co2[:, ilat, :] *= cell_area  # kg m-2 s-1 to kg s-1

# A = 2*pi*R^2 |sin(lat1)-sin(lat2)| |lon1-lon2|/360
# = (pi/180)R^2 |sin(lat1)-sin(lat2)| |lon1-lon2|

# Convert from CO2 kg s-1 to GtC:
co2 *= 365 * 24 * 60 * 60  # kg s-1 to kg
co2 /= 1e12  # kg to Gt
co2 /= 3.67  # GtCO2 to GtC

co2 = np.sum(co2, axis=(1, 2))

# Calculate cumulutive emissions:
cumulative_co2 = np.cumsum(co2)  # S t

# -------------------------------------------------------------------------------------------

# CONVERT ANNUAL TEMPERATURE FROM NorESM TO DIAM GRID
#
annual_temperature = calculate_annual_mean(in_temperature[1])

global_annual_temp = np.average(
    np.average(annual_temperature, axis=2), axis=1, weights=weights
)

# -------------------------------------------------------------------------------------------

# PLOT

cm = 1 / 2.54  # centimeters in inches

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6 * cm, 4.5 * cm))

ax.scatter(cumulative_co2, global_annual_temp - pi_temp, s=1, color="k")

ax.set_xlabel("Cumulative emissions (GtC)", fontsize=6)
ax.set_ylabel("Temperature change", fontsize=6)
ax.xaxis.set_tick_params(labelsize=5)
ax.yaxis.set_tick_params(labelsize=5)

fig.subplots_adjust(bottom=0.2, left=0.13, top=0.95, right=0.95)

fig.savefig("global_TCRE.pdf")

years = np.arange(1850, 1850 + nyears)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6 * cm, 4.5 * cm))

ax.scatter(
    years[50:],
    1000 * (global_annual_temp - pi_temp)[50:] / cumulative_co2[50:],
    s=0.5,
    color="k",
)

ax.hlines(0, 1900, 2100, linewidth=0.2)
ax.hlines(1, 1900, 2100, linewidth=0.2)
ax.hlines(2, 1900, 2100, linewidth=0.2)
ax.hlines(3, 1900, 2100, linewidth=0.2)
ax.hlines(4, 1900, 2100, linewidth=0.2)

ax.set_xlabel("Year", fontsize=6)
ax.set_ylabel("degC per GtC", fontsize=6)
ax.xaxis.set_tick_params(labelsize=5)
ax.yaxis.set_tick_params(labelsize=5)

fig.subplots_adjust(bottom=0.2, left=0.13, top=0.95, right=0.95)

fig.savefig("carbon-climate_response.pdf")

# -------------------------------------------------------------------------------------------
