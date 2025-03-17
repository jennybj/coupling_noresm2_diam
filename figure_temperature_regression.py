#-------------------------------------------------------------------------------------------

# IMPORT MODULES

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from netCDF4 import Dataset
import statsmodels.api as sm
from scipy import interpolate

from module_coupling import get_coordinate_data, sort_in_diam_order, sort_in_diam_order_3D, \
                            regrid_from_noresm_to_diam, calculate_annual_mean
#-------------------------------------------------------------------------------------------

# SPECIFY

path = '/home/jennybj/Documents/koding/filer/scenarios/'

pi_file = path + 'N1850_f19_tn14_20190730esm.nc'
histssp_file = path + 'onlyCO2.nc'
couple_e1_file = path + 'full_couple_baseline.nc'
couple_e2_file = path + 'full_couple_baseline_e2.nc'

hist_co2_file = '/home/jennybj/Documents/NorESM-DIAM/emissions-cmip6_CO2_anthro_surface_175001-201512_fv_1.9x2.5_c20181011.nc'
ssp_co2_file = '/home/jennybj/Documents/NorESM-DIAM/emissions-cmip6_CO2_anthro_surface_ScenarioMIP_IAMC-AIM-ssp370_201401-210112_fv_1.9x2.5_c20190207.nc'

temperature_files = [pi_file, histssp_file, couple_e1_file, couple_e2_file]
co2_files = [hist_co2_file, ssp_co2_file]

diam_latitudes = np.arange(-90., 90., 1)
diam_longitudes = np.arange(-180., 180., 1)
list_lats, list_longs = get_coordinate_data()

earth_radius = 6.3781e6
ncells = list_lats.shape[0]
print(ncells)

coordinates = list(zip(list_lats, list_longs))
weights = np.zeros(ncells)

for icell in range(ncells):

    if coordinates[icell] not in coordinates[:icell]:
        weights[icell] = np.cos(np.deg2rad(list_lats[icell]))


#-------------------------------------------------------------------------------------------

# READ IN DATA

cumulative_co2_e1 = np.loadtxt('full_couple_baseline_cumulative_emissions.txt', usecols=1)
cumulative_co2_e2 = np.loadtxt('full_couple_baseline_e2_cumulative_emissions.txt', usecols=1)

in_temperature = []
in_co2 = []

for i, temp_file in enumerate(temperature_files):

    ncfile = Dataset(temp_file)
    in_temperature.append(ncfile.variables['TREFHT'][:] - 273.15)

    if i == 0:
        latitudes = ncfile.variables['lat'][:]
        longitudes = ncfile.variables['lon'][:]

    ncfile.close()

ncfile = Dataset(hist_co2_file)
in_co2_hist = ncfile.variables['CO2_flux'][100 * 12:-2 * 12]
ncfile.close()

ncfile = Dataset(ssp_co2_file)
in_co2_ssp = ncfile.variables['CO2_flux'][12:-12]
ncfile.close()

month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
in_temperature_pi = np.average(in_temperature[0][:100 * 12, :, :],
                               axis=0,
                               weights=np.tile(month_lengths, 100))

in_co2 = np.concatenate((in_co2_hist, in_co2_ssp))

nyears = in_temperature[1].shape[0] // 12
nlats = latitudes.shape[0]
nlons = longitudes.shape[0]

diff_lon = longitudes[1] - longitudes[0]
diff_lat = latitudes[1] - latitudes[0]

pi_temp = np.average(np.average(in_temperature_pi, axis=1),
                     axis=0,
                     weights=np.cos(np.deg2rad(latitudes)))
print('PI temp: ', pi_temp)



#-------------------------------------------------------------------------------------------

# CALCULATE PRE-INDUSTRIAL TEMPERATURE AT DIAM GRID

in_temperature_pi = regrid_from_noresm_to_diam(in_temperature_pi)

temperature_pi = sort_in_diam_order(in_temperature_pi)  # T bar i

#-------------------------------------------------------------------------------------------

# CALCULATE ANNUAL CUMULATIVE EMISSIONS IN GtC

co2 = calculate_annual_mean(in_co2)

# Convert from kg m-2 s-1 to kg s-1:
for ilat in range(nlats):

    cell_area = np.pi / 180 * earth_radius**2 * np.abs(
        np.sin(np.deg2rad(latitudes[ilat] - diff_lat / 2)) -
        np.sin(np.deg2rad(latitudes[ilat] + diff_lat / 2))) * diff_lon

    co2[:, ilat, :] *= cell_area  # kg m-2 s-1 to kg s-1

# A = 2*pi*R^2 |sin(lat1)-sin(lat2)| |lon1-lon2|/360
# = (pi/180)R^2 |sin(lat1)-sin(lat2)| |lon1-lon2|

# Convert from CO2 kg s-1 to GtC:
co2 *= (365 * 24 * 60 * 60)  # kg s-1 to kg
co2 /= 1e12  # kg to Gt
co2 /= 3.67  # GtCO2 to GtC

co2 = np.sum(co2, axis=(1, 2))

# Calculate cumulutive emissions:
cumulative_co2 = np.cumsum(co2)  # S t

#-------------------------------------------------------------------------------------------

# CONVERT ANNUAL TEMPERATURE FROM NorESM TO DIAM GRID

temperatures = []

for i in range(1,len(in_temperature)):

    temperature = calculate_annual_mean(in_temperature[i])
    myears = temperature.shape[0]

    temp = np.zeros((myears, 180, 360))

    for iyear in range(myears):

        temp[iyear, :, :] = regrid_from_noresm_to_diam(temperature[iyear, :, :])
    temperatures.append(sort_in_diam_order_3D(diam_latitudes, diam_longitudes,
                                    temp))  # T it
    
print(temperatures[0].shape)

temp_pi_land = np.average(temperature_pi, weights=weights)
temp_land_all = np.average(temperatures[0], axis=0, weights=weights)
temp_land_e1 = np.average(temperatures[1], axis=0, weights=weights)
temp_land_e2 = np.average(temperatures[2], axis=0, weights=weights)
                                
#-------------------------------------------------------------------------------------------

# CALCULATE REGRESSION

x1 = np.zeros(nyears)
x1[1:] = cumulative_co2[:-1]

all_co2 = np.concatenate((cumulative_co2_e1[:-1], cumulative_co2_e2[:-1]))

rmse_emissions_residuals = np.zeros(ncells)
rsquared_emissions = np.zeros(ncells)
coeffs = np.zeros((ncells,2))

x2 = np.concatenate((all_co2, x1))
xs = np.vstack((x2,x2**2)).T

for icell in range(ncells):

    y = np.concatenate((temperatures[1][icell,:] -temperature_pi[icell], temperatures[2][icell,:] -temperature_pi[icell], temperatures[0][icell,:] - temperature_pi[icell]))
    
    model = sm.OLS(y, xs)
    regression = model.fit()
    coeffs[icell,:] = regression.params[:]

expected_temperature = np.zeros((ncells, nyears))

for icell in range(ncells):
    expected_temperature[icell, :] = coeffs[icell,0] * cumulative_co2 + coeffs[icell,1]*cumulative_co2**2  + coeffs[icell,0]

expected_temp_land = np.average(expected_temperature, axis=0, weights=weights)

#-------------------------------------------------------------------------------------------

# PLOT

cm = 1/2.54  # centimeters in inches

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6*cm, 4.5*cm))

ax.scatter(cumulative_co2, temp_land_all - temp_pi_land, s=1, color='k', label='HIST+SSP370')
ax.scatter(cumulative_co2_e1[:-1], temp_land_e1 - temp_pi_land, s=1, color='grey', label='1990-2100 run 1')
ax.scatter(cumulative_co2_e2[:-1], temp_land_e2 - temp_pi_land, s=1, color='silver', label='1990-2100 run 2')
ax.plot(cumulative_co2, expected_temp_land, color='r', label='Simple model', linewidth=1)

ax.set_xlabel('Cumulative emissions (GtC)', fontsize=6)
ax.set_ylabel('Temperature change', fontsize=6)
ax.xaxis.set_tick_params(labelsize=5)
ax.yaxis.set_tick_params(labelsize=5)

ax.legend(fontsize=5)
fig.subplots_adjust(bottom=0.2, left=0.13, top=0.95, right=0.95)

fig.savefig('figures/figure_temperature_regression.pdf')

#-------------------------------------------------------------------------------------------
