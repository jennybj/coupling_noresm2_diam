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
        noresm_latitudes = ncfile.variables['lat'][:]
        noresm_longitudes = ncfile.variables['lon'][:]

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
nlats = noresm_latitudes.shape[0]
nlons = noresm_longitudes.shape[0]

diff_lon = noresm_longitudes[1] - noresm_longitudes[0]
diff_lat = noresm_latitudes[1] - noresm_latitudes[0]

pi_temp = np.average(np.average(in_temperature_pi, axis=1),
                     axis=0,
                     weights=np.cos(np.deg2rad(noresm_latitudes)))
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
        np.sin(np.deg2rad(noresm_latitudes[ilat] - diff_lat / 2)) -
        np.sin(np.deg2rad(noresm_latitudes[ilat] + diff_lat / 2))) * diff_lon

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

# CONVERT  ANNUAL TEMPERATURE FROM NorESM TO DIAM GRID

temperatures = []

for i in range(1,len(in_temperature)):

    temperature = calculate_annual_mean(in_temperature[i])
    print(temperature.shape)
    myears = temperature.shape[0]

    temp = np.zeros((myears, 180, 360))

    for iyear in range(myears):

        temp[iyear, :, :] = regrid_from_noresm_to_diam(temperature[iyear, :, :])
    temperatures.append(sort_in_diam_order_3D(diam_latitudes, diam_longitudes,
                                    temp))  # T it

                                

print(temperatures[1].shape, temperatures[2].shape)

temp_pi_land = np.average(temperature_pi, weights=weights)
temp_all_land = np.average(temperatures[0], axis=0, weights=weights)
dtemp_all_land = temp_all_land - temp_pi_land

#-------------------------------------------------------------------------------------------

# WRITE TEMPERATURES TO FILES

file = open('NorESM2_HIST_SSP370_regional_temperatures_v2.txt', 'w')

for icell in range(ncells):

    file.writelines(['%16.1f' % list_lats[icell]])
    file.writelines(['%16.1f' % list_longs[icell]])
    file.writelines(['%16.7f' % item for item in temperatures[0][icell, :]])
    file.write('\n')

file.close()

file = open('NorESM2_picontrol_regional_temperatures_v2.txt', 'w')

for icell in range(ncells):

    file.writelines(['%16.1f' % list_lats[icell]])
    file.writelines(['%16.1f' % list_longs[icell]])
    file.writelines(['%16.7f' % temperature_pi[icell]])
    file.writelines(['%16.7f' % weights[icell]])
    file.write('\n')

file.close()

# WRITE CUMULATIVE EMISSIONS TO FILE

file = open('NorESM2_HIST_SSP370_cumulative_emissions_global_temperature_v2.txt',
            'w')

cum = np.zeros(nyears)
cum[1:] = cumulative_co2[:-1]

file.writelines('# Column 1: Year\n')
file.writelines('# Column 2: Cumulative emissions\n')
file.writelines('# Column 3: Area-weighted temperature of the grid cells included in the model\n')
file.writelines('# Column 4: Area-weighted temperature change from pre-industrial of the grid cells included in the model\n')


for iyear in range(nyears):

    year = 1850 + iyear

    file.writelines(['% 8d' % year])
    file.writelines(['%16.4f' % cum[iyear]])
    file.writelines(['%16.4f' % temp_all_land[iyear]])
    file.writelines(['%16.4f' % dtemp_all_land[iyear]])
    file.write('\n')

#-------------------------------------------------------------------------------------------

# CALCULATE REGRESSION

x1 = np.zeros(nyears)
x1[1:] = cumulative_co2[:-1]

all_co2 = np.concatenate((cumulative_co2_e1[:-1], cumulative_co2_e2[:-1]))

rmse_emissions_residuals = np.zeros(ncells)
rsquared_emissions = np.zeros(ncells)
coeffs = np.zeros((ncells,3))

x2 = np.concatenate((all_co2, x1))
xs = np.vstack((x2,x2**2)).T

for icell in range(ncells):

    y = np.concatenate((temperatures[1][icell,:] -temperature_pi[icell], temperatures[2][icell,:] -temperature_pi[icell], temperatures[0][icell,:] - temperature_pi[icell]))
    
    model = sm.OLS(y, xs)
    regression = model.fit()
    coeffs[icell,1:] = regression.params[:]

expected_temperature_all = np.zeros((ncells, nyears))
expected_temperature_e1 = np.zeros((ncells, cumulative_co2_e1[:-1].shape[0]))
expected_temperature_e2 = np.zeros((ncells, cumulative_co2_e2[:-1].shape[0]))

for icell in range(ncells):
    expected_temperature_all[icell, :] = coeffs[icell,1] * cumulative_co2 + coeffs[icell,2]*cumulative_co2**2  + coeffs[icell,0]
    expected_temperature_e1[icell, :] = coeffs[icell,1] * cumulative_co2_e1[:-1] + coeffs[icell,2]*cumulative_co2_e1[:-1]**2  + coeffs[icell,0]
    expected_temperature_e2[icell, :] = coeffs[icell,1] * cumulative_co2_e2[:-1] + coeffs[icell,2]*cumulative_co2_e2[:-1]**2  + coeffs[icell,0]

#-------------------------------------------------------------------------------------------


# CALCULATE AUTOREGRESSION OF TEMPERATURE DEVIATION

zi_all = np.zeros(temperatures[0].shape)
zi_e1 = np.zeros(temperatures[1].shape)
zi_e2 = np.zeros(temperatures[2].shape)

for icell in range(ncells):

    zi_all[icell,:] = temperatures[0][icell,:] - temperature_pi[icell] - expected_temperature_all[icell,:]
    zi_e1[icell,:] = temperatures[1][icell,:] - temperature_pi[icell] - expected_temperature_e1[icell,:]
    zi_e2[icell,:] = temperatures[2][icell,:] - temperature_pi[icell] - expected_temperature_e2[icell,:]
    
print(np.max(zi_all), np.min(zi_all), np.mean(zi_all))
print(np.max(zi_e1), np.min(zi_e1), np.mean(zi_e1))
print(np.max(zi_e2), np.min(zi_e2), np.mean(zi_e2))

rhos = np.zeros(ncells)

for icell in range(ncells):

    x = np.concatenate((zi_all[icell,:-1], zi_e1[icell,:-1], zi_e2[icell,:-1]))
    y = np.concatenate((zi_all[icell,1:], zi_e1[icell,1:], zi_e2[icell,1:]))

    model = sm.OLS(y, x)
    regression = model.fit()
    rhos[icell] = regression.params[:]

#-------------------------------------------------------------------------------------------

# WRITE COEFFICIENTS AND RMSE TO FILE

file = open('NorESM2_HIST_SSP370_coefficients_v2.txt', 'w')

file.writelines('# Column 1: Latitude\n')
file.writelines('# Column 2: Longitude\n')
file.writelines('# Column 3: Gamma linear coefficient\n')
file.writelines('# Column 4: Gamma quadratic coefficient\n')
file.writelines('# Column 5: rho\n')


for icell in range(ncells):

    file.writelines(['%16.1f' % list_lats[icell]])
    file.writelines(['%16.1f' % list_longs[icell]])
    file.writelines(['%16.12f' % coeffs[icell,1]])
    file.writelines(['%16.12f' % coeffs[icell,2]])
    file.writelines(['%16.12f' % rhos[icell]])
    file.write('\n')

#-------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------
