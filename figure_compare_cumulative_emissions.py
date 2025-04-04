# Plot cumulative emissions against time. 
# Compare the coupled NorESM2-DIAM run emissions with the SSPs.

#-------------------------------------------------------------------------------------------

# IMPORT MODULES

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys
sys.path.insert(0, '../modules')
from module_coupling import calculate_annual_mean,  regrid_from_noresm_to_diam, sort_in_diam_order_3D, \
                            sort_in_diam_order, from_diam_order_to_grid

#-------------------------------------------------------------------------------------------

# SPECIFY

path = '../NorESM-DIAM/'

hist_co2_file = path + 'emissions-cmip6_CO2_anthro_surface_175001-201512_fv_1.9x2.5_c20181011.nc'
ssp_co2_files = [
    path + 'emissions-cmip6_CO2_anthro_surface_ScenarioMIP_IAMC-IMAGE-ssp126_201401-210112_fv_1.9x2.5_c20221115.nc',
    path + 'emissions-cmip6_CO2_anthro_surface_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245_201401-210112_fv_1.9x2.5_c20230419.nc',
    path + 'emissions-cmip6_CO2_anthro_surface_ScenarioMIP_IAMC-AIM-ssp370_201401-210112_fv_1.9x2.5_c20190207.nc',
    path + 'emissions-cmip6_CO2_anthro_surface_ScenarioMIP_IAMC-REMIND-MAGPIE-ssp585_201401-210112_fv_1.9x2.5_c20190207_djlo20200102.nc'
]  #
diam_co2_file1 = '../Smith_PreDoc_2024/decisions1/time_series.txt'
diam_co2_file2 = '../Smith_PreDoc_2024/decisions2/time_series.txt'

colors = ['lightsteelblue', 'cornflowerblue', 'royalblue', 'navy']

earth_radius = 6.3781e6

#-------------------------------------------------------------------------------------------

# READ IN DATA

# Read in DIAM cumulative emissions:
cumulative_co2_diam1 = np.loadtxt(diam_co2_file1, usecols=5)
cumulative_co2_diam2 = np.loadtxt(diam_co2_file2, usecols=5)
dyears = cumulative_co2_diam1.shape[0]

# Read in NorESM2 historical emissions:
ncfile = Dataset(hist_co2_file)
in_co2_hist = ncfile.variables['CO2_flux'][
    100 * 12:-12]  # last year repeated (not include 2015)
latitudes = ncfile.variables['lat'][:]
longitudes = ncfile.variables['lon'][:]
ncfile.close()

# Read in NorESM2 scenario emissions
in_co2_ssps = []
for file in ssp_co2_files:
    ncfile = Dataset(file)
    in_co2_ssps.append(
        ncfile.variables['CO2_flux'][12:-12])  # use 2014 from historical
    ncfile.close()

nyears = in_co2_hist.shape[0] // 12 + in_co2_ssps[0].shape[0] // 12
nlats = latitudes.shape[0]

diff_lon = longitudes[1] - longitudes[0]
diff_lat = latitudes[1] - latitudes[0]

cumulative_co2_gcb = np.array([
    4572663000, 4769414700, 4968083000, 5175497000, 5392617500, 5647651000,
    5907755000, 6185051600, 6464983600, 6749208600, 7050690000, 7381652500,
    7729438000, 8083927000, 8462107600, 8869834000, 9302693000, 9749246000,
    10227407000, 10718959000, 11240913000, 11774251000, 12341143000,
    12968444000, 13634925000, 14259158000, 14936272000, 15623051000,
    16323456000, 17029429000, 17785420000, 18639335000, 19522880000,
    20456241000, 21448995000, 22453130000, 23464765000, 24492184000,
    25570930000, 26765283000, 27959906000, 29260270000, 30622020000,
    31994415000, 33349892000, 34752913000, 36240343000, 37776710000,
    39385940000, 41083110000, 42936816000, 44889027000, 46905770000,
    48973455000, 51227554000, 53507300000, 55934620000, 58469350000,
    61357670000, 64134110000, 67024610000, 70058700000, 73138200000,
    76374160000, 79880440000, 83060790000, 86195650000, 89581500000,
    93121930000, 96615195000, 99644200000, 103165630000, 106255730000,
    109496040000, 113172275000, 116859494000, 120586530000, 124247560000,
    128243690000, 132222450000, 136493360000, 140431950000, 143951860000,
    147125340000, 150466850000, 154101100000, 157915890000, 162101410000,
    166575640000, 170782540000, 175233800000, 180095140000, 185065950000,
    190025530000, 195067230000, 200185140000, 204442270000, 209088400000,
    214234190000, 219659140000, 224840140000, 230769480000, 237150880000,
    243618320000, 250269190000, 257058680000, 264502540000, 272428530000,
    280614140000, 289031300000, 297885960000, 307272400000, 316686760000,
    326433300000, 336699360000, 347523000000, 358832670000, 370694400000,
    382930700000, 395835050000, 409595870000, 424494040000, 439996840000,
    456220280000, 473303780000, 490313150000, 507363130000, 525348370000,
    543843020000, 562907000000, 582510000000, 601992400000, 621009400000,
    639874760000, 658860700000, 678502400000, 698807500000, 719415000000,
    740660800000, 762738200000, 785117150000, 807869900000, 831099300000,
    853666560000, 876464900000, 899499700000, 923024160000, 947274300000,
    971670300000, 996001200000, 1020834700000, 1046336000000, 1072010900000,
    1098259230000, 1125907800000, 1154528000000, 1184120200000, 1214725900000,
    1246225600000, 1278267600000, 1309760800000, 1343067100000, 1377503700000,
    1412439100000, 1447671500000, 1483137700000, 1518601000000, 1554061000000,
    1590086500000, 1626853300000, 1663893500000, 1698901300000, 1735717800000,
    1772867600000
])

cumulative_co2_gcb = cumulative_co2_gcb - cumulative_co2_gcb[0]
cumulative_co2_gcb = cumulative_co2_gcb / 3.664e9
cyears = cumulative_co2_gcb.shape[0]

#-------------------------------------------------------------------------------------------

# CALCULATE ANNUAL CUMULATIVE EMISSIONS IN GtC

runs = [in_co2_hist] + in_co2_ssps
names = ['hist', 'ssp126', 'ssp245', 'ssp370', 'ssp585']

for i, run in enumerate(runs):

    co2_noresm = calculate_annual_mean(run)
    name = names[i]

    # Convert from kg m-2 s-1 to kg s-1:
    for ilat in range(nlats):

        cell_area = np.pi / 180 * earth_radius**2 * np.abs(
            np.sin(np.deg2rad(latitudes[ilat] - diff_lat / 2)) -
            np.sin(np.deg2rad(latitudes[ilat] + diff_lat / 2))) * diff_lon

        co2_noresm[:, ilat, :] *= cell_area  # kg m-2 s-1 to kg s-1

    # A = 2*pi*R^2 |sin(lat1)-sin(lat2)| |lon1-lon2|/360
    # = (pi/180)R^2 |sin(lat1)-sin(lat2)| |lon1-lon2|

    # Convert from CO2 kg s-1 to GtC:
    co2_noresm = co2_noresm * (365 * 24 * 60 * 60) / 3.664e12  # kg s-1 to GtC

    co2_noresm = np.sum(co2_noresm, axis=(1, 2))

    # Calculate cumulutive emissions:
    if i == 0:
        cumulative_co2_noresm = np.concatenate(
            (np.array([0]), np.cumsum(co2_noresm)))
    else:
        cumulative_co2_noresm = np.cumsum(
            co2_noresm) + cumulative_co2_noresm_hist[-1]

    exec('cumulative_co2_noresm_' + names[i] + '= cumulative_co2_noresm')

diff_emiss = cumulative_co2_diam1[0] - cumulative_co2_noresm_hist[140]
print(diff_emiss)

#-------------------------------------------------------------------------------------------

# PLOT

years_noresm = np.arange(1850, 1850 + nyears)
years_diam = np.arange(1990, 1990 + dyears)
years_gcb = np.arange(1850, 1850 + cyears)
nyears_hist = cumulative_co2_noresm_hist.shape[0]

fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(14, 10))

ax1.plot(years_noresm[130:nyears_hist],
         cumulative_co2_noresm_hist[130:] + diff_emiss,
         label='Historical',
         color='k',
         linewidth=3)
ax1.plot(years_noresm[nyears_hist:],
         cumulative_co2_noresm_ssp126[:-1] + diff_emiss,
         label='SSP1-2.6',
         color=colors[0],
         linewidth=3)
ax1.plot(years_noresm[nyears_hist:],
         cumulative_co2_noresm_ssp245[:-1] + diff_emiss,
         label='SSP2-4.5',
         color=colors[1],
         linewidth=3)
ax1.plot(years_noresm[nyears_hist:],
         cumulative_co2_noresm_ssp370[:-1] + diff_emiss,
         label='SSP3-7.0',
         color=colors[2],
         linewidth=3)
ax1.plot(years_noresm[nyears_hist:],
         cumulative_co2_noresm_ssp585[:-1] + diff_emiss,
         label='SSP5-8.5',
         color=colors[3],
         linewidth=3)
ax1.plot(years_diam[:110],
         cumulative_co2_diam1[:110],
         label='NorESM2-DIAM 1%',
         color='#f4d570',
         linewidth=3)
ax1.plot(years_diam[:110],
         cumulative_co2_diam2[:110],
         label='NorESM2-DIAM 2.5%',
         color='#d4a610',
         linewidth=3)
"""
ax1.plot(years_gcb[130:],
         cumulative_co2_gcb[130:],
         label='Global Carbon Budget (2023)',
         color='slategrey',
         linewidth=3)
"""
ax1.legend(fontsize=20)
ax1.set_xlabel('Year', fontsize=20)
ax1.set_ylabel('Cumulative emissions (GtC)', fontsize=20)
ax1.xaxis.set_tick_params(labelsize=16)
ax1.yaxis.set_tick_params(labelsize=16)

fig1.savefig('figures/figure_compare_cumulative_emissions.pdf')

# Where does DIAM cross SSP126:
a = np.abs(cumulative_co2_noresm_ssp126[:-1] - cumulative_co2_diam1[25:110])
b = np.where(a == a.min())[0]
print(years_noresm[nyears_hist + b])

# Where does DIAM cross SSP585:
a = np.abs(cumulative_co2_noresm_ssp585[:-1] - cumulative_co2_diam2[25:110])
b = np.where(a == a.min())[0]
print(years_noresm[nyears_hist + b])

print(cumulative_co2_noresm_hist[140:])
print(cumulative_co2_noresm_ssp585)

#-------------------------------------------------------------------------------------------
