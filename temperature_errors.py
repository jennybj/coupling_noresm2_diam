import numpy as np
import matplotlib.pyplot as plt

from module_coupling import *

#--------------------------------------------------------------------------------------------------

# READ IN DATA

path = '/home/jennybj/uio/home/coupling/'

regtemp = np.loadtxt(path + 'full_couple_baseline_v2_regtemp.txt', skiprows=2)
emissions = np.loadtxt(path + 'emissions.txt', usecols=1)

# Calculate cumulative emissions:
cumulative_emissions_1990 = np.array([216.865])
cumulative_emissions = np.cumsum(np.concatenate((cumulative_emissions_1990, emissions)))

nyears = regtemp.shape[0]
ncells = regtemp.shape[1]

gamma1, gamma2, rho = get_coefficients()
list_latitudes, list_longitudes = get_coordinate_data()
pi_temp = get_pi_temperature()

expected_temp = np.zeros((nyears,ncells))

for iyear in range(nyears):
    expected_temp[iyear,:] = pi_temp + gamma1*cumulative_emissions[iyear] + gamma2*cumulative_emissions[iyear]**2


#--------------------------------------------------------------------------------------------------

# CALCULATE ERRORS

temp_errors = np.abs(regtemp - expected_temp)
grid_errors = np.average(temp_errors, axis=0)#/(np.max(regtemp, axis=0) - np.min(regtemp, axis=0))

indices_sorted = grid_errors.argsort()
indices_max = indices_sorted[-10:][::-1]
indices_min = indices_sorted[:5]
indices_both = np.concatenate((indices_max, indices_min))



#--------------------------------------------------------------------------------------------------

# PLOT SOME REGIONS

years = np.arange(1990, 1990 + nyears)

for i in indices_both:

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

    ax.plot(years,
            regtemp[:, i],
            label='NorESM2-DIAM')
    ax.plot(years,
            expected_temp[:, i],
            label='Expectation')

    ax.legend()

    fig.savefig('figures/temperature_region_' + str(list_latitudes[i]) + '_' +
                str(list_longitudes[i]) + '.pdf')

plt.close()

#--------------------------------------------------------------------------------------------------