#--------------------------------------------------------------------------------------

#import sys as sys
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
from collections import defaultdict
import seaborn.apionly as sns
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from scipy.interpolate import interp1d, RegularGridInterpolator
import statsmodels.api as sm

#sys.path.insert(0, '../modules')
from module_coupling import  *

#--------------------------------------------------------------------------------------

alpha = 0.36  # capital’s share of income (capital share + labor share = 1)
delta = 0.06  # The (annual) rate of depreciation of the capital stock
grate = 0.01  # rate of (assumed?) energy increase for each year
price = get_price()

file_path = '/home/jennybj/uio/home/'

#--------------------------------------------------------------------------------------

ncells = 20249

global_pi_temperature = 14.460473280816053

pi_temperatures = get_pi_temperature()
population = get_initial_population()
chi = get_chit()

diam_latitudes, diam_longitudes = get_coordinate_data()

#--------------------------------------------------------------------------------------

# READ IN DATA

years, ss_emissions = np.loadtxt(file_path + 'coupling/emissions.txt', unpack=True)

output_files = sorted(
    glob.glob(file_path + 'coupling/output_year_*.txt'))
nyears = len(output_files)
fp_output_files = sorted(glob.glob(file_path +
                                   'coupling/fp_output_year_*.txt'))

# Make empty arrays:
expected_temperature = np.zeros((nyears, ncells))
temperature = np.zeros((nyears, ncells))
wealth_scaled = np.zeros((nyears, ncells))
capital_scaled = np.zeros((nyears, ncells))
ai = np.zeros((nyears, ncells))
energy_scaled = np.zeros((nyears, ncells))
expected_emissions = np.zeros((nyears, ncells))
actual_emissions = np.zeros((nyears, ncells))
fp_wealth_scaled = np.zeros((nyears, ncells))
fp_capital_scaled = np.zeros((nyears, ncells))
fp_ai = np.zeros((nyears, ncells))
fp_energy_scaled = np.zeros((nyears, ncells))
fp_emissions = np.zeros((nyears, ncells))

# Read in output:


for i, file in enumerate(output_files[:nyears]):

    print(file)

    expected_temperature[i, :], temperature[
        i, :], wealth_scaled[i, :], capital_scaled[i, :], ai[
            i, :], energy_scaled[i, :], expected_emissions[
                i, :], actual_emissions[i, :] = np.loadtxt(
                    file,
                    skiprows=15,
                    usecols=(2, 3, 5, 6, 8, 10, 12, 13),
                    unpack=True)

for i, file in enumerate(fp_output_files[:nyears]):

    print(file)

    fp_wealth_scaled[i, :], fp_capital_scaled[i, :], fp_ai[
        i, :], fp_energy_scaled[i, :], fp_emissions[i, :] = np.loadtxt(
            file, skiprows=15, usecols=(5, 6, 8, 10, 12), unpack=True)

global_temperature = np.loadtxt(
    file_path + '/coupling/full_couple_baseline_v2_global_temp.txt', usecols=1)

#--------------------------------------------------------------------------------------

# FUNCTIONS


def damages(regtemp, tstar=12.609, scale1=0.00327721, scale2=0.00362887):
    """ The regional damage function. Already raised to the power of 1/(1 - alpha)"""

    # Define constants:
    pbound = 0.02
    toler = 1.0e-4

    diff = regtemp - tstar

    myears = regtemp.shape[0]
    mcells = regtemp.shape[1]

    if mcells != ncells:
        print('Number of cells is ', mcells, ' not ', ncells)

    fval = np.zeros((myears, mcells))

    #((1 - d) * exp(-κ_minus * (t - T) ^ 2) + d) ^ (1 / (1 - α))

    for iyear in range(myears):

        for icell in range(mcells):

            if diff[iyear, icell] < 0:
                fval[iyear, icell] = (
                    np.exp(-scale1 * diff[iyear, icell] * diff[iyear, icell]) *
                    (1 - pbound) + pbound)**(1 / (1 - alpha))
            else:
                fval[iyear, icell] = (
                    np.exp(-scale2 * diff[iyear, icell] * diff[iyear, icell]) *
                    (1 - pbound) + pbound)**(1 / (1 - alpha))

            if fval[iyear, icell] < toler:
                fval[iyear, icell] = toler

    return fval


def descale(in_variable, in_ai):

    out_variable = in_variable * (population * in_ai)

    return out_variable



#--------------------------------------------------------------------------------------

# LOOK AT DIFFERENCE BETWEEN ACTUAL AND EXECTED TEMPERATURE

temp_diffs = np.mean(temperature - expected_temperature, axis=1)
print(temp_diffs.shape)

bins = np.histogram(temp_diffs, bins=100)[1]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))

ax.hist(temp_diffs, bins=bins)

ax.legend()

fig.savefig('figures/mean_diff_actual_and_expected_temperature.pdf')
plt.close()

#--------------------------------------------------------------------------------------

# CALCULATIONS


# Calculate wealth:
wealth = descale(wealth_scaled, ai)
fp_wealth = descale(fp_wealth_scaled, ai)
sum_wealth = np.sum(wealth, axis=1)
sum_wealth_scaled = np.sum(wealth_scaled, axis=1)

# Calculate emissions:
cumulative_emissions_1990 = np.array([216.865])
sum_actual_emissions = np.sum(actual_emissions, axis=1)
actual_cumulative_emissions = np.cumsum(np.concatenate((cumulative_emissions_1990, sum_actual_emissions / 1e3)))
ss_cumulative_emissions = np.cumsum(
    np.concatenate((cumulative_emissions_1990, ss_emissions)))

diff_cumulative_emissions = (actual_cumulative_emissions -
                             ss_cumulative_emissions[:nyears + 1])  # * 1e3
diff_emissions = sum_actual_emissions / 1e3 - ss_emissions[:nyears]  #* 1e3

# Calculate aggregate energy use:
energy_use = descale(energy_scaled, ai)
aggregate_energy_use = np.sum(energy_use, axis=1)
fp_energy_use = descale(fp_energy_scaled, fp_ai)
fp_aggregate_energy_use = np.sum(fp_energy_use, axis=1)

fp_sum_emissions = np.sum(fp_emissions, axis=1)

ss_energy_use = ss_emissions[:nyears] * 1e3 / chi[:nyears]

print('Total diff', np.sum(sum_actual_emissions - fp_sum_emissions))

# Read and calculate GDP:
capital = descale(capital_scaled, ai)
gdp = wealth - (1 - delta) * capital
sum_gdp = np.sum(gdp, axis=1)
gdp_scaled = wealth_scaled - (1 - delta) * capital_scaled
sum_gdp_scaled = np.sum(gdp_scaled, axis=1)

fp_gdp = descale(fp_wealth_scaled,
                 fp_ai) - (1 - delta) * descale(fp_capital_scaled, fp_ai)
fp_sum_gdp = np.sum(fp_gdp, axis=1)


# Detrend the GDP:
gdp_detrended = np.zeros((nyears, ncells))
sum_gdp_detrended = np.zeros((nyears))
fp_gdp_detrended = np.zeros((nyears, ncells))
fp_sum_gdp_detrended = np.zeros((nyears))

for iyear in range(nyears):

    gdp_detrended[iyear, :] = gdp[iyear, :] / (1 + grate)**iyear
    sum_gdp_detrended[iyear] = sum_gdp[iyear] / (1 + grate)**iyear
    fp_gdp_detrended[iyear, :] = fp_gdp[iyear, :] / (1 + grate)**iyear
    fp_sum_gdp_detrended[iyear] = fp_sum_gdp[iyear] / (1 + grate)**iyear

print('Percentage change in GDP (detrended):')
print(
    100 * (sum_gdp_detrended - sum_gdp_detrended[0]) /
    sum_gdp_detrended[0],
    np.mean(100 * (sum_gdp_detrended[-10:] - sum_gdp_detrended[0]) /
            sum_gdp_detrended[0]))
print(
    100 * (fp_sum_gdp_detrended - fp_sum_gdp_detrended[0]) /
    fp_sum_gdp_detrended[0],
    np.mean(100 * (fp_sum_gdp_detrended[-10:] - fp_sum_gdp_detrended[0]) /
            fp_sum_gdp_detrended[0]))



# Calculate population weighted temperature:
pop_temp = np.average(temperature - pi_temperatures,
                      axis=1,
                      weights=population)
expected_pop_temp = np.average(expected_temperature - pi_temperatures,
                               axis=1,
                               weights=population)

# Calculate area weighted temperature:
area_temp = np.average(temperature - pi_temperatures,
                       axis=1,
                       weights=np.cos(np.deg2rad(diam_latitudes)))
expected_area_temp = np.average(expected_temperature - pi_temperatures,
                                axis=1,
                                weights=np.cos(np.deg2rad(diam_latitudes)))

cell_temp = np.average(temperature - pi_temperatures, axis=1)

#--------------------------------------------------------------------------------------

# DEFINE AND EXTRACT COUNTRIES/REGIONS:

country_names = get_country_names()

# Make empty dictionary with empty lists:
country_indices = defaultdict(list)
country_latitudes = defaultdict(list)
country_pops = defaultdict(list)
country_pop = defaultdict(float)
country_gdp = defaultdict(float)
fp_country_gdp = defaultdict(float)

all_indices = []

# Sort all indices for each country into the dictionary:
for index, country in enumerate(country_names):
    country_indices[country].append(index)
    country_latitudes[country].append(diam_latitudes[index])
    country_pop[country] = country_pop[country] + population[index]
    country_pops[country].append(population[index])
    country_gdp[country] = country_gdp[country] + gdp_detrended[:, index]
    fp_country_gdp[country] = fp_country_gdp[country] + fp_gdp_detrended[:, index]


# Make list of all countries without duplicates:
all_countries = list(country_indices.keys())
n_countries = np.arange(len(all_countries))

# Remove some regions:
for c, country in enumerate(country_indices.keys()):

    # Remove if pop under 250k and GDP under 2:
    if country_pop[country] < 250 and country_gdp[0, 0, country] < 2:

        all_countries.remove(country)

# Make list of chosen countries:
chosen_countries = all_countries

# Make arrays with the GDP, damages, and PI temperature of the chosen countries:
gdp_country = np.zeros((nyears, len(chosen_countries)))
fp_gdp_country = np.zeros((nyears, len(chosen_countries)))
pi_temp_countries = np.zeros((len(chosen_countries)))
temp_countries = np.zeros((nyears, len(chosen_countries)))
expected_temp_countries = np.zeros((nyears, len(chosen_countries)))

for c, country in enumerate(chosen_countries):

    indices = country_indices[country]
    pops = np.asarray(country_pops[country])

    gdp_country[:, c] = country_gdp[country] * 1e9
    fp_gdp_country[:, c] = fp_country_gdp[country] * 1e9

    pi_temp_countries[c] = calculate_regional_mean(pi_temperatures[:],
                                                   indices,
                                                   weights=pops)


    for iyear in range(nyears):

        temp_countries[iyear,
                    c] = calculate_regional_mean(temperature[iyear, :],
                                                    indices,
                                                    weights=pops)
        expected_temp_countries[iyear, c] = calculate_regional_mean(
            expected_temperature[iyear, :], indices, weights=pops)

#--------------------------------------------------------------------------------------

# STATISTICS
"""
std_dtemp_countries = np.std(temp_countries - pi_temp_countries, axis=1)
std_expected_dtemp_countries = np.std(expected_temp_countries -
                                      pi_temp_countries,
                                      axis=1)
std_dtemp = np.std(temperature - pi_temperatures, axis=1)
std_expected_dtemp = np.std(expected_temperature - pi_temperatures, axis=1)

expected_dgdp_countries = 100 * (fp_gdp_country -
                                 fp_gdp_country[0, :]) / fp_gdp_country[0, :]
dgdp_countries = 100 * (gdp_country - gdp_country[0, :]) / gdp_country[0, :]

expected_dgdp = 100 * (fp_gdp_detrended -
                       fp_gdp_detrended[0, :]) / fp_gdp_detrended[0, :]
dgdp = 100 * (gdp_detrended - gdp_detrended[0, :]) / gdp_detrended[0, :]

std_gdp_countries = np.std(dgdp_countries, axis=1)
std_expected_gdp_countries = np.std(expected_dgdp_countries, axis=1)

std_gdp = np.std(dgdp, axis=1)
std_expected_gdp = np.std(expected_dgdp, axis=1)
"""
#--------------------------------------------------------------------------------------

# LOOK AT CHANGE IN PRODUCTIVITY
"""
chosen_years = [2, 20, 40, 50, 80]

damage = damages(temperature)
expected_damage = damages(expected_temperature)

damage_diff = 100 * (damage - expected_damage) / expected_damage
dss = (damage / expected_damage)  #**(1 - alpha)

for chosen_year in chosen_years:

    idx_pos = []
    idx_neg = []
    emissions_pos = 0
    emissions_neg = 0

    # Sort negative and positive changes:
    for icell in range(ncells):

        if dss[chosen_year, icell] < 1:
            idx_neg.append(icell)
            emissions_neg += actual_emissions[chosen_year, icell]
        else:
            idx_pos.append(icell)
            emissions_pos += actual_emissions[chosen_year, icell]

    # Add text:
    textstr = '\n'.join(
        (r'$\Delta T=%.2f$' % (pop_temp[chosen_year], ),
         r'$\Delta \bar{T}=%.2f$' % (expected_pop_temp[chosen_year], )))
    textstr2 = r'Difference in emissions: %.2f' % (
        100 * diff_emissions[chosen_year] / ss_emissions[chosen_year], )
    neg_string = r'$%.1f$ of emissions' % (100 * emissions_neg /
                                           sum_actual_emissions[chosen_year], )
    pos_string = r'$%.1f$ of emissions' % (100 * emissions_pos /
                                           sum_actual_emissions[chosen_year], )

    bins = np.histogram(damage_diff[chosen_year, :], bins=50)[1]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))

    ax.hist(damage_diff[chosen_year, :], bins=bins)
    ax.set_xlim(-100, 100)
    ax.set_title('Year ' + str(1990 + chosen_year))
    ax.set_xlabel('Deviation from expected damage (%)', fontsize=14)
    ax.text(0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top')
    ax.text(0.55,
            0.95,
            textstr2,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top')
    ax.text(0.05,
            0.75,
            neg_string,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top')
    ax.text(0.55,
            0.75,
            pos_string,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top')

    fig.savefig('Histogram_damage_diff_year_' + str(1990 + chosen_year) +
                '.pdf')
    bins = np.histogram(dss[chosen_year, :], bins=50)[1]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))

    ax.hist(dss[chosen_year, :], bins=bins)
    #ax.set_xlim(-100, 100)
    ax.set_title('Year ' + str(1990 + chosen_year))
    ax.set_xlabel('dss', fontsize=14)
    ax.text(0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top')
    ax.text(0.55,
            0.95,
            textstr2,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top')
    ax.text(0.05,
            0.75,
            neg_string,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top')
    ax.text(0.55,
            0.75,
            pos_string,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top')

    fig.savefig('Histogram_dss_year_' + str(1990 + chosen_year) + '.pdf')

plt.close()
"""
#--------------------------------------------------------------------------------------

# PLOT

years = np.arange(1990, 1990 + nyears + 1)

fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
bx1 = ax1.twinx()

fig1.subplots_adjust(right=0.85)
linestyles = ['-', '--']


ax1.plot(years[:-1],
            100 * diff_emissions / ss_emissions[:nyears],
            linewidth=2,
            color='k')
bx1.plot(years[:-1],
            100 * diff_cumulative_emissions[:-1] /
            ss_cumulative_emissions[:nyears],
            linewidth=2,
            color='grey')

ax1.set_xlabel('Year', fontsize=14)
ax1.set_ylabel('Difference in yearly emissions (%)', fontsize=14)
bx1.set_ylabel('Difference in cumulative emissions (%)',
               fontsize=14,
               color='grey')
ax1.legend()
#ax1.set_title('1% growth', fontsize=14)

ax1.xaxis.set_tick_params(labelsize=12)
ax1.yaxis.set_tick_params(labelsize=12)
bx1.yaxis.set_tick_params(labelsize=12, color='grey')

fig1.savefig('figures/difference_emissions.pdf')

print('Sum of emission difference: ', np.sum(diff_emissions))


#--------------------------------------------------------------------------------------

fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(14, 10))

ax2.plot(years[:-1],
            pop_temp,
            label='NorESM2-DIAM',
            linewidth=3,
            color='darkblue')
ax2.scatter(years[:-1], pop_temp, color='darkblue', s=75)

ax2.plot(years[:-1],
            expected_pop_temp,
            label='DIAM expectation',
            linewidth=3,
            color='cornflowerblue')

    

ax2.set_xlabel('Year', fontsize=20)
ax2.set_ylabel('Temperature change (\N{DEGREE SIGN}C)', fontsize=20)
ax2.xaxis.set_tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelsize=16)
ax2.legend(fontsize=20)

#print('Average difference in population-weigthed temperature:',
#      np.mean(pop_temp2) - np.mean(pop_temp1))

fig2.savefig('figures/population_weighted_temperature.pdf')
fig2.savefig('figures/population_weighted_temperature.png')

fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(14, 10))


ax2.plot(years[:-1],
            expected_area_temp,
            label='DIAM expectation',
            linewidth=3,
            color='cornflowerblue')
ax2.plot(years[:-1],
            area_temp,
            label='NorESM2-DIAM',
            linewidth=3,
            color='darkblue')

ax2.scatter(years[:-1], area_temp, color='darkblue', s=75)

ax2.set_xlabel('Year', fontsize=20)
ax2.set_ylabel('Temperature change (\N{DEGREE SIGN}C)', fontsize=20)
ax2.xaxis.set_tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelsize=16)
ax2.legend(fontsize=20)

fig2.savefig('figures/area_weighted_temperature.pdf')

#print('Average difference in area-weigthed temperature:',
#      np.mean(area_temp2) - np.mean(area_temp1))

#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------

# PLOT EVERY YEAR

polyline = np.linspace(-34, 24, 100)

expected_dtemp = expected_temp_countries - pi_temp_countries
expected_dgdp = 100 * (fp_gdp_country -
                       fp_gdp_country[0, :]) / fp_gdp_country[0, :]
dtemp = temp_countries - pi_temp_countries
dgdp = 100 * (gdp_country - fp_gdp_country[0, :]) / fp_gdp_country[0, :]

print(np.max(dtemp), np.min(dtemp))
print(np.max(dgdp), np.min(dgdp))
print(np.max(expected_dgdp), np.min(expected_dgdp))

colors = sns.color_palette('plasma', 11).as_hex()
my_cmap = ListedColormap(colors)

for iyear in range(nyears):  #nyears):

    fig4, ax4 = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.5))

    pscat1 = ax4[0].scatter(expected_dtemp[iyear, :],
                            expected_dgdp[iyear, :],
                            cmap=my_cmap,
                            vmin=-3,
                            vmax=30,
                            edgecolors='none',
                            alpha=0.8,
                            label=None,
                            c=pi_temp_countries,
                            s=np.sqrt(gdp_country[0, :] / 1e7))

    pscat2 = ax4[1].scatter(dtemp[iyear, :],
                            dgdp[iyear, :],
                            cmap=my_cmap,
                            vmin=-3,
                            vmax=30,
                            edgecolors='none',
                            alpha=0.8,
                            label=None,
                            c=pi_temp_countries,
                            s=np.sqrt(gdp_country[0, :] / 1e7))
    """
    for c, country in enumerate(chosen_countries):

        # Add country names:
        ax4[0].text(expected_temp_countries[0, :] - pi_temp_countries,
                    expected_gdp_country[0, :] - expected_gdp_country[0, :],
                    country,
                    fontsize=5)
        ax4[1].text(temp_countries[0, :] - pi_temp_countries,
                    gdp_country[0, :] - gdp_country[0, :],
                    country,
                    fontsize=5)
    """
    """
        # Add circle around chosen courtries:
        pscat = ax4.scatter(dtemp_countries,
                            gdp_countries_change_damage,
                            cmap=my_cmap,
                            edgecolors='k',
                            linewidth=0.2,
                            alpha=0.8,
                            label=None,
                            c=pi_temp_countries,
                            s=np.sqrt(gdp_country_1990[c] * 1e3))
    """

    # Add global value:
    pscat3 = ax4[0].scatter(
        np.average(expected_temperature[iyear, :],
                    weights=population) -
        np.average(pi_temperatures, weights=population),
        100 * (fp_sum_gdp_detrended[iyear] - fp_sum_gdp_detrended[0]) /
        fp_sum_gdp_detrended[0],
        c='black',
        s=50,
        alpha=0.8)
    pscat4 = ax4[1].scatter(
        np.average(temperature[iyear, :], weights=population) -
        np.average(pi_temperatures, weights=population),
        100 *
        (sum_gdp_detrended[iyear] - sum_gdp_detrended[0]) /
        sum_gdp_detrended[0],
        c='black',
        s=50,
        alpha=0.8)

    # Generate color bar to indicate 2000 temperature:
    cbar_ax = fig4.add_axes([0.93, 0.13, 0.02, 0.39])
    cbar = fig4.colorbar(pscat1,
                            ticks=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27],
                            cax=cbar_ax)
    cbar.set_label('Temperature (\N{DEGREE SIGN}C)',
                    fontsize=12,
                    rotation=270,
                    labelpad=18)
    cbar.ax.tick_params(labelsize=10)

    # Generate legend to indicate GDP size:
    gdp_ax = fig4.add_axes([0.92, 0.52, 0.02, 0.4], frameon=False)
    gdp_ax.set_yticks([]), gdp_ax.set_xticks([])
    gdp_labels = [
        '10$^9$', '10$^{10}$', '10$^{11}$', '10$^{12}$', '10$^{13}$'
    ]
    for igdp, gdp in enumerate([1e9, 1e10, 1e11, 1e12, 1e13]):
        gdp_ax.scatter([], [],
                        c='',
                        edgecolor='k',
                        linewidths=0.7,
                        s=np.sqrt(gdp / 1e7),
                        label=gdp_labels[igdp])
    glegend = gdp_ax.legend(scatterpoints=1,
                            frameon=False,
                            labelspacing=0.7,
                            title='GDP ($)',
                            loc=2,
                            fontsize=10)
    glegend.get_title().set_fontsize('12')

    ax4[0].xaxis.set_tick_params(labelsize=12)
    ax4[0].yaxis.set_tick_params(labelsize=12)
    ax4[1].xaxis.set_tick_params(labelsize=12)
    ax4[1].yaxis.set_tick_params(labelsize=12)

    ax4[0].set_xlim(-1.3, 6)
    ax4[0].set_ylim(-50, 60)
    ax4[1].set_xlim(-1.3, 6)
    ax4[1].set_ylim(-50, 60)

    # Add the 0-line:
    ax4[0].axhline(0, color='grey', alpha=0.6, linestyle='--', linewidth=1)
    ax4[1].axhline(0, color='grey', alpha=0.6, linestyle='--', linewidth=1)

    ax4[0].set_xlabel(r'$\Delta$temperature ' + '(\N{DEGREE SIGN}C)',
                        fontsize=14)
    ax4[0].set_ylabel(r'$\Delta$GDP (%)', fontsize=14)
    ax4[1].set_xlabel(r'$\Delta$temperature ' + '(\N{DEGREE SIGN}C)',
                        fontsize=14)

    ax4[0].set_title('DIAM expectation', fontsize=14)
    ax4[1].set_title('NorESM2-DIAM', fontsize=14)

    fig4.subplots_adjust(left=0.06, right=0.9, top=0.9, bottom=0.1)

    fig4.suptitle('Year {:d}'.format(1990 + iyear), x=0.06)

    fig4.savefig('figures/countries_year_{:d}.png'.format(
        1990 + iyear))

    plt.close()

#--------------------------------------------------------------------------------------

# PLOT EVERY DECADE

colors = sns.color_palette('plasma', 11).as_hex()
my_cmap = ListedColormap(colors)

polyline = np.linspace(-60, 60, 100)

temp_start = np.average(temp_countries[:10, :], axis=0)
expected_temp_start = np.average(expected_temp_countries[:10, :], axis=0)
expected_gdp_start = np.average(fp_gdp_country[:10, :], axis=0)
gdp_start = np.average(gdp_country[:10, :], axis=0)

ndecades1 = nyears // 10

text_countires = [
    'Norway', 'United States', 'Russia', 'United Kingdom', 'China', 'Somalia',
    'South Africa', 'Germany', 'Sudan', 'Canada', 'New Zealand', 'Spain',
    'Senegal', 'Somalia', 'Argentina', 'Peru', 'India', 'Saudi Arabia', 'Iraq',
    'Iceland'
]  #'Algeria', 'Indonesia'



for idec in range(1, ndecades1):

    print('Decade:', 1990 + idec * 10, '-', 2000 + idec * 10)

    expected_dtemp_countries = np.average(
        expected_temp_countries[10 * idec:10 * (idec + 1), :],
        axis=0) - expected_temp_start
    dtemp_countries = np.average(temp_countries[10 * idec:10 *
                                                (idec + 1), :],
                                    axis=0) - temp_start

    print(np.max(expected_dtemp_countries),
            all_countries[np.argmax(expected_dtemp_countries)])

    expected_dgdp_countries = 100 * (
        np.average(fp_gdp_country[10 * idec:10 * (idec + 1), :], axis=0) -
        expected_gdp_start) / expected_gdp_start
    dgdp_countries = 100 * (
        np.average(gdp_country[10 * idec:10 *
                                (idec + 1), :], axis=0) -
        gdp_start) / gdp_start

    # Degree 2 polynomial fit or quadratic fit:
    print(dgdp_countries.shape, dtemp_countries.shape)
    expected_model = np.poly1d(
        np.polyfit(expected_dgdp_countries, expected_dtemp_countries, 2))
    model = np.poly1d(np.polyfit(dgdp_countries, dtemp_countries, 2))
    #print(expected_model)
    #print(model)

    #fig5, ax5 = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.5))
    fig5, ax5 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))
    fig6, ax6 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))

    pscat1 = ax5.scatter(expected_dtemp_countries,
                            expected_dgdp_countries,
                            cmap=my_cmap,
                            vmin=-3,
                            vmax=30,
                            edgecolors='none',
                            alpha=0.8,
                            label=None,
                            c=temp_start,
                            s=np.sqrt(gdp_country[0, :] / 1e7))

    pscat2 = ax6.scatter(dtemp_countries,
                            dgdp_countries,
                            cmap=my_cmap,
                            vmin=-3,
                            vmax=30,
                            edgecolors='none',
                            alpha=0.8,
                            label=None,
                            c=temp_start,
                            s=np.sqrt(gdp_country[0, :] / 1e7))

    for c, country in enumerate(all_countries):

        if country in text_countires:

            # Add country names:
            ax5.text(expected_dtemp_countries[c],
                        expected_dgdp_countries[c],
                        country,
                        fontsize=5)
            ax6.text(dtemp_countries[c],
                        dgdp_countries[c],
                        country,
                        fontsize=5)
            """
            # Add circle around chosen courtries:
            ax5[0].scatter(expected_dtemp_countries[c],
                        expected_dgdp_countries[c],
                        cmap=my_cmap,
                        edgecolors='k',
                        linewidth=0.2,
                        alpha=0.8,
                        label=None,
                        c=temp_start,
                        s=np.sqrt(gdp_country[0, c] / 1e7))
            ax5[1].scatter(dtemp_countries[c],
                        dgdp_countries[c],
                        cmap=my_cmap,
                        edgecolors='k',
                        linewidth=0.2,
                        alpha=0.8,
                        label=None,
                        c=temp_start,
                        s=np.sqrt(gdp_country[0, c] / 1e7))
            """

    ax5.plot(expected_model(polyline), polyline)
    ax6.plot(model(polyline), polyline)

    # Add global value:
    ax5.scatter(np.average(
        np.average(expected_temperature[10 * idec:10 *
                                        (idec + 1), :],
                    weights=population,
                    axis=1)) - np.average(
                        np.average(expected_temperature[:10, :],
                                    weights=population,
                                    axis=1)),
                100 * (np.average(fp_sum_gdp_detrended[10 * idec:10 *
                                                        (idec + 1)]) -
                        np.average(fp_sum_gdp_detrended[:10])) /
                np.average(fp_sum_gdp_detrended[:10]),
                c='black',
                s=50,
                alpha=0.8)
    ax6.scatter(
        np.average(
            np.average(temperature[10 * idec:10 * (idec + 1), :],
                        weights=population,
                        axis=1)) -
        np.average(
            np.average(
                temperature[:10, :], weights=population, axis=1)),
        100 * (np.average(sum_gdp_detrended[10 * idec:10 *
                                            (idec + 1)]) -
                np.average(sum_gdp_detrended[:10])) /
        np.average(sum_gdp_detrended[:10]),
        c='black',
        s=50,
        alpha=0.8)

    for fig, ax in zip([fig5, fig6], [ax5, ax6]):

        # Add the 0-line:
        ax.axhline(0, color='grey', alpha=0.6, linestyle='--', linewidth=1)

        # Generate legend to indicate GDP size:
        gdp_ax = fig.add_axes([0.86, 0.52, 0.02, 0.4], frameon=False)
        gdp_ax.set_yticks([]), gdp_ax.set_xticks([])
        gdp_labels = [
            '10$^9$', '10$^{10}$', '10$^{11}$', '10$^{12}$', '10$^{13}$'
        ]
        for igdp, gdp in enumerate([1e9, 1e10, 1e11, 1e12, 1e13]):
            gdp_ax.scatter([], [],
                            c='',
                            edgecolor='k',
                            linewidths=0.7,
                            s=np.sqrt(gdp / 1e7),
                            label=gdp_labels[igdp])
        glegend = gdp_ax.legend(scatterpoints=1,
                                frameon=False,
                                labelspacing=0.7,
                                title='GDP ($)',
                                loc=2,
                                fontsize=10)
        glegend.get_title().set_fontsize('12')

        # Generate color bar to indicate 2000 temperature:
        cbar_ax = fig.add_axes([0.88, 0.13, 0.02, 0.39])
        cbar = fig.colorbar(pscat1,
                            ticks=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27],
                            cax=cbar_ax)
        cbar.set_label('Temperature (\N{DEGREE SIGN}C)',
                        fontsize=12,
                        rotation=270,
                        labelpad=18)
        cbar.ax.tick_params(labelsize=10)

        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)

        ax.set_xlabel(r'$\Delta$temperature ' + '(\N{DEGREE SIGN}C)',
                        fontsize=14)
        ax.set_ylabel(r'$\Delta$GDP (%)', fontsize=14)

        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-50, 50)

        fig.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)
    """
    ax5[0].set_title('DIAM expectation: {:d}-{:d}'.format(
        1990 + idec * 10, 1999 + idec * 10),
                    fontsize=14)
    ax5[1].set_title('NorESM2-DIAM: {:d}-{:d}'.format(1990 + idec * 10,
                                                    1999 + idec * 10),
                    fontsize=14)

    fig5.subplots_adjust(left=0.06, right=0.9, top=0.9, bottom=0.1)

    fig5.savefig('country_gdp_change_damage_compare_{:d}_{:d}.pdf'.format(
        1990 + idec * 10, 2000 + idec * 10))
    """

    fig5.savefig(
        'figures/country_gdp_change_expectation_same_axes_{:d}_{:d}.pdf'
        .format(1990 + idec * 10, 1999 + idec * 10))
    fig5.savefig(
        'figures/country_gdp_change_expectation_same_axes_{:d}_{:d}.png'
        .format(1990 + idec * 10, 1999 + idec * 10))
    fig6.savefig(
        'figures/country_gdp_change_damage_noresm2-diam_same_axes_{:d}_{:d}.pdf'
        .format(1990 + idec * 10, 1999 + idec * 10))
    fig6.savefig(
        'figures/country_gdp_change_damage_noresm2-diam_same_axes_{:d}_{:d}.png'
        .format(1990 + idec * 10, 1999 + idec * 10))

    plt.close()
exit()

#--------------------------------------------------------------------------------------

a = np.average(expected_temperature1[-10:, :], axis=0) - np.average(
    expected_temperature1[:10, :], axis=0)
b = np.average(temperature1[-10:, :], axis=0) - np.average(
    temperature1[:10, :], axis=0)

c = 100 * (np.average(expected_gdp_detrended1[-10:, :], axis=0) -
           np.average(expected_gdp_detrended1[:10, :], axis=0)) / np.average(
               expected_gdp_detrended1[:10, :], axis=0)
d = 100 * (np.average(gdp_detrended1[-10:, :], axis=0) -
           np.average(gdp_detrended1[:10, :], axis=0)) / np.average(
               gdp_detrended1[:10, :], axis=0)

print(a, b, c, d)
"""
fig6, ax6 = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.5))

pscat1 = ax6[0].scatter(a,
                        c,
                        cmap=my_cmap,
                        vmin=-3,
                        vmax=30,
                        edgecolors='none',
                        alpha=0.8,
                        label=None,
                        c=np.average(expected_temperature1[:10, :], axis=0))

pscat2 = ax6[1].scatter(b,
                        d,
                        cmap=my_cmap,
                        vmin=-3,
                        vmax=30,
                        edgecolors='none',
                        alpha=0.8,
                        label=None,
                        c=np.average(temperature1[:10, :], axis=0))

# Add global value:
ax6[0].scatter(
    np.average(
        np.average(expected_temperature1[-10:, :], weights=population, axis=1))
    - np.average(
        np.average(expected_temperature1[:10, :], weights=population, axis=1)),
    100 * (np.average(sum_expected_gdp_detrended1[-10:]) -
           np.average(sum_expected_gdp_detrended1[:10])) /
    np.average(sum_expected_gdp_detrended1[:10]),
    c='black',
    s=50,
    alpha=0.8)
ax6[1].scatter(
    np.average(np.average(temperature1[-10:, :], weights=population, axis=1)) -
    np.average(np.average(temperature1[:10, :], weights=population, axis=1)),
    100 * (np.average(sum_gdp_detrended1[-10:]) -
           np.average(sum_gdp_detrended1[:10])) /
    np.average(sum_gdp_detrended1[:10]),
    c='black',
    s=50,
    alpha=0.8)

# Add the 0-line:
ax6[0].axhline(0, color='grey', alpha=0.6, linestyle='--', linewidth=1)
ax6[1].axhline(0, color='grey', alpha=0.6, linestyle='--', linewidth=1)

# Generate color bar to indicate 2000 temperature:
cbar_ax = fig6.add_axes([0.93, 0.13, 0.02, 0.39])
cbar = fig6.colorbar(pscat1,
                     ticks=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27],
                     cax=cbar_ax)
cbar.set_label('Temperature (\N{DEGREE SIGN}C)',
               fontsize=12,
               rotation=270,
               labelpad=18)
cbar.ax.tick_params(labelsize=10)

ax6[0].xaxis.set_tick_params(labelsize=12)
ax6[0].yaxis.set_tick_params(labelsize=12)
ax6[1].xaxis.set_tick_params(labelsize=12)
ax6[1].yaxis.set_tick_params(labelsize=12)

ax6[0].set_xlabel(r'$\Delta$temperature ' + '(\N{DEGREE SIGN}C)', fontsize=14)
ax6[0].set_ylabel(r'$\Delta$GDP (%)', fontsize=14)
ax6[1].set_xlabel(r'$\Delta$temperature ' + '(\N{DEGREE SIGN}C)', fontsize=14)

ax6[0].set_title('DIAM expectation', fontsize=14)
ax6[1].set_title('NorESM2-DIAM', fontsize=14)

fig6.subplots_adjust(left=0.06, right=0.9, top=0.9, bottom=0.1)

fig6.savefig('region_gdp_change_damage_compare.pdf')
"""
#--------------------------------------------------------------------------------------

fig7, ax7 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))

ax7.plot(years[:-1],
         np.std(expected_temperature1, axis=1),
         label='DIAM expectation')
ax7.plot(years[:-1], np.std(temperature1, axis=1), label='NorESM2-DIAM')

ax7.set_xlabel('Year')
ax7.set_ylabel('Temperature (degC)')

ax7.legend()

fig7.savefig('std_temperature.pdf')
fig7.set_facecolor('#66cefc')
fig7.savefig('phd_std_temperature.png')

#--------------------------------------------------------------------------------------

fig8, ax8 = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.5))

ax8[0].plot(years[:-1],
            np.percentile(expected_gdp_detrended1 / population, 75, axis=1) /
            np.percentile(expected_gdp_detrended1 / population, 25, axis=1),
            label='DIAM fixed point')
ax8[0].plot(years[:-1],
            np.percentile(gdp_detrended1 / population, 75, axis=1) /
            np.percentile(gdp_detrended1 / population, 25, axis=1),
            label='NorESM2-DIAM')
ax8[0].set_title('Regional GDP per capita: 75-25 ratio')

ax8[1].plot(years[:-1],
            np.percentile(expected_gdp_detrended1 / population, 90, axis=1) /
            np.percentile(expected_gdp_detrended1 / population, 10, axis=1),
            label='DIAM fixed point')
ax8[1].plot(years[:-1],
            np.percentile(gdp_detrended1 / population, 90, axis=1) /
            np.percentile(gdp_detrended1 / population, 10, axis=1),
            label='NorESM2-DIAM')
ax8[1].set_title('Regional GDP per capita: 90-10 ratio')

#ax8[0].set_xlabel('GDP change (%)')
#ax8[1].set_xlabel('GDP change (%)')

ax8[0].legend()

fig8.savefig('gdp_percentiles.pdf')
fig8.set_facecolor('#66cefc')
fig8.savefig('phd_gdp_percentiles.png')

fig8, ax8 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))

ax8.plot(years[:-1],
         np.std(expected_gdp_detrended1 / population, axis=1),
         label='DIAM expectation')
ax8.plot(years[:-1],
         np.std(gdp_detrended1 / population, axis=1),
         label='NorESM2-DIAM')

ax8.set_xlabel('Year')
ax8.set_ylabel('GDP per capita')

ax8.legend()

fig8.savefig('gdp_per_capita_std.pdf')
fig8.set_facecolor('#66cefc')
fig8.savefig('phd_gdp_per_capita_std.png')

#--------------------------------------------------------------------------------------

temp_bins = np.histogram(np.hstack((a, b)), bins=50)[1]
gdp_bins = np.histogram(np.hstack((c, d)), bins=50)[1]

fig9, ax9 = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.5))

ax9[0].hist(a, bins=temp_bins, alpha=0.5, label='DIAM expectation')
ax9[0].hist(b, bins=temp_bins, alpha=0.5, label='NorESM2-DIAM')
ax9[0].set_title('Temperature', fontsize=14)

ax9[1].hist(c, bins=gdp_bins, alpha=0.5, label='DIAM expectation')
ax9[1].hist(d, bins=gdp_bins, alpha=0.5, label='NorESM2-DIAM')
ax9[1].set_title('GDP', fontsize=14)

ax9[0].set_xlabel(r'$\Delta$temperature ' + '(\N{DEGREE SIGN}C)', fontsize=14)
ax9[1].set_xlabel(r'$\Delta$GDP (%)', fontsize=14)

ax9[1].legend()

fig9.savefig('histogram_temp_and_gdp.pdf')
fig9.set_facecolor('#66cefc')
fig9.savefig('phd_histogram_temp_and_gdp.png')

#--------------------------------------------------------------------------------------
