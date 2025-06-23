# --------------------------------------------------------------------------------------

# import sys as sys
import glob as glob
import os as os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn.apionly as sns
from matplotlib.colors import ListedColormap, TwoSlopeNorm, LinearSegmentedColormap

# sys.path.insert(0, '../modules')
from module_coupling import *

# --------------------------------------------------------------------------------------

alpha = 0.36  # capital’s share of income (capital share + labor share = 1)
delta = 0.06  # The (annual) rate of depreciation of the capital stock
price = get_price()

file_path = ""  # "/home/jennybj/uio/home/"

# --------------------------------------------------------------------------------------

global_pi_temperature = 14.460473280816053

pi_temperatures = get_pi_temperature()
population = get_population()
chi = get_chit()

diam_latitudes, diam_longitudes = get_coordinate_data()

ncells = diam_latitudes.shape[0]

# --------------------------------------------------------------------------------------

# READ IN DATA

years, ss_emissions = np.loadtxt(file_path + "coupling/emissions.txt", unpack=True)

output_files = sorted(glob.glob(file_path + "coupling/output_year_*.txt"))
nyears = len(output_files)
fp_output_files = sorted(glob.glob(file_path + "coupling/fp_output_year_*.txt"))

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

    (
        expected_temperature[i, :],
        temperature[i, :],
        wealth_scaled[i, :],
        capital_scaled[i, :],
        ai[i, :],
        energy_scaled[i, :],
        expected_emissions[i, :],
        actual_emissions[i, :],
    ) = np.loadtxt(file, skiprows=15, usecols=(2, 3, 5, 6, 8, 10, 12, 13), unpack=True)

for i, file in enumerate(fp_output_files[:nyears]):
    print(file)

    (
        fp_wealth_scaled[i, :],
        fp_capital_scaled[i, :],
        fp_ai[i, :],
        fp_energy_scaled[i, :],
        fp_emissions[i, :],
    ) = np.loadtxt(file, skiprows=15, usecols=(5, 6, 8, 10, 12), unpack=True)

# global_temperature = np.loadtxt(
#    file_path + "/coupling/full_couple_population_global_temp.txt", usecols=1
# )

# --------------------------------------------------------------------------------------

# FUNCTIONS


def damages(regtemp, tstar=12.609, scale1=0.00327721, scale2=0.00362887):
    """The regional damage function. Already raised to the power of 1/(1 - alpha)"""

    # Define constants:
    pbound = 0.02
    toler = 1.0e-4

    diff = regtemp - tstar

    myears = regtemp.shape[0]
    mcells = regtemp.shape[1]

    if mcells != ncells:
        print("Number of cells is ", mcells, " not ", ncells)

    fval = np.zeros((myears, mcells))

    # ((1 - d) * exp(-κ_minus * (t - T) ^ 2) + d) ^ (1 / (1 - α))

    for iyear in range(myears):
        for icell in range(mcells):
            if diff[iyear, icell] < 0:
                fval[iyear, icell] = (
                    np.exp(-scale1 * diff[iyear, icell] * diff[iyear, icell])
                    * (1 - pbound)
                    + pbound
                ) ** (1 / (1 - alpha))
            else:
                fval[iyear, icell] = (
                    np.exp(-scale2 * diff[iyear, icell] * diff[iyear, icell])
                    * (1 - pbound)
                    + pbound
                ) ** (1 / (1 - alpha))

            if fval[iyear, icell] < toler:
                fval[iyear, icell] = toler

    return fval


def descale(in_variable, in_ai):
    nyears = in_variable.shape[0]
    out_variable = np.zeros((nyears, ncells))

    for iyear in range(nyears):
        out_variable[iyear, :] = in_variable[iyear, :] * (
            population[iyear, :] * in_ai[iyear, :]
        )

    return out_variable


def add_bubble_label(fig, position, labels, label_values, title):

    # Generate legend to indicate GDP size:
    ax = fig.add_axes(position, frameon=False)
    ax.set_yticks([]), ax.set_xticks([])
    for i, value in enumerate(label_values):
        ax.scatter(
            [],
            [],
            c='None',
            edgecolor='black',
            linewidths=0.7,
            s=value,
            label=labels[i],
        )
    legend = ax.legend(
        scatterpoints=1,
        frameon=False,
        labelspacing=0.7,
        title=title,
        loc=2,
        fontsize=10,
    )
    legend.get_title().set_fontsize("12")


# --------------------------------------------------------------------------------------

# CALCULATIONS


# Calculate wealth:
wealth = descale(wealth_scaled, ai)
fp_wealth = descale(fp_wealth_scaled, ai)
sum_wealth = np.sum(wealth, axis=1)
sum_wealth_scaled = np.sum(wealth_scaled, axis=1)

# Calculate emissions:
cumulative_emissions_1990 = np.array([216.865])
sum_actual_emissions = np.sum(actual_emissions, axis=1)
actual_cumulative_emissions = np.cumsum(
    np.concatenate((cumulative_emissions_1990, sum_actual_emissions / 1e3))
)
ss_cumulative_emissions = np.cumsum(
    np.concatenate((cumulative_emissions_1990, ss_emissions))
)

diff_cumulative_emissions = (
    actual_cumulative_emissions - ss_cumulative_emissions[: nyears + 1]
)  # * 1e3
diff_emissions = sum_actual_emissions / 1e3 - ss_emissions[:nyears]  # * 1e3

# Calculate aggregate energy use:
energy_use = descale(energy_scaled, ai)
aggregate_energy_use = np.sum(energy_use, axis=1)
fp_energy_use = descale(fp_energy_scaled, fp_ai)
fp_aggregate_energy_use = np.sum(fp_energy_use, axis=1)

fp_sum_emissions = np.sum(fp_emissions, axis=1)

ss_energy_use = ss_emissions[:nyears] * 1e3 / chi[:nyears]

print("Total diff", np.sum(sum_actual_emissions - fp_sum_emissions))

# Read and calculate GDP:
capital = descale(capital_scaled, ai)
gdp = wealth - (1 - delta) * capital
sum_gdp = np.sum(gdp, axis=1)
gdp_scaled = wealth_scaled - (1 - delta) * capital_scaled
sum_gdp_scaled = np.sum(gdp_scaled, axis=1)

fp_gdp_scaled = fp_wealth_scaled - (1 - delta) * fp_capital_scaled
fp_gdp = descale(fp_wealth_scaled, fp_ai) - (1 - delta) * descale(
    fp_capital_scaled, fp_ai
)
fp_sum_gdp = np.sum(fp_gdp, axis=1)


# Detrend the GDP:
gdp_detrended = np.zeros((nyears, ncells))
fp_gdp_detrended = np.zeros((nyears, ncells))
sum_gdpper_detrended = np.zeros(nyears)
sum_fp_gdpper_detrended = np.zeros(nyears)

fp_gdp_per_capita = fp_gdp / population[:nyears, :]

for iyear in range(nyears):
    gdp_detrended[iyear, :] = gdp[iyear, :] / (1 + ga) ** iyear
    fp_gdp_detrended[iyear, :] = fp_gdp[iyear, :] / (1 + ga) ** iyear
    sum_gdpper_detrended[iyear] = (
        np.sum(gdp[iyear, :])
        * 1e9
        / ((1 + ga) ** iyear * np.sum(population[iyear, :] * 1e3))
    )
    sum_fp_gdpper_detrended[iyear] = np.sum(fp_gdp[iyear, :] * 1e9) / (
        (1 + ga) ** iyear * np.sum(population[iyear, :] * 1e3)
    )

sum_gdp_detrended = np.sum(gdp_detrended, axis=1)
sum_fp_gdp_detrended = np.sum(fp_gdp_detrended, axis=1)

print("Percentage change in GDP (detrended):")
print(
    100 * (sum_gdpper_detrended - sum_gdpper_detrended[0]) / sum_gdpper_detrended[0],
)
print(
    100
    * (sum_fp_gdpper_detrended - sum_fp_gdpper_detrended[0])
    / sum_fp_gdpper_detrended[0]
)
global_change = (
    100
    * (sum_fp_gdpper_detrended[-1] - sum_fp_gdpper_detrended[0])
    / sum_fp_gdpper_detrended[0]
)
print(global_change)

# Calculate population weighted temperature:
pop_temp = np.zeros((nyears))
expected_pop_temp = np.zeros((nyears))
expected_pop_temp2 = np.zeros((nyears))

for iyear in range(nyears):
    pop_temp[iyear] = np.average(
        temperature[iyear, :] - pi_temperatures, weights=population[iyear, :]
    )
    expected_pop_temp[iyear] = np.average(
        expected_temperature[iyear, :] - pi_temperatures, weights=population[iyear, :]
    )
    expected_pop_temp2[iyear] = np.average(
        expected_temperature[iyear, :], weights=population[iyear, :]
    )

expected_pop_temp_start = np.average(expected_pop_temp2[:10])

expected_const_pop_temp = np.average(
    expected_temperature - pi_temperatures, axis=1, weights=population[0, :]
)

# Calculate area weighted temperature:
area_temp = np.average(
    temperature - pi_temperatures, axis=1, weights=np.cos(np.deg2rad(diam_latitudes))
)
expected_area_temp = np.average(
    expected_temperature - pi_temperatures,
    axis=1,
    weights=np.cos(np.deg2rad(diam_latitudes)),
)

cell_temp = np.average(temperature - pi_temperatures, axis=1)

# --------------------------------------------------------------------------------------

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
    country_pop[country] = country_pop[country] + population[:, index]
    country_pops[country].append(population[:, index])
    country_gdp[country] = country_gdp[country] + gdp[:, index]
    fp_country_gdp[country] = fp_country_gdp[country] + fp_gdp[:, index]

# Make list of all countries without duplicates:
all_countries = list(country_indices.keys())
n_countries = np.arange(len(all_countries))

# Remove some regions:
for c, country in enumerate(country_indices.keys()):
    # Remove if pop under 250k and GDP under 2:
    if country_pop[country][0] < 250 and country_gdp[0, 0, country] < 2:
        all_countries.remove(country)

# Make list of chosen countries:
chosen_countries = all_countries

# Make arrays with the GDP, damages, and PI temperature of the chosen countries:
gdp_country = np.zeros((nyears, len(chosen_countries)))
fp_gdp_country = np.zeros((nyears, len(chosen_countries)))
gdpper_country = np.zeros((nyears, len(chosen_countries)))
fp_gdpper_country = np.zeros((nyears, len(chosen_countries)))
pi_temp_countries_area = np.zeros((len(chosen_countries)))
dtemp_countries = np.zeros((nyears, len(chosen_countries)))
expected_dtemp_countries = np.zeros((nyears, len(chosen_countries)))
expected_temp_countries = np.zeros((nyears, len(chosen_countries)))
population_countries = np.zeros((nyears, len(chosen_countries)))

for c, country in enumerate(chosen_countries):
    indices = country_indices[country]
    pops = np.asarray(country_pops[country]).T
    lat_weight = np.cos(np.deg2rad(country_latitudes[country]))
    population_countries[:, c] = np.asarray(country_pop[country][:nyears])

    gdp_country[:, c] = country_gdp[country] * 1e9
    fp_gdp_country[:, c] = fp_country_gdp[country] * 1e9
    for iyear in range(nyears):
        gdpper_country[iyear, c] = gdp_country[iyear, c] / (
            (1 + ga) ** iyear * population_countries[iyear, c] * 1e3
        )
        fp_gdpper_country[iyear, c] = fp_gdp_country[iyear, c] / (
            (1 + ga) ** iyear * population_countries[iyear, c] * 1e3
        )

    pi_temp_countries_area[c] = calculate_regional_mean(
        pi_temperatures[:], indices, weights=lat_weight
    )

    for iyear in range(nyears):
        dtemp_countries[iyear, c] = calculate_regional_mean(
            temperature[iyear, :] - pi_temperatures, indices, weights=pops[iyear, :]
        )
        expected_dtemp_countries[iyear, c] = calculate_regional_mean(
            expected_temperature[iyear, :] - pi_temperatures,
            indices,
            weights=pops[iyear, :],
        )
        expected_temp_countries[iyear, c] = calculate_regional_mean(
            expected_temperature[iyear, :],
            indices,
            weights=pops[iyear, :],
        )

start_temp_countries = np.average(expected_temp_countries[:10, :], axis=0)



# --------------------------------------------------------------------------------------

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
# --------------------------------------------------------------------------------------

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
# --------------------------------------------------------------------------------------

# PLOT

years = np.arange(1990, 1990 + nyears + 1)

fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
bx1 = ax1.twinx()

fig1.subplots_adjust(right=0.85)
linestyles = ["-", "--"]


ax1.plot(
    years[:-1], 100 * diff_emissions / ss_emissions[:nyears], linewidth=2, color='black'
)
bx1.plot(
    years[:-1],
    100 * diff_cumulative_emissions[:-1] / ss_cumulative_emissions[:nyears],
    linewidth=2,
    color="grey",
)

ax1.set_xlabel("Year", fontsize=14)
ax1.set_ylabel("Difference in yearly emissions (%)", fontsize=14)
bx1.set_ylabel("Difference in cumulative emissions (%)", fontsize=14, color="grey")
ax1.legend()
# ax1.set_title('1% growth', fontsize=14)

ax1.xaxis.set_tick_params(labelsize=12)
ax1.yaxis.set_tick_params(labelsize=12)
bx1.yaxis.set_tick_params(labelsize=12, color="grey")

fig1.savefig("figures/difference_emissions.pdf")

print("Sum of emission difference: ", np.sum(diff_emissions))


# --------------------------------------------------------------------------------------

fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(14, 10))

print(years.shape, pop_temp.shape)


ax2.plot(years[:-1], pop_temp, label="NorESM2-DIAM", linewidth=3, color="darkblue")
ax2.scatter(years[:-1], pop_temp, color="darkblue", s=75)

ax2.plot(
    years[:-1],
    expected_pop_temp,
    label="DIAM expectation",
    linewidth=3,
    color="cornflowerblue",
)
#ax2.plot(
#    years[:-1],
#    expected_const_pop_temp,
#    label="const pop",
#    linewidth=3,
#   color="grey",
#)

ax2.set_xlabel("Year", fontsize=20)
ax2.set_ylabel("Temperature change (\N{DEGREE SIGN}C)", fontsize=20)
ax2.xaxis.set_tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelsize=16)
ax2.legend(fontsize=20)

# print('Average difference in population-weigthed temperature:',
#      np.mean(pop_temp2) - np.mean(pop_temp1))

fig2.savefig("figures/population_weighted_temperature.pdf")
fig2.savefig("figures/population_weighted_temperature.png")

fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(14, 10))


ax2.plot(
    years[:-1],
    expected_area_temp,
    label="DIAM expectation",
    linewidth=3,
    color="cornflowerblue",
)
ax2.plot(years[:-1], area_temp, label="NorESM2-DIAM", linewidth=3, color="darkblue")

ax2.scatter(years[:-1], area_temp, color="darkblue", s=75)

ax2.set_xlabel("Year", fontsize=20)
ax2.set_ylabel("Temperature change (\N{DEGREE SIGN}C)", fontsize=20)
ax2.xaxis.set_tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelsize=16)
ax2.legend(fontsize=20)

fig2.savefig("figures/area_weighted_temperature.pdf")

# print('Average difference in area-weigthed temperature:',
#      np.mean(area_temp2) - np.mean(area_temp1))

# --------------------------------------------------------------------------------------

ncolors = 11
colors = sns.color_palette("plasma", ncolors).as_hex()
vmin = -3
vmax = 30
color_bins = np.linspace(vmin,vmax,ncolors+1)
index_global_color = np.where(color_bins < expected_pop_temp_start)[0][-1]
color_global = colors[index_global_color]

gdp_cmap = ListedColormap(colors)

#cvals  = [-2., 0,  2]
#colors = ["dimgray","salmon","red"]
#norm=plt.Normalize(min(cvals),max(cvals))
#tuples = list(zip(map(norm,cvals), colors))
#cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

cdict3 = {
    'red': (
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.5, 0.8, 1.0),
        (0.75, 1.0, 1.0),
        (1.0, 0.4, 1.0),
    ),
    'green': (
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.5, 0.9, 0.9),
        (0.75, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
    'blue': (
        (0.0, 0.0, 0.4),
        (0.25, 1.0, 1.0),
        (0.5, 1.0, 0.8),
        (0.75, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    )
}

cmap = LinearSegmentedColormap('BlueRed3', cdict3)

population_cmap = cmap #'RdYlBu'
vmin2=-100
vmax2=1000
divnorm = TwoSlopeNorm(vmin=vmin2, vcenter=0, vmax=vmax2)

# --------------------------------------------------------------------------------------

# PLOT EVERY YEAR

expected_dgdp = (
    100 * (fp_gdpper_country - fp_gdpper_country[0, :]) / fp_gdpper_country[0, :]
)
dgdp = 100 * (gdpper_country - fp_gdpper_country[0, :]) / fp_gdpper_country[0, :]

os.makedirs("figures/gdpper_percent_yearly", exist_ok=True)

for iyear in range(nyears):  # nyears):
    fig4, ax4 = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.5))

    pscat1 = ax4[0].scatter(
        expected_dtemp_countries[iyear, :],
        expected_dgdp[iyear, :],
        cmap=gdp_cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        alpha=0.8,
        label=None,
        c=pi_temp_countries_area,
        s=np.sqrt(fp_gdpper_country[0, :]),  # s=np.sqrt(gdp_country[0, :] / 1e7),
    )

    pscat2 = ax4[1].scatter(
        dtemp_countries[iyear, :],
        dgdp[iyear, :],
        cmap=gdp_cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        alpha=0.8,
        label=None,
        c=pi_temp_countries_area,
        s=np.sqrt(fp_gdpper_country[0, :]),  # s=np.sqrt(gdp_country[0, :] / 1e7),
    )
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
                            cmap=gdp_cmap,
                            edgecolors='k',
                            linewidth=0.2,
                            alpha=0.8,
                            label=None,
                            c=pi_temp_countries,
                            s=np.sqrt(gdp_country_1990[c] * 1e3))
    """

    # Add global value:
    pscat3 = ax4[0].scatter(
        expected_pop_temp[iyear],
        100
        * (sum_fp_gdpper_detrended[iyear] - sum_fp_gdpper_detrended[0])
        / sum_fp_gdpper_detrended[0],
        c="black",
        s=np.sqrt(sum_fp_gdpper_detrended[0]),
        alpha=0.8,
    )
    pscat4 = ax4[1].scatter(
        pop_temp[iyear],
        100
        * (sum_gdpper_detrended[iyear] - sum_fp_gdpper_detrended[0])
        / sum_fp_gdpper_detrended[0],
        c="black",
        s=np.sqrt(sum_fp_gdpper_detrended[0]),
        alpha=0.8,
    )

    # Generate color bar to indicate 2000 temperature:
    cbar_ax = fig4.add_axes([0.93, 0.13, 0.02, 0.39])
    cbar = fig4.colorbar(
        pscat1, ticks=color_bins[1:-1], cax=cbar_ax
    )
    cbar.set_label(
        "Temperature (\N{DEGREE SIGN}C)", fontsize=12, rotation=270, labelpad=18
    )
    cbar.ax.tick_params(labelsize=10)

    # Generate legend to indicate GDP size:
    add_bubble_label(fig=fig4, position=[0.92, 0.52, 0.02, 0.4], labels=["100$", "1000$", "10 000$", "100 000$"], label_values=[np.sqrt(1e2), np.sqrt(1e3), np.sqrt(1e4), np.sqrt(1e5)], title="GDP/capita ($)")

    ax4[0].xaxis.set_tick_params(labelsize=12)
    ax4[0].yaxis.set_tick_params(labelsize=12)
    ax4[1].xaxis.set_tick_params(labelsize=12)
    ax4[1].yaxis.set_tick_params(labelsize=12)

    ax4[0].set_xlim(-1.3, 6)
    ax4[0].set_ylim(-50, 60)
    ax4[1].set_xlim(-1.3, 6)
    ax4[1].set_ylim(-50, 60)

    # Add the 0-line:
    ax4[0].axhline(0, color="grey", alpha=0.6, linestyle="--", linewidth=1)
    ax4[1].axhline(0, color="grey", alpha=0.6, linestyle="--", linewidth=1)

    ax4[0].set_xlabel(r"$\Delta$temperature " + "(\N{DEGREE SIGN}C)", fontsize=14)
    ax4[0].set_ylabel(r"$\Delta$GDP/capita (%)", fontsize=14)
    ax4[1].set_xlabel(r"$\Delta$temperature " + "(\N{DEGREE SIGN}C)", fontsize=14)

    ax4[0].set_title("DIAM expectation", fontsize=14)
    ax4[1].set_title("NorESM2-DIAM", fontsize=14)

    fig4.subplots_adjust(left=0.06, right=0.9, top=0.9, bottom=0.1)

    fig4.suptitle("Year {:d}".format(1990 + iyear), x=0.06)

    fig4.savefig(
        "figures/gdpper_percent_yearly/countries_gdpper_percent_year_{:d}.png".format(
            1990 + iyear
        )
    )

    plt.close()

# PLOT ABSOLUTE DIFFERENCE

expected_dgdp = fp_gdpper_country - fp_gdpper_country[0, :]
dgdp = gdpper_country - fp_gdpper_country[0, :]

os.makedirs("figures/gdpper_absolute_yearly", exist_ok=True)

for iyear in range(nyears):  # nyears):
    fig4, ax4 = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.5))

    pscat1 = ax4[0].scatter(
        expected_dtemp_countries[iyear, :],
        expected_dgdp[iyear, :],
        cmap=gdp_cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        alpha=0.8,
        label=None,
        c=pi_temp_countries_area,
        s=np.sqrt(fp_gdpper_country[0, :]),  # s=np.sqrt(gdp_country[0, :] / 1e7),
    )

    pscat2 = ax4[1].scatter(
        dtemp_countries[iyear, :],
        dgdp[iyear, :],
        cmap=gdp_cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        alpha=0.8,
        label=None,
        c=pi_temp_countries_area,
        s=np.sqrt(fp_gdpper_country[0, :]),  # s=np.sqrt(gdp_country[0, :] / 1e7),
    )

    # Add global value:
    pscat3 = ax4[0].scatter(
        expected_pop_temp[iyear],
        sum_fp_gdpper_detrended[iyear] - sum_fp_gdpper_detrended[0],
        c="black",
        s=np.sqrt(sum_fp_gdpper_detrended[0]),
        alpha=0.8,
    )
    pscat4 = ax4[1].scatter(
        pop_temp[iyear],
        sum_gdpper_detrended[iyear] - sum_fp_gdpper_detrended[0],
        c="black",
        s=np.sqrt(sum_fp_gdpper_detrended[0]),
        alpha=0.8,
    )

    # Generate color bar to indicate 2000 temperature:
    cbar_ax = fig4.add_axes([0.93, 0.13, 0.02, 0.39])
    cbar = fig4.colorbar(
        pscat1, ticks=color_bins[1:-1], cax=cbar_ax
    )
    cbar.set_label(
        "Temperature (\N{DEGREE SIGN}C)", fontsize=12, rotation=270, labelpad=18
    )
    cbar.ax.tick_params(labelsize=10)

    # Generate legend to indicate GDP size:
    add_bubble_label(fig=fig4, position=[0.92, 0.52, 0.02, 0.4], labels=["100$", "1000$", "10 000$", "100 000$"], label_values=[np.sqrt(1e2), np.sqrt(1e3), np.sqrt(1e4), np.sqrt(1e5)], title="GDP/capita ($)")

    ax4[0].xaxis.set_tick_params(labelsize=12)
    ax4[0].yaxis.set_tick_params(labelsize=12)
    ax4[1].xaxis.set_tick_params(labelsize=12)
    ax4[1].yaxis.set_tick_params(labelsize=12)

    ax4[0].set_xlim(-1.3, 6)
    ax4[0].set_ylim(-6000, 3000)
    ax4[1].set_xlim(-1.3, 6)
    ax4[1].set_ylim(-6000, 3000)

    # Add the 0-line:
    ax4[0].axhline(0, color="grey", alpha=0.6, linestyle="--", linewidth=1)
    ax4[1].axhline(0, color="grey", alpha=0.6, linestyle="--", linewidth=1)

    ax4[0].set_xlabel(r"$\Delta$temperature " + "(\N{DEGREE SIGN}C)", fontsize=14)
    ax4[0].set_ylabel(r"$\Delta$GDP/capita", fontsize=14)
    ax4[1].set_xlabel(r"$\Delta$temperature " + "(\N{DEGREE SIGN}C)", fontsize=14)

    ax4[0].set_title("DIAM expectation", fontsize=14)
    ax4[1].set_title("NorESM2-DIAM", fontsize=14)

    fig4.subplots_adjust(left=0.06, right=0.9, top=0.9, bottom=0.1)

    fig4.suptitle("Year {:d}".format(1990 + iyear), x=0.06)

    fig4.savefig(
        "figures/gdpper_absolute_yearly/countries_gdpper_absolute_year_{:d}.png".format(
            1990 + iyear
        )
    )

    plt.close()

expected_dgdp = 100 * (fp_gdp_country - fp_gdp_country[0, :]) / fp_gdp_country[0, :]
dgdp = 100 * (gdp_country - fp_gdp_country[0, :]) / fp_gdp_country[0, :]

os.makedirs("figures/gdp_percent_yearly", exist_ok=True)

for iyear in range(nyears):  # nyears):
    fig4, ax4 = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.5))

    pscat1 = ax4[0].scatter(
        expected_dtemp_countries[iyear, :],
        expected_dgdp[iyear, :],
        cmap=gdp_cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        alpha=0.8,
        label=None,
        c=pi_temp_countries_area,
        s=np.sqrt(fp_gdp_country[0, :] / 1e7),
    )

    pscat2 = ax4[1].scatter(
        dtemp_countries[iyear, :],
        dgdp[iyear, :],
        cmap=gdp_cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        alpha=0.8,
        label=None,
        c=pi_temp_countries_area,
        s=np.sqrt(fp_gdp_country[0, :] / 1e7),
    )
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
                            cmap=gdp_cmap,
                            edgecolors='k',
                            linewidth=0.2,
                            alpha=0.8,
                            label=None,
                            c=pi_temp_countries,
                            s=np.sqrt(gdp_country_1990[c] * 1e3))
    """

    # Add global value:
    pscat3 = ax4[0].scatter(
        expected_pop_temp[iyear],
        100
        * (sum_fp_gdp_detrended[iyear] - sum_fp_gdp_detrended[0])
        / sum_fp_gdp_detrended[0],
        c="black",
        s=np.sqrt(sum_fp_gdp_detrended[0]),
        alpha=0.8,
    )
    pscat4 = ax4[1].scatter(
        pop_temp[iyear],
        100
        * (sum_gdp_detrended[iyear] - sum_fp_gdp_detrended[0])
        / sum_fp_gdp_detrended[0],
        c="black",
        s=np.sqrt(sum_fp_gdpper_detrended[0]),
        alpha=0.8,
    )

    # Generate color bar to indicate 2000 temperature:
    cbar_ax = fig4.add_axes([0.93, 0.13, 0.02, 0.39])
    cbar = fig4.colorbar(
        pscat1, ticks=color_bins[1:-1], cax=cbar_ax
    )
    cbar.set_label(
        "Temperature (\N{DEGREE SIGN}C)", fontsize=12, rotation=270, labelpad=18
    )
    cbar.ax.tick_params(labelsize=10)

    # Generate legend to indicate GDP size:
    add_bubble_label(fig=fig4, position=[0.92, 0.52, 0.02, 0.4], labels=["10$^9$", "10$^{10}$", "10$^{11}$", "10$^{12}$", "10$^{13}$"], label_values=[np.sqrt(1e9/1e7), np.sqrt(1e10/1e7), np.sqrt(1e11/1e7), np.sqrt(1e12/1e7), np.sqrt(1e13/1e7)], title="GDP ($)")

    ax4[0].xaxis.set_tick_params(labelsize=12)
    ax4[0].yaxis.set_tick_params(labelsize=12)
    ax4[1].xaxis.set_tick_params(labelsize=12)
    ax4[1].yaxis.set_tick_params(labelsize=12)

    ax4[0].set_xlim(-1.3, 6)
    ax4[0].set_ylim(-50, 60)
    ax4[1].set_xlim(-1.3, 6)
    ax4[1].set_ylim(-50, 60)

    # Add the 0-line:
    ax4[0].axhline(0, color="grey", alpha=0.6, linestyle="--", linewidth=1)
    ax4[1].axhline(0, color="grey", alpha=0.6, linestyle="--", linewidth=1)

    ax4[0].set_xlabel(r"$\Delta$temperature " + "(\N{DEGREE SIGN}C)", fontsize=14)
    ax4[0].set_ylabel(r"$\Delta$GDP (%)", fontsize=14)
    ax4[1].set_xlabel(r"$\Delta$temperature " + "(\N{DEGREE SIGN}C)", fontsize=14)

    ax4[0].set_title("DIAM expectation", fontsize=14)
    ax4[1].set_title("NorESM2-DIAM", fontsize=14)

    fig4.subplots_adjust(left=0.06, right=0.9, top=0.9, bottom=0.1)

    fig4.suptitle("Year {:d}".format(1990 + iyear), x=0.06)

    fig4.savefig(
        "figures/gdp_percent_yearly/countries_gdp_percent_year_{:d}.png".format(
            1990 + iyear
        )
    )

    plt.close()

# --------------------------------------------------------------------------------------

# PLOT EVERY DECADE

polyline = np.linspace(-60, 60, 100)

expected_gdpper_start = np.average(fp_gdpper_country[:10, :], axis=0)
gdpper_start = np.average(gdpper_country[:10, :], axis=0)

expected_gdp_start = np.average(fp_gdp_country[:10, :], axis=0)
gdp_start = np.average(gdp_country[:10, :], axis=0)


os.makedirs("figures/gdpper_percent_decade", exist_ok=True)
os.makedirs("figures/gdpper_percent_population_decade", exist_ok=True)
os.makedirs("figures/gdp_percent_decade", exist_ok=True)
os.makedirs("figures/gdp_percent_population_decade", exist_ok=True)
os.makedirs("figures/gdpper_percent_decade_all", exist_ok=True)
os.makedirs("figures/gdpper_percent_population_decade_all", exist_ok=True)
os.makedirs("figures/gdp_percent_decade_all", exist_ok=True)
os.makedirs("figures/gdp_percent_population_decade_all", exist_ok=True)

ndecades1 = nyears // 10

text_countires = [
    "Norway",
    "United States",
    "Russia",
    "United Kingdom",
    "China",
    "Somalia",
    "South Africa",
    "Germany",
    "Sudan",
    "Canada",
    "New Zealand",
    "Spain",
    "Senegal",
    "Somalia",
    "Argentina",
    "Peru",
    "India",
    "Saudi Arabia",
    "Iraq", "Niger"
]  #'Algeria', 'Indonesia'
text_countires = all_countries

edgecolors = []
for c, country in enumerate(all_countries):
    if country in text_countires:
        edgecolors.append('black')
    else:
        edgecolors.append("none")

for idec in range(1, ndecades1):
    print("Decade:", 1990 + idec * 10, "-", 2000 + idec * 10)

    expected_dtemp_countries_decade = np.average(
        expected_dtemp_countries[10 * idec : 10 * (idec + 1), :], axis=0
    )

    dtemp_countries_decade = np.average(
        dtemp_countries[10 * idec : 10 * (idec + 1), :], axis=0
    )

    dpopulation_country_decade = (
        100
        * (
            np.average(population_countries[10 * idec : 10 * (idec + 1), :], axis=0)
            - population_countries[0, :]
        )
        / population_countries[0, :]
    )
    if idec == ndecades1- 1:
        for c in range(len(all_countries)):
            print(all_countries[c], dpopulation_country_decade[c])

    expected_dgdpper_countries = (
        100
        * (
            np.average(fp_gdpper_country[10 * idec : 10 * (idec + 1), :], axis=0)
            - expected_gdpper_start
        )
        / expected_gdpper_start
    )
    dgdpper_countries = (
        100
        * (
            np.average(gdpper_country[10 * idec : 10 * (idec + 1), :], axis=0)
            - expected_gdpper_start
        )
        / expected_gdpper_start
    )

    expected_dgdp_countries = (
        100
        * (
            np.average(fp_gdp_country[10 * idec : 10 * (idec + 1), :], axis=0)
            - expected_gdp_start
        )
        / expected_gdp_start
    )
    dgdp_countries = (
        100
        * (
            np.average(gdp_country[10 * idec : 10 * (idec + 1), :], axis=0)
            - expected_gdp_start
        )
        / expected_gdp_start
    )

    # Degree 2 polynomial fit or quadratic fit:
    expected_model = np.poly1d(
        np.polyfit(expected_dgdp_countries, expected_dtemp_countries_decade, 2)
    )
    model = np.poly1d(np.polyfit(dgdpper_countries, dtemp_countries_decade, 2))

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))
    fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))
    fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))
    fig5, ax5 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))
    fig6, ax6 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))
    fig7, ax7 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))
    fig8, ax8 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))

    pscat1 = ax1.scatter(
        expected_dtemp_countries_decade,
        expected_dgdpper_countries,
        cmap=gdp_cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors=edgecolors,
        linewidth=0.2,
        alpha=0.8,
        label=None,
        c=start_temp_countries,
        s=np.sqrt(expected_gdpper_start),
    )

    pscat2 = ax2.scatter(
        dtemp_countries_decade,
        dgdpper_countries,
        cmap=gdp_cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors=edgecolors,
        linewidth=0.2,
        alpha=0.8,
        label=None,
        c=start_temp_countries,
        s=np.sqrt(expected_gdpper_start),
    )

    pscat3 = ax3.scatter(
        expected_dtemp_countries_decade,
        expected_dgdpper_countries,
        cmap=population_cmap,
        norm=divnorm,
        edgecolors=edgecolors,
        linewidth=0.2,
        alpha=0.8,
        label=None,
        c=dpopulation_country_decade,
        s=np.sqrt(population_countries[0, :] * 1e3 / 1e3),
    )

    pscat4 = ax4.scatter(
        dtemp_countries_decade,
        dgdpper_countries,
        cmap=population_cmap,
        norm=divnorm,
        edgecolors=edgecolors,
        linewidth=0.2,
        alpha=0.8,
        label=None,
        c=dpopulation_country_decade,
        s=np.sqrt(population_countries[0, :] * 1e3 / 1e3),
    )

    pscat5 = ax5.scatter(
        expected_dtemp_countries_decade,
        expected_dgdp_countries,
        cmap=gdp_cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors=edgecolors,
        linewidth=0.2,
        alpha=0.8,
        label=None,
        c=start_temp_countries,
        s=np.sqrt(expected_gdp_start / 1e7),
    )

    pscat6 = ax6.scatter(
        dtemp_countries_decade,
        dgdp_countries,
        cmap=gdp_cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors=edgecolors,
        linewidth=0.2,
        alpha=0.8,
        label=None,
        c=start_temp_countries,
        s=np.sqrt(expected_gdp_start / 1e7),
    )

    pscat7 = ax7.scatter(
        expected_dtemp_countries_decade,
        expected_dgdp_countries,
        cmap=population_cmap,
        norm=divnorm,
        edgecolors=edgecolors,
        linewidth=0.2,
        alpha=0.8,
        label=None,
        c=dpopulation_country_decade,
        s=np.sqrt(population_countries[0, :] * 1e3 / 1e3),
    )

    pscat8 = ax8.scatter(
        dtemp_countries_decade,
        dgdp_countries,
        cmap=population_cmap,
        norm=divnorm,
        edgecolors=edgecolors,
        linewidth=0.2,
        alpha=0.8,
        label=None,
        c=dpopulation_country_decade,
        s=np.sqrt(population_countries[0, :] * 1e3 / 1e3),
    )

    # Add country names:
    for c, country in enumerate(all_countries):
        if country in text_countires:
            ax1.text(
                expected_dtemp_countries_decade[c],
                expected_dgdpper_countries[c],
                country,
                fontsize=5,
            )
            ax2.text(dtemp_countries_decade[c], dgdpper_countries[c], country, fontsize=5)
            ax3.text(
                expected_dtemp_countries_decade[c],
                expected_dgdpper_countries[c],
                country,
                fontsize=5,
            )
            ax4.text(dtemp_countries_decade[c], dgdpper_countries[c], country, fontsize=5)
            ax5.text(
                    expected_dtemp_countries_decade[c],
                    expected_dgdp_countries[c],
                    country,
                    fontsize=5,
                    )
            ax6.text(dtemp_countries_decade[c], dgdp_countries[c], country, fontsize=5)
            ax7.text(
                    expected_dtemp_countries_decade[c],
                    expected_dgdp_countries[c],
                    country,
                    fontsize=5,
                    )
            ax8.text(dtemp_countries_decade[c], dgdp_countries[c], country, fontsize=5)

    # ax5.plot(expected_model(polyline), polyline)
    # ax6.plot(model(polyline), polyline)

    # Add global value:
    ax1.scatter(
        np.average(expected_pop_temp[10 * idec : 10 * (idec + 1)]),
        100
        * (
            np.average(sum_fp_gdpper_detrended[10 * idec : 10 * (idec + 1)])
            - np.average(sum_fp_gdpper_detrended[:10])
        )
        / np.average(sum_fp_gdpper_detrended[:10]),
        c=color_global,
        linewidth=1,
        edgecolor='black',
        s=np.sqrt(np.average(sum_fp_gdpper_detrended[:10])),
        alpha=0.8,
    )
    ax2.scatter(
        np.average(pop_temp[10 * idec : 10 * (idec + 1)]),
        100
        * (
            np.average(sum_gdpper_detrended[10 * idec : 10 * (idec + 1)])
            - np.average(sum_fp_gdpper_detrended[:10])
        )
        / np.average(sum_fp_gdpper_detrended[:10]),
        c=color_global,
        linewidth=1,
        edgecolor='black',
        s=np.sqrt(np.average(sum_fp_gdpper_detrended[:10])),
        alpha=0.8,
    )
    ax3.scatter(
        np.average(expected_pop_temp[10 * idec : 10 * (idec + 1)]),
        100
        * (
            np.average(sum_fp_gdpper_detrended[10 * idec : 10 * (idec + 1)])
            - np.average(sum_fp_gdpper_detrended[:10])
        )
        / np.average(sum_fp_gdpper_detrended[:10]),
        c='black',
        linewidth=1,
        edgecolor='black',
        s=50,
        alpha=0.8,
    )
    ax4.scatter(
        np.average(pop_temp[10 * idec : 10 * (idec + 1)]),
        100
        * (
            np.average(sum_gdpper_detrended[10 * idec : 10 * (idec + 1)])
            - np.average(sum_fp_gdpper_detrended[:10])
        )
        / np.average(sum_fp_gdpper_detrended[:10]),
        c='black',
        linewidth=1,
        edgecolor='black',
        s=50,
        alpha=0.8,
    )
    ax5.scatter(
        np.average(expected_pop_temp[10 * idec : 10 * (idec + 1)]),
        100
        * (
            np.average(sum_fp_gdp_detrended[10 * idec : 10 * (idec + 1)])
            - np.average(sum_fp_gdp_detrended[:10])
        )
        / np.average(sum_fp_gdp_detrended[:10]),
        c=expected_pop_temp_start,
        linewidth=1,
        edgecolor='black',
        s=50,
        alpha=0.8,
    )
    ax6.scatter(
        np.average(pop_temp[10 * idec : 10 * (idec + 1)]),
        100
        * (
            np.average(sum_gdp_detrended[10 * idec : 10 * (idec + 1)])
            - np.average(sum_fp_gdp_detrended[:10])
        )
        / np.average(sum_fp_gdp_detrended[:10]),
        c=expected_pop_temp_start,
        linewidth=1,
        edgecolor='black',
        s=50,
        alpha=0.8,
    )
    ax7.scatter(
        np.average(expected_pop_temp[10 * idec : 10 * (idec + 1)]),
        100
        * (
            np.average(sum_fp_gdp_detrended[10 * idec : 10 * (idec + 1)])
            - np.average(sum_fp_gdp_detrended[:10])
        )
        / np.average(sum_fp_gdp_detrended[:10]),
        c='black',
        linewidth=1,
        edgecolor='black',
        s=50,
        alpha=0.8,
    )
    ax8.scatter(
        np.average(pop_temp[10 * idec : 10 * (idec + 1)]),
        100
        * (
            np.average(sum_gdp_detrended[10 * idec : 10 * (idec + 1)])
            - np.average(sum_fp_gdp_detrended[:10])
        )
        / np.average(sum_fp_gdp_detrended[:10]),
        c='black',
        linewidth=1,
        edgecolor='black',
        s=50,
        alpha=0.8,
    )

    ax1.text(
        np.average(expected_pop_temp[10 * idec : 10 * (idec + 1)]),
        100
        * (
            np.average(sum_fp_gdpper_detrended[10 * idec : 10 * (idec + 1)])
            - np.average(sum_fp_gdpper_detrended[:10])
        )
        / np.average(sum_fp_gdpper_detrended[:10]),
        "GLOBAL",
        fontsize=5,
    )
    ax2.text(
        np.average(pop_temp[10 * idec : 10 * (idec + 1)]),
        100
        * (
            np.average(sum_gdpper_detrended[10 * idec : 10 * (idec + 1)])
            - np.average(sum_fp_gdpper_detrended[:10])
        )
        / np.average(sum_fp_gdpper_detrended[:10]),
        "GLOBAL",
        fontsize=5,
    )
    ax3.text(
        np.average(expected_pop_temp[10 * idec : 10 * (idec + 1)]),
        100
        * (
            np.average(sum_fp_gdpper_detrended[10 * idec : 10 * (idec + 1)])
            - np.average(sum_fp_gdpper_detrended[:10])
        )
        / np.average(sum_fp_gdpper_detrended[:10]),
        "GLOBAL",
        fontsize=5,
    )
    ax4.text(
        np.average(pop_temp[10 * idec : 10 * (idec + 1)]),
        100
        * (
            np.average(sum_gdpper_detrended[10 * idec : 10 * (idec + 1)])
            - np.average(sum_fp_gdpper_detrended[:10])
        )
        / np.average(sum_fp_gdpper_detrended[:10]),
        "GLOBAL",
        fontsize=5,
    )
    ax5.text(
        np.average(expected_pop_temp[10 * idec : 10 * (idec + 1)]),
        100
        * (
            np.average(sum_fp_gdp_detrended[10 * idec : 10 * (idec + 1)])
            - np.average(sum_fp_gdp_detrended[:10])
        )
        / np.average(sum_fp_gdp_detrended[:10]),
        "GLOBAL",
        fontsize=5,
    )
    ax6.text(
        np.average(pop_temp[10 * idec : 10 * (idec + 1)]),
        100
        * (
            np.average(sum_gdp_detrended[10 * idec : 10 * (idec + 1)])
            - np.average(sum_fp_gdp_detrended[:10])
        )
        / np.average(sum_fp_gdp_detrended[:10]),
        "GLOBAL",
        fontsize=5,
    )
    ax7.text(
        np.average(expected_pop_temp[10 * idec : 10 * (idec + 1)]),
        100
        * (
            np.average(sum_fp_gdp_detrended[10 * idec : 10 * (idec + 1)])
            - np.average(sum_fp_gdp_detrended[:10])
        )
        / np.average(sum_fp_gdp_detrended[:10]),
        "GLOBAL",
        fontsize=5,
    )
    ax8.text(
        np.average(pop_temp[10 * idec : 10 * (idec + 1)]),
        100
        * (
            np.average(sum_gdp_detrended[10 * idec : 10 * (idec + 1)])
            - np.average(sum_fp_gdp_detrended[:10])
        )
        / np.average(sum_fp_gdp_detrended[:10]),
        "GLOBAL",
        fontsize=5,
    )

    for fig, ax in zip([fig1, fig2], [ax1, ax2]):
        # Add the 0-line:
        ax.axhline(0, color="grey", alpha=0.6, linestyle="--", linewidth=1)

        # Generate legend to indicate GDP size:
        add_bubble_label(fig=fig, position=[0.82, 0.52, 0.02, 0.4], labels=["100$", "1000$", "10 000$", "100 000$"], label_values=[np.sqrt(1e2), np.sqrt(1e3), np.sqrt(1e4), np.sqrt(1e5)], title="Initial\nGDP/capita")

        # Generate color bar to indicate 2000 temperature:
        cbar_ax = fig.add_axes([0.88, 0.13, 0.02, 0.39])
        cbar = fig.colorbar(
            pscat1, ticks=color_bins[1:-1], cax=cbar_ax
        )
        cbar.set_label(
            "Initial temperature (\N{DEGREE SIGN}C)",
            fontsize=12,
            rotation=270,
            labelpad=18,
        )
        cbar.ax.tick_params(labelsize=10)

        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)

        ax.set_xlabel(r"$\Delta$temperature " + "(\N{DEGREE SIGN}C)", fontsize=14)
        ax.set_ylabel(r"$\Delta$GDP/capita (%)", fontsize=14)

        ax.set_xlim(-0.5, 7)
        ax.set_ylim(-60, 50)

        fig.subplots_adjust(left=0.11, right=0.83, top=0.95, bottom=0.1)

    for fig, ax in zip([fig3, fig4], [ax3, ax4]):
        # Add the 0-line:
        ax.axhline(0, color="grey", alpha=0.6, linestyle="--", linewidth=1)

        # Generate legend to indicate population size:
        add_bubble_label(fig=fig, position=[0.82, 0.52, 0.02, 0.4], labels=["10$^5$", "10$^{6}$", "10$^{7}$", "10$^{8}$", "10$^{9}$"], label_values=[np.sqrt(1e5/1e3), np.sqrt(1e6/1e3), np.sqrt(1e7/1e3), np.sqrt(1e8/1e3), np.sqrt(1e9/1e3)], title="Initial\npopulation")

        # Generate color bar to indicate 2000 temperature:
        cbar_ax = fig.add_axes([0.88, 0.13, 0.02, 0.39])
        cbar = fig.colorbar(pscat3, cax=cbar_ax)
        cbar.set_label(
            "Population change (%)",
            fontsize=12,
            rotation=270,
            labelpad=18,
        )
        cbar.ax.tick_params(labelsize=10)

        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)

        ax.set_xlabel(r"$\Delta$temperature " + "(\N{DEGREE SIGN}C)", fontsize=14)
        ax.set_ylabel(r"$\Delta$GDP/capita (%)", fontsize=14)

        ax.set_xlim(-0.5, 7)
        ax.set_ylim(-60, 50)

        fig.subplots_adjust(left=0.11, right=0.83, top=0.95, bottom=0.1)

    for fig, ax in zip([fig5, fig6], [ax5, ax6]):
        # Add the 0-line:
        ax.axhline(0, color="grey", alpha=0.6, linestyle="--", linewidth=1)

        # Generate legend to indicate GDP size:
        add_bubble_label(fig=fig, position=[0.86, 0.52, 0.02, 0.4], labels=["10$^9$", "10$^{10}$", "10$^{11}$", "10$^{12}$", "10$^{13}$"], label_values=[np.sqrt(1e9/1e7), np.sqrt(1e10/1e7), np.sqrt(1e11/1e7), np.sqrt(1e12/1e7), np.sqrt(1e13/1e7)], title="Initial\nGDP ($)")

        # Generate color bar to indicate 2000 temperature:
        cbar_ax = fig.add_axes([0.88, 0.13, 0.02, 0.39])
        cbar = fig.colorbar(
            pscat5, ticks=color_bins[1:-1], cax=cbar_ax
        )
        cbar.set_label(
            "Initial temperature (\N{DEGREE SIGN}C)",
            fontsize=12,
            rotation=270,
            labelpad=18,
        )
        cbar.ax.tick_params(labelsize=10)

        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)

        ax.set_xlabel(r"$\Delta$temperature " + "(\N{DEGREE SIGN}C)", fontsize=14)
        ax.set_ylabel(r"$\Delta$GDP (%)", fontsize=14)

        ax.set_xlim(-0.5, 7)
        ax.set_ylim(-100, 2500)

        fig.subplots_adjust(left=0.11, right=0.85, top=0.95, bottom=0.1)

    for fig, ax in zip([fig7, fig8], [ax7, ax8]):
        # Add the 0-line:
        ax.axhline(0, color="grey", alpha=0.6, linestyle="--", linewidth=1)

        # Generate legend to indicate population size:
        add_bubble_label(fig=fig, position=[0.82, 0.52, 0.02, 0.4], labels=["10$^5$", "10$^{6}$", "10$^{7}$", "10$^{8}$", "10$^{9}$"], label_values=[np.sqrt(1e5/1e3), np.sqrt(1e6/1e3), np.sqrt(1e7/1e3), np.sqrt(1e8/1e3), np.sqrt(1e9/1e3)], title="Initial\npopulation")

        # Generate color bar to indicate 2000 temperature:
        cbar_ax = fig.add_axes([0.88, 0.13, 0.02, 0.39])
        cbar = fig.colorbar(pscat7, cax=cbar_ax)
        cbar.set_label(
            "Population change (%)",
            fontsize=12,
            rotation=270,
            labelpad=18,
        )
        cbar.ax.tick_params(labelsize=10)

        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)

        ax.set_xlabel(r"$\Delta$temperature " + "(\N{DEGREE SIGN}C)", fontsize=14)
        ax.set_ylabel(r"$\Delta$GDP (%)", fontsize=14)

        ax.set_xlim(-0.5, 7)
        ax.set_ylim(-100, 2500)

        fig.subplots_adjust(left=0.11, right=0.83, top=0.95, bottom=0.1)

    fig1.savefig(
        "figures/gdpper_percent_decade_all/country_gdpper_percent_expectation_same_axes_{:d}_{:d}.pdf".format(
            1990 + idec * 10, 1999 + idec * 10
        )
    )
    fig1.savefig(
        "figures/gdpper_percent_decade_all/country_gdpper_percent_expectation_same_axes_{:d}_{:d}.png".format(
            1990 + idec * 10, 1999 + idec * 10
        )
    )
    fig2.savefig(
        "figures/gdpper_percent_decade_all/country_gdpper_percent_noresm2-diam_same_axes_{:d}_{:d}.pdf".format(
            1990 + idec * 10, 1999 + idec * 10
        )
    )
    fig2.savefig(
        "figures/gdpper_percent_decade_all/country_gdpper_percent_noresm2-diam_same_axes_{:d}_{:d}.png".format(
            1990 + idec * 10, 1999 + idec * 10
        )
    )

    fig3.savefig(
        "figures/gdpper_percent_population_decade_all/country_gdpper_percent_population_expectation_same_axes_{:d}_{:d}.pdf".format(
            1990 + idec * 10, 1999 + idec * 10
        )
    )
    fig3.savefig(
        "figures/gdpper_percent_population_decade_all/country_gdpper_percent_population_expectation_same_axes_{:d}_{:d}.png".format(
            1990 + idec * 10, 1999 + idec * 10
        )
    )
    fig4.savefig(
        "figures/gdpper_percent_population_decade_all/country_gdpper_percent_population_noresm2-diam_same_axes_{:d}_{:d}.pdf".format(
            1990 + idec * 10, 1999 + idec * 10
        )
    )
    fig4.savefig(
        "figures/gdpper_percent_population_decade_all/country_gdpper_percent_population_noresm2-diam_same_axes_{:d}_{:d}.png".format(
            1990 + idec * 10, 1999 + idec * 10
        )
    )

    fig5.savefig(
        "figures/gdp_percent_decade_all/country_gdp_percent_expectation_same_axes_{:d}_{:d}.pdf".format(
            1990 + idec * 10, 1999 + idec * 10
        )
    )
    fig5.savefig(
        "figures/gdp_percent_decade_all/country_gdp_percent_expectation_same_axes_{:d}_{:d}.png".format(
            1990 + idec * 10, 1999 + idec * 10
        )
    )
    fig6.savefig(
        "figures/gdp_percent_decade_all/country_gdp_percent_noresm2-diam_same_axes_{:d}_{:d}.pdf".format(
            1990 + idec * 10, 1999 + idec * 10
        )
    )
    fig6.savefig(
        "figures/gdp_percent_decade_all/country_gdp_percent_noresm2-diam_same_axes_{:d}_{:d}.png".format(
            1990 + idec * 10, 1999 + idec * 10
        )
    )
    fig7.savefig(
        "figures/gdp_percent_population_decade_all/country_gdp_percent_population_expectation_same_axes_{:d}_{:d}.pdf".format(
            1990 + idec * 10, 1999 + idec * 10
        )
    )
    fig7.savefig(
        "figures/gdp_percent_population_decade_all/country_gdp_percent_population_expectation_same_axes_{:d}_{:d}.png".format(
            1990 + idec * 10, 1999 + idec * 10
        )
    )
    fig8.savefig(
        "figures/gdp_percent_population_decade_all/country_gdp_percent_population_noresm2-diam_same_axes_{:d}_{:d}.pdf".format(
            1990 + idec * 10, 1999 + idec * 10
        )
    )
    fig8.savefig(
        "figures/gdp_percent_population_decade_all/country_gdp_percent_population_noresm2-diam_same_axes_{:d}_{:d}.png".format(
            1990 + idec * 10, 1999 + idec * 10
        )
    )

    plt.close()



# --------------------------------------------------------------------------------------


fig8, ax8 = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.5))


ax8[0].plot(
    years[:-2],
    np.percentile(fp_gdp_detrended[:nyears-1] / population[:nyears-1, :], 75, axis=1)
    / np.percentile(fp_gdp_detrended[:nyears-1] / population[:nyears-1, :], 25, axis=1),
    label="DIAM fixed point",
)
ax8[0].plot(
    years[:-2],
    np.percentile(gdp_detrended[:nyears-1] / population[:nyears-1, :], 75, axis=1)
    / np.percentile(gdp_detrended[:nyears-1] / population[:nyears-1, :], 25, axis=1),
    label="NorESM2-DIAM",
)
ax8[0].set_title("Regional GDP per capita: 75-25 ratio")

ax8[1].plot(
    years[:-2],
    np.percentile(fp_gdp_detrended[:nyears-1] / population[:nyears-1, :], 90, axis=1)
    / np.percentile(fp_gdp_detrended[:nyears-1] / population[:nyears-1, :], 10, axis=1),
    label="DIAM fixed point",
)
ax8[1].plot(
    years[:-2],
    np.percentile(gdp_detrended[:nyears-1] / population[:nyears-1, :], 90, axis=1)
    / np.percentile(gdp_detrended[:nyears-1] / population[:nyears-1, :], 10, axis=1),
    label="NorESM2-DIAM",
)
ax8[1].set_title("Regional GDP per capita: 90-10 ratio")

# ax8[0].set_xlabel('GDP change (%)')
# ax8[1].set_xlabel('GDP change (%)')

ax8[0].legend()

fig8.savefig("figures/gdp_percentiles.pdf")

fig8, ax8 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))

ax8.plot(
    years[:-1],
    np.std(fp_gdp_detrended / population[:nyears, :], axis=1),
    label="DIAM expectation",
)
ax8.plot(
    years[:-1],
    np.std(gdp_detrended / population[:nyears, :], axis=1),
    label="NorESM2-DIAM",
)

ax8.set_xlabel("Year")
ax8.set_ylabel("GDP per capita")
ax8.legend()

fig8.savefig("figures/gdp_per_capita_std.pdf")

exit()

# --------------------------------------------------------------------------------------

a = np.average(expected_temperature1[-10:, :], axis=0) - np.average(
    expected_temperature1[:10, :], axis=0
)
b = np.average(temperature1[-10:, :], axis=0) - np.average(temperature1[:10, :], axis=0)

c = (
    100
    * (
        np.average(expected_gdp_detrended1[-10:, :], axis=0)
        - np.average(expected_gdp_detrended1[:10, :], axis=0)
    )
    / np.average(expected_gdp_detrended1[:10, :], axis=0)
)
d = (
    100
    * (
        np.average(gdp_detrended1[-10:, :], axis=0)
        - np.average(gdp_detrended1[:10, :], axis=0)
    )
    / np.average(gdp_detrended1[:10, :], axis=0)
)

print(a, b, c, d)
"""
fig6, ax6 = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.5))

pscat1 = ax6[0].scatter(a,
                        c,
                        cmap=gdp_cmap,
                        vmin=vmin,
                        vmax=vmax,
                        edgecolors='none',
                        alpha=0.8,
                        label=None,
                        c=np.average(expected_temperature1[:10, :], axis=0))

pscat2 = ax6[1].scatter(b,
                        d,
                        cmap=gdp_cmap,
                        vmin=vmin,
                        vmax=vmax,
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
                     ticks=color_bins[1:-1],
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
# --------------------------------------------------------------------------------------

fig7, ax7 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))

ax7.plot(years[:-1], np.std(expected_temperature1, axis=1), label="DIAM expectation")
ax7.plot(years[:-1], np.std(temperature1, axis=1), label="NorESM2-DIAM")

ax7.set_xlabel("Year")
ax7.set_ylabel("Temperature (degC)")

ax7.legend()

fig7.savefig("std_temperature.pdf")
fig7.set_facecolor("#66cefc")
fig7.savefig("phd_std_temperature.png")

# --------------------------------------------------------------------------------------


temp_bins = np.histogram(np.hstack((a, b)), bins=50)[1]
gdp_bins = np.histogram(np.hstack((c, d)), bins=50)[1]

fig9, ax9 = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.5))

ax9[0].hist(a, bins=temp_bins, alpha=0.5, label="DIAM expectation")
ax9[0].hist(b, bins=temp_bins, alpha=0.5, label="NorESM2-DIAM")
ax9[0].set_title("Temperature", fontsize=14)

ax9[1].hist(c, bins=gdp_bins, alpha=0.5, label="DIAM expectation")
ax9[1].hist(d, bins=gdp_bins, alpha=0.5, label="NorESM2-DIAM")
ax9[1].set_title("GDP", fontsize=14)

ax9[0].set_xlabel(r"$\Delta$temperature " + "(\N{DEGREE SIGN}C)", fontsize=14)
ax9[1].set_xlabel(r"$\Delta$GDP (%)", fontsize=14)

ax9[1].legend()

fig9.savefig("histogram_temp_and_gdp.pdf")
fig9.set_facecolor("#66cefc")
fig9.savefig("phd_histogram_temp_and_gdp.png")

# --------------------------------------------------------------------------------------
