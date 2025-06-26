# --------------------------------------------------------------------------------------

# import sys as sys
import glob as glob
import os as os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn.apionly as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm

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

global_population = np.sum(population, axis=1)

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

# --------------------------------------------------------------------------------------

# FUNCTIONS


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
            c="None",
            edgecolor="black",
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


def add_global_value(
    ax, x, y, color, size, cmap, vmin="None", vmax="None", norm="None", text=True
):
    ax.scatter(
        x,
        y,
        cmap=gdp_cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        c=color,
        linewidth=1,
        edgecolor="black",
        s=size,
        alpha=0.8,
    )

    if text == True:
        ax.text(
            x,
            y,
            "GLOBAL",
            fontsize=12,
        )


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

# PLOT

years = np.arange(1990, 1990 + nyears + 1)

fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
bx1 = ax1.twinx()

fig1.subplots_adjust(right=0.85)
linestyles = ["-", "--"]


ax1.plot(
    years[:-1], 100 * diff_emissions / ss_emissions[:nyears], linewidth=2, color="black"
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

# PLOT AVERAGE TEMPERATURE AGAINST TIME

fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(14, 10))

ax2.plot(years[:-1], pop_temp, label="NorESM2-DIAM", linewidth=3, color="darkblue")
ax2.scatter(years[:-1], pop_temp, color="darkblue", s=75)

ax2.plot(
    years[:-1],
    expected_pop_temp,
    label="DIAM expectation",
    linewidth=3,
    color="cornflowerblue",
)

ax2.set_xlabel("Year", fontsize=20)
ax2.set_ylabel("Temperature change (\N{DEGREE SIGN}C)", fontsize=20)
ax2.xaxis.set_tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelsize=16)
ax2.legend(fontsize=20)

fig2.savefig("figures/population_weighted_temperature.pdf")
fig2.savefig("figures/population_weighted_temperature.png")

fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(14, 10))

ax3.plot(
    years[:-1],
    expected_area_temp,
    label="DIAM expectation",
    linewidth=3,
    color="cornflowerblue",
)
ax3.plot(years[:-1], area_temp, label="NorESM2-DIAM", linewidth=3, color="darkblue")

ax3.scatter(years[:-1], area_temp, color="darkblue", s=75)

ax3.set_xlabel("Year", fontsize=20)
ax3.set_ylabel("Temperature change (\N{DEGREE SIGN}C)", fontsize=20)
ax3.xaxis.set_tick_params(labelsize=16)
ax3.yaxis.set_tick_params(labelsize=16)
ax3.legend(fontsize=20)

fig3.savefig("figures/area_weighted_temperature.pdf")

# --------------------------------------------------------------------------------------

# SPECIFY COLOUR MAPS

ncolors = 11
colors = sns.color_palette("YlOrRd", ncolors).as_hex()
vmin = -3
vmax = 30
color_bins = np.linspace(vmin, vmax, ncolors + 1)
index_global_color = np.where(color_bins < expected_pop_temp_start)[0][-1]
color_global = colors[index_global_color]

gdp_cmap = ListedColormap(colors)

# https://matplotlib.org/stable/gallery/color/custom_cmap.html
cdict3 = {
    "red": (
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.5, 0.8, 1.0),
        (0.75, 1.0, 1.0),
        (1.0, 0.4, 1.0),
    ),
    "green": (
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.5, 0.9, 0.9),
        (0.75, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
    "blue": (
        (0.0, 0.0, 0.4),
        (0.25, 1.0, 1.0),
        (0.5, 1.0, 0.8),
        (0.75, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
}

cmap = LinearSegmentedColormap("BlueRed3", cdict3)

population_cmap = cmap  #'RdYlBu'
vmin2 = -100
vmax2 = 1000
divnorm = TwoSlopeNorm(vmin=vmin2, vcenter=0, vmax=vmax2)

# --------------------------------------------------------------------------------------

# PLOT CHANGE OF LAST DECADE

polyline = np.linspace(-60, 60, 100)
ndecades = nyears // 10

print("Decade:", 1990 + (ndecades - 1) * 10, "-", 2000 + (ndecades - 1) * 10)
print(ndecades, 10 * (ndecades - 1), 10 * ndecades)


expected_gdpper_start = np.average(fp_gdpper_country[:10, :], axis=0)
gdpper_start = np.average(gdpper_country[:10, :], axis=0)
expected_dgdpper_countries = (
    100
    * (
        np.average(fp_gdpper_country[10 * (ndecades - 1) : 10 * ndecades, :], axis=0)
        - expected_gdpper_start
    )
    / expected_gdpper_start
)
dgdpper_countries = (
    100
    * (
        np.average(gdpper_country[10 * (ndecades - 1) : 10 * ndecades, :], axis=0)
        - expected_gdpper_start
    )
    / expected_gdpper_start
)

expected_dtemp_countries_decade = np.average(
    expected_dtemp_countries[10 * (ndecades - 1) : 10 * ndecades, :], axis=0
)
dtemp_countries_decade = np.average(
    dtemp_countries[10 * (ndecades - 1) : 10 * ndecades, :], axis=0
)

dpopulation_country_decade = (
    100
    * (
        np.average(population_countries[10 * (ndecades - 1) : 10 * ndecades, :], axis=0)
        - np.average(population_countries[0:10, :], axis=0)
    )
    / np.average(population_countries[0:10, :], axis=0)
)

diff_gdpper = dgdpper_countries - expected_dgdpper_countries
ind = np.argsort(diff_gdpper)
for i in range(len(all_countries)):
    print(all_countries[ind[i]], diff_gdpper[ind[i]])

# Degree 2 polynomial fit or quadratic fit:
expected_model = np.poly1d(
    np.polyfit(expected_dgdpper_countries, expected_dtemp_countries_decade, 2)
)
model = np.poly1d(np.polyfit(dgdpper_countries, dtemp_countries_decade, 2))


text_countires = [
    "Norway",
    "United States",
    "Russia",
    "United Kingdom",
    "China",
    "Somalia",
    "Germany",
    "Sudan",
    "Canada",
    "New Zealand",
    "Spain",
    "Somalia",
    "Brazil",
    "India",
    "Saudi Arabia",
    "Iraq",
    "Niger",
    "Mali",
    "Namibia",
]  #'Algeria', 'Indonesia'
# text_countires = all_countries

edgecolors = []
zorders = []
for c, country in enumerate(all_countries):
    if country in text_countires:
        edgecolors.append("black")
        zorders.append(2)
    else:
        edgecolors.append("none")
        zorders.append(1)


fig4, ax4 = plt.subplots(nrows=2, ncols=2, figsize=(14, 11))

pscat1 = ax4[0, 0].scatter(
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

pscat2 = ax4[0, 1].scatter(
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

pscat3 = ax4[1, 0].scatter(
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

pscat4 = ax4[1, 1].scatter(
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

# Add country names:
for c, country in enumerate(all_countries):
    if country in text_countires:
        ax4[0, 0].text(
            expected_dtemp_countries_decade[c],
            expected_dgdpper_countries[c],
            country,
            fontsize=10,
            zorder=2,
        )
        ax4[0, 1].text(
            dtemp_countries_decade[c],
            dgdpper_countries[c],
            country,
            fontsize=10,
            zorder=2,
        )
        ax4[1, 0].text(
            expected_dtemp_countries_decade[c],
            expected_dgdpper_countries[c],
            country,
            fontsize=10,
            zorder=2,
        )
        ax4[1, 1].text(
            dtemp_countries_decade[c],
            dgdpper_countries[c],
            country,
            fontsize=10,
            zorder=2,
        )
# ax4[0, 0].plot(expected_model(polyline), polyline)
# ax4[0, 1].plot(model(polyline), polyline)

# Add global value:
add_global_value(
    ax=ax4[0, 0],
    x=np.average(expected_pop_temp[10 * (ndecades - 1) : 10 * ndecades]),
    y=100
    * (
        np.average(sum_fp_gdpper_detrended[10 * (ndecades - 1) : 10 * ndecades])
        - np.average(sum_fp_gdpper_detrended[:10])
    )
    / np.average(sum_fp_gdpper_detrended[:10]),
    color=color_global,
    size=np.sqrt(np.average(sum_fp_gdpper_detrended[:10])),
    cmap=gdp_cmap,
    vmin=vmin,
    vmax=vmax,
    text=True,
)
add_global_value(
    ax=ax4[0, 1],
    x=np.average(pop_temp[10 * (ndecades - 1) : 10 * ndecades]),
    y=100
    * (
        np.average(sum_gdpper_detrended[10 * (ndecades - 1) : 10 * ndecades])
        - np.average(sum_fp_gdpper_detrended[:10])
    )
    / np.average(sum_fp_gdpper_detrended[:10]),
    color=color_global,
    size=np.sqrt(np.average(sum_fp_gdpper_detrended[:10])),
    cmap=gdp_cmap,
    vmin=vmin,
    vmax=vmax,
    text=True,
)
add_global_value(
    ax=ax4[1, 0],
    x=np.average(expected_pop_temp[10 * (ndecades - 1) : 10 * ndecades]),
    y=100
    * (
        np.average(sum_fp_gdpper_detrended[10 * (ndecades - 1) : 10 * ndecades])
        - np.average(sum_fp_gdpper_detrended[:10])
    )
    / np.average(sum_fp_gdpper_detrended[:10]),
    color=100
    * (
        np.average(global_population[10 * (ndecades - 1) : 10 * ndecades])
        - np.average(global_population[:10])
    )
    / np.average(global_population[:10]),
    size=50,
    cmap=population_cmap,
    vmin=-200,
    vmax=1000,
    norm=divnorm,
    text=True,
)
add_global_value(
    ax=ax4[1, 1],
    x=np.average(pop_temp[10 * (ndecades - 1) : 10 * ndecades]),
    y=100
    * (
        np.average(sum_gdpper_detrended[10 * (ndecades - 1) : 10 * ndecades])
        - np.average(sum_fp_gdpper_detrended[:10])
    )
    / np.average(sum_fp_gdpper_detrended[:10]),
    color=100
    * (
        np.average(global_population[10 * (ndecades - 1) : 10 * ndecades])
        - np.average(global_population[:10])
    )
    / np.average(global_population[:10]),
    size=50,
    cmap=population_cmap,
    vmin=-100,
    vmax=1000,
    norm=divnorm,
    text=True,
)

# Generate legend to indicate GDP size:
add_bubble_label(
    fig=fig4,
    position=[0.9, 0.55, 0.02, 0.4],
    labels=["100$", "1000$", "10 000$", "100 000$"],
    label_values=[np.sqrt(1e2), np.sqrt(1e3), np.sqrt(1e4), np.sqrt(1e5)],
    title="Initial\nGDP/capita",
)

# Generate legend to indicate population size:
add_bubble_label(
    fig=fig4,
    position=[0.9, 0.08, 0.02, 0.4],
    labels=["10$^5$", "10$^{6}$", "10$^{7}$", "10$^{8}$", "10$^{9}$"],
    label_values=[
        np.sqrt(1e5 / 1e3),
        np.sqrt(1e6 / 1e3),
        np.sqrt(1e7 / 1e3),
        np.sqrt(1e8 / 1e3),
        np.sqrt(1e9 / 1e3),
    ],
    title="Initial\npopulation",
)

# Generate color bar to indicate 2000 temperature:
cbar_ax = fig4.add_axes([0.91, 0.52, 0.02, 0.25])
cbar = fig4.colorbar(pscat1, ticks=color_bins[1:-1], cax=cbar_ax)
cbar.set_label(
    "Initial temperature (\N{DEGREE SIGN}C)",
    fontsize=12,
    rotation=270,
    labelpad=18,
)
cbar.ax.tick_params(labelsize=10)

# Generate color bar to indicate population change:
cbar_ax = fig4.add_axes([0.91, 0.05, 0.02, 0.25])
cbar = fig4.colorbar(pscat3, cax=cbar_ax)
cbar.set_label(
    "Population change (%)",
    fontsize=12,
    rotation=270,
    labelpad=18,
)
cbar.ax.tick_params(labelsize=10)

for ax in ax4.flatten():
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-60, 50)

    # Add the 0-line:
    ax.axhline(0, color="grey", alpha=0.6, linestyle="--", linewidth=1)

ax4[1, 0].set_xlabel(r"$\Delta$temperature " + "(\N{DEGREE SIGN}C)", fontsize=14)
ax4[1, 1].set_xlabel(r"$\Delta$temperature " + "(\N{DEGREE SIGN}C)", fontsize=14)
ax4[0, 0].set_ylabel(r"$\Delta$GDP/capita (%)", fontsize=14)
ax4[1, 0].set_ylabel(r"$\Delta$GDP/capita (%)", fontsize=14)

ax4[0, 0].set_title("DIAM expectation", fontsize=14)
ax4[0, 1].set_title("NorESM2-DIAM", fontsize=14)

fig4.text(0.01, 0.98, "(a)", fontsize=12, wrap=True)
fig4.text(0.47, 0.98, "(b)", fontsize=12, wrap=True)
fig4.text(0.01, 0.48, "(c)", fontsize=12, wrap=True)
fig4.text(0.47, 0.48, "(d)", fontsize=12, wrap=True)

fig4.subplots_adjust(
    left=0.055, right=0.9, top=0.95, bottom=0.05, hspace=0.1, wspace=0.1
)

fig4.savefig("figures/country_gdpper_percent_all_noresm2-diam_2090_2099.pdf")
fig4.savefig("figures/country_gdpper_percent_all_noresm2-diam_2090_2099.png")

plt.close()


# --------------------------------------------------------------------------------------
