# Write to file cumulative emissions data from ourworldindata.org

# -------------------------------------------------------------------------------------------

# IMPORT MODULES
import numpy as np

# -------------------------------------------------------------------------------------------

# SPECIFY

path = "../data/input_emission_data/"  # CHANGE to local path

years_with_land, cum_emissions_with_land = np.genfromtxt(
    path + "cumulative-co2-including-land.csv",
    delimiter=",",
    skip_header=1,
    usecols=(2, 3),
    unpack=True,
)
years_without_land, cum_emissions_without_land = np.genfromtxt(
    path + "cumulative-co-emissions.csv",
    delimiter=",",
    skip_header=100,
    usecols=(2, 3),
    unpack=True,
)


nyears = years_with_land.shape[0]
years_with_land = np.concatenate((years_with_land, np.array([2024])))

cum_emissions_with_land -= cum_emissions_with_land[0]
cum_emissions_without_land -= cum_emissions_without_land[0]

cum_emissions_with_land = cum_emissions_with_land / 3.664e9
cum_emissions_without_land = cum_emissions_without_land / 3.664e9

with open("cumulative_emissions_from_ourworldindata.txt", "w") as f:
    f.write("Column 1: Year \n")
    f.write(
        "Column 2: Cumulative emissions from 1850 without land use change, not including given year \n"
    )

    for iyear in range(nyears):
        f.write("%16i" % years_with_land[iyear])
        f.write("%16.8f" % cum_emissions_without_land[iyear])
        f.write("\n")

# -------------------------------------------------------------------------------------------
