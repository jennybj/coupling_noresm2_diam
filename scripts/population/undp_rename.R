# This script reads in UNDP population data, renames countries to match the GECON data base and outputs
# national population projections in long format
rm(list = ls())
library(tidyverse)
library(readxl)
setwd("/Users/henricornec/Dropbox/noresm/population_growth/shares_method")

# Read in historical data up to 2023
hist_df <- read_xlsx("undp_pop_growth_2024.xlsx", sheet = 1, skip = 16) %>%
  select("Region, subregion, country or area *", "Year", "Total Population, as of 1 January (thousands)")%>%
  dplyr::rename("country" = "Region, subregion, country or area *") %>%
  dplyr::rename("population" = "Total Population, as of 1 January (thousands)") %>%
  dplyr::rename("year" = "Year") 

# Read in medium variant (most commonly reported) population projections
proj_df <- read_xlsx("undp_pop_growth_2024.xlsx", sheet = 2, skip = 16) %>%
  select("Region, subregion, country or area *", "Year", "Total Population, as of 1 January (thousands)")%>%
  dplyr::rename("country" = "Region, subregion, country or area *") %>%
  dplyr::rename("population" = "Total Population, as of 1 January (thousands)") %>%
  dplyr::rename("year" = "Year") 

df <- rbind(hist_df, proj_df)

# Rename countries to match G-ECON
df <- df %>%
  mutate(country = case_when(
    country == "Russian Federation" ~ "Russia",
    country == "China, Hong Kong SAR" ~ "Hong Kong",
    country == "Republic of Korea" ~ "South Korea",
    country == "Iran (Islamic Republic of)" ~ "Iran",
    country == "China, Taiwan Province of China"  ~ "Taiwan",
    country == "Venezuela (Bolivarian Republic of)"  ~ "Venezuela",
    country == "Türkiye"  ~ "Turkey",
    country == "Brunei Darussalam" ~ "Brunei",
    country == "Viet Nam"  ~ "Vietnam",
    country == "Türkiye"  ~ "Turkey",
    country == "Dem. People's Republic of Korea" ~ "North Korea",
    country == "China, Macao SAR"  ~ "Macau",
    country == "Syrian Arab Republic"  ~ "Syria",
    country == "Réunion" ~ "Reunion",
    country == "Republic of Moldova" ~ "Moldova",
    country == "Kyrgyzstan"~ "Kyrgyztan", 
    country == "Democratic Republic of the Congo" ~ "Democratic Republic of Congo",
    country == "Côte d'Ivoire"~"Cote d'Ivoire",
    country == "United States of America" ~ "United States",
    country == "Bolivia (Plurinational State of)"  ~ "Bolivia",
    country == "State of Palestine" ~ "West Bank and Gaza",
    country == "Lao People's Democratic Republic" ~ "Laos",
    country == "Cabo Verde" ~ "Cape Verde",
    country == "Czechia" ~ "Czech Republic",
    country == "United Republic of Tanzania" ~ "Tanzania",
    TRUE ~ country)) %>%
  filter(year >= 1990) %>% # Only populations after 1990 are of interest for our purposes
  filter(country != "Latin America and the Caribbean") %>%
  filter(country != "Australia/New Zealand")

# Serbia and Montenegro is reported as one country in G-ECON, must be dealt with separately
serbia_montenegro <- df %>%
  filter(country == "Serbia" | country == "Montenegro") %>%
  mutate(year = as.numeric(year)) %>%
  mutate(population = as.numeric(population)) %>%
  group_by(year) %>%
  summarise(population = sum(population)) %>%
  mutate("country" = "Serbia and Montenegro") %>%
  select(country, year, population)




df <- rbind(df, serbia_montenegro)
write.csv(df,"renamed_undp.csv")
  
