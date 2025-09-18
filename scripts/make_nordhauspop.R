rm(list = ls())
library(tidyverse)
library(readxl)
setwd("/Users/henricornec/Dropbox/noresm/population_growth/shares_method")

file_path <- "~/Dropbox/noresm/population_growth/shares_method/nordhaus_v40_population.csv" # Update this with your actual file path

data <- read_csv(file_path)



## This is the original G-ECON Nordhaus dataset where cell populations are divided up between countries
df <- data %>%
  rename("1990" = "pop_1990") %>%
  rename("1995" = "pop_1995") %>%
  rename("2000" = "pop_2000") %>%
  rename("2005" = "pop_2005")

df<- pivot_longer(df, cols = "1990":"2005", names_to = "year", values_to = "population") 


df <- as.data.frame(apply(df,2, function(x) gsub("\\s+", "", x))) %>%
  mutate(country = case_when(
    country == "UnitedStates" ~ "United States",
    country == "UnitedKingdom" ~ "United Kingdom",
    country == "HongKong" ~ "Hong Kong",
    country == "SouthKorea" ~ "South Korea",
    country == "SaudiArabia" ~ "Saudi Arabia",
    country == "SouthAfrica" ~ "South Africa",
    country == "PuertoRico" ~ "Puerto Rico",
    country == "CzechRepublic" ~ "Czech Republic",
    country == "UnitedArabEmirates" ~ "United Arab Emirates",
    country == "NewZealand" ~ "New Zealand",
    country == "ElSalvador" ~ "El Salvador",
    country == "SriLanka" ~ "Sri Lanka",
    country == "TrinidadandTobago" ~ "Trinidad and Tobago",
    country == "NorthKorea" ~ "North Korea",
    country == "DominicanRepublic" ~ "Dominican Republic",
    country == "CostaRica" ~ "Costa Rica",
    country == "Bosnia&Herzegovina" ~ "Bosnia and Herzegovina",
    country == "DemocraticRepublicofCongo" ~ "Democratic Republic of Congo",
    country == "Coted'Ivoire" ~ "Cote d'Ivoire",
    country == "FrenchPolynesia" ~ "French Polynesia",
    country == "Swaziland" ~ "Eswatini",
    country == "Macedonia" ~ "North Macedonia",
    country == "SerbiaandMontenegro" ~ "Serbia and Montenegro", 
    country == "WestBankandGaza" ~ "West Bank and Gaza", 
    country == "NewCaledonia" ~"New Caledonia",
    country == "PapuaNewGuinea" ~ "Papua New Guinea",
    country == "CentralAfricanRepublic" ~ "Central African Republic", 
    country == "BurkinaFaso" ~ "Burkina Faso",
    country == "AntiguaandBarbuda" ~ "Antigua and Barbuda",
    country == "St.Lucia" ~ "Saint Lucia",
    country == "SierraLeone" ~ "Sierra Leone",
    country == "St.VincentandtheGrenadines" ~ "Saint Vincent and the Grenadines",
    country == "St.KittsandNevis" ~ "Saint Kitts and Nevis",
    country == "FrenchGuiana" ~ "French Guiana" ,
    country == "GuineaBissau" ~ "Guinea-Bissau",
    country == "CapeVerde" ~ "Cape Verde",
    country == "EquatorialGuinea" ~ "Equatorial Guinea",
    country == "SaoTomeandPrincipe" ~ "Sao Tome and Principe" ,
    country == "SolomonIslands"  ~ "Solomon Islands", 
    TRUE ~ country  
  )) %>%
  mutate(id = as.numeric(id)) %>%
  mutate(population = as.numeric(population)) %>%
  mutate(lon = as.numeric(lon)) %>%
  mutate(lat = as.numeric(lat))

df1 <- pivot_wider(df, names_from = year, values_from = population)

write_csv(df1, "nord40_gpw_populations.csv")
