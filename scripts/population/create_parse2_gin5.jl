using CSV, DataFrames, Plots, DelimitedFiles
include("io3.jl")
include("spline1.jl")
df_04 = CSV.read("/Users/henricornec/Dropbox/noresm/population_growth/nordhaus_update/Gecon40_post_final-2.csv", DataFrame)

select!(df_04, [:COUNTRY, :LAT, :LONGITUDE, :POPGPW_1990_40, :POPGPW_1995_40, :POPGPW_2000_40, :POPGPW_2005_40, :PPP1990_40, :RIG_xi0710, :AREA, :TEMP_NEW]) #, :MER1995_40, :MER2000_40, :MER2005_40])

rename!(df_04, :COUNTRY => :c)
rename!(df_04, :LAT => :lati)
rename!(df_04, :LONGITUDE => :loni)
rename!(df_04, :RIG_xi0710 => :rigi)
rename!(df_04, :AREA => :areai)
rename!(df_04, :TEMP_NEW => :avg_temp)
rename!(df_04, :POPGPW_1990_40 => :pop_1990)
rename!(df_04, :POPGPW_1995_40 => :pop_1995)
rename!(df_04, :POPGPW_2000_40 => :pop_2000)
rename!(df_04, :POPGPW_2005_40 => :pop_2005) 
rename!(df_04, :PPP1990_40 => :gdp_1990)



## Remove GDP that cannot be parsed 
# G-ECON 4.0 loses 6405 regions on parse, 721 on positivity condition
df_04 = filter(row -> !isnothing(tryparse(Float64, row.gdp_1990)), df_04)
df_04.gdp_1990 = parse.(Float64, df_04.gdp_1990)
df_04 = filter(row -> row.gdp_1990 > 0, df_04)

## Remove population that cannot be parsed
# G-ECON 4.0 loses 9 regions on parse
df_04 = dropmissing(df_04, :pop_1990)
df_04 = dropmissing(df_04, :pop_1995)
df_04 = dropmissing(df_04, :pop_2000)
df_04 = dropmissing(df_04, :pop_2005)

df_04.pop_1990 = collect(df_04[:, :pop_1990])/1000
df_04 = filter(row -> row.pop_1990 >= 0.001, df_04)
df_04.pop_1995 = collect(df_04[:, :pop_1995])/1000
df_04.pop_2000 = collect(df_04[:, :pop_2000])/1000
df_04.pop_2005 = collect(df_04[:, :pop_2005])/1000
df_04 = filter(row -> row.pop_1995 >= 0.001, df_04)
df_04 = filter(row -> row.pop_2000 >= 0.001, df_04)
df_04 = filter(row -> row.pop_2005 >= 0.001, df_04)

df_04 = sort(df_04, :gdp_1990, rev=true)



namei = collect(df_04.c)
lati = collect(df_04.lati)
loni = collect(df_04.loni)
areai = collect(df_04.areai)
areai = replace.(areai, "," => "")
areai = parse.(Float64, areai)./1000
areai = round.(areai, digits = 4)
rigi  = collect(df_04.rigi)
rigi = round.(rigi, digits = 3)
avgtempi = collect(df_04.avg_temp)
avgtempi = Float64.(coalesce.(avgtempi, 0.0))
avgtempi = round.(avgtempi, digits = 2)
blank = " "
popi = collect(df_04.pop_1990)

popi95 = collect(df_04.pop_1995)

popi00 = collect(df_04.pop_2000)

popi05 = collect(df_04.pop_2005)


gdpneti = collect(df_04.gdp_1990)

gdpnetperi = gdpneti ./ popi * 1000
gdpneti = round.(gdpneti, digits = 10)
gdpnetperi = round.(gdpnetperi, digits = 10)

popi = round.(popi, digits = 10)
popi95 = round.(popi95, digits = 10)
popi00 = round.(popi00, digits = 10)
popi05 = round.(popi05, digits = 10)


ncells = 19240

io = open("parse2.gin5", "w")
writearrays(io, (8, 5, 20, (2, 5), (9, 15.4)), blank, namei, lati, loni, areai, rigi, avgtempi, popi, popi95, popi00, popi05,gdpneti, gdpnetperi)
close(io)





