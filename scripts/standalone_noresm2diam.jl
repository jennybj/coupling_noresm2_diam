using Optim, Roots
using LinearAlgebra, Statistics
using Interpolations, Plots, ProgressMeter
using DelimitedFiles, CSV, DataFrames
using Formatting, BenchmarkTools
using FastGaussQuadrature
using Random, Distributions
# Helper scripts
include("/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Code/io3.jl")
include("/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Code/spline1.jl")

# File paths
input_path = "/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Input Files/"
#decision_path = "/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Output Files/decision_rules/"

decision_path = "/Users/henricornec/Dropbox/noresm/Decision Rules/new_decision_rules/"
output_path = "/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Output Files/"
coef_file = "NorESM2_HIST_SSP370_coefficients_and_RMSE_v4.txt"
io = open(input_path * coef_file, "r")
datamatrix = readdlm(io)
close(io)


# γ_1 and γ_2 are temperature sensitivity parameters
γ_1 = datamatrix[:, 4]

γ_2 = datamatrix[:, 5]


# ρ and ϵ are region-specific AR(1) shock parameters
ρ = datamatrix[:, 6] 
ϵ = datamatrix[:, 7]

# Load pre-industrial temperatures by region
T_preind = readdlm(open(input_path * "NorESM2_picontrol_regional_temperatures_v4.txt"), skipstart=0)[:, 4]

const chi_horizon = 150 # For t>t_horizon, χ(t) = 0
const gn_horizon = 150 # For t> gn_horizon, gn = 0.0
const startyear = 1990
if chi_horizon > gn_horizon
    simul_time = chi_horizon
else
    simul_time = gn_horizon
end
const N_w = 21    # wealth grid size
const N_z = 21    # shock grid size
const T_horizon = 150 # model simulation length
const ncells = 19240  # number of regions
const ga = 0.015  # TFP growth
const β = 0.985   # discount factor
const δ = 0.06    # depreciation
const α = 0.36    # capital share
const energyshare = 0.062
const rss = (1 + ga) / β - 1
const θ = 1 / (1 + energyshare)
const b = 0.4     # production scaling

# Emission levels for year 1 calibration
const cumstock1990 = 216.8650
const cumstock1991 = 222.9089
escale = 1e3 # energy scaler
yscale = 1e-3 # output scaler

# Load emissions from previous run to cut down on iterations
emis_scaled = readdlm(open(output_path * "emissions.txt", "r"), skipstart=0)[:, 2]

# Energy greening path
η_0001, η_05 = 10, 75
χ(t) = inv(1 + exp(log(0.01 / 0.99) * (t - η_05) / (η_0001 - η_05)))
pctdirty = χ.(creategrid(1, 200, 200)) / χ(1)

countryi = fill("",ncells)
lati = fill(0,ncells)
loni = fill(0,ncells)
areai = fill(0.0,ncells)
rigi = fill(0.0,ncells)
avgtempi = fill(0.0,ncells)
popregi1990 = fill(0.0,ncells)
gdpregi1990 = fill(0.0,ncells)
gdpperregi1990 = fill(0.0,ncells)
d = fill(0.0,ncells)

#filename = "c:/Dropbox/noresm/endogenous_grid_method/Input Files/parse2.gin5"
filen = input_path*"parse2.gin6"
io = open(filen,"r")
for i in 1:ncells
d[i],lati[i],loni[i],countryi[i],areai[i],rigi[i],avgtempi[i],popregi1990[i],
    gdpregi1990[i],gdpperregi1990[i] = readio(io,(3,"b3","a40",6))
end
close(io)

# Population Parameters
io = open(input_path*"regpop4.pop", "r")
datamatrix = readdlm(io, skipstart = 0)
close(io)
popi = datamatrix[:,4:end]

io = open(input_path*"regpop4.grate", "r")
datamatrix = readdlm(io, skipstart = 0)
close(io)
gn_mat = datamatrix[:,4:end]


# Load cumulative historical emissions for calibration
cum_emissions = readdlm(open(input_path * "NorESM2_HIST_SSP370_cumulative_emissions_global_temperature_v3.txt", "r"), skipstart=4)[:, 2]

# Compute annual emissions from cumulative path
orig_emissions = [cum_emissions[i + 1] - cum_emissions[i] for i in 1:(length(cum_emissions) - 1)]
append!(orig_emissions, fill(orig_emissions[end], 40))  # extend with flat tail


# Rescale GDP and compute emissions per dirty energy unit
gdpnetperi = round.(gdpperregi1990 * yscale, digits = 8)
globalgdp1990 = sum(gdpregi1990)
x1990 = escale * orig_emissions[141] / pctdirty[1] 
const p = energyshare * globalgdp1990 / x1990  # price of energy (fixed)

capitali = α * gdpnetperi / (rss + δ)
xi = ((1 - θ) / (θ * p)) * gdpnetperi
ai = (b * (1 - θ) * (capitali .^ (α * θ)) .* (xi .^ (-θ)) / p) .^ (1 / (θ * (α - 1)))

# Steady-state capital (for representative agent = 1)
function calcki(ai)
    return ((((rss + δ) / (α * θ)) * (b^(-1 / θ)) * ((p / (1 - θ))^((1 - θ) / θ))) ^ (1 / (α - 1))) * ai
end
k_ss = calcki(1)

function readrules(i)
    year = string(startyear+i-1)
    rulematrix = CSV.read(decision_path*"decrule_"*year*".csv", DataFrame)
    rulek = zeros(ncells, N_w, N_z)
    for i in 1:ncells
        endrow = N_w*i 
        beginrow = N_w*(i-1) +1
        for j in 1:N_z
            zcol = rulematrix[!, j+3]
            rulek[i, :, j] = zcol[ beginrow:endrow]
        end
        
    end
    return rulek
end

function generate_grids(ρ, ϵ, T; N_w=N_w, N_z=N_z, k_ss=k_ss)
    k_grid = creategrid(0.5 * k_ss, 1.5 * k_ss, N_w)
    stdev = (ϵ == 0) ? 1.0 : sqrt(ϵ^2 / (1 - ρ^2))
    z_grid = creategrid(-3 * stdev, 3 * stdev, N_z)
    w_grid = creategrid(0.5 * G(k_ss, 0, T), 1.5 * G(k_ss, 0, T), N_w)
    return k_grid, z_grid, w_grid
end

# Damage Function
function D(t; α = α) 
    T = 12.609
    d = 0.02
    κ_plus = 0.00362887
    κ_minus =  0.00327721
    if t <= T
        return ((1-d) * exp(-κ_minus*(t-T)^2) + d)^(1/(1-α))
    else return  ((1-d) * exp(-κ_plus*(t-T)^2) + d)^(1/(1-α))
    end
end

d(T1, T2) = D(T2)/D(T1) # damage from year-to-year transition in temperature
d_shock(T, z) = D(T+z) / D(T) # damage from stochastic temperature shock
h(k, z, T) = ((1-θ)*b/p)^(1/θ) * k^α * d_shock(T, z)^(1-α) # energy choice function

# Steady-state energy chocie x_ss. Note that x_ss = x_0 and k_ss = k_0 for the forward simulation
x_ss = h(k_ss, 0, 10.2)

# Production function
F(k, x, z, T) = b * k^(α*θ) * d_shock(T, z)^(θ - α*θ)*x^(1-θ) - p*x


# Wealth function
G(k, z, T) = F(k, h(k, z, T), z, T) + (1-δ)*k

# Partial Derivative of wealth function
partial_G_k(k, z, T) = α  * b * k^(α-1) * d_shock(T, z)^(1-α)*((1-θ)*b/p)^((1-θ)/θ) - p*α*((1-θ)*b/p)^(1/θ)*k^(α-1)*d_shock(T, z)^(1-α)+ (1-δ)

function h_hat(w, z, kprime, w_grid, z_grid)
    interp = interpolate((w_grid, z_grid), kprime, Gridded(Linear()))
    extp = extrapolate(interp, Line())
    val = extp(w, z)
    if val <= 0
        val = 0.001
    end

    return val  # Handles both in and out-of-bounds cases
end

Random.seed!(123)
function draw_shocks(last_z)
    this_z = zeros(ncells)
    for i in 1:ncells
        this_z[i] = ρ[i]*last_z[i] + rand(Normal(0, ϵ[i]^2))
    end
    return this_z
end

emis_scaled = readdlm(open(output_path * "emissions.txt", "r"), skipstart=0)[:, 2]

function simulate_forward(expected_emissions; nyears = T_horizon)

    carbonstock = zeros(nyears+1)
    average_temp = zeros(nyears+1)
    pop_temp = zeros(nyears+1)
    carbonstock[1] = cumstock1990
    proportions = areai .* rigi ./ sum(areai .* rigi)
    pop_proportions = zeros(ncells, nyears+1)

    for i in 1:nyears+1
        pop_proportions[:,i] =  popi[:,i]./ sum(popi[:,i])
    end

    expected_temp = zeros(ncells, nyears+1)
    expected_carbonstock = cumsum(vcat(cumstock1990, expected_emissions))
    emissions = zeros(ncells, nyears)
    total_emissions = zeros(nyears)

    k = fill(k_ss, ncells, nyears+1)
    x = fill(x_ss, ncells, nyears+1)
    e = zeros(ncells, nyears+1)
    w = zeros(ncells, nyears)
    y = zeros(ncells, nyears)

    z = zeros(ncells, nyears)
    reg_gdp = zeros(ncells, nyears)
    reg_gdp_unsc = zeros(ncells, nyears)
    gdp = zeros(nyears)
    gdp_unsc = zeros(nyears)
    reg_temp = zeros(ncells, nyears+1)
    actual_damages = zeros(ncells, nyears+1)

    for i in 1:nyears+1
        expected_temp[:, i] .= T_preind .+ γ_1 .* expected_carbonstock[i] .+ γ_2 .* expected_carbonstock[i]^2
    end
    expected_damages = D.(expected_temp)
    
    # Calculate Productivity Path
    a_i = similar(expected_temp)
    a_i[:, 1] .= ai
    for i in 2:nyears
        a_i[:, i] .= (1 + ga) .* expected_damages[:, i] ./ expected_damages[:, i - 1] .* a_i[:, i - 1]
    end

    @showprogress for i in 1:nyears
        emissions[:, i] .= a_i[:, i] .* popi[:,i] .* x[:, i] .* pctdirty[i] ./ escale # Unscale Emissions
        total_emissions[i] = sum(emissions[:, i])
        carbonstock[i+1] = carbonstock[i] + total_emissions[i]
        if i == 1
            z[:,i] = draw_shocks(zeros(ncells))
        else
            z[:,i] = draw_shocks(z[:,i-1])
        end

        reg_temp[:, i] .= T_preind .+ γ_1 * carbonstock[i] .+ γ_2 .* carbonstock[i]^2 + z[:,i] #Calculate Regional Temeprature
        average_temp[i] = sum(proportions .* reg_temp[:, i]) # Calculate area-weighted average temperature
        pop_temp[i] = sum(pop_proportions[:,i] .* reg_temp[:,i])
        actual_damages[:, i] .= D.(reg_temp[:, i])

         for j in 1:ncells
            T1 = reg_temp[j, i] # Experienced temperature
            y[j, i] = F(k[j, i], h(k[j, i], z[j,i], T1), z[j,i], T1) # Update output
            w[j, i] = G(k[j, i], z[j,i], T1) # Update Wealth
            reg_gdp[j, i] = y[j, i] * a_i[j, i] * popi[j,i] # Unscaled Regional GDP
            reg_gdp_unsc[j, i] = y[j,i] * a_i[j,1]# Scaled Regional GDP
        end

        gdp[i] = sum(reg_gdp[:, i]) # Unscaled Global GDP
        gdp_unsc[i] = sum(reg_gdp_unsc[:, i]) # Scaled Global GDP

        rulek = readrules(i)

        for j in 1:ncells
            T1 = reg_temp[j, i]
            k_grid, z_grid, w_grid = generate_grids(ρ[j], ϵ[j], T1)
            kprime = rulek[j, :, :] # Select correct array for policy function
            k[j, i+1] = h_hat(w[j, i], ρ[j] * z[j,i], kprime, w_grid, z_grid) # Interpolate savings decision
            x[j, i+1] = h(k[j, i+1], ρ[j]*z[j,i], T1) # Expected energy use next year
        end

    end


    return carbonstock, average_temp, pop_temp, total_emissions, emissions, reg_temp, reg_gdp, reg_gdp_unsc, a_i, k, w, x, z
end

carbonstock, average_temp, pop_temp, total_emissions, emissions, reg_temp, reg_gdp, reg_gdp_unsc, a_i, k, w, x, z = simulate_forward(emis_scaled)

simul_path = "/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Output Files/simul/"
io = open(simul_path*"reg_gdp_fp.txt", "w")
writearrays(io, (12 ,18.8), reg_gdp_unsc)
close(io)

io = open(simul_path*"reg_gdp.txt", "w")
writearrays(io, (12 ,18.8), reg_gdp)
close(io)

io = open(simul_path*"carbonstock.txt", "w")
writearrays(io, (18, 15.8), carbonstock)
close(io)

io = open(simul_path*"area_temp.txt", "w")
writearrays(io, (18, 15.8), average_temp)
close(io)

io = open(simul_path*"pop_temp.txt", "w")
writearrays(io, (18, 15.8), pop_temp)
close(io)

io = open(simul_path*"reg_temp.txt", "w")
writearrays(io, (18, 15.8), reg_temp)
close(io)