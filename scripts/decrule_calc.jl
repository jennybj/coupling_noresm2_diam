# Load distributed computing libraries with SLURM
using Distributed, SlurmClusterManager
addprocs(SlurmManager())

@everywhere begin
    # Base packages
    using Optim, Roots
    using LinearAlgebra, Statistics
    using Interpolations, Plots, ProgressMeter
    using DelimitedFiles, CSV, DataFrames
    using Formatting, BenchmarkTools
    using FastGaussQuadrature

    # Helper scripts
    include("/gpfs/gibbs/project/tsmith/jhc84/Endogrid/Code/io3.jl")
    include("/gpfs/gibbs/project/tsmith/jhc84/Endogrid/Code/spline1.jl")
    
    # File paths
    input_path = "/gpfs/gibbs/project/tsmith/jhc84/Endogrid/Input Files/"
    output_path = "/gpfs/gibbs/project/tsmith/jhc84/Endogrid/Output Files/"
    
    
    
end

# =====================================
# Load Coefficients and Regional Parameters
# =====================================

# Load regression coefficients for temperature
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

@everywhere begin
    # ===============================
    # Model-Wide Constants
    # ===============================
    const chi_horizon = 150 # For t>t_horizon, χ(t) = 0
    const gn_horizon = 150 # For t> gn_horizon, gn = 0.0

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

    # ==================================
    # Load region-specific parameters
    # ==================================
    countryi = fill("",ncells)
    lati = fill(0,ncells)
    loni = fill(0,ncells)
    areai = fill(0.0,ncells)
    rigi = fill(0.0,ncells)
    avgtempi = fill(0.0,ncells)
    popregi1990 = fill(0.0,ncells)
    gdpregi1990 = fill(0.0,ncells)
    gdpperregi1990 = fill(0.0,ncells)
    dump = fill(0.0,ncells)
    
    #filename = "c:/Dropbox/noresm/endogenous_grid_method/Input Files/parse2.gin5"
    filename = input_path*"parse2.gin6"
    io = open(filename,"r")
    for i in 1:ncells
    dump[i],lati[i],loni[i],countryi[i],areai[i],rigi[i],avgtempi[i],popregi1990[i],
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

    # ==================================
    # Functions (see Section 2 and Appendix for additional details)
    # ==================================

    # Helper Function to generate the three grids used in code
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
    
    # ===============================
    # Irregulat Wealth Grid calculation
    # ===============================
    function irreg_w_grid(H, k_grid, T1, T2, nregion, nyear; N_w=N_w, N_z=N_z)
        w = similar(H)
        if nyear >= gn_horizon
            gn = 0.0
        else
            gn = gn_mat[nregion, nyear]
        end
        dT = d(T1, T2)
        for j in 1:N_w, l in 1:N_z
            w[j, l] = β^(-1)*(1 + ga)*dT*inv(H[j, l])+(1 + gn)*(1 + ga)*dT*k_grid[j]
        end
        return w
    end

    # ===============================
    # Regularization of wealth grid 
    # ===============================
    function reg_w_grid(w_irreg_grid, w_grid, k_grid)
        L = size(w_irreg_grid, 2)
        k_prime = similar(w_irreg_grid)
        for l in 1:L
            itp = linear_interpolation(w_irreg_grid[:, l], k_grid, extrapolation_bc=Line())
            k_prime[:, l] .= itp.(w_grid)
        end
        return k_prime
    end


    # ===============================
    # Interpolated policy function ĥ(w, z)
    # ===============================
    function h_hat(w, z, kprime, w_grid, z_grid)
        interp = interpolate((w_grid, z_grid), kprime, Gridded(Linear()))
        extp = extrapolate(interp, Line())
        val = extp(w, z)
        if val <= 0
            val = 0.001
        end

        return val  # Handles both in and out-of-bounds cases
    end

    # ===============================
    # Value function (Lambda)
    # ===============================
    function Λ(k, z, kprime, w_grid, z_grid, T1, T2, nregion, nyear)
        
        if nyear >= gn_horizon
            gn  = 0.0
        else
            gn = gn_mat[nregion, nyear]
        end
        Gval = G(k, z, T1)
        hval = h_hat(Gval, z, kprime, w_grid, z_grid)
        return inv(Gval - (1 + ga) * (1 + gn) * d(T1, T2) * hval)
    end

    # ===============================
    # Expectation update for H
    # ===============================
    function H_update(ρ, ϵ, kprime, k_grid, w_grid, z_grid, T1, T2, nregion, nyear, abscissa, weights; N_w=N_w, N_z=N_z)
        H1 = zeros(N_w, N_z)
        M = length(abscissa)

        if ϵ == 0
            for j in 1:N_w, l in 1:N_z
                z_val = z_grid[l] * ρ
                H1[j, l] = Λ(k_grid[j], z_val, kprime, w_grid, z_grid, T1, T2, nregion, nyear) *
                           partial_G_k(k_grid[j], z_val, T1)
            end
        else
            z_i_m = zeros(M)
            λ_vec = zeros(M)
            for j in 1:N_w, l in 1:N_z
                for m in 1:M
                    z_i_m[m] = ρ * z_grid[l] + sqrt(2) * ϵ * abscissa[m]
                    λ_vec[m] = Λ(k_grid[j], z_i_m[m], kprime, w_grid, z_grid, T1, T2, nregion, nyear)
                end
                G_partials = partial_G_k.(k_grid[j], z_i_m, T1)
                H1[j, l] = π^(-0.5) * sum(weights .* λ_vec .* G_partials)
            end
        end
        return H1
    end

    # ===============================
    # Steady-state routine for backwards iteration starting point
    # ===============================
    function compute_steady_state(M, ρ, ϵ, T, nregion; maxiter=500, toler=1e-8, nyear = 151)
        err = Inf
        niter = 0

        k_grid, z_grid, w_grid = generate_grids(ρ, ϵ, T)
        abscissa, weights = gausshermite(M)
        H0 = ones(N_w, N_z)
        H1 = similar(H0)
        kprime = nothing

        while err > toler && niter < maxiter
            niter += 1

            # Step 1: Compute irregular wealth grid
            irreg_w = irreg_w_grid(H0, k_grid, T, T, nregion, nyear)

            # Step 2: Interpolate to regular grid
            kprime = reg_w_grid(irreg_w, w_grid, k_grid)

            # Step 3: Update H
            H1 .= H_update(ρ, ϵ, kprime, k_grid, w_grid, z_grid, T, T, nregion, nyear, abscissa, weights)

            # Step 4: Compute error and update guess
            err = maximum(abs.(H1 .- H0))
            H0 .= H1
        end

        return H1, kprime
    end


    # ===============================
    # Backward Iteration
    # ===============================
    function iterate_backwards(emissions, M, γ_1, γ_2, ρ, ϵ, T_preind, nregion; nyears=T_horizon)
        abscissa, weights = gausshermite(M)
        k_grid, z_grid, w_grid = generate_grids(ρ, ϵ, T_preind)

        # Compute carbon stock path from emissions
        carbon_stock = cumsum(vcat(cumstock1990, emissions))

        # Calculate regional temperature from carbon stock
        temp_path = T_preind .+ γ_1 .* carbon_stock .+ γ_2 .* carbon_stock.^2
        

        # Allocate output arrays
        H_matrix = zeros(N_w, N_z, nyears+1)
        kprime_matrix = zeros(N_w, N_z, nyears+1)

        # Final period steady state
        H_matrix[:, :, end], kprime_matrix[:, :, end] = compute_steady_state(M, ρ, ϵ, temp_path[end], nregion)

        # Temporary arrays for quadrature
        z_i_m = zeros(M)
        λ_vec = similar(z_i_m)
        G_vals = similar(z_i_m)

        # Backward pass
        for t in nyears:-1:1
            T1, T2 = temp_path[t], temp_path[t+1]

            irreg_w = irreg_w_grid(H_matrix[:, :, t+1], k_grid, T1, T2, nregion, t)
            kprime_matrix[:, :, t] = reg_w_grid(irreg_w, w_grid, k_grid)

            # Quadrature Routin
            for j in 1:N_w, l in 1:N_z
                for m in 1:M
                    z_i_m[m] = ρ * z_grid[l] + sqrt(2) * ϵ * abscissa[m]
                    λ_vec[m] = Λ(k_grid[j], z_i_m[m], kprime_matrix[:, :, t], w_grid, z_grid, T1, T2, nregion, t)
                    G_vals[m] = partial_G_k(k_grid[j], z_i_m[m], T1)
                end
                # Update on H using quadrature
                H_matrix[j, l, t] = π^(-0.5) * sum(weights .* λ_vec .* G_vals)
            end
        end

        return H_matrix, kprime_matrix
    end

    # ===============================
    # Forward Simulation
    # ===============================
    function simulate_forward(kprime_mat, expected_emissions; nyears=T_horizon)
        # Proportions is used to calculate area-weighted average temperature
        proportions = areai .* rigi ./ sum(areai .* rigi)
        pop_proportions = zeros(ncells, nyears+1)

        for i in 1:nyears+1
            pop_proportions[:,i] =  popi[:,i]./ sum(popi[:,i])
        end

        # Pre-allocate large arrays
        carbonstock = zeros(nyears+1)
        average_temp = zeros(nyears+1)
        pop_temp = zeros(nyears+1)
        carbonstock[1] = cumstock1990

        expected_temp = zeros(ncells, nyears+1)
        expected_carbonstock = cumsum(vcat(cumstock1990, expected_emissions))
        emissions = zeros(ncells, nyears)
        total_emissions = zeros(nyears)

        k = fill(k_ss, ncells, nyears+1)
        x = fill(x_ss, ncells, nyears+1)
        e = zeros(ncells, nyears+1)
        w = zeros(ncells, nyears)
        y = zeros(ncells, nyears)

        reg_gdp = zeros(ncells, nyears)
        reg_gdp_unsc = zeros(ncells, nyears)
        gdp = zeros(nyears)
        gdp_unsc = zeros(nyears)
        reg_temp = zeros(ncells, nyears+1)
        actual_damages = zeros(ncells, nyears+1)


        # Calculate Expected Temperature and related damages
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
            carbonstock[i+1] = carbonstock[i] + total_emissions[i] # Update Carbonstock

            reg_temp[:, i] .= T_preind .+ γ_1 * carbonstock[i] .+ γ_2 .* carbonstock[i]^2 #Calculate Regional Temeprature
            average_temp[i] = sum(proportions .* reg_temp[:, i]) # Calculate area-weighted average temperature
            pop_temp[i] = sum(pop_proportions[:,i] .* reg_temp[:,i]) # Calculate population-weighted average temperature
            actual_damages[:, i] .= D.(reg_temp[:, i]) # Calculate damages from experienced (not expected) temperature

            for j in 1:ncells
                T1 = reg_temp[j, i] # Experienced temperature
                z = reg_temp[j, i] - expected_temp[j, i] # Stochastic Shock
                y[j, i] = F(k[j, i], h(k[j, i], z, T1), z, T1) # Update output
                w[j, i] = G(k[j, i], z, T1) # Update Wealth
                reg_gdp[j, i] = y[j, i] * a_i[j, i] * popi[j,i] # Unscaled Regional GDP
                reg_gdp_unsc[j, i] = y[j,i] * a_i[j,1]# Scaled Regional GDP
            end

            gdp[i] = sum(reg_gdp[:, i]) # Unscaled Global GDP
            gdp_unsc[i] = sum(reg_gdp_unsc[:, i]) # Scaled Global GDP

            for j in 1:ncells
                z = reg_temp[j, i] - expected_temp[j, i]
                T1 = reg_temp[j, i]
                k_grid, z_grid, w_grid = generate_grids(ρ[j], ϵ[j], T1)
                kprime = kprime_mat[j, :, :, i] # Select correct array for policy function
                k[j, i+1] = h_hat(w[j, i], z, kprime, w_grid, z_grid) # Interpolate savings decision
                x[j, i+1] = h(k[j, i+1], z, T1) # Expected energy use next year
            end
            reg_temp[:, i+1] .= T_preind .+ γ_1 * carbonstock[i+1] .+ γ_2 * carbonstock[i+1]^2 # Update t+1 regional temperature
            average_temp[i+1] = sum(proportions .* reg_temp[:, i+1]) # Update t+1 global average temperature

        end

        
        return carbonstock, average_temp, pop_temp, total_emissions, emissions, reg_temp, reg_gdp, reg_gdp_unsc, a_i, k, w, x
    end

    # ===============================
    # Helper Function for Euler Error
    # ===============================
    function ϕ_trans(w, z, k_this, k_next, T1, T2, T3, w_grid, z_grid, nyear, nregion)
        k_interp = h_hat(w, z, k_this, w_grid, z_grid)
        Gval = G(k_interp, z, T1)
        if nyear > gn_horizon-1
            gn = 0.0
        else
            gn = gn_mat[nregion, nyear+1]
        end

        h_interp = h_hat(Gval, z, k_next, w_grid, z_grid)
        return inv(Gval - (1 + ga) * (1 + gn) * d(T2, T3) * h_interp) * partial_G_k(k_interp, z, T1)
    end

    # ===============================
    # Euler Error Calculation
    # ===============================
    function transition_euler(kprime_mat, temppath, ϵ, ρ, nregion; M=5, nyears=T_horizon)
        abscissa, weights = gausshermite(M)
        k_grid, z_grid, w_grid = generate_grids(ρ, ϵ, 0.0)

        # Output Arrays
        rel_val = zeros(N_w, N_z, nyears+1)
        max_rel = zeros(nyears+1)
        mean_rel = zeros(nyears+1)
        true_mean_rel = zeros(nyears+1)

        # Quadrature Arrays
        z_i_m = zeros(M)
        phi_vec = zeros(M)

        for t in 1:nyears+1
            k_this = kprime_mat[:, :, t] # Select decision rule
            T1 = temppath[t]
            # Conditional statements for year t2 and beyond (see notes)
            T2 = t < nyears+1 ? temppath[t+1] : T1
            T3 = t < nyears ? temppath[t+2] : T2
            k_next = t < nyears+1 ? kprime_mat[:, :, t+1] : k_this

            if t > gn_horizon
                gn = 0.0
            else
                gn = gn_mat[nregion, t]
            end

            for j in 1:N_w, l in 1:N_z
                w = w_grid[j]; z = z_grid[l]
                # Quadrature Routine
                for m in 1:M
                    z_i_m[m] = ρ * z + sqrt(2) * ϵ * abscissa[m]
                    phi_vec[m] = ϕ_trans(w, z_i_m[m], k_this, k_next, T1, T2, T3, w_grid, z_grid, t, nregion)
                end
                phi = π^(-0.5) * sum(weights .* phi_vec)
                denom = w - (1 + ga) * (1 + gn) * d(T1, T2) * h_hat(w, z, k_this, w_grid, z_grid)
                rel_val[j, l, t] = 1 - ((β^(-1) * (1 + ga) * d(T1, T2) * inv(phi)) / denom) # Relative Euler Error

                if rel_val[j,l,t] > 0.006
                    println("Critical Error:     ", rel_val[j,l,t])
                    println("Region:       ", nregion)
                    println("N_w:       ", j)
                    println("N_z:       ", l)
                    println("Year:       ", t)
                end
            end

            max_rel[t] = maximum(abs.(rel_val[:, :, t])) # Maximum absolute Error
            mean_rel[t] = mean(abs.(rel_val[:, :, t])) # Mean absolute error
            true_mean_rel[t] = mean(rel_val[:, :, t]) # Mean Error
        end

        return max_rel, mean_rel, true_mean_rel
    end

    # ===============================
    # Iterative emissions adjustment
    # ===============================
    function iterate(M, γ_1, γ_2, ρ, ϵ, T_preind, emis_scaled; maxiter=500, toler=1e-8, nyears = T_horizon)
        # Pre-allocate large arrays
        H = zeros(ncells, N_w, N_z, nyears+1)
        kprime = zeros(ncells, N_w, N_z, nyears+1)
        emiss_0 = copy(emis_scaled)
        carbonstock = zeros(nyears+1)
        average_temp = zeros(nyears)
        pop_temp = zeros(nyears)
        total_emissions = zeros(nyears)
        reg_temp = zeros(ncells, nyears)
        emiss_0 = copy(emis_scaled)

        reg_gdp = zeros(ncells, nyears)
        reg_gdp_fp = zeros(ncells, nyears)
        reg_emissions = zeros(ncells, nyears)

        a_i = zeros(ncells, nyears)
        k = zeros(ncells, nyears+1)
        w = zeros(ncells, nyears)
        x = zeros(ncells, nyears+1)

        niter = 0
        err = Inf

        while niter <= maxiter && err > toler
            niter += 1
            # Parallel function call
            res = @showprogress "Iteration $niter" pmap(i -> iterate_backwards(emiss_0, M, γ_1[i], γ_2[i], ρ[i], ϵ[i], T_preind[i], i), 1:ncells)

            # Read parallel function return into usable format
            for i in 1:ncells
                H[i, :, :, :] .= res[i][1]
                kprime[i, :, :, :] .= res[i][2]
            end

            # Simulate forward to calculate new emissions trajectory
            carbonstock, average_temp, total_emissions, reg_emissions, reg_temp, reg_gdp, reg_gdp_fp, a_i, k, w, x = simulate_forward(kprime, emiss_0)
            
            # Calculate error
            err = maximum(abs.(emiss_0 .- total_emissions))
            println("Error: $err")

            # Update guess
            emiss_0 .= total_emissions
        end

        return carbonstock, average_temp, total_emissions, reg_emissions, reg_temp, H, kprime, reg_gdp, reg_gdp_fp, a_i, k, w, x
    end
end

# Fixed Point Iteration
# Note that iterate() can be run with any given emissions path. The first function call (commented out) will take longer to converge.

#carbonstock, average_temp, pop_temp, total_emissions, reg_emissions, reg_temp, H, kprime, reg_gdp, reg_gdp_fp, a_i, k, w, x = iterate(5, γ_1, γ_2, ρ, ϵ, T_preind, orig_emissions)
carbonstock, average_temp, pop_temp, total_emissions, reg_emissions, reg_temp, H, kprime, reg_gdp, reg_gdp_fp, a_i, k, w, x = iterate(5, γ_1, γ_2, ρ, ϵ, T_preind, emis_scaled)

# Write output files

io = open(output_path*"ai.txt", "w")
writearrays(io, (12 ,18.8), a_i)
close(io)

io = open(output_path*"chi.txt", "w")
writearrays(io, (6 ,18.8), pctdirty)
close(io)

io = open(output_path*"capital.txt", "w")
writearrays(io, (12 ,18.8), k)
close(io)

io = open(output_path*"wealth.txt", "w")
writearrays(io, (12 ,18.8), w)
close(io)

io = open(output_path*"reg_gdp.txt", "w")
writearrays(io, (12 ,18.8), reg_gdp)
close(io)

io = open(output_path*"reg_gdp_fp.txt", "w")
writearrays(io, (12 ,18.8), reg_gdp_fp)
close(io)

io = open(output_path*"regional_emissions.txt", "w")
writearrays(io, (12, 15.8), reg_emissions)
close(io)

io = open(output_path*"emissions.txt", "w")
writearrays(io, (8 ,15.8), total_emissions)
close(io)

io = open(output_path*"average_temperature.txt", "w")
writearrays(io, (8, 15.8), average_temp)
close(io)

io = open(output_path*"regional_temperature.txt", "w")
writearrays(io, (12, 15.8), reg_temp)
close(io)

io = open(output_path*"carbonstock.txt", "w")
writearrays(io, (18, 15.8), carbonstock)
close(io)




function write_to_csv(data::Array{Float64, 4}; ncells = ncells, nyears = 150)    
    w_grid = creategrid(0.5 * G(k_ss, 0, 0), 1.5*G(k_ss, 0, 0), N_w)
    rules_output = output_path*"decision_rules/"
    column_names = ["latitude", "longitude", "w"]  # Add new columns
    append!(column_names, ["z$i" for i in 1:N_z])
    # Iterate over the 4th dimension (150 slices)
    for t in 1:nyears
        # Extract the 19240 × 21 × 21 slice for the current `t`
        slice = round.(data[:, :, :, t], digits=8)
        
        # Create a DataFrame to represent the entire structure
        rows = []
        for i in 1:ncells # Iterate over the 20249 matrices
            matrix = slice[i, :, :] # Extract the 21 × 21 matrix
            for row in 1:N_w
                w_value = w_grid[row]
                push!(rows, (lati[i], loni[i], w_value, vec(matrix[row, :])...)) # Add each row of the matrix as a separate row
            end
        end
        # Convert the collected rows to a DataFrame
        df = DataFrame(rows, column_names)
        
        # Generate file name
        file_name = joinpath(rules_output, "decrule_$(t+1989).csv")
        
        # Write the DataFrame to a CSV file
        CSV.write(file_name, df)
    end
end

write_to_csv(kprime)






# Write Error Grid for later calculations
z_grid = zeros(ncells, N_z)
for i in 1:ncells
    stdev = (ϵ[i] == 0) ? 1.0 : sqrt(ϵ[i]^2 / (1 - ρ[i]^2))
    z_grid[i,:] = creategrid(-3 * stdev, 3 * stdev, N_z)
end

io = open(output_path*"z_grid.txt", "w")
writearrays(io, (12 ,15.8), z_grid)
close(io)

# Euler error calculation
res = @showprogress pmap((i->transition_euler(kprime[i,:,:,:], reg_temp[i,:], ϵ[i], ρ[i], i)), 1:ncells)
max_rel_euler = zeros(ncells, 151)
mean_rel_euler = zeros(ncells, 151)
true_mean_euler = zeros(ncells, 151)
for i in 1:ncells
    matrix1, matrix2, matrix3 = res[i]
    max_rel_euler[i,:] = matrix1
    mean_rel_euler[i,:]= matrix2
    true_mean_euler[i,:] = matrix3
end

# Write out Euler Error Files
io = open(output_path*"max_relative_err.txt", "w")
writearrays(io, (12 , 15.8), max_rel_euler)
close(io)

io = open(output_path*"mean_relative_err.txt", "w")
writearrays(io, (12 ,15.8), mean_rel_euler)
close(io)

io = open(output_path*"true_mean_relative_err.txt", "w")
writearrays(io, (12 ,15.8), true_mean_euler)
close(io)

