include("/Users/henricornec/Documents/PreDoc/04_24 Static Problem/io3.jl")
include("/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Output Files/latex_compile.jl")
code_path = "/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Code/"
## These packages help read in .txt files created by io3.jl
using DelimitedFiles
using Formatting

## Mapping Packages
using GMT

ENV["PATH"] *= ":/opt/homebrew/bin"
## Packages to create mp4 files of jpg images
using PrettyTables
using VideoIO
using FileIO
using ImageTransformations
using StatsBase, Plots
using CSV
output_path = "/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Output Files/Paper Plots/"
input_path = "/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Input Files/"
coef_file = "NorESM2_HIST_SSP370_coefficients_and_RMSE_v4.txt"


io = open(input_path * coef_file, "r")
datamatrix = readdlm(io)
close(io)

T_preind = readdlm(open(input_path * "NorESM2_picontrol_regional_temperatures_v4.txt"), skipstart=0)[:, 4]

# γ_1 and γ_2 are temperature sensitivity parameters
γ_1 = datamatrix[:, 4]

γ_2 = datamatrix[:, 5]


# ρ and ϵ are region-specific AR(1) shock parameters
ρ = datamatrix[:, 6] 
ϵ = datamatrix[:, 7]

lat_lon = "/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Input Files/NorESM2_HIST_SSP370_coefficients_and_RMSE_v4.txt"
dm = readdlm(lat_lon, skipstart = 0)
cols = [2,3]
dm1 = dm[:,cols ]
latitude = dm1[:,1] .+ 0.5
longitude = dm1[:,2] .+ 0.5
ncells = 19240

# Figure 1 #
################
# Log GDP 1990 #
################

gdp_expected = readdlm(open("/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Output Files/reg_gdp.txt", "r"), skipstart = 0)[:,2:101]
loggdp_1990 = log.(gdp_expected[:,1])

loggdp_1990_df = [longitude latitude loggdp_1990]
loggdp_1990_grd = xyz2grd(loggdp_1990_df, region=(-180, 180, -60, 75), registration=:p, spacing=(1.0, 1.0))
C = grd2cpt(loggdp_1990_grd, cmap=:roma, nlevels="128+c", inverse = true)
C2 = grd2cpt(loggdp_1990_grd, cmap=:roma, nlevels="8+c", inverse = true)
grdimage(loggdp_1990_grd, proj=:Miller, cmap=C,xaxis=(annot=60, ticks=60), yaxis=(annot=20, ticks=20))
colorbar!(par=(FORMAT_FLOAT_MAP="%.2f",),proj=:Miller,pos=(justify=:CT, size=(7,0.2), offset=(1,0)),region=(-180, 180, -60, 75), equal_size=true,nolines=true,cmap=C2)
colorbar!(par=(FONT_ANNOT=0.1,MAP_TICK_LENGTH=0.01),region=(-180, 180, -60, 75), proj=:Miller, equal_size=true, nolines=true,pos=(justify=:CT, size=(7,0.2), offset=(1,0)),cmap=C)
coast!(water="lightblue", region=(-180, 180, -60, 75), proj=:Miller, savefig=output_path*"loggdp_1990.pdf")

# Figure 2 #
##############
# Population # 
##############
pop_path = "/Users/henricornec/Dropbox/noresm/population_growth/"
pop = readdlm(open(pop_path*"regpop4.pop", "r"))[:,4:end]

ga_from_base = zeros(ncells)
for i in 1:19240
   ga_from_base[i] = (pop[i,111]./pop[i,1] .-1) * 100
end

pop2100 = [longitude latitude ga_from_base]
pop2100_grd = xyz2grd(pop2100, region=(-180, 180, -60, 75), registration=:p, spacing=(1.0, 1.0))
C = grd2cpt(pop2100_grd, cmap=:roma, nlevels="128+c", inverse = true)
C2 = grd2cpt(pop2100_grd, cmap=:roma, nlevels="8+c", inverse = true)

grdimage(pop2100_grd, proj=:Miller, cmap=C,xaxis=(annot=60, ticks=60), yaxis=(annot=20, ticks=20))
colorbar!(par=(FORMAT_FLOAT_MAP="%.0f",),proj=:Miller,pos=(justify=:CT, size=(7,0.2), offset=(1,0)),region=(-180, 180, -60, 75), equal_size=true,nolines=true,cmap=C2)
colorbar!(par=(FONT_ANNOT=0.1,MAP_TICK_LENGTH=0.01),region=(-180, 180, -60, 75), proj=:Miller, equal_size=true, nolines=true,pos=(justify=:CT, size=(7,0.2), offset=(1,0)),cmap=C)
coast!(water="lightblue", region=(-180, 180, -60, 75), proj=:Miller, savefig=output_path*"pop2100_roma.pdf")

# Figure 4 #
#####################
# Productivity 1990 #
#####################
expected_temperature = zeros(ncells, 100)
temperature = zeros(ncells, 100)
productivity = zeros(ncells, 100)

for i in 1:100
    year = 1989 + i
    df = readdlm(open("/Users/henricornec/Dropbox/noresm/Output/full_couple_population/output_year_$year.txt", "r"), skipstart = 16)

    expected_temperature[:,i] .= df[:,3]
    temperature[:,i] .= df[:,4]
end
temperature .= temperature .- expected_temperature[:,1]
expected_temperature .= expected_temperature .- expected_temperature[:,1]

function D(t; α = 0.36) 
    T = 12.609
    d = 0.02
    κ_plus = 0.00362887
    κ_minus =  0.00327721
    if t <= T
        return ((1-d) * exp(-κ_minus*(t-T)^2) + d)^(1/(1-α))
    else return  ((1-d) * exp(-κ_plus*(t-T)^2) + d)^(1/(1-α))
    end
end


T1990 = expected_temperature[:,1]
dmg_1990 = [longitude latitude D_1990]
dmg_1990_grd = xyz2grd(dmg_1990, region=(-180, 180, -60, 75), registration=:p, spacing=(1.0, 1.0))
C = makecpt(cmap=:viridis, range=(0, 1.0), inverse = true)
grdimage(dmg_1990_grd, proj=:Miller, cmap=C,xaxis=(annot=60, ticks=60), yaxis=(annot=20, ticks=20))
coast!(water="lightblue", region=(-180, 180, -60, 75), proj=:Miller, savefig=output_path*"productivity_1990.jpg");


# Figure 7 #
###############################################################
# Map for Regional Temp. Change after 1 ° global temp. change #
###############################################################
cumstock1990 = 216.8650

gamma_1_glob = sum(γ_1 .* proportions)
gamma_2_glob = sum(γ_2 .* proportions)
T_preind_glob = sum(T_preind .* proportions)
using Roots

t_1990 = sum(expected_temperature[:,1] .* proportions) - T_preind_glob
f(x) = gamma_1_glob * x .+ gamma_2_glob * x^2 .- t_1990 .- 1 

root = find_zero(f, (0, 1000))


a = gamma_2_glob
b = gamma_1_glob
c = - (gamma_1_glob * cumstock1990 + gamma_2_glob * cumstock1990^2 + 1)

root = (- b + sqrt(b^2 - 4a*c))/(2a)

reg_warm = γ_1 .* (root-cumstock1990) + γ_2 .* (root^2-cumstock1990^2)

reg_warming = [longitude latitude reg_warm]
reg_warming_grd = xyz2grd(reg_warming, region=(-180, 180, -60, 75), registration=:p, spacing=(1.0, 1.0))
C = grd2cpt(reg_warming_grd, cmap=:plasma, nlevels="128+c")
C2 = grd2cpt(reg_warming_grd, cmap=:plasma, nlevels="8+c")
grdimage(reg_warming_grd, proj=:Miller, cmap=C,xaxis=(annot=60, ticks=60), yaxis=(annot=20, ticks=20))
colorbar!(par=(FORMAT_FLOAT_MAP="%.3f", FONT_ANNOT=15),pos=(justify=:CT, size=(7,0.2), offset=(1,0)),region=(-180, 180, -60, 75), equal_size=true,nolines=true, cmap=C2)
colorbar!(par=(FONT_ANNOT=0.1,MAP_TICK_LENGTH=0.01), pos=(justify=:CT, size=(7,0.2), offset=(1,0)),region=(-180, 180, -60, 75), equal_size=true, nolines=true, cmap=C)
coast!(water="lightblue", region=(-180, 180, -60, 75), proj=:Miller, savefig=output_path*"reg_warming.pdf");

# Figure 11 #
############################
# Side-by-side Temperature #
############################
temp_noshocks_2090 = [longitude latitude expected_temperature[:,100]]
temp_noshocks_2090_grd = xyz2grd(temp_noshocks_2090, region=(-180, 180, -60, 75), registration=:p, spacing=(1.0, 1.0))

temp_noshocks2040 = [longitude latitude expected_temperature[:,50]]
temp_noshocks_2040_grd = xyz2grd(temp_noshocks2040, region=(-180, 180, -60, 75), registration=:p, spacing=(1.0, 1.0))

temp_2090 = [longitude latitude temperature[:,100]]
temp_2090_grd = xyz2grd(temp_2090, region=(-180, 180, -60, 75), registration=:p, spacing=(1.0, 1.0))
temp_2040 = [longitude latitude temperature[:,50]]
temp_2040_grd = xyz2grd(temp_2040, region=(-180, 180, -60, 75), registration=:p, spacing=(1.0, 1.0))

C = makecpt(cmap=:plasma, range=(-2, 10))
grdimage(temp_2090_grd, proj=:Miller, cmap=C,xaxis=(annot=60, ticks=60), yaxis=(annot=20, ticks=20),par=(FONT_ANNOT_PRIMARY=14,))
coast!(water="lightblue", region=(-180, 180, -60, 75), proj=:Miller, savefig=output_path*"temp2090_viridis.jpg");

#  NorESM 2040

grdimage(temp_2040_grd, proj=:Miller, cmap=C,xaxis=(annot=60, ticks=60), yaxis=(annot=20, ticks=20),par=(FONT_ANNOT_PRIMARY=14,))
coast!(water="lightblue", region=(-180, 180, -60, 75), proj=:Miller, savefig=output_path*"temp2040_viridis.jpg");

# Fixed Point 2090

grdimage(temp_noshocks_2090_grd, proj=:Miller, cmap=C,xaxis=(annot=60, ticks=60), yaxis=(annot=20, ticks=20),par=(FONT_ANNOT_PRIMARY=14,))
coast!(water="lightblue", region=(-180, 180, -60, 75), proj=:Miller, savefig=output_path*"temp_noshocks_2090_viridis.jpg");

# Fixed Point 2040
grdimage(temp_noshocks_2040_grd, proj=:Miller, cmap=C,xaxis=(annot=60, ticks=60), yaxis=(annot=20, ticks=20), par=(FONT_ANNOT_PRIMARY=14,))
coast!(water="lightblue", region=(-180, 180, -60, 75), proj=:Miller, savefig=output_path*"temp_noshocks_2040_viridis.jpg");


# Create a figure with just a colorbar
basemap(region=(0,30,0,0.48), frame=(axes=:wsne, annot=:auto, ticks=:auto), par=(:MAP_FRAME_TYPE,:inside))
colorbar!(xlabel="Temperature Change since 1990 (degree C)",position=(anchor=:MR,size=(3, 0.1)), par=(FONT_ANNOT=8, FONT_LABEL=10), ylabel="@.C", cmap=C,savefig = output_path*"colorbar_only.jpg")


using FileIO, ImageMagick, Images

function crop_colorbar_image(input_path::String, output_path::String; top=0, bottom=0, left=1800, right=1)
    img = load(input_path)
    cropped = img[top+1:end-bottom, left+1:end-right]  # rows, cols
    save(output_path, cropped)
end

crop_colorbar_image(output_path*"colorbar_only.jpg", output_path*"colorbar_only_cropped_temp.jpg")


function create_2x2_panel_with_shared_colorbar(img_dir::String, image_names::Vector{String}, titles::Vector{String}, colorbar_path::String, output_tex::String)
    img_dir = abspath(img_dir)
    full_paths = [joinpath(img_dir, img) for img in image_names]

    if length(full_paths) != 4 || length(titles) != 4
        println("Exactly 4 image filenames and 4 titles must be provided.")
        return
    end

    for (i, path) in enumerate(full_paths)
        if !isfile(path)
            println("Missing image: $(image_names[i])")
            return
        end
    end

    if !isfile(colorbar_path)
        println("Missing colorbar image: $colorbar_path")
        return
    end

    header = """
    \\documentclass{beamer}
    \\usepackage{graphicx}
    \\usepackage{caption}
    \\usepackage{adjustbox}
    \\usepackage{geometry}
    \\geometry{paperwidth=160mm,paperheight=100mm}  % Adjust as needed
    \\setbeamertemplate{navigation symbols}{}
    \\begin{document}
    """

    slide = """
    \\begin{frame}[t]
    \\centering
    \\begin{minipage}{0.85\\textwidth}
      \\begin{tabular}{cc}
        \\parbox{0.45\\linewidth}{\\centering \\tiny $(titles[1])\\\\
        \\includegraphics[width=0.95\\linewidth]{$(full_paths[1])}} &
        \\parbox{0.45\\linewidth}{\\centering \\tiny $(titles[2])\\\\
        \\includegraphics[width=0.95\\linewidth]{$(full_paths[2])}} \\\\
        \\parbox{0.45\\linewidth}{\\centering \\tiny $(titles[3])\\\\
        \\includegraphics[width=0.95\\linewidth]{$(full_paths[3])}} &
        \\parbox{0.45\\linewidth}{\\centering \\tiny $(titles[4])\\\\
        \\includegraphics[width=0.95\\linewidth]{$(full_paths[4])}} \\\\
      \\end{tabular}
    \\end{minipage}
    \\begin{minipage}{0.1\\textwidth}
      \\includegraphics[width=0.65\\linewidth]{$(colorbar_path)}
    \\end{minipage}
    \\end{frame}
    """

    footer = "\\end{document}"

    open(output_tex, "w") do f
        write(f, header * slide * footer)
    end

    println("LaTeX file with 2x2 panel and shared colorbar written to $output_tex")
end

function create_1x2_panel_with_colorbar(img_dir::String, image_names::Vector{String}, titles::Vector{String}, colorbar_path::String, output_tex::String)
    img_dir = abspath(img_dir)
    full_paths = [joinpath(img_dir, img) for img in image_names]

    if length(full_paths) != 2 || length(titles) != 2
        println("Exactly 2 image filenames and 2 titles must be provided.")
        return
    end

    if !isfile(colorbar_path)
        println("Missing colorbar image: $colorbar_path")
        return
    end

    header = """
    \\documentclass{beamer}
    \\usepackage{graphicx}
    \\usepackage{caption}
    \\usepackage{adjustbox}
    \\usepackage{geometry}
    \\geometry{paperwidth=100mm,paperheight=140mm}
    \\setbeamertemplate{navigation symbols}{}
    \\begin{document}
    """

    slide = """
    \\begin{frame}[t]
    \\centering
    \\begin{minipage}{0.8\\textwidth}
      \\begin{tabular}{c}
        \\parbox{\\linewidth}{\\centering \\tiny $(titles[1])\\\\
        \\includegraphics[width=0.95\\linewidth]{$(full_paths[1])}} \\\\
        \\parbox{\\linewidth}{\\centering \\tiny $(titles[2])\\\\
        \\includegraphics[width=0.95\\linewidth]{$(full_paths[2])}} \\\\
      \\end{tabular}
    \\end{minipage}
    \\begin{minipage}{0.075\\textwidth}
      \\includegraphics[height=0.4\\textheight]{$(colorbar_path)}
    \\end{minipage}
    \\end{frame}
    """

    footer = "\\end{document}"

    open(output_tex, "w") do f
        write(f, header * slide * footer)
    end

    println("LaTeX file with 1x2 panel and colorbar written to $output_tex")
end

function create_side_by_side_pdf_slide(pdf1::String, pdf2::String, output_tex::String)
    pdf1 = abspath(pdf1)
    pdf2 = abspath(pdf2)

    if !isfile(pdf1)
        println("File not found: $pdf1")
        return
    end
    if !isfile(pdf2)
        println("File not found: $pdf2")
        return
    end

    header = """
    \\documentclass{beamer}
    \\usepackage{graphicx}
    \\usepackage{geometry}
    \\geometry{paperwidth=160mm,paperheight=100mm}
    \\setbeamertemplate{navigation symbols}{}
    \\begin{document}
    """

    slide = """
    \\begin{frame}[t]
    \\centering
    \\includegraphics[width=0.48\\linewidth]{$pdf1}
    \\hspace{1mm}%
    \\includegraphics[width=0.48\\linewidth]{$pdf2}
    \\end{frame}
    """

    footer = "\\end{document}"

    open(output_tex, "w") do f
        write(f, header * slide * footer)
    end

    println("Side-by-side PDF slide written to $output_tex")
end




create_2x2_panel_with_shared_colorbar( "/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Output Files/Paper Plots",
    ["temp_noshocks_2040_viridis.jpg", "temp2040_viridis.jpg", "temp_noshocks_2090_viridis.jpg", "temp2090_viridis.jpg"], 
    ["a)  Standalone Model in 2040", "b) NorESM2-DIAM in 2040", "c) Standalone Model in 2090", "d) NorESM2-DIAM in 2090"], 
    output_path*"colorbar_only_cropped_temp.jpg",
    "temp.tex")
run(`/Library/TeX/texbin/pdflatex temp.tex`)



# Figure 12 #
####################
# Side-by-side GDP #
####################


ai = zeros(ncells, 100)
k = zeros(ncells, 100)
w = zeros(ncells, 100)





df = readdlm(open("/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Output Files/6_21_meeting_files/output_format/output_year_1990.txt", "r"), skipstart = 0)
for i in 1:100
    year = 1989 + i
    df = readdlm(open("/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Output Files/6_21_meeting_files/output_format/output_year_$year.txt", "r"), skipstart = 0)

    ai[:,i] .= df[:,7]
    k[:,i] = df[:,6]
    w[:,i] = df[:,5]
end

δ = 0.06 
gdp_fp = zeros(ncells, 100)
for i in 1:100
    gdp_fp[:,i] = (w[:,i] .- (1-δ)*k[:,i]) .* ai[:,i]./1.015^(i-1)
end

gdp_fp .= gdp_fp * 1e3 

gdp_fp_delta40 = zeros(ncells)
gdp_fp_delta90 = zeros(ncells)
for i in 1:100
    gdp_fp_delta40= 100*(gdp_fp[:,50]./gdp_fp[:,1] .-1)
    gdp_fp_delta90=100*(gdp_fp[:,100]./gdp_fp[:,1] .-1)
end


gdpfp_2090 = [longitude latitude gdp_fp_delta90]
gdpfp_2090_grd = xyz2grd(gdpfp_2090, region=(-180, 180, -60, 75), registration=:p, spacing=(1.0, 1.0))
gdpfp_2040 = [longitude latitude gdp_fp_delta40]
gdpfp_2040_grd = xyz2grd(gdpfp_2040, region=(-180, 180, -60, 75), registration=:p, spacing=(1.0, 1.0))

for i in 1:100
    year = 1989 + i
    df = readdlm(open("/Users/henricornec/Dropbox/noresm/Output/full_couple_population/output_year_$year.txt", "r"), skipstart = 16)

    ai[:,i] .= df[:,9]
    k[:,i] = df[:,7]
    w[:,i] = df[:,6]
end
gdp = zeros(ncells, 100)

for i in 1:100
    gdp[:,i] = (w[:,i] .- (1-δ)*k[:,i]) .* ai[:,i]./1.015^(i-1)
end

gdp .= gdp * 1e3
gdp_delta40 = zeros(ncells)
gdp_delta90 = zeros(ncells)
for i in 1:100
    gdp_delta40= 100*(gdp[:,50]./gdp_fp[:,1] .-1)
    gdp_delta90=100*(gdp[:,100]./gdp_fp[:,1] .-1)
end


gdp_2090 = [longitude latitude gdp_delta90]
gdp_2090_grd = xyz2grd(gdp_2090, region=(-180, 180, -60, 75), registration=:p, spacing=(1.0, 1.0))

gdp_2040 = [longitude latitude gdp_delta40]
gdp_2040_grd = xyz2grd(gdp_2040, region=(-180, 180, -60, 75), registration=:p, spacing=(1.0, 1.0))

C = grd2cpt(gdp_2090_grd, cmap=:roma, nlevels="56+c", inverse=true)
C2 = grd2cpt(gdp_2090_grd, cmap=:roma, nlevels="12+c", inverse=true)
basemap(region=(0,25,0,0.48), par=(:MAP_FRAME_TYPE,:inside))
colorbar!(xlabel="Change in GDP/capita since 1990 in %", frame=true, position=(anchor=:MR,size=(7, 0.1)), par=(FONT_ANNOT=8, FONT_LABEL=8,FORMAT_FLOAT_MAP="%.2f"), equal_size=true,nolines=true,cmap=C2)
colorbar!(equal_size=true,position=(anchor=:MR,size=(7, 0.1)), par=(FONT_ANNOT=0.1,MAP_TICK_LENGTH=0.0001), frame=true, cmap=C,savefig = output_path*"colorbar_only.jpg")
crop_colorbar_image(output_path*"colorbar_only.jpg", output_path*"colorbar_only_cropped.jpg")


#  NorESM 2090
grdimage(gdp_2090_grd, proj=:Miller, cmap=C,xaxis=(annot=60, ticks=60), yaxis=(annot=20, ticks=20))
coast!(water="lightblue", region=(-180, 180, -60, 75), proj=:Miller, savefig=output_path*"gdp2090_viridis.jpg");

#  NorESM 2040
grdimage(gdp_2040_grd, proj=:Miller, cmap=C,xaxis=(annot=60, ticks=60), yaxis=(annot=20, ticks=20))
coast!(water="lightblue", region=(-180, 180, -60, 75), proj=:Miller, savefig=output_path*"gdp2040_viridis.jpg");

# Fixed Point 2090
grdimage(gdpfp_2090_grd, proj=:Miller, cmap=C,xaxis=(annot=60, ticks=60), yaxis=(annot=20, ticks=20))
coast!(water="lightblue", region=(-180, 180, -60, 75), proj=:Miller, savefig=output_path*"gdp_noshocks_2090_viridis.jpg");


# Fixed Point 2040
grdimage(gdpfp_2040_grd, proj=:Miller, cmap=C,xaxis=(annot=60, ticks=60), yaxis=(annot=20, ticks=20))
coast!(water="lightblue", region=(-180, 180, -60, 75), proj=:Miller, savefig=output_path*"gdp_noshocks_2040_viridis.jpg");


create_2x2_panel_with_shared_colorbar("/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Output Files/Paper Plots", ["gdp_noshocks_2040_viridis.jpg", "gdp2040_viridis.jpg", 
"gdp_noshocks_2090_viridis.jpg","gdp2090_viridis.jpg"],
    ["a) Standalone Model in 2040", "b) NorESM-DIAM in 2040", "c) Standalone Model in 2090","d) NorESM-DIAM in 2090"], output_path*"colorbar_only_cropped.jpg", "gdp.tex")
run(`/Library/TeX/texbin/pdflatex gdp.tex`)


# Figure 14 #
######################################
# Standard Deviation of Regional GDP # 
######################################

gdp_expected = readdlm(open("/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Output Files/reg_gdp.txt", "r"), skipstart = 0)[:,2:101]

for i in 1:100
    gdp_expected[:, i] = gdp_expected[:,i]/1.015^(i-1)
end
simul_path = "/Users/henricornec/Dropbox/noresm/endogenous_grid_method/Output Files/simul/"
gdp_standalone = readdlm(open(simul_path*"reg_gdp.txt", "r"), skipstart = 0)[:,2:101]
for i in 1:100
    gdp_standalone[:,i] .= gdp_standalone[:,i] /1.015^(i-1)
end
ai = zeros(ncells, 100)
k = zeros(ncells, 100)
w = zeros(ncells, 100)
for i in 1:100
    year = 1989 + i
    df = readdlm(open("/Users/henricornec/Dropbox/noresm/Output/full_couple_population/output_year_$year.txt", "r"), skipstart = 16)

    ai[:,i] .= df[:,9]
    k[:,i] = df[:,7]
    w[:,i] = df[:,6]
end

δ = 0.06
gdp_noresm = zeros(ncells, 100)
for i in 1:100
    gdp_noresm[:,i] = (w[:,i] .- (1-δ)*k[:,i]) .*pop[:,i] .* ai[:,i]./1.015^(i-1)
end

sd_loggdp = zeros(ncells)

for i in 1:ncells
    sd_loggdp[i] = std(log.(gdp_noresm[i,:]) .- log.(gdp_expected[i,:])) *100
end


sd_loggdp_df = [longitude latitude sd_loggdp]
sd_loggdp_grd = xyz2grd(sd_loggdp_df, region=(-180, 180, -60, 75), registration=:p, spacing=(1.0, 1.0))


C = grd2cpt(sd_loggdp_grd, cmap=:roma, nlevels="128+c", inverse=true)
C2 = grd2cpt(sd_loggdp_grd, cmap=:roma, nlevels="8+c", inverse=true)
grdimage(sd_loggdp_grd, proj=:Miller, cmap=C,xaxis=(annot=60, ticks=60), yaxis=(annot=20, ticks=20))
colorbar!(par=(FORMAT_FLOAT_MAP="%.1f", FONT_ANNOT=15),pos=(justify=:CT, size=(7,0.2), offset=(1,0)),region=(-180, 180, -60, 75),  equal_size=true, nolines=true, frame=(xlabel = "S.D. GDP (%)",),cmap=C2)
colorbar!(par=(FONT_ANNOT=0.1,MAP_TICK_LENGTH=0.01), pos=(justify=:CT, size=(7,0.2), offset=(1,0)), region=(-180, 180, -60, 75), equal_size=true, nolines=true, cmap=C)
coast!(water="lightblue", region=(-180, 180, -60, 75), proj=:Miller, savefig=output_path*"sd_loggdp.pdf");

