---
Contributors:
  - Jenny Bjordal
  - Henri Cornec
  - Anthony A. Smith, Jr.
  - Trude Storelvmo
---

[![DOI](https://zenodo.org/badge/948414289.svg)](https://doi.org/10.5281/zenodo.17176879)


# README and Guidance

## Overview

This repository contains the code necessary to replicate the data presented in Bjordal et al., 2025, and in general, to run the coupled NorESM2-DIAM model. It also contains some of the input data (small size), and otherwise instructions as to where the remaining input data can be found.

## Data Availability and Provenance Statements

<!--
### Statement about Rights
- [x] I certify that the author(s) of the manuscript have legitimate access to and permission to use the data used in this manuscript.
- [x] I certify that the author(s) of the manuscript have documented permission to redistribute/publish the data contained within this replication package. Appropriate permission are documented in the [LICENSE.txt](LICENSE.txt) file.
-->

### License for Data

The data are licensed under a Creative Commons/CC-BY-NC license. See LICENSE.txt for details.

### Summary of Availability

- [x] All data **are** publicly available.
- [ ] Some data **cannot be made** publicly available.
- [ ] **No data can be made** publicly available.

### Details on each Data Source

| Data name                        | File name                  | Location   | Provided | Citation               |
|----------------------------------|----------------------------|------------|----------|------------------------|
| "World Population Prospects 2024"| `undp_pop_growth_2024.xlsx`| `data/raw/`| TRUE | UN DESA, Population Division, 2024 |
| "G-ECON v4.0"                    | `Gecon40_post_final.csv`   | `data/raw/`| TRUE     | Nordhaus et al. 2006 |
| Historical CO2 emissions         | `emissions-cmip6_CO2_anthro_surface_175001-201512_fv_1.9x2.5_c20181011.nc` | `data/input_emission_data/`| TRUE | U.S. National Science Foundation, 2023 |
| SSP1-2.6 CO2 emissions         | `emissions-cmip6_CO2_anthro_surface_ScenarioMIP_IAMC-IMAGE-ssp126_201401-210112_fv_1.9x2.5_c20221115.nc` | `data/input_emission_data/`| TRUE | The NorESM developers group, 2020 |
| SSP2-4.5 CO2 emissions         | `emissions-cmip6_CO2_anthro_surface_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245_201401-210112_fv_1.9x2.5_c20230419.nc` | `data/input_emission_data/`| TRUE | The NorESM developers group, 2020 |
| SSP3-7.0 CO2 emissions         | `emissions-cmip6_CO2_anthro_surface_ScenarioMIP_IAMC-AIM-ssp370_201401-210112_fv_1.9x2.5_c20190207.nc` | `data/input_emission_data/`| TRUE | The NorESM developers group, 2020 |
| SSP5-8.5 CO2 emissions         | `emissions-cmip6_CO2_anthro_surface_ScenarioMIP_IAMC-REMIND-MAGPIE-ssp585_201401-210112_fv_1.9x2.5_c20190207_djlo20200102.nc` | `data/input_emission_data/`| TRUE | The NorESM developers group, 2020 |


#### "World Population Prospects 2024"
Data on historical country-level population levels and future population projections were downloaded from the United Nations Department of Economic and Social Affairs [UN DESA], Population Division, 2024. We use the complete .xlsx file format. Data can be downloaded from [here](https://population.un.org/wpp/assets/Excel\%20Files/1_Indicator\%20(Standard)/EXCEL_FILES/1_General/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_FULL.xlsx). The data are licensed under a CC-BY 3.0.

Note that we have renamed the datafile: `data/raw/undp_pop_growth_2024.xlsx`

#### G-ECON v4.0

The paper uses sub-national economic and population data from the G-ECON v.4.0 database. Available for download [here](https://gecon.yale.edu/data-and-documentation-g-econ-project) under Point 5 in the Data Sets Section.

#### Historical CO2 emissions

The historical CO2 emissions file is available from the CESM input databsae. Accesssing and downloading CESM input data is described in the CESM Quickstart Guide (U.S. National Science Foundation, 2023). A direct download can be found [here](https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/atm/cam/ggas/).) 

#### SSP CO2 emissions

The four SSP CO2 emissions files are avaiable from the NorESM2 input database. Accessing and downloading NorESM2 input data is described in the NorESM2 User’s Guide (The NorESM developers group, 2020). A direct download can be found [here](https://ns16001b.web.sigma2.no/datalake/inputdata/atm/cam/ggas/).


## Dataset list

| Data file                                                        | Source                         | Location | Notes                                                                 |
|------------------------------------------------------------------|--------------------------------|----------|----------------------------------------------------------------------|
| `data/raw/emissions.txt`                                         | Model Output                   | GitHub   | Not required. Reduces iterations for `scripts/decrule_calc.jl`.      |
| `data/raw/undp_pop_growth_2024.xlsx`                             | UN DESA, Population Division, 2024 | GitHub | Historical Estimates and Medium Variant Projections used.             |
| `data/raw/Gecon40_post_final.csv`                                | Nordhaus et al. 2006           | GitHub   |                                                                      |
| `data/input_to_regression/N1850_f19_tn14_20190730esm.TREFHT.nc`  | NorESM2 output                 | GitHub  | PiControl temperature data.                                          |
| `data/input_to_regression/onlyCO2.nc`                            | NorESM2 output                 | GitHub  | HIST and SSP3-7.0 temperature data with only CO2 emissions.          |
| `data/input_to_regression/full_couple_baseline.nc`               | NorESM2 output                 | GitHub  | Temperature data from 1990–2100, only CO2 emissions.                 |
| `data/input_to_regression/full_couple_baseline_e2.nc`            | NorESM2 output                 | GitHub  | Temperature data from 1990–2100, only CO2 emissions.                 |
| `data/input_emission_data/*`                                     | NorESM2 input                  | GitHub  | Standard CO2 emissions input files. For details see The NorESM developers group (2020) |
| `onlyCO2/rest/1990-01-01-00000/*`                                | Bjordal, 2025c                 | Zenodo [DOI](10.5281/zenodo.17856602) | Restart files for running NorESM2 from 1990.                         |
| `data/output_coupling/*`                                         | Output from NorESM2-DIAM       | GitHub  | The output files are compress with 7-zip and need to be unpacked.* |
| `data/output_noresm2/full_couple_population.TREFHT.nc`           | Output from NorESM2            | GitHub  | NorESM2 output from the coupled run.** |

GitHub refer to this repository.

For the NorESM2 restart files, the full simulation from 1850 to 2100, with restart files every 10 years until 1990 and every 5 years thereafter is available from the NIRD RDA [DOI](https://doi.org/10.11582/2025.tdi6hhfl) (Bjordal, 2025b)

*Unpacking the compressed output files e.g. by:
```bash
7z x your_archive.7z
```
**The complete NorESM2 output with all variables are avaiable from the NIRD RDA [DOI](https://doi.org/10.11582/2025.31ney5y8) (Bjordal, 2025a).

## Computational requirements



### Software Requirements

The replication package contains two programs to install the necessary dependencies for Julia and Python. R scripts are installed within the scripts.

**Julia 1.10.4**

Run `setup/packages.jl` to install all necessary Julia packages.

**R 4.3.1**

The two R scripts automatically install 'tidyr' and 'readxl'.

**Python 3.7.3**

To run the Python scripts in `scripts/create_figures/` and `scripts/create_input_files/`, use the setup from `setup/environment.yml`.
The easiest way is to create a new conda environment from the `environment.yml` file.
This can be done in the terminal as follows:

```bash
conda env create -f environment.yml
conda activate base_env
```

The first command need only be run once, while the second activates the conda environment `base_env` (as specified by the file) and must be activated before running the scripts.

**Python 3.10.4 on HPC system**

To run the coupling scripts (which must be done on a HPC system), use the setup from `scripts/running_noresm2diam/environment_coupling.yml`. Move it to the system, and run:

```bash
conda env create -f environment_coupling.yml
```
The environment is activated by the script `couple_iterations.sh


To set up and run NorESM2, see [NorESM GitHub](https://github.com/NorESMhub/NorESM) and [NorESM documentation](https://noresm-docs.readthedocs.io/en/noresm2/).
We have used the version available under the tag `release-noresm2.0.9`.




### Controlled Randomness

- [x] Random seed is set at line 192 of program 'scripts/standalone\_noresm2diam.jl'.
- [x]  No Pseudo random generator is used elsewhere in the analysis described here.

### Memory, Runtime, Storage Requirements

Portions of this code were last run on a 6-core Apple M2-Pro laptop with MacOS version 15.5 with 50GB of free space.

Portions of the code were last run on a 3-node cluster (1x cascadelake, 2x icelake) with a SLURM cluster manager.

NorESM2 (including the coupling scripts) was run on an Atos BullSequana XH2000, using 10 CPU nodes (each with 128 cores and 256 GiB of memory). The machine, named Betzy, is provided by Sigma2 AS, and more details can be found [here](https://documentation.sigma2.no/hpc_machines/betzy.html). With this setup, the coupled model takes approximately one hour per year of simulation.

The rest of the python scripts (not for coupling) can be run on any laptop. Each script can be run in less than 5 minutes, and in total they produce output requiring storage of approximately 90 MB.


## Description of code

### General

- The programs in `scripts/julia_helper/` are auxiliary Julia scripts used in other portions of the code that simplify the workflow in other programs, e.g., by modifying the creation of arrays or creating more readable output files.
- The programs in `scripts/population/` generate a series of output files used to calculate subnational population levels and growth rates.
- The program `scripts/module_coupling.py` reads in various data and performs calculations used by the various Python scripts in `scripts/create_figures/` and `scripts/create_input_files/` as well as by the coupling script `scripts/run_noresm2diam/couple_with_decision_rules.py`.

### Generate input files
#### Generate Population Files
- The program `scripts/create_nordhaus_v40.jl` will create `nordhaus_v40_population.csv` used in `undp_rename.R`  and `make_nordhauspop.R`.
- The program `scripts/undp_rename.R` will create `undp_wide.csv` used in `regpop3.jl` and `regpop4.jl`.
- The programs `scripts/create_parse2_gin5.jl` and `scripts/make_nordhauspop.R` will create `parse2.gin5` and `nord40_gpw_population.csv` respectively. Both are used in in `regpop3.jl` and `regpop4.jl`.
- The program `scripts/regpop3.jl` will create `parse2.gin6` which is used in model calibration.
- The program `scripts/regpop4.jl` will create the regional population numbers and growth rates found in `regpop4.pop` and `regpop4.grate`. Must be run after `regpop3.jl`

#### Generate Emissions Files and Coefficients
- The program `scripts/create_input_files/create_initial_emissions_file.py` creates the emissions file used by NorESM2 in the first year, i.e., year 1990.
- The program `scripts/create_input_files/create_input_files_from_noresm_data.py` creates the input files `NorESM2_picontrol_regional_temperatures.txt`, `NorESM2_HIST_SSP370_cumulative_emissions_global_temperature.txt`, and `NorESM2_HIST_SSP370_coefficients_and_RMSE.txt`.

### Running stand-alone DIAM

- The program `scripts/decrule_calc.jl` will calculate decision rules used in the coupled run as well as generate the output files for a so-called fixed-point run where all shocks \( z_{it} \) are set to 0.
- The program `scripts/standalone_noresm2diam.jl` will initiate the standalone model run reported in the paper and generate a few corresponding output files.

### Running NorESM2-DIAM

- The program `scripts/run_noresm2diam/set_up_noresm_case.py` creates a new NorESM2 case (our simulation) to be used in the coupling.
- The program `scripts/run_noresm2diam/couple_iterations.sh` loads modules and activates the correct conda environment before initializing the coupling script `scripts/run_noresm2diam/couple_with_decision_rules.py`.
- The program `scripts/run_noresm2diam/couple_with_decision_rules.py` couples the two models. It reads in the last year temperature data from NorESM2, uses the decision rules as calculated by `scripts/decrule_calc.jl`, calculates the emissions of next year, and writes the emissions to an input file for NorESM2 to read.
- The program `scripts/run_noresm2diam/calculate_fixed_point_values.py` is not needed for the coupling. It simply calculates the same data as the DIAM standalone (the fixed point) and writes this output to files of the same format as the coupling script, to make future calculations/comparisons easier.

### Calculations and figures
- The program `scripts/create_figures/figure_damage_function.py` produces a figure showing the damage function.
- The program `scripts/create_figures/figure_greening_function.py` produces a figure showing the greening function.
- The program `scripts/create_figures/figure_compare_cumulative_emissions.py` reads in emissions from the CMIP6 scenarios, the Shared Socioeconomic Pathways (SSPs), as well as the emissions from the DIAM stand-alone model, calculates the cumulative emissions since 1850, and produces a figure showing these cumulative emission paths.
- The program `scripts/create_figures/figures_model_output.py` reads in the data produced by the coupled model, performs calculations—at grid cell, country, and global level—and produces figures.
- The program `scripts/create_figures/make_figures.jl` reads in output produced by the coupled model and the standalone model and creates the map figures in the paper.


### License for Code

The code is licensed under a MIT license. See [LICENSE.md](LICENSE.md) for details.


## Instructions to Replicators

When running the coupled NorESM2-DIAM we have to set up a NorESM2 case (detailed below), which is the simulation we run with NorESM2. This simulation needs a name, hereafter know as the `CASENAME`, which will need to be specified several places. In the code, this should be indicated by `# CHANGE`.

### Setup
- Before running any program in the replication package, make sure to edit the file paths provided in all the scripts. For the python scripts, the file paths are followed by the comment `# CHANGE so that you can search through the code before running it.
- Run the two programs in  `setup/` once on a new system to set up the
  working environment. Details provided above under [Software Requirements](#software-requirements).
- Download the data files referenced above and double-check that files are in the correct directories as specified by your file paths.
- Before running any of the python script, make sure that the conda environment is activated:
  ```bash
  conda activate base_env
  ```

### Generate input files

- Refer to the "Generate Population Input Files" Section and run the programs in the order described there. 
- Run `scripts/create_input_files/create_initial_emissions_file.py` to create the necessary emissions input file for NorESM2. Make sure to change `case_name` to the wanted `CASENAME`, so that the name of the file is `input_emissions_CASENAME.py`.
```bash
python create_initial_emissions.py
```
Run `scripts/create_input_files/create_input_files_from_noresm_data.py to create input files need by both DIAM standalone and the coupled NorESM2-DIAM.
```bash
python create_input_files_from_noresm_data.py
```

### Running stand-alone DIAM

- Run `scripts/decrule_calc.jl` to write decision rule files and fixed-point output.
- Run `scripts/standalone_noresm2diam.jl` to simulate the stand-alone model and create appropriate output files.

### Running NorESM2-DIAM

NorESM2 needs to be run on an HPC system.

- First, you need to download and set up the NorESM2 model code. This is described here: [NorESM2 Access Guide](https://noresm-docs.readthedocs.io/en/noresm2/access/access.html). For challenges with downloading and running NorESM2 in general, we refer to the *NorESM developers group*. For general NorESM2 input data (not specific to the coupling), we also refer to this group and their [User's guide](https://noresm-docs.readthedocs.io/en/noresm2/index.html).
  Note that setting up NorESM2 could potentially be challenging and might require help from the people that run the HPC system you use.
  It is a good idea to check if you manage to run a standard NorESM2 simulation (a few days or months) before trying the coupled version. The coupled version isn’t necessarily harder to run, but starting with a standard simulation can make it easier to troubleshoot any issues that come up later.

- Next, make sure that you copy all the needed input data for the coupling:

  - Restart data for the NorESM2 case must be downloaded from [DOI](https://doi.org/10.11582/2025.tdi6hhfl), from the folder `onlyCO2/rest/1990-01-01-00000/` and placed in a folder as specified in `set_up_noresm_case.py` as `restart_dir` or copied directly into the NorESM2 case's run folder.
  - The scripts needed for the coupling—`module_coupling.py`, `couple_with_decision_rules.py`, and `couple_iterations.sh`—must be placed in a folder as specified in `set_up_noresm_case.py` as `input_dir` or copied directly into the NorESM2 case's folder.
  - The decision rules, as created by `decrule_calc.jl`, must be placed in a folder as specified in `couple_with_decision_rules.py` as `dr_path`.
  - The emissions calculated from standalone DIAM—`emissions.txt`—must be placed in a folder as specified in `couple_with_decision_rules.py` as `file_path`. This is also where the output from the coupled run will be placed.
  - The initial emission files for NorESM2—`input_emissions_CASENAME.py`—must be placed in a folder as specified in `user_nl_cam` as `co2flux_fuel_file`. This is set both in `module_coupling.py` and `set_up_noresm_case.py`, so make sure these are the same.
  - The input files `NorESM2_picontrol_regional_temperatures.txt`, `NorESM2_HIST_SSP370_cumulative_emissions_global_temperature.txt`, `NorESM2_HIST_SSP370_coefficients_and_RMSE.txt`, and `parse2.gin6` must be placed in a folder as specified in `module_coupling.py` as `file_path`. These can be in the same folder as above, but it’s not required.

- Set up a NorESM2 case by running `set_up_noresm_case.py`:

```bash
python set_up_noresm_case.py CASENAME
```
This script is by no means fool proof, and might not work on your specific HPC system. (At least not without significant changes.) If not, follow the steps in the script and do them manually in the terminal. You can also see the [User's guide](https://noresm-docs.readthedocs.io/en/noresm2/index.html) for details on how to set up a case if you find the script confusing.
- Make sure that all the scripts and input files are in the correct folders!
- Start the coupled run:
```bash
cd /path/to/case_dir/CASENAME
./case.submit
```
The run is started from the case folder, which is the `case\_dir` you specified in `set_up_noresm_case.py` followed by the `CASENAME`.

### Calculations and figures

- Run the programs in `scripts/create_figures/` to create figures 3, 5, 6, 8, 9, 10, and 13 in the paper. (These also calculate the data presented in the figures.)

```bash
python figure***.py
julia make_figures.jl
```

### Details for selected scripts
- `scripts/decrule\_calc.jl`: Calculates decision rules and writes them to a .csv format. Note that the main function `iterate()` is called twice in the script. Once using the converged emissions path and once using the SSP 3-7.0 Emissions Pathway. Both will converge to the same fixed point, but the latter requires significantly more iterations.


### Details



## List of tables and programs


The provided code reproduces:


| Figure/Table #    | Program                  | Line Numbers | Output file                      | 
|-------------------|--------------------------|-------------|----------------------------------|
| Fig. 1 | `scripts/create_figures/make_figures.jl`| 53-63 | `figures/loggdp_1990.pdf` | 
| Fig. 2 |`scripts/create_figures/make_figures.jl`  | 69-85 | `figures/pop2100_roma.pdf` | 
| Fig. 3 | `scripts/create_figures/figure_damage_function.py`| 40-52 | `figures/figure_damage_function.pdf` | 
| Fig. 4 |`scripts/create_figures/make_figures.jl`  | 91-122 |`figures/productivity_1990.pdf`| 
| Fig. 5 | `scripts/create_figures/figure_greening_function.py`| 29-39 | `figures/figure_greening_function.pdf` | 
| Fig. 6 | `scripts/create_figures/figure_temperature_regression.py`| 202-217 | `figures/figure_temperature_regression.pdf` | 
| Fig. 7 |`scripts/create_figures/make_figures.jl` | 91-122 |`figures/reg_warming.pdf` | 
| Fig. 8 | `scripts/create_figures/figure_compare_cumulative_emissions.py`| 324-381 | `figures/figure_compare_cumulative_emissions.pdf` | 
| Fig. 9 | `scripts/create_figures/figures_model_output.py`| 419-446 | `figures/difference_emissions.pdf` | 
| Fig. 10 | `scripts/create_figures/figures_model_output.py`| 455-543 | `figures/population_weighted_temperature_and_GDP_per_capita.pdf` | 
| Fig. 11 | `scripts/create_figures/make_figures.jl` | 163-370 | `figures/temp.pdf`|
| Fig. 12 | `scripts/create_figures/make_figures.jl` |380-477| `figures/gdp.pdf`| 
| Fig. 13 | `scripts/create_figures/figures_model_output.py`| 952-1145 | `figures/country_gdpper_percent_all_noresm2-diam_2090_2099.pdf` | 
| Fig. 14 | `scripts/create_figures/make_figures.jl` |485-529|`figures/sd_loggdp.pdf` | 

## References

Bjordal, Smith, Cornec, and Storelvmo (2025). ***NorESM2–DIAM: A coupled model for investigating global and regional climate–economy interactions***. [Manuscript submitted for publication].

Bjordal, J. (2025). ***NorESM2-DIAM prototype simulation, NorESM2 standard output*** [Data set]. NIRD RDA. [DOI](https://doi.org/10.11582/2025.31ney5y8).

Bjordal, J. (2025). ***NorESM2-LME Historical and SSP3-7.0 with only CO2 emissions*** [Data set]. NIRD RDA. [DOI](https://doi.org/10.11582/2025.tdi6hhfl).

Bjordal, J. (2025). ***NorESM2 restart files to be used for NorESM2-DIAM*** [Data set]. Zenodo. [DOI](https://doi.org/10.5281/zenodo.17856602)

Nordhaus, Azam, Corderi, Hood, Makarova Victor, Mohammed,  Miltner, and Weiss (2006). ***The G-Econ Database on Gridded Output: Methods and Data, Yale Unversity***.

The NorESM developers group (2020). ***Welcome to the NorESM2 User’s Guide! — NorESM documentation***, https://noresm-docs.readthedocs.io/en/latest/.

United Nations, Department of Economic and Social Affairs, Population Division (2024). ***World Population Prospects 2024, Online Edition***.

 U.S. National Science Foundation (2023). ***CESM Quickstart Guide (CESM2.1)***, https://escomp.github.io/CESM/release-cesm2/index.html.


---

## Acknowledgements

The structure of this README is adapted from the README template by Villhuber, Koren, Llull, Connolly and Morrow. Available [by clicking here](https://github.com/social-science-data-editors/template_README/blob/release-candidate/templates/README.md).
