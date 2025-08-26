---
Contributors:
  - Jenny Bjordal
  - Henri Cornec
  - Anthony A. Smith, Jr.
  - Trude Storelvmo
---

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
| "World Population Prospects 2024"| `undp_pop_growth_2024.xlsx`| `data/raw/`| TRUE     | UN DESA, Population Division, 2024 |
| "G-ECON v4.0"                    | `Gecon40_post_final.csv`   | `data/raw/`| TRUE     | Nordhaus et al. 2006 |


#### "World Population Prospects 2024"
Data on historical country-level population levels and future population projections were downloaded from the United Nations Department of Economic and Social Affairs [UN DESA], Population Division, 2024. We use the complete .xlsx file format. Data can be downloaded from [here](https://population.un.org/wpp/assets/Excel\%20Files/1_Indicator\%20(Standard)/EXCEL_FILES/1_General/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_FULL.xlsx). The data are licensed under a CC-BY 3.0.

Datafile: `data/raw/undp_pop_growth_2024.xlsx`

#### G-ECON v4.0

The paper uses sub-national economic and population data from the G-ECON v.4.0 database. Available for download [here](https://gecon.yale.edu/data-and-documentation-g-econ-project) under Point 5 in the Data Sets Section.

## Dataset list

| Data file                                                        | Source                         | Location | Notes                                                                 |
|------------------------------------------------------------------|--------------------------------|----------|----------------------------------------------------------------------|
| `data/raw/emissions.txt`                                         | Model Output                   | GitHub   | Not required. Reduces iterations for `scripts/decrule_calc.jl`.      |
| `data/raw/undp_pop_growth_2024.xlsx`                             | UN DESA, Population Division, 2024 | Dropbox  | Historical Estimates and Medium Variant Projections used.             |
| `data/raw/Gecon40_post_final.csv`                                | Nordhaus et al. 2006           | GitHub   |                                                                      |
| `onlyCO2/rest/1990-01-01-00000/*`                                | Bjordal, 2025                  | [DOI](https://doi.org/10.11582/2025.tdi6hhfl) | Restart files for running NorESM2 from 1990.                         |
| `data/input_to_regression/N1850_f19_tn14_20190730esm.TREFHT.nc`  | NorESM2 output                 | Dropbox  | PiControl temperature data.                                          |
| `data/input_to_regression/onlyCO2.nc`                            | NorESM2 output                 | Dropbox  | HIST and SSP3-7.0 temperature data with only CO2 emissions.          |
| `data/input_to_regression/full_couple_baseline.nc`               | NorESM2 output                 | Dropbox  | Temperature data from 1990–2100, only CO2 emissions.                 |
| `data/input_to_regression/full_couple_baseline_e2.nc`            | NorESM2 output                 | Dropbox  | Temperature data from 1990–2100, only CO2 emissions.                 |
| `data/input_emission_data/*`                                     | NorESM2 input                  | Dropbox  | Standard CO2 emissions input files. For details see NorESM dev group |

GitHub refer to this repository and Dropbox refer to [this Dropbox folder](LINK HERE).

## Computational requirements



### Software Requirements

- [x] The replication package contains two programs to install the necessary dependencies.

**Julia 1.10.4**

Run `setup/packages.jl` to install all necessary Julia packages.

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

- [x] Random seed is set at line 192 of program \texttt{scripts/standalone\_noresm2diam.jl}.
- [x]  No Pseudo random generator is used elsewhere in the analysis described here.

### Memory, Runtime, Storage Requirements

Portions of this code were last run on a 6-core Apple M2-Pro laptop with MacOS version 15.5 with 50GB of free space.

Portions of the code were last run on a 3-node cluster (1x cascadelake, 2x icelake) with a SLURM cluster manager.

NorESM2 (including the coupling scripts) was run on an Atos BullSequana XH2000, using 10 CPU nodes (each with 128 cores and 256 GiB of memory). The machine, named Betzy, is provided by Sigma2 AS, and more details can be found [here](https://documentation.sigma2.no/hpc_machines/betzy.html). With this setup, the coupled model takes approximately one hour per year of simulation.

The rest of the python scripts (not for coupling) can be run on any laptop. Each script can be run in less than 5 minutes, and in total they produce output requiring storage of approximately 90 MB.


## Description of code

### General

- The programs in `scripts/julia_helper/` are auxiliary Julia scripts used in other portions of the code that simplify the workflow in other programs, e.g., by modifying the creation of arrays or creating more readable output files.
- The program `scripts/module_coupling.py` reads in various data and performs calculations used by the various Python scripts in `scripts/create_figures/` and `scripts/create_input_files/` as well as by the coupling script `scripts/run_noresm2diam/couple_with_decision_rules.py`.

### Generate input files

- The program `scripts/regpop4.jl` will create the regional population numbers and growth rates found in `regpop4.pop` and `regpop4.grate`, as well as `parse2.gin6` which is used in model calibration.
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



### License for Code

The code is licensed under a MIT license. See [LICENSE.md](LICENSE.md) for details.


## Instructions to Replicators



### Details



## List of tables and programs


The provided code reproduces:

- [ ] All numbers provided in text in the paper
- [ ] All tables and figures in the paper
- [ ] Selected tables and figures in the paper, as explained and justified below.


| Figure/Table #    | Program                  | Line Number | Output file                      | Note                            |
|-------------------|--------------------------|-------------|----------------------------------|---------------------------------|


## References



---

## Acknowledgements

The structure of this README is adapted from the README template by Villhuber, Koren, Llull, Connolly and Morrow. Available [by clicking here](https://github.com/social-science-data-editors/template_README/blob/release-candidate/templates/README.md).
