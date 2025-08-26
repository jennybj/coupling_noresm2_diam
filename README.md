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

> INSTRUCTIONS: Some estimation code uses random numbers, almost always provided by pseudorandom number generators (PRNGs). For reproducibility purposes, these should be provided with a deterministic seed, so that the sequence of numbers provided is the same for the original author and any replicators. While this is not always possible, it is a requirement by many journals' policies. The seed should be set once, and not use a time-stamp. If using parallel processing, special care needs to be taken. If using multiple programs in sequence, care must be taken on how to call these programs, ideally from a main program, so that the sequence is not altered. If no PRNG is used, check the other box.

- [x] Random seed is set at line 192 of program \texttt{scripts/standalone\_noresm2diam.jl}.
- [x]  No Pseudo random generator is used elsewhere in the analysis described here.

### Memory, Runtime, Storage Requirements


#### Summary

Approximate time needed to reproduce the analyses on a standard (CURRENT YEAR) desktop machine:

- [ ] <10 minutes
- [ ] 10-60 minutes
- [ ] 1-2 hours
- [ ] 2-8 hours
- [ ] 8-24 hours
- [ ] 1-3 days
- [ ] 3-14 days
- [ ] > 14 days

Approximate storage space needed:

- [ ] < 25 MBytes
- [ ] 25 MB - 250 MB
- [ ] 250 MB - 2 GB
- [ ] 2 GB - 25 GB
- [ ] 25 GB - 250 GB
- [ ] > 250 GB

- [ ] Not feasible to run on a desktop machine, as described below.

#### Details



## Description of programs/code



### (Optional, but recommended) License for Code

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
