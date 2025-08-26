---
contributors:
  - Jenny Bjordal
  - Henri Cornec
  - Anthony A. Smith, Jr.
  - Trude Storelvmo
---

# README and Guidance

## Overview

This repository contains the code necessary to replicate the data presented in Bjordal et al., 2025, and in general, to run the coupled NorESM2-DIAM model. It also contains some of the input data (small size), and otherwise instructions as to where the remaining input data can be found.

## Data Availability and Provenance Statements


### Statement about Rights

- [x] I certify that the author(s) of the manuscript have legitimate access to and permission to use the data used in this manuscript. 
- [x] I certify that the author(s) of the manuscript have documented permission to redistribute/publish the data contained within this replication package. Appropriate permission are documented in the [LICENSE.txt](LICENSE.txt) file.


### (Optional, but recommended) License for Data

The data are licensed under a Creative Commons/CC-BY-NC license. See LICENSE.txt for details.

### Summary of Availability

- [x] All data **are** publicly available.
- [ ] Some data **cannot be made** publicly available.
- [ ] **No data can be made** publicly available.

### Details on each Data Source


| Data.Name  | Data.Files | Location | Provided | Citation |
| -- | -- | -- | -- | -- | 



## Dataset list



| Data file | Source | Notes    |Provided |
|-----------|--------|----------|---------|



## Computational requirements



### Software Requirements

- [x]The replication package contains two programs to install necessary dependencies.



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
