#!/bin/bash

#SBATCH --account=nn9600k

module --force purge
module load StdEnv
module load Miniforge3/24.1.2-0
source ${EBROOTMINIFORGE3}/bin/activate
conda activate env_coupling

python couple_with_decision_rules.py
