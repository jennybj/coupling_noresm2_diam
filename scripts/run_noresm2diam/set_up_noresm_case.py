# Jenny Bjordal

# Script for setting up a NorESM2 case with active carbon cycle
# reading emissions from DIAM output.

# Before running the script:
# Make sure the restart files are in the correct location.

# Usage:
# python set_up_noresm_case.py CASENAME
# CASENAME is the name of the case/simulation

#------------------------------------------------------------------------------------------

import os, sys, glob
import fileinput
from shutil import copy2

#------------------------------------------------------------------------------------------

# SPECIFY

# Specify default arguments for new case:
compset = 'N1850frc2esm'
mach = 'betzy' # CHANGE machine
project = 'nn9600k' # CHANGE project
res = 'f19_tn14'
case_dir = '/cluster/home/jennybj/cases_noresm_diam/' # CHANGE path to where you want to create the case

if mach == 'betzy': # CHANGE if you need any machine specific arguments
    additional_arguments = ' --pecount X1'

# Specify restart case:
restart_case = 'onlyCO2'
restart_dir = '/cluster/home/jennybj/restart/onlyCO2/' # CHANGE path to where the restart files are located

# Specify input directory:
input_dir = '/cluster/home/jennybj/input_noresm_diam/' # CHANGE path to where couple_iterations.sh, couple_with_decision_rules.py, and module_coupling.py are located

# Specify environment changes:
batch_xml = {'JOB_WALLCLOCK_TIME': '02:00:00'}
run_xml = {
    'RUN_TYPE': 'branch',
    'RUN_REFCASE': restart_case,
    'RUN_REFDATE': '1990-01-01',
    'RUN_STARTDATE': '1990-01-01',
    'STOP_OPTION': 'nyears',
    'STOP_N': '1',
    'REST_OPTION': 'nyears',
    'REST_N': '1',
    'RESUBMIT': '110', # CHANGE to number of resubmits (must be resubmitted each year)
    'POSTRUN_SCRIPT': './couple_iterations.sh'
}

# Check for consistency:
if run_xml['RUN_REFCASE'] != restart_case:
    print('Restart case is inconsistent: ' + run_xml['RUN_REFCASE'] + ' and ' +
          restart_case)
    sys.exit()

#------------------------------------------------------------------------------------------

# CREATE A NEW CASE

# Check for needed argumet:
if len(sys.argv) != 2:

    print('Need one argument:')
    print('python set_up_noresm_case.py CASENAME')
    sys.exit()

else:

    case = sys.argv[1]

newcase_command = './create_newcase' + \
                  ' --case ' + case_dir +case + \
                  ' --compset ' + compset + \
                  ' --mach ' + mach + \
                  ' --res ' + res + \
                  ' --project ' + project + \
                  additional_arguments

os.chdir('/cluster/home/jennybj/NorESM2.0.9/cime/scripts') # CHANGE path to location_of_NorESM2/cime/scripts
err = os.system(newcase_command)
if err != 0:
    print('Failed to create new case.')
    sys.exit()

#------------------------------------------------------------------------------------------

# CONFIGURE THE CASE

os.chdir(case_dir + case)
err = os.system('./case.setup')
if err != 0:
    print('Failed to setup case.')
    sys.exit()

#------------------------------------------------------------------------------------------

# CHANGE ENVIRONMENT FILES

os.chdir(case_dir + case)

# Change env_batch.xml:
for variable, value in batch_xml.items():
    print('Setting ', variable, ' to ', value)
    command_batch = './xmlchange ' + variable + '=' + value + ' --subgroup case.run'
    err = os.system(command_batch)
    if err != 0:
        print('Failed to change an xml variable: ' + command_batch)
        sys.exit()

# Change env_run.xml:
for variable, value in run_xml.items():
    print('Setting ', variable, ' to ', value)
    command_run = './xmlchange ' + variable + '=' + value
    err = os.system(command_run)
    if err != 0:
        print('Failed to change an xml variable: ' + command_run)
        sys.exit()


#------------------------------------------------------------------------------------------

# CHANGE NAMELIST FILES

os.chdir(case_dir + case)

with open("user_nl_cam", "a") as f:
  f.write("\n")
  f.write("&co2_cycle_nl\n")
  f.write(" co2_flag               = .true.\n")
  f.write(" co2_readflux_fuel      = .true.\n")
  f.write(" co2flux_fuel_file      = '/cluster/home/jennybj/input_emissions_" + case + ".nc'\n") # CHANGE path to where the emissions file will be located. Make sure that this is the same path as specified in make_emissions_file in module_coupling.py.

with open("user_nl_clm", "a") as f:
  f.write("\n")
  f.write("use_init_interp=.true.")

err = os.system('./preview_namelists')
if err != 0:
    print('Failed to run preview_namelist')
    sys.exit()

#------------------------------------------------------------------------------------------

# COPY RESTART FILES TO RUN DIRECTORY

for file in glob.glob(restart_dir + '*'):

    try:
        copy2(file, '/cluster/work/users/jennybj/noresm/' + case + '/run/')
        print('Copied: ', file,
              ' to /cluster/work/users/jennybj/noresm/' + case + '/run/')

    except:
        print('Failed to copy file: ', file)
        sys.exit()

#------------------------------------------------------------------------------------------

# BUILD THE CASE

os.chdir(case_dir + case)
err = os.system('./case.build')
if err != 0:
    print('Failed to build the case.')
    sys.exit()

#------------------------------------------------------------------------------------------

# COPY AND PREPARE COUPLE/POST SCRIPT

# Copy coupling scripts:
coupling_scripts = ['couple_iterations.sh', 'couple_with_decision_rules.py', 'module_coupling.py']
for file in coupling_scripts:

    try:
        copy2(input_dir + file, case_dir + case)
        print('Copied: ', file, ' to ', case_dir + case)

    except:
        print('Failed to copy file: ', file)
        sys.exit()

# Make script executable:
err = os.system('chmod +x couple_iterations.sh')
if err != 0:
    print('Failed to make couple_iterations.sh executable')
    sys.exit()

#------------------------------------------------------------------------------------------
