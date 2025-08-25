#--------------------------------------------------------------------------------------

#import sys as sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime

#sys.path.insert(0, '../modules')
from module_coupling import *

#--------------------------------------------------------------------------------------

start_time = datetime.timestamp(datetime.now())

cumulative_emissions_1990 = 216.8650  # GtC not including 1990

price = get_price()
chi = get_chit()
ga, beta, delta, alpha, energyshare, rss, theta, b = get_constants()

pi_temperature = get_pi_temperature()
gamma1, gamma2, rhos = get_coefficients()

latitudes, longitudes = get_coordinate_data()

ncells = latitudes.shape[0]

file_path = '/cluster/home/jennybj/coupling/'  # CHANGE path to location of "emissions.txt" (and output from this script)
dr_path = '/cluster/work/users/jennybj/coupling/' # CHANGE path to location of decision rules

#--------------------------------------------------------------------------------------

# READ IN DATA

years, expected_emissions = np.loadtxt(file_path + 'emissions.txt',
                                       unpack=True)
expected_cumulative_emissions = np.concatenate(
    (np.array([cumulative_emissions_1990]),
     np.cumsum(expected_emissions) + cumulative_emissions_1990))

# Population data:

nyears = 150

population = get_population()

#--------------------------------------------------------------------------------------


def get_ai(iyear):

    ts = datetime.timestamp(datetime.now())

    ai = get_initial_ai()
    ai_last = ai

    if iyear != 1990:

        expected_temperature_last = get_expected_temperature(1990)
        expected_damages = damages(expected_temperature_last)

        nyears = iyear
        for i in range(1990, nyears):

            ai_last = ai
            expected_damages_last = expected_damages

            expected_temperature = get_expected_temperature(i + 1)
            expected_damages = damages(expected_temperature)

            ai = (1 + ga) * ai_last * (expected_damages /
                                          expected_damages_last)

            dt = datetime.timestamp(datetime.now())

            print(i + 1, ai, dt - ts)

    return ai_last, ai


def get_expected_temperature(iyear):
    """ Calculate the expected temperature using the expected cumulative emissions and
    regression coefficients """

    # Check that we use correct year:
    if not iyear - 1989 == years[iyear - 1990]:

        print(
            'Using wrong year of expected cumulative emissions. Should have ',
            iyear, ', using ', years[iyear - 1990]) + 1989
        stop_due_to_error()

    else:

        expected_temperature = pi_temperature + gamma1 * expected_cumulative_emissions[
            iyear - 1990] + gamma2 * expected_cumulative_emissions[iyear - 1990]**2

    return expected_temperature


def damages(regtemp, tstar=12.609, scale1=0.00327721, scale2=0.00362887):
    """ The regional damage function. Already raised to the power of 1/(1 - alpha)"""

    # Define constants:
    pbound = 0.02
    toler = 1.0e-4

    diff = regtemp - tstar

    mcells = regtemp.shape[0]

    if mcells != ncells:
        print('Number of cells is ', mcells, ' not ', ncells)

    fval = np.zeros(mcells)

    #((1 - d) * exp(-κ_minus * (t - T) ^ 2) + d) ^ (1 / (1 - α))

    for icell in range(mcells):

        if diff[icell] < 0:
            fval[icell] = ((1 - pbound) * np.exp(-scale1 * diff[icell]**2) +
                           pbound)**(1 / (1 - alpha))
        else:
            fval[icell] = ((1 - pbound) * np.exp(-scale2 * diff[icell]**2) +
                           pbound)**(1 / (1 - alpha))

        if fval[icell] < toler:
            fval[icell] = toler

    return fval


def cobb_douglas(x, y):
    """ Cobb Douglas production function """

    return x**theta * y**(1 - theta)


def F(x, y):
    """ Production function """

    return b * cobb_douglas(x, y)


def scale(in_variable, iyear):

    dum, ai = get_ai(iyear)
    expected_temperature = get_expected_temperature(iyear)

    out_variable = in_variable / (population[:, iyear - 1990] * ai *
                                  damages(expected_temperature))

    return out_variable


def descale(in_variable, iyear):

    dum, ai = get_ai(iyear)
    expected_temperature = get_expected_temperature(iyear)

    out_variable = in_variable * (population[:, iyear - 1990] * ai *
                                  damages(expected_temperature))

    return out_variable


def calculate_energy_use_scaled(capital_scaled, z, expected_temperature):

    regtemp = expected_temperature + z

    energy_use = (
        (1 - theta) * b / price)**(1 / theta) * capital_scaled**alpha * (
            damages(regtemp) / damages(expected_temperature))**(1 - alpha)

    return energy_use


def my_bisect(target, values):

    print(target)
    aa = 0
    bb = values.shape[0] - 1

    while bb - aa > 1:
        cc = round((bb - aa) / 2) + aa
        if values[cc] > target:
            bb = cc
        else:
            aa = cc

    return (aa, bb)


def get_capital_scaled(iyear, wealth_scaled, z):

    ts = datetime.timestamp(datetime.now())

    npoints = 21
    capital = np.zeros(ncells)
    z_points = np.loadtxt(dr_path + 'z_grid.txt')[:, 1:]

    # Read in decision rules:
    file = dr_path + 'decrule_' + str(iyear) + '.csv'
    data = np.loadtxt(file, skiprows=1, delimiter=',')  # (wealth,shock)
    lats = data[::npoints, 0]
    lons = data[::npoints, 1]
    wealth_points = data[:, 2].reshape((ncells, npoints))
    decision_rules = data[:, 3:].reshape((ncells, npoints, npoints))

    nextrap = 0

    for icell in range(ncells):

        f_interp = RegularGridInterpolator((wealth_points[icell,:], z_points[icell, :]),
                                           decision_rules[icell, :, :],
                                           bounds_error=False,
                                           fill_value=None)

        if wealth_scaled[icell] > wealth_points[icell, -1] or wealth_scaled[
                icell] < wealth_points[icell, 0] or z[icell] > z_points[
                    icell, -1] or z[icell] < z_points[icell, 0]:

            nextrap += 1

        capital[icell] = f_interp(np.array([wealth_scaled[icell], z[icell]]))

        """
        # Test the interpolation method:
        if icell == 0:

            i1, i2 = my_bisect(wealth_scaled[icell], wealth_points[icell,:])
            j1, j2 = my_bisect(z[icell], z_points[icell, :])

            print(wealth_scaled[icell], z[icell])
            print(wealth_points[icell, i1], wealth_points[icell, i2], z_points[icell, j1],
                  z_points[icell, j2])

            mycap = decision_rules[icell, i1, j1] * (
                1 - (wealth_scaled[icell] - wealth_points[icell, i1]) /
                (wealth_points[icell, i2] - wealth_points[icell, i1])) * (
                    1 - (z[icell] - z_points[icell, j1]) /
                    (z_points[icell, j2] - z_points[icell, j1])
                ) + decision_rules[icell, i2, j1] * (
                    1 - (wealth_points[icell, i2] - wealth_scaled[icell]) /
                    (wealth_points[icell, i2] - wealth_points[icell, i1])) * (
                        1 - (z[icell] - z_points[icell, j1]) /
                        (z_points[icell, j2] - z_points[icell, j1])
                    ) + decision_rules[icell, i1, j2] * (
                        1 - (wealth_scaled[icell] - wealth_points[icell, i1]) /
                        (wealth_points[icell, i2] - wealth_points[icell, i1])) * (
                            1 - (z_points[icell, j2] - z[icell]) /
                            (z_points[icell, j2] - z_points[icell, j1])
                        ) + decision_rules[icell, i2, j2] * (
                            1 - (wealth_points[icell, i2] - wealth_scaled[icell]) /
                            (wealth_points[icell, i2] - wealth_points[icell, i1])) * (
                                1 - (z_points[icell, j2] - z[icell]) /
                                (z_points[icell, j2] - z_points[icell, j1]))
            print(mycap, capital[icell])
        """

    print('Decision rules done in ', datetime.timestamp(datetime.now()) - ts)
    print('Extrapolated ', nextrap, ' grid cells')

    return capital


def calculate_wealth_scaled(iyear, regtemp, capital_scaled, energy_scaled):
    """ Calculate the scaled wealth """

    expected_temperature = get_expected_temperature(iyear)

    wealth_scaled = F(
        capital_scaled**alpha *
        (damages(regtemp) / damages(expected_temperature))**(1 - alpha),
        energy_scaled) - price * energy_scaled + (1 - delta) * capital_scaled

    print(
        F(
            capital_scaled**alpha *
            (damages(regtemp) / damages(expected_temperature))**(1 - alpha),
            energy_scaled), price * energy_scaled,
        (1 - delta) * capital_scaled)

    return wealth_scaled


#--------------------------------------------------------------------------------------

# Start in year 1990 = 0
# Know capital and emissions in year 1990
# Run NorESM2 --> get temperatures
# Use temperature to calculate wealth
# Use decision rules to get energy use
# Calculate emissions and write to file
#

case_name = get_case_name()  # Name of the NorESM2 run
if case_name.endswith('_cont'):
    file_name = case_name[:-5]
else:
    file_name = case_name

# Name of the in/out files:
expected_emissions_file = file_path + file_name + '_expected_emissions.txt'
actual_emissions_file = file_path + file_name + '_actual_emissions.txt'
expected_global_emissions_file = file_path + file_name + '_expected_global_emissions.txt'
cumulative_emissions_file = file_path + file_name + '_cumulative_emissions.txt'
global_emissions_file = file_path + file_name + '_global_emissions.txt'
wealth_file = file_path + file_name + '_wealth.txt'
capital_file = file_path + file_name + '_capital.txt'
regtemp_file = file_path + file_name + '_regtemp.txt'
global_temp_file = file_path + file_name + '_global_temp.txt'

year_current = get_year_current(case_name)
year_next = year_current + 1

#--------------------------------------------------------------------------------------

# GET CURRENT VALUES

if year_current == 1990:

    # Get initial values for year 1990:
    ai_current = get_initial_ai()

    kss = ((((rss + delta) / (alpha * theta)) * (b**(-1 / theta)) *
            ((price / (1 - theta))**((1 - theta) / theta)))**(1 / (alpha - 1)))
    print('kss', kss)
    xss = calculate_energy_use_scaled(kss, 0, np.array([0]))
    print('xss', xss)

    expected_emissions_current = ai_current * population[:,
                                                         0] * xss  #population[:,0] * ai_current * initial_energy_use * chi[0]  # MtC
    print(expected_emissions_current)

    cumulative_emissions_current = cumulative_emissions_1990
    capital_current_scaled = np.full(ncells, kss)

    expected_global_emissions_current = np.sum(expected_emissions_current)/1000
    print(expected_global_emissions_current)

    # Write initial values to files:
    write_to_txt_file(year_current, expected_emissions_file,
                      expected_emissions_current)
    write_to_txt_file(year_current, expected_global_emissions_file,
                      expected_global_emissions_current)
    write_to_txt_file(year_current, cumulative_emissions_file,
                      cumulative_emissions_current)
    write_to_txt_file(year_current, capital_file, capital_current_scaled)

else:

    # Get values calculated at the last time step:
    capital_current_scaled = read_last_value_in_file(capital_file)
    expected_emissions_current = read_last_value_in_file(
        expected_emissions_file)
    cumulative_emissions_current = read_last_value_in_file(
        cumulative_emissions_file)

expected_temperature_current = get_expected_temperature(year_current)

ai_current, ai_next = get_ai(year_next)

global_temp_current, regtemp_current = get_noresm_regional_temperatures(
    year_current, case_name)
z_current = regtemp_current - expected_temperature_current
print('z: ', np.mean(z_current), np.max(np.abs(z_current)))

actual_energy_current_scaled = calculate_energy_use_scaled(
    capital_current_scaled, z_current, expected_temperature_current)
actual_emissions_current = actual_energy_current_scaled * chi[
    year_current - 1990] * ai_current * population[:, year_current - 1990]

wealth_current_scaled = calculate_wealth_scaled(year_current, regtemp_current,
                                                capital_current_scaled,
                                                actual_energy_current_scaled)

wealth_current = descale(wealth_current_scaled, year_current)

#--------------------------------------------------------------------------------------

# CALCULATE NEXT VALUES

cumulative_emissions_next = cumulative_emissions_current + np.sum(
    actual_emissions_current) / 1000
global_emissions_current = np.sum(actual_emissions_current) / 1000

print('Emissions current: ',
      np.sum(expected_emissions_current) / 1000,
      np.sum(actual_emissions_current) / 1000)

expected_temperature_next = get_expected_temperature(year_next)

capital_next_scaled = get_capital_scaled(year_current, wealth_current_scaled,
                                         z_current)

expected_energy_next_scaled = calculate_energy_use_scaled(
    capital_next_scaled, rhos * z_current, expected_temperature_next)
expected_emissions_next = expected_energy_next_scaled * chi[
    year_next -
    1990] * ai_next * population[:, year_next -
                                 1990] + actual_emissions_current - expected_emissions_current

expected_global_emissions_next = np.sum(expected_emissions_next) / 1000
print('Emissions next: ', expected_global_emissions_next)

#--------------------------------------------------------------------------------------

# WRITE TO FILES

write_to_txt_file(year_current, wealth_file, wealth_current)  # not necessary
write_to_txt_file(year_current, actual_emissions_file,
                  actual_emissions_current)
write_to_txt_file(year_current, global_emissions_file,
                  global_emissions_current)
write_to_txt_file(year_next, expected_emissions_file, expected_emissions_next)
write_to_txt_file(year_next, expected_global_emissions_file, expected_global_emissions_next)
write_to_txt_file(year_next, cumulative_emissions_file,
                  cumulative_emissions_next)
write_to_txt_file(year_next, capital_file, capital_next_scaled)
write_to_txt_file(year_current, regtemp_file, regtemp_current)
write_to_txt_file(year_current, global_temp_file, global_temp_current)

make_emissions_file(file_name, expected_emissions_file)

# Run NorESM2 for next year

output_file = file_path + 'output_year_' + str(year_current) + '.txt'

with open(output_file, 'wt') as f:

    f.write('Column 1: latitude \n')
    f.write('Column 2: longitude \n')
    f.write('Column 3: expected temperature current year \n')
    f.write('Column 4: NorESM2 temperature current year \n')
    f.write('Column 5: z = NorESM2 temperature - expected temperature \n')
    f.write('Column 6: scaled wealth current year \n')
    f.write('Column 7: scaled capital current year \n')
    f.write('Column 8: scaled capital next year \n')
    f.write('Column 9: ai current year \n')
    f.write('Column 10: ai next year \n')
    f.write('Column 11: actual scaled energy current year \n')
    f.write('Column 12: scaled energy next year \n')
    f.write('Column 13: expected emissions current year \n')
    f.write('Column 14: actual emissions current year \n')
    f.write('Column 15: expected emissions next year \n')
    f.write('\n')

    for icell in range(ncells):
        f.write('%16.8f' % latitudes[icell])
        f.write('%16.8f' % longitudes[icell])
        f.write('%16.8f' % expected_temperature_current[icell])
        f.write('%16.8f' % regtemp_current[icell])
        f.write('%16.8f' % z_current[icell])
        f.write('%16.8f' % wealth_current_scaled[icell])
        f.write('%16.8f' % capital_current_scaled[icell])
        f.write('%16.8f' % capital_next_scaled[icell])
        f.write('%16.8f' % ai_current[icell])
        f.write('%16.8f' % ai_next[icell])
        f.write('%16.8f' % actual_energy_current_scaled[icell])
        f.write('%16.8f' % expected_energy_next_scaled[icell])
        f.write('%16.8f' % expected_emissions_current[icell])
        f.write('%16.8f' % actual_emissions_current[icell])
        f.write('%16.8f' % expected_emissions_next[icell])
        f.write('\n')

print('Coupling script finished! ',
      datetime.timestamp(datetime.now()) - start_time)

#--------------------------------------------------------------------------------------
