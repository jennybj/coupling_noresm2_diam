#--------------------------------------------------------------------------------------

#import sys as sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime

#sys.path.insert(0, '../modules') # CHANGE path
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
nyears = years.shape[0]
population = get_population()

#--------------------------------------------------------------------------------------


def get_expected_temperature():
    """ Calculate the expected temperature using the expected cumulative emissions and
    regression coefficients """

    expected_temperature = np.zeros((nyears, ncells))

    for iyear in range(nyears):

        expected_temperature[
            iyear, :] = pi_temperature + gamma1 * expected_cumulative_emissions[
                iyear]+ gamma2 * expected_cumulative_emissions[
                iyear]**2

    return expected_temperature


def get_ai():

    ai = np.zeros((nyears, ncells))
    expected_temperature = get_expected_temperature()
    expected_damages = damages(expected_temperature)

    ai[0, :] = get_initial_ai()

    for iyear in range(1, nyears):

        ai[iyear, :] = (1 + ga) * ai[iyear - 1, :] * (
            expected_damages[iyear, :] / expected_damages[iyear - 1, :])

    return ai


def damages(regtemp, tstar=12.609, scale1=0.00327721, scale2=0.00362887):
    """ The regional damage function. Already raised to the power of 1/(1 - alpha)"""

    # Define constants:
    pbound = 0.02
    toler = 1.0e-4

    diff = regtemp - tstar

    if regtemp.ndim == 1:
        myears = 1
        mcells = regtemp.shape[0]
        diff.reshape((1, mcells))
    elif regtemp.ndim == 2:
        myears = regtemp.shape[0]
        mcells = regtemp.shape[1]
    else:
        exit('Too many dimensions in temperature given to damage function.')

    if mcells != ncells:
        print('Number of cells is ', mcells, ' not ', ncells)

    fval = np.zeros((myears, mcells))

    #((1 - d) * exp(-κ_minus * (t - T) ^ 2) + d) ^ (1 / (1 - α))

    for iyear in range(myears):

        for icell in range(mcells):

            if diff[iyear, icell] < 0:
                fval[iyear, icell] = (
                    np.exp(-scale1 * diff[iyear, icell] * diff[iyear, icell]) *
                    (1 - pbound) + pbound)**(1 / (1 - alpha))
            else:
                fval[iyear, icell] = (
                    np.exp(-scale2 * diff[iyear, icell] * diff[iyear, icell]) *
                    (1 - pbound) + pbound)**(1 / (1 - alpha))

            if fval[iyear, icell] < toler:
                fval[iyear, icell] = toler

    if myears == 1:
        fval.reshape((mcells))

    return fval


def cobb_douglas(x, y):
    """ Cobb Douglas production function """

    return x**theta * y**(1 - theta)


def F(x, y):
    """ Production function """

    return b * cobb_douglas(x, y)


def scale(in_variable, iyear):

    ai = get_ai()

    out_variable = in_variable / (population * ai)

    return out_variable


def descale(in_variable, iyear):

    ai = get_ai()

    out_variable = in_variable * (population * ai)

    return out_variable


def calculate_energy_use_scaled(capital_scaled, z=0, expected_temperature=0):

    energy_use = ((1 - theta) * b /
                  price)**(1 / theta) * capital_scaled**alpha  # NB! simplified

    #regtemp = expected_temperature + z

    #energy_use = (
    #    (1 - theta) * b / price)**(1 / theta) * capital_scaled**alpha * (
    #        damages(regtemp) / damages(expected_temperature))**(1 - alpha)

    return energy_use


def get_capital_scaled(iyear, wealth_scaled, z):

    ts = datetime.timestamp(datetime.now())

    iyear = iyear + 1990

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
    #decision_rules2 = data[:, 13].reshape((ncells, npoints))

    nextrap = 0

    for icell in range(ncells):

        f_interp = RegularGridInterpolator(
            (wealth_points[icell, :], z_points[icell, :]),
            decision_rules[icell, :, :],
            bounds_error=False,
            fill_value=None)

        if wealth_scaled[icell] > wealth_points[icell, -1] or wealth_scaled[
                icell] < wealth_points[icell, 0] or z[icell] > z_points[
                    icell, -1] or z[icell] < z_points[icell, 0]:

            nextrap += 1

        capital[icell] = f_interp(np.array([wealth_scaled[icell], z[icell]]))
        #capital2 = np.interp(wealth_scaled[icell], wealth_points[icell,:], decision_rules2[icell,:])

    print('Decision rules done in ', datetime.timestamp(datetime.now()) - ts)
    print('Extrapolated ', nextrap, ' grid cells')

    return capital


def calculate_wealth_scaled(capital_scaled, energy_scaled, regtemp=0):
    """ Calculate the scaled wealth """

    wealth_scaled = F(capital_scaled**alpha,
                      energy_scaled) - price * energy_scaled + (
                          1 - delta) * capital_scaled  # NB! simplified

    #expected_temperature = get_expected_temperature()

    #wealth_scaled = F(
    #    capital_scaled**alpha *
    #    (damages(regtemp) / damages(expected_temperature))**(1 - alpha),
    #    energy_scaled) - price * energy_scaled + (1 - delta) * capital_scaled

    return wealth_scaled


#--------------------------------------------------------------------------------------

# SPECIFY FILES

file_name = 'fixed_point'

# Name of the in/out files:
emissions_file = file_path + file_name + '_emissions.txt'
global_emissions_file = file_path + file_name + '_global_emissions.txt'
cumulative_emissions_file = file_path + file_name + '_cumulative_emissions.txt'
wealth_file = file_path + file_name + '_wealth.txt'
capital_file = file_path + file_name + '_capital.txt'
regtemp_file = file_path + file_name + '_regtemp.txt'

#--------------------------------------------------------------------------------------

energy_scaled = np.zeros((nyears, ncells))
emissions = np.zeros((nyears, ncells))
capital_scaled = np.zeros((nyears, ncells))
wealth_scaled = np.zeros((nyears, ncells))
wealth = np.zeros((nyears, ncells))
cumulative_emissions = np.zeros(nyears)
z = np.zeros(ncells)

ai = get_ai()
expected_temperature = get_expected_temperature()
expected_damages = damages(expected_temperature)

kss = ((((rss + delta) / (alpha * theta)) * (b**(-1 / theta)) *
        ((price / (1 - theta))**((1 - theta) / theta)))**(1 / (alpha - 1)))
xss = calculate_energy_use_scaled(kss, 0, np.array([0]))
print('kss', kss)
print('xss', xss)

#emissions[0,:] = ai[0,:] * population[0,:] * xss   # MtC
cumulative_emissions[0] = cumulative_emissions_1990
capital_scaled[0, :] = np.full(ncells, kss)

write_to_txt_file(1990, cumulative_emissions_file, cumulative_emissions[0])
write_to_txt_file(1990, capital_file, capital_scaled[0, :])

for iyear in range(0, nyears):

    energy_scaled[iyear, :] = calculate_energy_use_scaled(
        capital_scaled[iyear], z, expected_temperature[iyear, :])
    emissions[iyear, :] = energy_scaled[iyear, :] * chi[iyear] * ai[
        iyear, :] * population[:,iyear]
    wealth_scaled[iyear, :] = calculate_wealth_scaled(capital_scaled[iyear, :],
                                                      energy_scaled[iyear, :])

    cumulative_emissions[
        iyear +
        1] = cumulative_emissions[iyear] + np.sum(emissions[iyear, :]) / 1000
    capital_scaled[iyear + 1, :] = get_capital_scaled(iyear,
                                                      wealth_scaled[iyear, :],
                                                      z)

    wealth = wealth_scaled[iyear, :] * (population[:, iyear] * ai[iyear, :])
    print(wealth.shape)

    print(iyear, np.sum(emissions[iyear, :]) - expected_emissions[iyear]*1e3)

    #----------------------------------------------------------------------------------

    # WRITE TO FILES

    write_to_txt_file(iyear + 1990, wealth_file, wealth)  # not necessary
    write_to_txt_file(iyear + 1990, emissions_file, emissions[iyear, :])
    write_to_txt_file(iyear + 1990, global_emissions_file, np.sum(emissions[iyear,:])/1000)
    write_to_txt_file(iyear + 1991, cumulative_emissions_file,
                      cumulative_emissions[iyear + 1])
    write_to_txt_file(iyear + 1991, capital_file, capital_scaled[iyear + 1, :])
    write_to_txt_file(iyear + 1990, regtemp_file,
                      expected_temperature[iyear, :])

    output_file = file_path + 'fp_output_year_' + str(iyear + 1990) + '.txt'

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
            f.write('%16.8f' % expected_temperature[iyear, icell])
            f.write('%16.8f' % expected_temperature[iyear, icell])
            f.write('%16.8f' % z[icell])
            f.write('%16.8f' % wealth_scaled[iyear, icell])
            f.write('%16.8f' % capital_scaled[iyear, icell])
            f.write('%16.8f' % capital_scaled[iyear + 1, icell])
            f.write('%16.8f' % ai[iyear, icell])
            f.write('%16.8f' % ai[iyear + 1, icell])
            f.write('%16.8f' % energy_scaled[iyear, icell])
            f.write('%16.8f' % z[icell])
            f.write('%16.8f' % emissions[iyear, icell])
            f.write('%16.8f' % z[icell])
            f.write('%16.8f' % z[icell])
            f.write('\n')

#--------------------------------------------------------------------------------------
