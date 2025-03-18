#----------------------------------------------------------------------------------------

import numpy as np
import numpy.ma as ma
import os as os
import sys as sys
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from datetime import datetime

#-----------------------------------------------------------------------------------------------

file_path = ''  # '/cluster/home/jennybj/coupling/'

# READ IN ECONOMIC DATA:

# Read lines from file:
with open(file_path + 'parse2.gin5', 'r') as myfile:
    lines = myfile.readlines()

ncells = len(lines)
all_indices = np.arange(ncells)

# Arrays:
diam_latitudes = np.zeros(ncells)
diam_longitudes = np.zeros(ncells)
pop = np.zeros(ncells)
gdpnetper = np.zeros(ncells)
country_names = []

# Add coordinates:
for i in range(ncells):

    line = lines[i].split()

    diam_latitudes[i] = float(line[1])
    diam_longitudes[i] = float(line[2])
    pop[i] = float(line[-3])
    gdpnetper[i] = float(line[-1]) * 1e-3

    name = ' '.join(line[3:-6])
    country_names.append(name)



cumulative_emissions = np.loadtxt(
    file_path +
    'NorESM2_HIST_SSP370_cumulative_emissions_global_temperature.txt',
    usecols=1)

orig_emissions = cumulative_emissions[1:] - cumulative_emissions[:-1]

#----------------------------------------------------------------------------------------

# DEFINE CONSTANTS

ga = 0.01
beta = 0.985
delta = 0.06
alpha = 0.36
energyshare = 0.062
rss = (1 + ga) / beta - 1
theta = 1 / (1 + energyshare)
b = 0.4

#----------------------------------------------------------------------------------------

nlats = 96
nlons = 144

noresm_latitudes = np.linspace(-90, 90, nlats)
noresm_longitudes = np.linspace(0, 357.5, nlons)

month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

#----------------------------------------------------------------------------------------

# FUNCTIONS


def stop_due_to_error():

    stop_command = './xmlchange RESUBMIT=0'

    os.system(stop_command)
    sys.exit()

    return


def get_chit():

    n_01 = 10
    n_05 = 75
    years = np.arange(1, 201, 1)

    chi = (1 + np.exp(np.log(0.01 / 0.99) * (years - n_05) /
                      (n_01 - n_05)))**(-1)
    chit = chi / chi[0]

    return chit


def get_price():

    chit = get_chit()

    globalgdp1990 = np.sum(pop * gdpnetper)
    x1990 = 1e3 * (orig_emissions[140]) / chit[0]
    price = energyshare * globalgdp1990 / x1990

    return price


def get_initial_population():
    return pop


def get_initial_ai():

    price = get_price()
    capitali = alpha * gdpnetper / (rss + delta)
    xi = ((1 - theta) / (theta * price)) * gdpnetper

    ai = (b * (1 - theta) * (capitali**(alpha * theta)) * (xi**(-theta)) /
          price)**(1 / (theta * (alpha - 1)))

    return ai


def get_pi_temperature():

    pi_temperature = np.loadtxt(file_path +
                                'NorESM2_picontrol_regional_temperatures.txt',
                                usecols=2)

    return pi_temperature


def get_coefficients():

    coefficients_emissions, coefficients_shocks = np.loadtxt(
        file_path + 'NorESM2_HIST_SSP370_coefficients_and_RMSE.txt',
        usecols=(2, 4),
        unpack=True)

    return coefficients_emissions, coefficients_shocks


def get_country_names():
    return country_names


def get_case_name():
    """ Find the name of the NorESM2 case. """

    directory_path = os.getcwd()
    case_name = os.path.basename(directory_path)

    return case_name


def get_number_of_line_in_file(file_name):

    return sum(1 for i in open(file_name, 'rb'))


def get_year_current(case_name):
    """ Get the latest year run by NorESM2. Year 1990 is the initial year. """

    file = 'rpointer.atm'  # '/cluster/work/users/jennybj/noresm/' + case_name + '/run/rpointer.atm'
    place = len(case_name) + 7

    if not os.path.exists(file):
        print('Cannot get current year because ' + file + ' does not exist.')
        stop_due_to_error()

    with open(file, 'r') as f:
        year_current = int(f.readlines()[0][place:place + 4]) - 1

    print('The current year is ', year_current)

    return year_current


def get_coordinate_data():

    return diam_latitudes, diam_longitudes


def damage_function(temp, temp_opt, kappa_plus, kappa_minus):
    """ Calculating the productivity as function of temperature. """

    lower_bound = 0.02

    N = len(temp)
    productivity = np.zeros(N)

    for i in range(N):

        if temp[i] > temp_opt:
            productivity[i] = ((1 - lower_bound) * np.exp(
                -kappa_plus * (temp[i] - temp_opt)**2) + lower_bound)**(1 / (1 - alpha))

        elif temp[i] <= temp_opt:
            productivity[i] = ((1 - lower_bound) * np.exp(
                -kappa_minus * (temp[i] - temp_opt)**2) + lower_bound)**(1 / (1 - alpha))

        else:
            print('Problem with temperature value: ', temp[i])

    return productivity



def regrid_from_noresm_to_diam(noresm_variable):
    """ Function for changing grid of noresm_variable from NorESM2s 2x2 grid to DIAMs 1x1 grid. 
    noresm_variable must be 2D with (lat,lon)"""

    # Stagger ESM grid to match DIAM:
    stag_lat = (noresm_latitudes[1] - noresm_latitudes[0]) / 2
    stag_lon = (noresm_longitudes[1] - noresm_longitudes[0]) / 2

    stag_noresm_latitudes = noresm_latitudes - stag_lat
    stag_noresm_longitudes = noresm_longitudes - stag_lon

    # Create lat and lon with spacing that fits with the DIAM data (1x1):
    interp_lat = np.arange(-90., 90., 1)
    interp_lon = np.arange(0., 360., 1)  # DIAM has lon -180 to 179
    X, Y = np.meshgrid(interp_lat, interp_lon, indexing='ij')

    dlats = 180
    dlons = 360

    variable_out = np.zeros((dlats, dlons))

    # Interpolate data to 1x1:
    f = RegularGridInterpolator(
        (stag_noresm_latitudes, stag_noresm_longitudes),
        noresm_variable,
        bounds_error=False,
        fill_value=None)
    interp_variable = f((X, Y))

    # Shift coordinates to match DAIM:
    variable_out[:, 0:180] = interp_variable[:, 180:]
    variable_out[:, 180:] = interp_variable[:, 0:180]

    return variable_out


def sort_in_diam_order(variable):

    in_latitudes = np.arange(-90., 90., 1)
    in_longitudes = np.arange(-180., 180., 1)

    out_variable = np.zeros(ncells)
    icell = 0

    for lat, lon in zip(diam_latitudes, diam_longitudes):

        ilat = np.where(in_latitudes == lat)[0]
        ilon = np.where(in_longitudes == lon)[0]

        out_variable[icell] = variable[ilat, ilon]
        icell += 1

    return out_variable


def sort_in_diam_order_3D(in_latitudes, in_longitudes, in_variable):

    ncells = diam_latitudes.shape[0]
    nyears = in_variable.shape[0]

    out_variable = np.zeros((ncells, nyears))
    icell = 0

    for lat, lon in zip(diam_latitudes, diam_longitudes):

        ilat = np.where(in_latitudes == lat)[0]
        ilon = np.where(in_longitudes == lon)[0]

        out_variable[icell, :] = in_variable[:, ilat, ilon].reshape(nyears)
        icell += 1

    return out_variable


def from_diam_order_to_grid(in_variable):

    ncells = in_variable.shape[0]
    na = -999.9  # fill values for cells without data

    out_latitudes = np.arange(-90., 90., 1)
    out_longitudes = np.arange(-180., 180., 1)

    diam_latitudes, diam_longitudes = read_coordinate_data()

    out_variable = np.full((180, 360), na)

    for icell in range(ncells):

        index_lat = np.where(out_latitudes == diam_latitudes[icell])
        index_lon = np.where(out_longitudes == diam_longitudes[icell])

        out_variable[index_lat, index_lon] = in_variable[icell]

    # Make into masked array:
    out_variable = ma.masked_equal(out_variable, na)

    return out_variable


def get_noresm_regional_temperatures(year_current, case_name):
    """ Read in monthly temperature data from NorESM2, and calculate annual temperature. """

    ts = datetime.timestamp(datetime.now())

    path = ''  # '/cluster/work/users/jennybj/noresm/' + case_name + '/run/'
    file_name = case_name + '.cam.h0.' + str(year_current) + '-'
    months = [
        '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'
    ]
    temps = np.zeros((12, 96, 144))

    # Read in temperature data:
    for i, month in enumerate(months):

        ncfile = Dataset(path + file_name + month + '.nc')
        temps[i, :, :] = ncfile.variables['TREFHT'][:]
        ncfile.close()

    # Calculate annual average:
    regtemp = np.average(temps,
                         axis=0,
                         weights=np.array([
                             31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
                         ])) - 273.15

    global_temp = np.average(np.average(regtemp, axis=1),
                             axis=0,
                             weights=np.cos(np.deg2rad(noresm_latitudes)))

    # Convert to DIAM grid:
    regtemp = regrid_from_noresm_to_diam(regtemp)
    regtemp = sort_in_diam_order(regtemp)

    #regtemp = regtemp2000  # CHANGE

    print('NorESM2 temperatures read in: ',
          datetime.timestamp(datetime.now()) - ts)

    return global_temp, regtemp


def calculate_annual_mean(variable):

    dim_shape = variable.shape
    nyears = dim_shape[0] // 12

    if len(dim_shape) == 1:
        out = np.zeros(nyears)
    else:
        out = np.zeros((nyears, ) + dim_shape[1:])

    # Calculate annual average:
    for iyear in range(nyears):
        out[iyear] = np.average(variable[iyear * 12:(iyear + 1) * 12],
                                axis=0,
                                weights=month_lengths)

    return out


def calculate_regional_mean(variable, indices, weights):
    """ 
    Calculate the mean for a region. 
    Indices are the indices to calculate average for. 
    Weighting must be specified. 
    """

    ncells = len(indices)
    out = np.zeros(ncells)

    for icell in range(ncells):

        index = indices[icell]

        out[icell] = variable[index]

    average_out = np.average(out, weights=weights)

    return average_out


def read_last_value_in_file(file_name):
    """ Read in last value from file. """

    # Check number of lines is at least 2:
    nlines = get_number_of_line_in_file(file_name)
    if nlines < 2:
        print('No data in ' + file_name)
        stop_due_to_error()

    value = np.loadtxt(file_name, skiprows=nlines - 1)

    if value.shape[0] == 2:
        value = value[-1]

    return value


def write_to_txt_file(year_current, file_name, variable):

    ts = datetime.timestamp(datetime.now())

    if isinstance(variable, float):
        write_to_txt_file_global(year_current, file_name, variable)

    elif variable.shape[0] == ncells:
        write_to_txt_file_gridded(year_current, file_name, variable)

    else:
        print('The variable is neither global nor for all grid cells')
        stop_due_to_error()

    print('Written to file in: ', datetime.timestamp(datetime.now()) - ts)

    return


def write_to_txt_file_gridded(year_current, file_name, variable):

    assert len(variable) == ncells

    save_kwargs = dict(fmt="%16.8f",
                       delimiter="",
                       newline="",
                       footer="\n",
                       comments="")

    if year_current == 1990:

        try:
            os.remove(file_name)
        except OSError:
            pass

        with open(file_name, 'wt') as f:
            np.savetxt(f, diam_latitudes, **save_kwargs)
            np.savetxt(f, diam_longitudes, **save_kwargs)
            np.savetxt(f, variable, **save_kwargs)

    else:

        # Check if file exists:
        if not os.path.exists(file_name):
            print(file_name, ' does not exist')
            stop_due_to_error()

        # Check that number of lines is correct:
        nlines = get_number_of_line_in_file(file_name)
        if nlines - 2 != year_current - 1990:
            print('File contains the wrong number of lines/years')
            stop_due_to_error()

        with open(file_name, 'at') as f:
            np.savetxt(f, variable, **save_kwargs)

    return


def write_to_txt_file_global(year_current, file_name, variable):

    if year_current == 1990:

        try:
            os.remove(file_name)
        except OSError:
            pass

        with open(file_name, 'wt') as f:
            f.write('% 8d' % (year_current))
            f.write('%16.8f' % variable)
            f.write('\n')

    else:

        # Check if file exists:
        if not os.path.exists(file_name):
            print(file_name, ' does not exist')
            stop_due_to_error()

        # Check that number of lines is correct:
        nlines = get_number_of_line_in_file(file_name)
        if nlines != year_current - 1990:
            print('File contains the wrong number of lines/years')
            stop_due_to_error()

        with open(file_name, 'at') as f:
            f.write('% 8d' % (year_current))
            f.write('%16.8f' % variable)
            f.write('\n')

    return


def make_emissions_file(case_name, emissions_file):
    """ Make NetCDF4 file to read in NorESM2. """

    ts = datetime.timestamp(datetime.now())

    # File names:
    outfile = 'input_emissions_' + case_name + '.nc'  # '/cluster/home/jennybj/input_emissions_' + case_name + '.nc'

    # Constants:
    earth_radius = 6.3781e6
    extra_years = 5

    #-------------------------------------------------------------------------

    # READ IN CO2 DATA

    # Read CO2 vaules from file:
    diam_co2 = np.loadtxt(emissions_file)
    nyears = diam_co2.shape[0] - 2 + extra_years

    # Extract the values:
    diam_lat = diam_co2[0, :]
    diam_lon = diam_co2[1, :]
    diam_co2 = diam_co2[2:, :].transpose()

    extra_co2 = np.zeros((ncells, extra_years))
    for iyear in range(extra_years):
        extra_co2[:, iyear] = diam_co2[:, -1]

    diam_co2 = np.concatenate((diam_co2, extra_co2), axis=1)
    print(diam_co2.shape)

    global_emissions = np.sum(diam_co2, axis=0)
    print('Global emissions in MtC: ', global_emissions)

    #-------------------------------------------------------------------------

    # PUT EMISSIONS INTO GRID

    # Put the DIAM emission data into an array (time, lat, lon):
    gridded_lat = np.arange(-90, 91)
    gridded_lon = np.arange(-180, 180)
    gridded_co2 = np.zeros((nyears, len(gridded_lat), len(gridded_lon)))

    for i in range(ncells):

        # Find index of coordinates in the new array:
        index_lat = np.where(diam_lat[i] == gridded_lat)
        index_lon = np.where(diam_lon[i] == gridded_lon)

        # Place emission data in correct place in array:
        gridded_co2[:, index_lat[0][0], index_lon[0][0]] += diam_co2[i, :]

#-------------------------------------------------------------------------

# INTERPOLATE EMISSIONS FROM DIAM TO NorESM GRID

# Specify arrays:
    interp_lat = gridded_lat
    interp_lon = np.arange(0., 360., 1)  # DIAM has lon -180 to 179
    noresm_co2 = np.zeros((nyears, nlats, nlons))

    # Shift longitude coordinates to match NorESM:
    interp_co2 = np.zeros(gridded_co2.shape)
    interp_co2[:, :, 0:180] = gridded_co2[:, :, 180:]
    interp_co2[:, :, 180:] = gridded_co2[:, :, 0:180]

    X, Y = np.meshgrid(noresm_latitudes, noresm_longitudes, indexing='ij')

    # Interpolate:
    for i in range(gridded_co2.shape[0]):

        smoothing = gaussian_filter(interp_co2[i, :, :],
                                    sigma=1)  # to improve interpolation

        f = RegularGridInterpolator((interp_lat, interp_lon),
                                    smoothing,
                                    bounds_error=False,
                                    fill_value=None)
        interpolated = f((X, Y))

        # Remove emissions from +/- 90 degrees latitude:
        interpolated[0, :] = 0
        interpolated[-1, :] = 0

        # Make sure the sum of the emissions stays the same:
        ratio = global_emissions[i] / np.sum(interpolated)
        noresm_co2[i, :, :] = interpolated * ratio

        print(np.sum(noresm_co2[i, :, :]))

    #-------------------------------------------------------------------------

    # CHANGE CO2 DATA TO CORRECT FORMAT

    # Convert from ktC to CO2 kg s-1:
    noresm_co2 *= 3.67  # MtC to MtCO2
    noresm_co2 *= 1e9  # Mt to kg
    noresm_co2 /= (365 * 24 * 60 * 60)  # kg to kg s-1

    dlats = noresm_latitudes[1] - noresm_latitudes[0]

    # Convert from kg s-1 to kg m-2 s-1:
    for i in range(1, noresm_latitudes.shape[0] - 1):

        lat = noresm_latitudes[i]

        cell_area = np.pi / 180 * earth_radius**2 * np.abs(
            np.sin(np.deg2rad(lat - 0.5 * dlats)) -
            np.sin(np.deg2rad(lat + 0.5 * dlats))) * 2.5

        noresm_co2[:, i, :] /= cell_area  # kg s-1 to kg m-2 s-2

    # A = 2*pi*R^2 |sin(lat1)-sin(lat2)| |lon1-lon2|/360
    # = (pi/180)R^2 |sin(lat1)-sin(lat2)| |lon1-lon2|

    # Repeat value, as file must have monthly values and at least 1 extra year
    temp = np.zeros((nyears * 12, nlats, nlons))
    for i in range(nyears):
        temp[i * 12:(i + 1) * 12, :, :] = noresm_co2[i, :, :]
    noresm_co2 = temp

    print(np.sum(noresm_co2[0, :, :]))

    #-------------------------------------------------------------------------

    # MAKE RELEVANT TIME VARIABLES

    # Create time and date values (middle of every month):
    time_val = []
    date_val = []
    bound_val = []
    start_day = np.array(
        [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    start_year = np.array([
        19900116, 19900215, 19900316, 19900416, 19900516, 19900616, 19900716,
        19900816, 19900916, 19901016, 19901116, 19901216
    ])
    start_bound = np.array([[0, 31], [31, 59], [59, 90], [90, 120], [120, 151],
                            [151, 181], [181, 212], [212, 243], [243, 273],
                            [273, 304], [304, 334], [334, 365]])

    print('The file is created for ', nyears, 'years from ', start_year[0])

    for year in range(nyears):
        time_val += (start_day + year * 365).tolist()
        date_val += (start_year + year * 10000).tolist()
        bound_val += (start_bound + year * 365).tolist()

    #-------------------------------------------------------------------------

    # MAKE OUTPUT FILE

    # Delete if file exists:
    if os.path.exists(outfile):
        os.remove(outfile)

    # Create new files open for writing:
    file = Dataset(outfile, 'w')

    # Create the dimensions:
    file.createDimension('time', None)
    file.createDimension('bound', 2)
    file.createDimension('lat', nlats)
    file.createDimension('lon', nlons)

    # Create time variable:
    time = file.createVariable(varname='time',
                               datatype='d',
                               dimensions=('time'))
    time[:] = time_val  # fill values
    time.units = 'days since 1990-01-01 00:00:00'  # set attributes
    time.long_name = 'time'
    time.calendar = 'noleap'
    time.axis = 'T'
    time.bounds = 'time_bnds'
    time.realtopology = 'linear'
    time.standard_name = 'time'

    # Create date variable:
    date = file.createVariable(varname='date',
                               datatype='i',
                               dimensions=('time', ))
    date[:] = date_val
    date.long_name = 'date'
    date.units = 'YYYYMMDD'

    # Create time bounds variable:
    time_bnds = file.createVariable(varname='time_bnds',
                                    datatype='d',
                                    dimensions=('time', 'bound'))
    time_bnds[:] = bound_val

    # Create the latitude variable:
    lat = file.createVariable(varname='lat',
                              datatype='d',
                              dimensions=('lat', ))
    lat[:] = noresm_latitudes
    lat.long_name = 'latitude'
    lat.units = 'degrees_north'

    # Create the longitude variable:
    lon = file.createVariable(
        varname='lon', datatype='d',
        dimensions=('lon', ))  # create dimension variable
    lon[:] = noresm_longitudes
    lon.long_name = 'longitude'
    lon.units = 'degrees_east'

    # Create co2 flux variable:
    noresm_co2 = noresm_co2.tolist()
    CO2_flux = file.createVariable(varname='CO2_flux',
                                   datatype='f',
                                   dimensions=('time', 'lat', 'lon'),
                                   fill_value=1e20)
    CO2_flux[:] = noresm_co2
    CO2_flux.missing_vale = np.array(1e20, dtype=np.float32)
    CO2_flux.cell_method = 'time: mean'
    CO2_flux.long_name = 'CO2 Anthropogenic Emissions'
    CO2_flux.units = 'kg m-2 s-1'

    file.close()

    print('Created emissions file in: ',
          datetime.timestamp(datetime.now()) - ts)

    return


#----------------------------------------------------------------------------------------
