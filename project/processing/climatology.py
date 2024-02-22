import numpy as np
import xarray as xr
import glob
import re
import os
import pickle
from calendar import monthrange
from utils.utils import *

def get_climatology(file, longitude, latitude, levels, level, vars):
    """
    Extracts the climatology from a month file
    
    Parameters:
    - file (str): The file path to extract the climatology from.

    Returns:
    - numpy.ndarray: The climatology data.
    """
    
    data = xr.open_dataset(file)

    # Select the region of interest
    data = data.isel(
        longitude=slice(longitude[0], longitude[1]),
        latitude=slice(latitude[0], latitude[1])
    )
    if level == 'upper':
        data = data.isel(level=slice(levels[0], levels[1]))

    # Remove unwanted variables
    vars_to_remove = [var for var in data.variables if var not in vars]
    data = data.drop_vars(vars_to_remove)

    # Compute climatology
    climatology = data.groupby('time.day').mean('time')
    climatology = climatology.to_array().values
    # Place vars in the last dimension
    num_dims = climatology.ndim
    axes = tuple(range(1, num_dims)) + (0,)
    climatology = np.transpose(climatology, axes)

    # If not a leap year, extend february to 29 days
    match = re.search(r'/(\d{4})-02-(surface|upper)\.nc$', file)
    
    if match and not is_leap_year(int(match.group(1))):
        last_day = climatology[-1]
        last_day_expanded = np.expand_dims(last_day, axis=0)
        climatology = np.concatenate((climatology, last_day_expanded), axis=0)

    return climatology

def process_month(path: str, month: int, level: str, longitude, latitude, levels, vars):
    """
    Processes the monthly data for a given month and level.
    
    Parameters:
    - path (str): The path to the directory containing the netCDF files.
    - month (int): The month to process.
    - levels (str): 'Surface' or 'upper'.

    Returns:
    - numpy.ndarray: The monthly data.
    """

    latitudes = latitude[1] - latitude[0]
    longitudes = longitude[1] - longitude[0]
    pressure_levels = levels[1] - levels[0]
    variables = len(vars) - 1 # Remove time

    if level == 'surface':
        monthly_data = np.zeros((monthrange(2000, month)[1], latitudes, longitudes, variables))
    elif level == 'upper':
        monthly_data = np.zeros((monthrange(2000, month)[1], pressure_levels, latitudes, longitudes, variables))
    else:
        raise ValueError('level must be "upper" or "surface"')

    file_pattern = f"{path}/*-{str(month).zfill(2)}-{level}.nc"
    files = sorted(glob.glob(file_pattern))

    # TODO: parallelize this
    for file in files:
        np_climatology = get_climatology(file, longitude, latitude, levels, level, vars)
        monthly_data += np_climatology

    return monthly_data / len(files)

def process_climatology(config: dict):
    """
    Processes the climatology data for all months and levels.
    
    Parameters:
    - config (dict): Configuration dictionary.
    """

    # Extraction bounds
    longitude = config['preprocessing']['limits']['longitude']
    latitude = config['preprocessing']['limits']['latitude']
    levels = config['preprocessing']['limits']['levels']

    # Vars to keep
    upper_var_names = config['preprocessing']['upper_var_names']
    surface_var_names = config['preprocessing']['surface_var_names']

    raw_data_path = os.path.join(
        config['global']['path'], config['global']['raw_data_path'])
    constants_path = os.path.join(
        config['global']['path'], config['global']['constants_path'])

    os.makedirs(constants_path, exist_ok=True)

    # Compute climatology
    surface_climatologies = {}
    upper_climatologies = {}

    # TODO: parallelize this
    for month in range(1, 13):
        surface_climatologies[month] = process_month(raw_data_path, month, 'surface', longitude, latitude, levels, surface_var_names)
        upper_climatologies[month] = process_month(raw_data_path, month, 'upper', longitude, latitude, levels, upper_var_names)

    # Save climatologies
    climatologies = {
        'surface_climatologies': surface_climatologies,
        'upper_climatologies': upper_climatologies
    }
    with open(os.path.join(constants_path, 'climatologies.pickle'), 'wb') as handle:
        pickle.dump(climatologies, handle, protocol=pickle.HIGHEST_PROTOCOL)