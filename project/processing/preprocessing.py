import netCDF4 as nc
import numpy as np
import glob
import os
import pickle
from calendar import monthrange
from datetime import datetime, timedelta
from utils.utils import *

def extract_data(path: str, var_names: list, level: str, levels: list, latitude: list, longitude: list):
    """
    Extracts specified variables from netCDF files and concatenates them into a single array.

    Parameters:
    - path (str): Path to the directory containing the netCDF files.
    - var_names (list): List of variable names to extract.
    - level (str): The atmospheric level ('upper' or 'surface') to filter files by.
    - levels (list): The range of vertical levels to slice (for 'upper' data).
    - latitude (list): The latitude range to slice.
    - longitude (list): The longitude range to slice.

    Returns:
    - dict: A dictionary where keys are variable names and values are concatenated data arrays.

    Raises:
    - ValueError: If the specified level is not 'upper' or 'surface'.
    """
    # Determine file pattern based on atmospheric level
    if level == 'upper':
        files = sorted(glob.glob(path + '/*-*-upper.nc'))

    elif level == 'surface':
        files = sorted(glob.glob(path + '/*-*-surface.nc'))
    else:
        raise ValueError('level must be "upper" or "surface"')

    # Dictionary to store concatenated data
    data = {var: [] for var in var_names}

    for file in files:
        with nc.Dataset(file, 'r') as nc_file:
            for var in var_names:
                if var == 'time':
                    data[var].append(nc_file.variables[var][:])
                    data[var] = data[var]
                else:
                    # Slice data according to specified levels and geographical bounds
                    if level == 'upper':
                        data[var].append(nc_file.variables[var][:, levels[0]:levels[1],
                                         latitude[0]:latitude[1], longitude[0]:longitude[1]])
                    elif level == 'surface':
                        data[var].append(
                            nc_file.variables[var][:, latitude[0]:latitude[1], longitude[0]:longitude[1]])

    # Concatenate data for each variable
    for var in var_names:
        data[var] = np.concatenate(data[var], axis=0)
        data[var] = data[var].filled(np.nan)
        data[var] = data[var].astype(np.float32)

    return data


def save_constants(raw_data_path: str, constants_path: str, levels: list, latitude: list, longitude: list):
    """
    Extracts and saves constant values from netCDF files as NumPy arrays.

    Parameters:
    - raw_data_path (str): Path to the directory containing raw netCDF files.
    - constants_path (str): Destination path to save the extracted constant arrays.
    - levels (list): Indices of levels to extract (for upper level data).
    - latitude (list): Indices of latitude range to extract.
    - longitude (list): Indices of longitude range to extract.
    """

    # Extract paths for upper and surface files randomly
    upper_files = glob.glob(os.path.join(raw_data_path, '*-upper.nc'))
    surface_files = glob.glob(os.path.join(raw_data_path, '*-surface.nc'))

    # Process and save constants from an upper level file
    with nc.Dataset(upper_files[-1], 'r') as nc_file:
        np.save(os.path.join(constants_path, 'latitude.npy'),
                nc_file.variables["latitude"][latitude[0]:latitude[1]+1].filled(np.nan).astype(np.float16))
        np.save(os.path.join(constants_path, 'longitude.npy'),
                nc_file.variables["longitude"][longitude[0]:longitude[1]+1].filled(np.nan).astype(np.float16))
        np.save(os.path.join(constants_path, 'level.npy'),
                nc_file.variables["level"][levels[0]:levels[1]+1].filled(np.nan).astype(np.uint16))

    # Process and save constants from a surface level file
    with nc.Dataset(surface_files[-1], 'r') as nc_file:
        np.save(os.path.join(constants_path, 'land_sea_mask.npy'), 
                nc_file.variables["lsm"][:, latitude[0]:latitude[1]+1, longitude[0]:longitude[1]+1].filled(np.nan).astype(np.float32))


def preprocess_data(config: dict):
    """
    Main preprocessing function that extracts data and constants from netCDF files and saves
    them in Numpy arrays.

    Parameters:
    - config (dict): Configuration dictionary.
    """

    # Extract configuration settings
    upper_var_names = config['preprocessing']['upper_var_names']
    surface_var_names = config['preprocessing']['surface_var_names']
    raw_data_path = os.path.join(config['global']['path'], config['global']['raw_data_path'])
    constants_path = os.path.join(config['global']['path'], config['global']['constants_path'])
    processed_data_path = os.path.join(config['global']['path'], config['global']['processed_data_path'])
    
    # Extraction bounds
    longitude = config['preprocessing']['limits']['longitude']
    latitude = config['preprocessing']['limits']['latitude']
    levels = config['preprocessing']['limits']['levels']

    # Ensure output directories exist
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(constants_path, exist_ok=True)

    # Extract data
    upper_data = extract_data(
        raw_data_path, upper_var_names, 'upper', levels, latitude, longitude)
    surface_data = extract_data(
        raw_data_path, surface_var_names, 'surface', levels, latitude, longitude)

    # Save constants and time
    save_constants(raw_data_path, constants_path, levels, latitude, longitude)
    time = upper_data['time']
    np.save(os.path.join(processed_data_path, 'time.npy'), time.astype(np.uint32))

    del upper_data['time']
    del surface_data['time']

    upper_var_names.remove('time')
    surface_var_names.remove('time')

    # Dictionary to single numpy array
    combined_upper_data = dictionary_to_array(upper_data)
    combined_surface_data = dictionary_to_array(surface_data)

    del upper_data
    del surface_data

    # Remove seasonality
    with open(os.path.join(constants_path, 'climatologies.pickle'), 'rb') as handle:
        climatologies = pickle.load(handle)

    surface_climatologies = climatologies['surface_climatologies']
    upper_climatologies = climatologies['upper_climatologies']

    for i, time_step in enumerate(time):
        month = hour_to_datetime(time_step).month
        day = hour_to_datetime(time_step).day - 1 # Adjust for 0-based indexing

        combined_upper_data[i] -= upper_climatologies[month][day]
        combined_surface_data[i] -= surface_climatologies[month][day]

    del surface_climatologies
    del upper_climatologies
    del time

    # Save data
    np.save(os.path.join(processed_data_path, 'surface.npy'), combined_surface_data)
    np.save(os.path.join(processed_data_path, 'upper.npy'), combined_upper_data)

    del combined_upper_data
    del combined_surface_data
