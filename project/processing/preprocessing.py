import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def dictionary_to_array(dictionary):
    """ Converts dictionary to numpy array """

    array = np.stack([dictionary[key] for key in dictionary.keys()], axis=-1)

    return array


def extract_data(path:str, var_names:list, level:str, levels:list, latitude:list, longitude:list):
    """ Extracts data from netCDF files and concatenates it into a single array """

    if level == 'upper':
        files = sorted(glob.glob(path + '/*-*-upper.nc'))

    elif level == 'surface':
        files = sorted(glob.glob(path + '/*-*-surface.nc'))
    else:
        raise ValueError('level must be "upper" or "surface"')

    # Dictionary to store concatenated data
    data = {var: [] for var in var_names}

    # Itearte over each file
    for file in files:
        with nc.Dataset(file, 'r') as nc_file:
            for var in var_names:
                if var == 'time':
                    data[var].append(nc_file.variables[var][:])
                    data[var] = data[var]
                else:
                    if level == 'upper':
                        data[var].append(nc_file.variables[var][:,levels[0]:levels[1]+1, latitude[0]:latitude[1]+1, longitude[0]:longitude[1]+1])
                    elif level == 'surface':
                        data[var].append(nc_file.variables[var][:,latitude[0]:latitude[1]+1, longitude[0]:longitude[1]+1])

    # Concatenate data for each variable
    for var in var_names:
        data[var] = np.concatenate(data[var], axis=0)
        data[var] = data[var].filled(np.nan)
        data[var] = data[var].astype(np.float32)

    return data


def save_constants(raw_data_path:str, constants_path:str, levels:list, latitude:list, longitude:list):
    """ Extracts constants from netCDF files and saves them as numpy arrays """
    
    # Obtener archivos aleatorios
    upper_files = glob.glob(os.path.join(raw_data_path, '*-upper.nc'))
    surface_files = glob.glob(os.path.join(raw_data_path, '*-surface.nc'))

    # Procesar archivo upper
    with nc.Dataset(upper_files[-1], 'r') as nc_file:
        np.save(os.path.join(constants_path, 'latitude.npy'), nc_file.variables["latitude"][latitude[0]:latitude[1]+1].filled(np.nan).astype(np.float16))
        np.save(os.path.join(constants_path, 'longitude.npy'), nc_file.variables["longitude"][longitude[0]:longitude[1]+1].filled(np.nan).astype(np.float16))
        np.save(os.path.join(constants_path, 'level.npy'), nc_file.variables["level"][levels[0]:levels[1]+1].filled(np.nan).astype(np.uint16))
    
    # Procesar archivo surface
    with nc.Dataset(surface_files[-1], 'r') as nc_file:
        np.save(os.path.join(constants_path, 'land_sea_mask.npy'), nc_file.variables["lsm"][:,latitude[0]:latitude[1]+1, longitude[0]:longitude[1]+1].filled(np.nan).astype(np.float32))


def preprocess_data(config:dict):
    """ Extracts data from netCDF files and saves it as numpy arrays """

    upper_var_names = config['preprocessing']['upper_var_names']
    surface_var_names = config['preprocessing']['surface_var_names']
    raw_data_path = os.path.join(config['global']['path'], config['global']['raw_data_path'])
    constants_path = os.path.join(config['global']['path'], config['global']['constants_path'])
    processed_data_path = os.path.join(config['global']['path'], config['global']['processed_data_path'])

    longitude = config['preprocessing']['limits']['longitude']
    latitude = config['preprocessing']['limits']['latitude']
    levels = config['preprocessing']['limits']['levels']

    # Extract data
    upper_data = extract_data(raw_data_path, upper_var_names, 'upper', levels, latitude, longitude)
    surface_data = extract_data(raw_data_path, surface_var_names, 'surface', levels, latitude, longitude)

    # Save constants
    save_constants(raw_data_path, constants_path, levels, latitude, longitude)

    # Save time
    np.save(os.path.join(processed_data_path,'time.npy'), upper_data['time'].astype(np.uint32))

    del upper_data['time']
    del surface_data['time']

    upper_var_names.remove('time')
    surface_var_names.remove('time')

    # Dictionary to single numpy array
    combined_upper_data = dictionary_to_array(upper_data)
    combined_surface_data = dictionary_to_array(surface_data)

    del upper_data
    del surface_data

    # Save data
    np.save(os.path.join(processed_data_path,'surface_processed.npy'), combined_surface_data)
    np.save(os.path.join(processed_data_path,'upper_processed.npy'), combined_upper_data)