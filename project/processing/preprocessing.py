import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob
import os, sys

project_directory = os.path.abspath('..')
sys.path.append(project_directory)

from project_directory.utils.config_loader import load_config

PATH = '/home/samuel/Documents/local/projects/UPM/TFM/project/raw_data'

LONGITUDE = (18, 35)
LATITUDE = (10, 28)
LEVELS = (0, 8)

def extract_data(path, var_names, level):
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
                else:
                    if level == 'upper':
                        data[var].append(nc_file.variables[var][:,LEVELS[0]:LEVELS[1]+1, LATITUDE[0]:LATITUDE[1]+1, LONGITUDE[0]:LONGITUDE[1]+1])
                    elif level == 'surface':
                        data[var].append(nc_file.variables[var][:,LATITUDE[0]:LATITUDE[1]+1, LONGITUDE[0]:LONGITUDE[1]+1])

    # Concatenate data for each variable
    for var in var_names:
        data[var] = np.concatenate(data[var], axis=0)
        data[var] = data[var].filled(np.nan)

    return data


if __name__ == '__main__':

    upper_var_names = ['time', 'u', 'v', 'q', 't', 'd', 'z', 'w', 'vo']
    surface_var_names = ['time', 'u100', 'v100', 'u10', 'v10', 'd2m', 't2m', 'z', 'msl']

    upper_data = extract_data(PATH, upper_var_names, 'upper')
    surface_data = extract_data(PATH, surface_var_names, 'surface')


    with nc.Dataset('/home/samuel/Documents/local/projects/UPM/TFM/project/raw_data/2021-02-upper.nc', 'r') as nc_file:
        np.save('./config/latitude.npy', nc_file.variables["latitude"][:].filled(np.nan).astype(np.float16))
        np.save('./config/longitude.npy', nc_file.variables["longitude"][:].filled(np.nan).astype(np.float16))
        np.save('./config/level.npy', nc_file.variables["level"][:].filled(np.nan).astype(np.uint16))
        
    with nc.Dataset('/home/samuel/Documents/local/projects/UPM/TFM/project/raw_data/2021-02-surface.nc', 'r') as nc_file:
        np.save('./config/land_sea_mask.npy', nc_file.variables["lsm"][:].filled(np.nan).astype(np.float32))

    np.save('./data/time.npy', upper_data['time'].astype('uint32'))

    del upper_data['time']
    del surface_data['time']

    upper_var_names.remove('time')
    surface_var_names.remove('time')


    for var in upper_data.keys():
        upper_data[var] = upper_data[var].astype(np.float32)

    for var in surface_data.keys():
        surface_data[var] = surface_data[var].astype(np.float32)

    surface_data_arrays = []
    upper_data_arrays = []

    for var in upper_data.keys():
        print(var, upper_data[var].shape)
        upper_data_arrays.append(upper_data[var])

    for var in surface_data.keys():
        print(var, surface_data[var].shape)
        surface_data_arrays.append(surface_data[var])


    combined_upper_data = np.stack(upper_data_arrays, axis=-1)
    combined_surface_data = np.stack(surface_data_arrays, axis=-1)

    del surface_data_arrays
    del upper_data_arrays
    del upper_data
    del surface_data

    np.save('./data/surface.npy', combined_surface_data)
    np.save('./data/upper.npy', combined_upper_data)