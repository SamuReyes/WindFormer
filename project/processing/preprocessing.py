import netCDF4 as nc
import numpy as np
import h5py
import glob
import os
import pickle
from utils.utils import dictionary_to_array, hour_to_datetime


def extract_data(path: str, var_names: list, level: str, levels: list, latitude: list, longitude: list, year: int):
    """
    Extracts specified variables from netCDF files and concatenates them into a single array.

    Parameters:
    - path (str): Path to the directory containing the netCDF files.
    - var_names (list): List of variable names to extract.
    - level (str): The atmospheric level ('upper' or 'surface') to filter files by.
    - levels (list): The range of vertical levels to slice (for 'upper' data).
    - latitude (list): The latitude range to slice.
    - longitude (list): The longitude range to slice.
    - year (int): The year to extract data for.

    Returns:
    - dict: A dictionary where keys are variable names and values are concatenated data arrays.
    """

    # Determine file pattern based on atmospheric level and year
    files_pattern = f"{path}/{year}-*-{level}.nc"
    files = sorted(glob.glob(files_pattern))

    # Check if the list of files is empty
    if not files:
        print(
            f"No data files found for the year {year}. Possible missing data for this period.")

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
                        data[var].append(nc_file.variables[var][:, latitude[0]:latitude[1], longitude[0]:longitude[1]])

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


def join_hdf5_data(directory):
    """
    Reads upper and surface data from multiple HDF5 files and organizes them into a single HDF5 file.
    The data is structured into groups by year, each containing subgroups for 'upper' and 'surface' data.

    Parameters:
    - directory: The directory where the HDF5 files are located and where the output file will be saved.
    """
    # Patterns to search for upper and surface files
    upper_files_pattern = os.path.join(directory, "*-upper.hdf5")
    surface_files_pattern = os.path.join(directory, "*-surface.hdf5")

    # List the files matching the patterns
    upper_files = sorted(glob.glob(upper_files_pattern))
    surface_files = sorted(glob.glob(surface_files_pattern))

    # Output file path
    output_file_path = os.path.join(directory, "data.hdf5")

    # Create or open the output HDF5 file
    with h5py.File(output_file_path, 'w') as output_file:
        # Process upper files
        for file_path in upper_files:
            year = os.path.basename(file_path).split('-')[0]
            with h5py.File(file_path, 'r') as input_file:
                for dataset_name in input_file:
                    data = input_file[dataset_name][()]
                    # Create a subgroup for upper data within the year group
                    year_group = output_file.require_group(year)
                    upper_group = year_group.require_group('upper')
                    # Use a generic or simplified dataset name
                    upper_group.create_dataset('data', data=data)
            # Delete the file after processing
            os.remove(file_path)

        # Process surface files
        for file_path in surface_files:
            year = os.path.basename(file_path).split('-')[0]
            with h5py.File(file_path, 'r') as input_file:
                for dataset_name in input_file:
                    data = input_file[dataset_name][()]
                    # Create a subgroup for surface data within the year group
                    year_group = output_file.require_group(year)
                    surface_group = year_group.require_group('surface')
                    # Use a generic or simplified dataset name
                    surface_group.create_dataset('data', data=data)
            # Delete the file after processing
            os.remove(file_path)


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
    years = config['preprocessing']['limits']['years']

    # Ensure output directories exist
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(constants_path, exist_ok=True)

    # Load climatologies
    with open(os.path.join(constants_path, 'climatologies.pickle'), 'rb') as handle:
        climatologies = pickle.load(handle)

    surface_climatologies = climatologies['surface_climatologies']
    upper_climatologies = climatologies['upper_climatologies']

    # Extract and save data for each year
    for year in range(years[0], years[1] + 1):
        # Extract data
        upper_data = extract_data(
            raw_data_path, upper_var_names, 'upper', levels, latitude, longitude, year)
        surface_data = extract_data(
            raw_data_path, surface_var_names, 'surface', levels, latitude, longitude, year)

        time = upper_data['time']
        del upper_data['time']
        del surface_data['time']

        # Dict to array
        upper_data = dictionary_to_array(upper_data)
        surface_data = dictionary_to_array(surface_data)

        for i, time_step in enumerate(time):
            month = hour_to_datetime(time_step).month
            # Adjust for 0-based indexing
            day = hour_to_datetime(time_step).day - 1
            upper_data[i] -= upper_climatologies[month][day]
            surface_data[i] -= surface_climatologies[month][day]

        # Save data
        with h5py.File(os.path.join(processed_data_path, f"{year}-upper.hdf5"), 'w') as f:
            f.create_dataset(f"{year}-upper", data=upper_data)
        with h5py.File(os.path.join(processed_data_path, f"{year}-surface.hdf5"), 'w') as f:
            f.create_dataset(f"{year}-surface", data=surface_data)

    join_hdf5_data(processed_data_path)

    # Save constants and time
    save_constants(raw_data_path, constants_path, levels, latitude, longitude)
