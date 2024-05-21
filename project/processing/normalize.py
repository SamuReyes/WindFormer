import numpy as np
import os
import h5py
import pickle


def normalize_data(config: dict):
    """
    Normalizes surface and upper layer data arrays using their mean and standard deviation.

    Parameters:
    - config (dict): Configuration dictionary.
    """

    data_file = os.path.join(config['global']['path'], config['global']['processed_data_path'], config['global']['data_file'])
    constants_path = os.path.join(config['global']['path'], config['global']['constants_path'])

    upper_vars = len(config['preprocessing']['upper_var_names']) - 1  # Remove time
    surface_vars = len(config['preprocessing']['surface_var_names']) - 1  # Remove time
    years = config['preprocessing']['limits']['years']
    levels = config['preprocessing']['limits']['levels'][1] + 1  # Add surface level

    # Dictionaries to store mean and standard deviation
    mean_dict = {level: np.zeros(upper_vars) if level > 0 else np.zeros(surface_vars) for level in range(levels)}
    std_dict = {level: np.zeros(upper_vars) if level > 0 else np.zeros(surface_vars) for level in range(levels)}
    mean_dict_years = {
        year: {level: np.zeros(8) if level > 0 else np.zeros(9) for level in range(levels)} for year in range(years[0], years[1] + 1)
    }

    # Number of data points to compute mean and std
    N_std = 0

    # Compute mean from HDF5
    with h5py.File(data_file, 'r') as file:
        for year in range(years[0], years[1] + 1):
            # Compute the mean
            subset = np.array(file[str(year)]["surface"]["data"])
            mean_dict_years[year][0] += subset.mean(axis=(0, 1, 2))

            for level in range(1, levels):
                subset = np.array(file[str(year)]["upper"]["data"])[:, level-1, :, :, :]
                mean_dict_years[year][level] += subset.mean(axis=(0, 1, 2))

    # Aggregate mean from years
    for year in range(years[0], years[1] + 1):
        for level in range(levels):
            mean_dict[level] += mean_dict_years[year][level]

    # Compute mean from years
    for key in mean_dict:
        mean_dict[key] /= (years[1] - years[0] + 1)

    # Compute standard deviation from data
    with h5py.File(data_file, 'r') as file:
        for year in range(years[0], years[1] + 1):
            # Get the number of data points
            subset = file[str(year)]["surface"]["data"]
            N_std += subset.shape[0] * subset.shape[-3] * subset.shape[-2]

            # Compute the standard deviation
            subset = np.array(file[str(year)]["surface"]["data"])
            subset = (subset - mean_dict[0])**2
            std_dict[0] += subset.sum(axis=(0, 1, 2))
            for level in range(1, levels):
                subset = np.array(file[str(year)]["upper"]["data"])[:, level-1, :, :, :]
                subset = (subset - mean_dict[level])**2
                std_dict[level] += subset.sum(axis=(0, 1, 2))
    for key in std_dict:
        std_dict[key] = np.sqrt(std_dict[key] / N_std)

    # Normalize data using mean and standard deviation computed
    with h5py.File(data_file, "r+") as f:
        for year in f.keys():
            f[year]["surface"]["data"][:] = (
                f[year]["surface"]["data"][:] - mean_dict[0]) / std_dict[0]

            for level in range(1, levels):
                f[year]["upper"]["data"][:, level-1, :, :, :] = (
                    f[year]["upper"]["data"][:, level-1, :, :, :] - mean_dict[level]) / std_dict[level]

    # Save mean and std
    statistics = {
        'mean': mean_dict,
        'std': std_dict
    }
    with open(os.path.join(constants_path, 'statistics.pickle'), 'wb') as handle:
        pickle.dump(statistics, handle, protocol=pickle.HIGHEST_PROTOCOL)
