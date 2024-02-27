import numpy as np
import os


def get_labels(config: dict):
    """
    Extracts specific variables from surface and upper data and combines them into a single label dataset.

    :param config: Dictionary containing configuration paths and variable names to be used for extracting labels.
    """

    # Load surface and upper data
    surface_data = np.load(os.path.join(
        config['global']['path'], config['global']['processed_data_path'], 'surface.npy'))
    upper_data = np.load(os.path.join(
        config['global']['path'], config['global']['processed_data_path'], 'upper.npy'))

    # Retrieve the names of variables from the configuration for surface and upper data
    upper_var_names = config['preprocessing']['upper_var_names']
    surface_var_names = config['preprocessing']['surface_var_names']

    # Determine the indices of 'u10', 'v10', 'u100', 'v100' for surface and 'u', 'v' for upper
    surface_vars_indices = [surface_var_names.index(var) for var in [
        'u10', 'v10']]
    middle_vars_indices = [surface_var_names.index(var) for var in [
        'u100', 'v100']]
    upper_vars_indices = [upper_var_names.index(var) for var in ['u', 'v']]

    middle_data = surface_data[:, :, :, middle_vars_indices]
    surface_data = surface_data[:, :, :, surface_vars_indices]
    upper_data = upper_data[:, :, :, :, upper_vars_indices]

    # Expand dimensions of surface data and combine it with upper data to form a unified label dataset
    surface_data = np.expand_dims(surface_data, axis=1)
    middle_data = np.expand_dims(middle_data, axis=1)
    combined_labels = np.concatenate((surface_data, middle_data, upper_data), axis=1)

    # Save the combined label dataset to a file
    np.save(os.path.join(config['global']['path'], config['global']
            ['processed_data_path'], 'labels.npy'), combined_labels)
