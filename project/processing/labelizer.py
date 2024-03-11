import numpy as np
import os
import h5py


def extract_labels(surface_data: np.ndarray, upper_data: np.ndarray, surface_indices: list, middle_indices: list, upper_indices: list) -> np.ndarray:
    """
    Extracts labels from surface and upper data and combines them into a single label dataset.

    :param surface_data: Numpy array representing 'surface' data of shape (T, W, H, C).
    :param upper_data: Numpy array representing 'upper' data of shape (T, L, W, H, C).
    :param surface_indices: List of indices for 'u10' and 'v10' in the surface data.
    :param middle_indices: List of indices for 'u100' and 'v100' in the surface data.
    :param upper_indices: List of indices for 'u' and 'v' in the upper data.

    :return: Numpy array representing the combined label dataset of shape (T, L, W, H, C).
    """

    middle_data = surface_data[:, :, :, middle_indices]
    surface_data = surface_data[:, :, :, surface_indices]
    upper_data = upper_data[:, :, :, :, upper_indices]

    # Expand dimensions of surface data and combine it with upper data to form a unified label dataset
    surface_data = np.expand_dims(surface_data, axis=1)
    middle_data = np.expand_dims(middle_data, axis=1)
    combined_labels = np.concatenate(
        (surface_data, middle_data, upper_data), axis=1)

    return combined_labels


def get_labels(config: dict):
    """
    Extracts labels from the processed data and saves them to the data file.

    :param config: Dictionary containing the configuration settings.
    """

    filename = os.path.join(
        config['global']['path'], config['global']['processed_data_path'], 'data.hdf5')

    # Retrieve the names of variables from the configuration for surface and upper data
    upper_var_names = config['preprocessing']['upper_var_names']
    surface_var_names = config['preprocessing']['surface_var_names']

    # Determine the indices of 'u10', 'v10', 'u100', 'v100' for surface and 'u', 'v' for upper
    surface_indices = [surface_var_names.index(var) for var in ['u10', 'v10']]
    middle_indices = [surface_var_names.index(var) for var in ['u100', 'v100']]
    upper_indices = [upper_var_names.index(var) for var in ['u', 'v']]

    with h5py.File(filename, "a") as f:
        # Iterate over years
        for key in f.keys():
            # Compute labels
            labels = extract_labels(f[key]["surface"]["data"], f[key]["upper"]["data"],
                                    surface_indices, middle_indices, upper_indices)

            # Save labels into data file
            if "labels" not in f[key]:
                f[key].create_dataset("labels", data=labels)
