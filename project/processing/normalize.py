import numpy as np
import os


def normalize_data(config: dict):
    """
    Normalizes surface and upper layer data arrays using their mean and standard deviation.

    Parameters:
    - config (dict): Configuration dictionary.
    """

    processed_data_path = os.path.join(
        config['global']['path'], config['global']['processed_data_path'])

    # Load data
    surface_data = np.load(os.path.join(processed_data_path, 'surface.npy'))
    upper_data = np.load(os.path.join(processed_data_path, 'upper.npy'))

    statistics = {}

    # Calculate mean and standard deviation for both datasets
    statistics["upper_mean"] = np.mean(upper_data, axis=(0, 1, 2, 3))
    statistics["upper_std"] = np.std(upper_data, axis=(0, 1, 2, 3))
    statistics["surface_mean"] = np.mean(surface_data, axis=(0, 1, 2))
    statistics["surface_std"] = np.std(surface_data, axis=(0, 1, 2))

    np.savez(os.path.join(config['global']['path'], config['global']
             ['constants_path'], 'statistics.npz'), **statistics)

    # Reshape the mean and std deviation for broadcasting during normalization
    upper_mean_reshaped = statistics["upper_mean"].reshape(1, 1, 1, 8)
    upper_std_reshaped = statistics["upper_std"].reshape(1, 1, 1, 8)
    surface_mean_reshaped = statistics["surface_mean"].reshape(1, 1, 1, 9)
    surface_std_reshaped = statistics["surface_std"].reshape(1, 1, 1, 9)

    # Normalize the data
    surface_data = (surface_data - surface_mean_reshaped) / \
        surface_std_reshaped
    upper_data = (upper_data - upper_mean_reshaped) / upper_std_reshaped

    np.save(os.path.join(processed_data_path, 'surface.npy'), surface_data)
    np.save(os.path.join(processed_data_path, 'upper.npy'), upper_data)
