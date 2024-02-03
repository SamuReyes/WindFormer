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

    # Initialize dictionary to store statistics
    statistics = {}

    # Process surface data
    surface_data_path = os.path.join(processed_data_path, 'surface.npy')
    surface_data = np.load(surface_data_path)
    statistics["surface_mean"] = np.mean(surface_data, axis=(0, 1, 2))
    statistics["surface_std"] = np.std(surface_data, axis=(0, 1, 2))
    # Reshape for broadcasting
    surface_mean_reshaped = statistics["surface_mean"].reshape(1, 1, 1, -1)
    surface_std_reshaped = statistics["surface_std"].reshape(1, 1, 1, -1)
    # Normalize
    surface_data = (surface_data - surface_mean_reshaped) / \
        surface_std_reshaped
    np.save(surface_data_path, surface_data)
    # Clear surface data from memory
    del surface_data

    # Process upper data
    upper_data_path = os.path.join(processed_data_path, 'upper.npy')
    upper_data = np.load(upper_data_path)
    statistics["upper_mean"] = np.mean(upper_data, axis=(0, 1, 2, 3))
    statistics["upper_std"] = np.std(upper_data, axis=(0, 1, 2, 3))
    # Reshape for broadcasting
    upper_mean_reshaped = statistics["upper_mean"].reshape(1, 1, 1, -1)
    upper_std_reshaped = statistics["upper_std"].reshape(1, 1, 1, -1)
    # Normalize
    upper_data = (upper_data - upper_mean_reshaped) / upper_std_reshaped
    np.save(upper_data_path, upper_data)
    # Clear upper data from memory
    del upper_data

    # Save the calculated statistics for future use
    np.savez(os.path.join(config['global']['path'], config['global']
             ['constants_path'], 'statistics.npz'), **statistics)
