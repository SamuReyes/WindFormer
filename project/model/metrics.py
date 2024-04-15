import numpy as np
import pickle
import os

def calculate_rmse(config, results):
    """
    Calculates the RMSE for 'upper' and 'surface' predictions from a list of dictionaries containing
    predictions and ground truth values, along with a type identifier.

    Parameters:
    - config (dict): Configuration dictionary.
    - results (list): List of dictionaries with 'type', 'prediction', and 'ground_truth' keys.

    Returns:
    - rmse_dict (dict): Dictionary with RMSE values for 'upper' and 'surface'.
    """
    upper_predictions = np.array([result['prediction'] for result in results if result['type'] == 'upper'])
    upper_ground_truths = np.array([result['ground_truth'] for result in results if result['type'] == 'upper'])
    
    surface_predictions = np.array([result['prediction'] for result in results if result['type'] == 'surface'])
    surface_ground_truths = np.array([result['ground_truth'] for result in results if result['type'] == 'surface'])

    # Desnormalize data
    statistics_path = os.path.join(config['global']['path'], config['global']['constants_path'], 'statistics.pickle')
    with open(statistics_path, 'rb') as file:
        statistics = pickle.load(file)

    surface_predictions_desnorm = surface_predictions * statistics['std'][0] + statistics['mean'][0]
    surface_ground_truths_desnorm = surface_ground_truths * statistics['std'][0] + statistics['mean'][0]

    levels = config['preprocessing']['limits']['levels']
    for level in range(*levels):
        upper_predictions[:, level, :, :, :] = upper_predictions[:, level, :, :, :] * statistics['std'][level + 1] + statistics['mean'][level + 1]
        upper_ground_truths[:, level, :, :, :] = upper_ground_truths[:, level, :, :, :] * statistics['std'][level + 1] + statistics['mean'][level + 1]

    # Compute RMSE
    squared_errors_surface = (surface_predictions_desnorm - surface_ground_truths_desnorm) ** 2
    squared_errors_upper = (upper_predictions - upper_ground_truths) ** 2

    mean_squared_errors_surface = np.mean(squared_errors_surface, axis=(0, 1, 2, 3))
    mean_squared_errors_upper = np.mean(squared_errors_upper, axis=(0, 1, 2, 3, 4))

    surface_rmse = np.sqrt(mean_squared_errors_surface)
    upper_rmse = np.sqrt(mean_squared_errors_upper)
    
    rmse = {
        'upper_rmse': upper_rmse,
        'surface_rmse': surface_rmse
    }

    return rmse
