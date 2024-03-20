import numpy as np

def calculate_rmse(results):
    """
    Calculates the RMSE for 'upper' and 'surface' predictions from a list of dictionaries containing
    predictions and ground truth values, along with a type identifier.

    Parameters:
    - results (list): List of dictionaries with 'type', 'prediction', and 'ground_truth' keys.

    Returns:
    - rmse_dict (dict): Dictionary with RMSE values for 'upper' and 'surface'.
    """
    upper_predictions = np.array([result['prediction'] for result in results if result['type'] == 'upper'])
    upper_ground_truths = np.array([result['ground_truth'] for result in results if result['type'] == 'upper'])
    
    surface_predictions = np.array([result['prediction'] for result in results if result['type'] == 'surface'])
    surface_ground_truths = np.array([result['ground_truth'] for result in results if result['type'] == 'surface'])
    
    # Upper RMSE
    upper_mse = np.mean((upper_predictions - upper_ground_truths) ** 2)
    upper_rmse = np.sqrt(upper_mse)
    
    #Surface RMSE
    surface_mse = np.mean((surface_predictions - surface_ground_truths) ** 2)
    surface_rmse = np.sqrt(surface_mse)
    
    rmse = {
        'upper_rmse': upper_rmse,
        'surface_rmse': surface_rmse
    }

    return rmse
