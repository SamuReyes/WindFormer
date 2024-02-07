import numpy as np
import wandb


def calculate_rmse(results):
    """
    Calculates the RMSE from a list of dictionaries containing
    predictions and ground truthvalues.

    Parameters:
    - results (list): List of dictionaries with 'prediction' and 'ground_truth' keys.

    Returns:
    - rmse (float): Root Mean Square Error.
    """
    # Extract predictions and actual values
    predictions = np.array([result['prediction'] for result in results])
    ground_truths = np.array([result['ground_truth'] for result in results])

    # Calculate MSE and then RMSE
    mse = np.mean((predictions - ground_truths) ** 2)
    rmse = np.sqrt(mse)

    wandb.log({"rmse": rmse})

    return rmse
