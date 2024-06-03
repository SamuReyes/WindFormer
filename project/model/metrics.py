import numpy as np
import pickle
import os
import torch

#TODO: To delete
def weighted_rmse(config, results, t=1):
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

    # Number of latitude and longitude coordinates: 
    N_lat = upper_predictions.shape[4]
    N_lon = upper_predictions.shape[3]

    # Calculate latitude weights:
    latitudes = np.load(os.path.join(config['global']['path'], config['global']['constants_path'], 'latitude.npy'))
    weights = np.deg2rad(np.arange(latitudes[0],latitudes[-1], -0.25))
    weights = torch.from_numpy(N_lat*np.cos(weights)/np.sum(np.cos(weights))).view(1, 1, N_lat)

    # Desnormalize data
    statistics_path = os.path.join(config['global']['path'], config['global']['constants_path'], 'statistics.pickle')
    with open(statistics_path, 'rb') as file:
        statistics = pickle.load(file)

    surface_predictions = torch.tensor(surface_predictions[:, t-1:, :, :, :] * statistics['std'][0] + statistics['mean'][0], dtype=torch.float32)
    surface_ground_truths = torch.tensor(surface_ground_truths[:, t-1:, :, :, :] * statistics['std'][0] + statistics['mean'][0], dtype=torch.float32)

    levels = config['preprocessing']['limits']['levels']
    for level in range(*levels):
        upper_predictions[:, t-1:, level, :, :, :] = upper_predictions[:, t-1:, level, :, :, :] * statistics['std'][level + 1] + statistics['mean'][level + 1]
        upper_ground_truths[:, t-1:, level, :, :, :] = upper_ground_truths[:, t-1:, level, :, :, :] * statistics['std'][level + 1] + statistics['mean'][level + 1]
    
    upper_predictions = torch.tensor(upper_predictions, dtype=torch.float32)
    upper_ground_truths = torch.tensor(upper_ground_truths, dtype=torch.float32)

    air_keys = config['preprocessing']['upper_var_names'][1:]
    pressure_level_keys = config['preprocessing']['pressure_levels']
    surface_keys = config['preprocessing']['surface_var_names'][1:]

    # Initialize dictionary to hold RMSE values:
    RMSE_upper = {}
    RMSE_surface = {}
    # Loop over air-variables:
    for i in range(upper_predictions.shape[5]):
        RMSE_upper[air_keys[i]] = {}
        # Loop over pressure levels:
        for j in range(upper_predictions.shape[2]):
            rmse = torch.sqrt(torch.sum(torch.pow(upper_predictions[:,:,j,:,:,i]-upper_ground_truths[:,:,j,:,:,i], 2)*weights, dim=(2,3))/(N_lat*N_lon))
            RMSE_upper[air_keys[i]][pressure_level_keys[j]] = torch.mean(rmse).item()

    # Loop over surface variables:
    for k in range(surface_predictions.shape[4]):
        rmse = torch.sqrt(torch.sum(torch.pow(surface_predictions[:,:,:,:,k]-surface_ground_truths[:,:,:,:,k], 2)*weights, dim=(2,3))/(N_lat*N_lon))
        RMSE_surface[surface_keys[k]] = torch.mean(rmse).item()

    return RMSE_surface, RMSE_upper

# TODO: Refactor code
def weighted_rmse_acc_future(config, results, t=0):
    """
    Calculates the RMSE for 'upper' and 'surface' predictions from a list of dictionaries containing
    predictions and ground truth values, along with a type identifier.

    Parameters:
    - config (dict): Configuration dictionary.
    - results (list): List of dictionaries with 'type', 'prediction', and 'ground_truth' keys.
    - t (int): Time index to calculate RMSE for.

    Returns:
    - rmse_dict (dict): Dictionary with RMSE values for 'upper' and 'surface'.
    """

    upper_predictions = [result['prediction'] for result in results if result['type'] == 'upper' and result['index'] == t]
    upper_ground_truths = [result['ground_truth'] for result in results if result['type'] == 'upper' and result['index'] == t]

    surface_predictions = [result['prediction'] for result in results if result['type'] == 'surface' and result['index'] == t]
    surface_ground_truths = [result['ground_truth'] for result in results if result['type'] == 'surface' and result['index'] == t]

    upper_predictions = np.concatenate(upper_predictions, axis=0)
    upper_ground_truths = np.concatenate(upper_ground_truths, axis=0)

    surface_predictions = np.concatenate(surface_predictions, axis=0)
    surface_ground_truths = np.concatenate(surface_ground_truths, axis=0)

    # Number of latitude and longitude coordinates: 
    N_lat = upper_predictions.shape[3]
    N_lon = upper_predictions.shape[2]

    # Calculate latitude weights:
    latitudes = np.load(os.path.join(config['global']['path'], config['global']['constants_path'], 'latitude.npy'))
    weights = np.deg2rad(np.arange(latitudes[0],latitudes[-1], -0.25))
    weights = torch.from_numpy(N_lat*np.cos(weights)/np.sum(np.cos(weights))).view(1, 1, N_lat)

    # Desnormalize data
    statistics_path = os.path.join(config['global']['path'], config['global']['constants_path'], 'statistics.pickle')
    with open(statistics_path, 'rb') as file:
        statistics = pickle.load(file)

    surface_predictions = torch.tensor(surface_predictions * statistics['std'][0] + statistics['mean'][0], dtype=torch.float32)
    surface_ground_truths = torch.tensor(surface_ground_truths * statistics['std'][0] + statistics['mean'][0], dtype=torch.float32)

    levels = config['preprocessing']['limits']['levels']
    for level in range(*levels):
        upper_predictions[:, level, :, :, :] = upper_predictions[:, level, :, :, :] * statistics['std'][level + 1] + statistics['mean'][level + 1]
        upper_ground_truths[:, level, :, :, :] = upper_ground_truths[:, level, :, :, :] * statistics['std'][level + 1] + statistics['mean'][level + 1]
    
    upper_predictions = torch.tensor(upper_predictions, dtype=torch.float32)
    upper_ground_truths = torch.tensor(upper_ground_truths, dtype=torch.float32)

    air_keys = config['preprocessing']['upper_var_names'][1:]
    pressure_level_keys = config['preprocessing']['pressure_levels']
    surface_keys = config['preprocessing']['surface_var_names'][1:]

    # Initialize dictionary to hold RMSE values:
    RMSE_upper = {}
    RMSE_surface = {}
    # Loop over air-variables:
    for i in range(upper_predictions.shape[4]):
        RMSE_upper[air_keys[i]] = {}
        # Loop over pressure levels:
        for j in range(upper_predictions.shape[1]):
            rmse = torch.sqrt(torch.sum(torch.pow(upper_predictions[:,j,:,:,i]-upper_ground_truths[:,j,:,:,i], 2)*weights, dim=(1,2))/(N_lat*N_lon))
            RMSE_upper[air_keys[i]][pressure_level_keys[j]] = torch.mean(rmse).item()

    # Loop over surface variables:
    for k in range(surface_predictions.shape[3]):
        rmse = torch.sqrt(torch.sum(torch.pow(surface_predictions[:,:,:,k]-surface_ground_truths[:,:,:,k], 2)*weights, dim=(1,2))/(N_lat*N_lon))
        RMSE_surface[surface_keys[k]] = torch.mean(rmse).item()

    # Initialize dictionary to hold ACC values:
    ACC_upper = {}
    ACC_surface = {}
    # Loop over air-variables:
    for i in range(upper_predictions.shape[4]):
        ACC_upper[air_keys[i]] = {}
        # Loop over pressure levels:
        for j in range(upper_predictions.shape[1]):
            acc = torch.sum(upper_predictions[:,j,:,:,i]*upper_ground_truths[:,j,:,:,i]*weights, dim=(1,2)) / torch.sqrt(torch.sum(torch.pow(upper_predictions[:,j,:,:,i], 2)*weights, dim=(1,2)) * torch.sum(torch.pow(upper_ground_truths[:,j,:,:,i], 2)*weights, dim=(1,2)))
            ACC_upper[air_keys[i]][pressure_level_keys[j]] = torch.mean(acc).item()

    # Loop over surface variables:
    for k in range(surface_predictions.shape[3]):
        acc = torch.sum(surface_predictions[:,:,:,k]*surface_ground_truths[:,:,:,k]*weights, dim=(1,2)) / torch.sqrt(torch.sum(torch.pow(surface_predictions[:,:,:,k], 2)*weights, dim=(1,2)) * torch.sum(torch.pow(surface_ground_truths[:,:,:,k], 2)*weights, dim=(1,2)))
        ACC_surface[surface_keys[k]] = torch.mean(acc).item()

    return RMSE_surface, RMSE_upper, ACC_surface, ACC_upper