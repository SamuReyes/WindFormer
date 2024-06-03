import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from model.dataset import HDF5CustomDataset
from model.model_instance import init_model, device


def infer(config):
    """
    Performs inference on a test dataset using a pretrained model.

    Parameters:
    - config (dict): Configuration dictionary.

    Returns:
    - results (list of dicts): A list of dictionaries containing model's predictions and 'ground_truth'.
    """

    # Initialize the model
    model = init_model(config)
    model_path = os.path.join(config['global']['path'], config['global']['checkpoints_path'], config['train']['model_name'] + '.pth')
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model.eval()

    # Load configuration parameters
    sequence_length = config['train']['sequence_length']
    batch_size = config['train']['batch_size']
    data_path = os.path.join(config['global']['path'], config['global']['processed_data_path'], config['global']['data_file'])
    test_split = config['train']['test_split']
    prefetch_factor = config['train']['prefetch']
    workers = config['train']['workers']

    # Convert the split years into strings
    test_split = [str(year) for year in range(test_split[0], test_split[-1] + 1)] if len(test_split) > 1 else [str(test_split[0])]

    # Create dataset and dataloader
    test_dataset = HDF5CustomDataset(hdf5_path=data_path, sequence_length=sequence_length, years=test_split)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True
    )

    results = []
    # Perform inference
    with torch.no_grad():
        for data in test_loader:
            upper, surface, upper_label, surface_label, year, index = data['upper'].to(device), data['surface'].to(device), data['upper_label'].to(device), data['surface_label'].to(device), data['year'], data['index']
            upper_output, surface_output = model(upper, surface)
            # For each sample in the batch save the prediction and ground truth
            for u_pred, u_gt, s_pred, s_gt, y, i in zip(upper_output.cpu().numpy(), upper_label.cpu().numpy(), surface_output.cpu().numpy(), surface_label.cpu().numpy(), year, index):
                results.append({
                    'type': 'upper',
                    'year': y,
                    'index': i,
                    'prediction': u_pred,
                    'ground_truth': u_gt,
                })
                results.append({
                    'type': 'surface',
                    'year': y,
                    'index': i,
                    'prediction': s_pred,
                    'ground_truth': s_gt,
                })

    return results

def infer_future(config, future_steps):
    """
    Performs multi-step inference into the future using a pretrained model.
    Parameters:
    - config (dict): Configuration dictionary.
    - future_steps (int): Number of future steps to predict.
    Returns:
    - results (list of dicts): A list of dictionaries containing predictions for each future step.
    """

    # Initialize and load the model
    model = init_model(config)
    model_path = os.path.join(config['global']['path'], config['global']['checkpoints_path'], config['train']['model_name'] + '.pth')
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model.eval()

    # Initialize device
    model.to(device)

    # Load test data
    data_path = os.path.join(config['global']['path'], config['global']['processed_data_path'], config['global']['data_file'])
    sequence_length = config['train']['sequence_length']
    batch_size = config['train']['batch_size']
    test_split = config['train']['test_split']
    test_split = [str(year) for year in range(test_split[0], test_split[-1] + 1)] if len(test_split) > 1 else [str(test_split[0])]
    prefetch_factor = config['train']['prefetch']
    workers = config['train']['workers']

    test_dataset = HDF5CustomDataset(hdf5_path=data_path, sequence_length=sequence_length+future_steps, years=test_split)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True
    )

    results = []
    with torch.no_grad():
        for data in test_loader:
            
            upper_input, surface_input, upper_label, surface_label, year, index = data['upper'], data['surface'], data['upper_label'], data['surface_label'], data['year'], data['index']

            upper_seq = upper_input[:, :sequence_length]
            surface_seq = surface_input[:, :sequence_length]

            for step in range(future_steps):
                
                # Predict the next time step
                upper_output, surface_output = model(upper_seq.to(device), surface_seq.to(device))

                results.append({
                    'type': 'upper',
                    'year': year,
                    'index': step,
                    'prediction': upper_output[:,-1].cpu(),
                    'ground_truth': upper_label[:, sequence_length + step].cpu(), #!revisar esto
                })
                results.append({
                    'type': 'surface',
                    'year': year,
                    'index': step,
                    'prediction': surface_output[:,-1].cpu(),
                    'ground_truth': surface_label[:, sequence_length + step].cpu(),
                })

                # Update the input sequence with the new prediction
                if step < future_steps - 1:

                    new_upper = torch.cat((upper_seq[:, 1:, :].cpu(), upper_output[:,-1:].cpu()), dim=1)
                    upper_seq = new_upper

                    new_surface = torch.cat((surface_seq[:, 1:, :].cpu(), surface_output[:,-1:].cpu()), dim=1)
                    surface_seq = new_surface
            
    return results