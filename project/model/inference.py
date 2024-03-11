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

    # Load the model weights
    model.load_state_dict(torch.load(os.path.join(
        config['global']['path'], config['global']['checkpoints_path'], config['train']['model_name'] + '.pth'), map_location=device))
    model.eval()

    # Load configuration parameters
    sequence_length = config['train']['sequence_length']
    batch_size = config['train']['batch_size']
    data_path = os.path.join(config['global']['path'], config['global']['processed_data_path'], 'data.hdf5')
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
            upper, surface, labels = data['upper'].to(device), data['surface'].to(device), data['label'].to(device)
            outputs = model(upper, surface)

            # For each sample in the batch save the prediction and ground truth
            for pred, gt in zip(outputs.cpu().numpy(), labels.cpu().numpy()):
                results.append({
                    'prediction': pred,
                    'ground_truth': gt,
                })

    return results
