import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from model.dataset import CustomDataset
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

    # Load the test data
    test_data_path = os.path.join(
        config['global']['path'], config['global']['train_data_path'])
    upper_test = np.load(os.path.join(test_data_path, 'upper_test.npy'))
    surface_test = np.load(os.path.join(test_data_path, 'surface_test.npy'))
    labels_test = np.load(os.path.join(test_data_path, 'labels_test.npy'))

    # Create dataset and dataloader for testing
    test_dataset = CustomDataset(upper_test, surface_test, labels_test,
                                 sequence_length=config['train']['sequence_length'], delay=config['model']['delay'])
    test_loader = DataLoader(
        test_dataset, batch_size=config['train']['batch_size'], shuffle=False)

    results = []
    # Perform inference
    with torch.no_grad():
        for data in test_loader:
            upper, surface, labels = data['upper'].to(
                device), data['surface'].to(device), data['label'].to(device)
            outputs = model(upper, surface)

            # For each sample in the batch save the prediction and ground truth
            for pred, gt in zip(outputs.cpu().numpy(), labels.cpu().numpy()):
                results.append({
                    'prediction': pred,
                    'ground_truth': gt,
                })

    return results
