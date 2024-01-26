import numpy as np
import os
import torch
import torch.nn.functional as F
from model.model import ViViT
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    """
    Custom dataset for handling paired upper and surface data along with labels.

    This dataset is designed to work with sequential data, where each sequence 
    has a specified length and is associated with a delayed label. It's suitable 
    for time-series predictions where the goal is to predict future values 
    based on past sequences of data.

    :param upper_data: Numpy array representing 'Upper' data of shape (T, L, W, H, C).
    :param surface_data: Numpy array representing 'surface' data of shape (T, W, H, C).
    :param labels: Numpy array representing labels of shape (T, L, W, H, C).
    :param sequence_length: Integer representing the length of the input sequence.
    :param delay: Integer representing the delay for the label in time steps.
    """

    def __init__(self, upper_data, surface_data, labels, sequence_length, delay):
        self.upper_data = upper_data
        self.surface_data = surface_data
        self.labels = labels
        self.sequence_length = sequence_length
        self.delay = delay

    def __len__(self):
        # Calculate the length of the dataset accounting for sequence length and delay
        return len(self.upper_data) - self.sequence_length - self.delay

    def __getitem__(self, idx):
        # Ensure that the requested index is within the bounds of the dataset
        if idx + self.sequence_length + self.delay >= len(self.upper_data):
            raise IndexError("Index out of range")

        # Extract sequences for upper and surface data, and the corresponding label
        upper_sequence = self.upper_data[idx: idx + self.sequence_length]
        surface_sequence = self.surface_data[idx: idx + self.sequence_length]
        label = self.labels[idx + self.sequence_length + self.delay]

        # Convert numpy arrays to torch tensors and return as a dictionary
        data = {
            'upper': torch.from_numpy(upper_sequence).float(),
            'surface': torch.from_numpy(surface_sequence).float(),
            'label': torch.from_numpy(label).float()
        }
        return data


def train_model(config: dict):
    """
    Trains the ViViT model based on the provided configuration.

    This function initializes the ViViT model with the given parameters, loads
    the training and validation datasets, and performs training for a specified
    number of epochs. It also handles validation and model saving.

    :param config: Dictionary containing the configuration settings.
    """

    # Extract model and training parameters from the config
    image_size_3d = config['model']['image_size_3d']
    patch_size_3d = config['model']['patch_size_3d']
    image_size_2d = config['model']['image_size_2d']
    patch_size_2d = config['model']['patch_size_2d']
    output_dim = config['model']['output_dim']
    dim = config['model']['dim']
    depth = config['model']['depth']
    heads = config['model']['heads']
    dim_head = config['model']['dim_head']
    dropout = config['model']['dropout']
    emb_dropout = config['model']['emb_dropout']
    scale_dim = config['model']['scale_dim']
    delay = config['model']['delay']

    sequence_length = config['train']['sequence_length']
    batch_size = config['train']['batch_size']
    learning_rate = config['train']['learning_rate']
    epochs = config['train']['epochs']

    # Load data
    train_data_path = os.path.join(
        config['global']['path'], config['global']['train_data_path'])

    upper_train = np.load(os.path.join(train_data_path, 'upper_train.npy'))
    surface_train = np.load(os.path.join(train_data_path, 'surface_train.npy'))
    labels_train = np.load(os.path.join(train_data_path, 'labels_train.npy'))

    upper_val = np.load(os.path.join(train_data_path, 'upper_val.npy'))
    surface_val = np.load(os.path.join(train_data_path, 'surface_val.npy'))
    labels_val = np.load(os.path.join(train_data_path, 'labels_val.npy'))

    # Create dataset and dataloader for training and validation
    train_dataset = CustomDataset(
        upper_train, surface_train, labels_train, sequence_length=sequence_length, delay=delay)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = CustomDataset(
        upper_val, surface_val, labels_val, sequence_length=sequence_length, delay=delay)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model and move it to the configured device
    model = ViViT(image_size_3d, patch_size_3d, image_size_2d, patch_size_2d, output_dim,
                  dim, depth, heads, dim_head, dropout, emb_dropout, scale_dim).to(device)

    # Loss function and optimizer setup
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        # Training iteration over the dataset
        for data in train_loader:
            upper, surface, labels = data['upper'].to(
                device), data['surface'].to(device), data['label'].to(device)
            outputs = model(upper, surface)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            # Iterate over validation data
            for data in val_loader:
                upper, surface, labels = data['upper'].to(
                    device), data['surface'].to(device), data['label'].to(device)
                outputs = model(upper, surface)
                val_loss += loss_fn(outputs, labels).item()
            val_loss /= len(val_loader)

        # Save the model periodically and print training progress
        if config['train']['save_model']:
            torch.save(model.state_dict(), os.path.join(
                config['global']['path'], config['global']['checkpoints_path'], config['train']['model_name'] + f'_epoch_{epoch+1}.pth'))
        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}, Val Loss: {val_loss}")

    # Save the final model
    if config['train']['save_model']:
        torch.save(model.state_dict(), os.path.join(
            config['global']['path'], config['global']['checkpoints_path'], config['train']['model_name'] + '.pth'))
