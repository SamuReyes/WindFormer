import numpy as np
import os
import torch
import torch.nn.functional as F
from model.model import ViViT
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 2

class CustomDataset(Dataset):
    def __init__(self, upper_data, surface_data, labels, sequence_length):
        self.upper_data = upper_data
        self.surface_data = surface_data
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.upper_data) - self.sequence_length

    def __getitem__(self, idx):
        if idx + self.sequence_length >= len(self.upper_data):
            raise IndexError("Index out of range")

        upper_sequence = self.upper_data[idx : idx + self.sequence_length]
        surface_sequence = self.surface_data[idx : idx + self.sequence_length]
        label = self.labels[idx + self.sequence_length]

        data = {
            'upper': torch.from_numpy(upper_sequence).float(),
            'surface': torch.from_numpy(surface_sequence).float(),
            'label': torch.from_numpy(label).float()  # Asumiendo que labels ya es un ndarray
        }
        return data

def train_model(config: dict):
    
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

    sequence_length = config['train']['sequence_length']
    batch_size = config['train']['batch_size']
    learning_rate = config['train']['learning_rate']
    epochs = config['train']['epochs']

    # Load data
    upper_data = np.load(os.path.join(config['global']['path'], config['global']['normalized_data_path'], 'upper_normalized.npy'))
    surface_data = np.load(os.path.join(config['global']['path'], config['global']['normalized_data_path'], 'surface_normalized.npy'))
    labels = np.load(os.path.join(config['global']['path'], config['global']['normalized_data_path'], 'labels_normalized.npy'))

    # Create dataset and dataloader
    dataset = CustomDataset(upper_data, surface_data, labels, sequence_length=sequence_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Create an instance of the model
    model = ViViT(image_size_3d, patch_size_3d, image_size_2d, patch_size_2d, output_dim, dim, depth, heads, dim_head, dropout, emb_dropout, scale_dim).to(device)
    
    # Loss function and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train cycle
    for epoch in range(epochs):
        for data in data_loader:
            upper, surface, labels = data['upper'].to(device), data['surface'].to(device), data['label'].to(device)

            # Forward pass
            outputs = model(upper, surface)
            loss = loss_fn(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

