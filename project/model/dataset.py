import torch
from torch.utils.data import Dataset


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
    """

    def __init__(self, upper_data, surface_data, labels, sequence_length):
        self.upper_data = upper_data
        self.surface_data = surface_data
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        # Calculate the length of the dataset accounting for sequence length
        return len(self.upper_data) - self.sequence_length - 1

    def __getitem__(self, idx):
        # Ensure that the requested index is within the bounds of the dataset
        if idx + self.sequence_length + 1 >= len(self.upper_data):
            raise IndexError("Index out of range")

        # Extract sequences for upper and surface data, and the corresponding label
        upper_sequence = self.upper_data[idx: idx + self.sequence_length]
        surface_sequence = self.surface_data[idx: idx + self.sequence_length]
        label = self.labels[idx + 1: idx + self.sequence_length + 1]

        # Convert numpy arrays to torch tensors and return as a dictionary
        data = {
            'upper': torch.from_numpy(upper_sequence).float(),
            'surface': torch.from_numpy(surface_sequence).float(),
            'label': torch.from_numpy(label).float()
        }
        return data
