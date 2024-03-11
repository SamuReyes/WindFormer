import torch
import itertools
import h5py
import torch
from torch.utils.data import Dataset


class HDF5CustomDataset(Dataset):
    """
    Custom dataset for handling data stored in an HDF5 file.
    This version allows specifying which years to include and handles varying lengths due to leap years.
    """

    def __init__(self, hdf5_path, sequence_length, years=None):
        self.hdf5_path = hdf5_path
        self.sequence_length = sequence_length

        # Open the HDF5 file and get the list of years, filtered by the ones provided if any
        with h5py.File(self.hdf5_path, 'r') as file:
            self.years = years

            # Calculate lengths per year
            self.lengths_per_year = [file[year]['upper']['data'].shape[0] -
                                     (self.sequence_length + 1) for i, year in enumerate(self.years)]

        # Calculate the total length
        self.total_length = sum(self.lengths_per_year)

        # Pre-calculate the cumulative sum of lengths to help with index mapping
        self.cumulative_lengths = [
            0] + list(itertools.accumulate(self.lengths_per_year))

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find which year this idx falls into, adjusted for sequence length
        year_idx = next(i for i, total in enumerate(self.cumulative_lengths) if idx < total) - 1
        within_year_idx = idx - self.cumulative_lengths[year_idx]

        # Open the HDF5 file and extract the sequences and label
        with h5py.File(self.hdf5_path, 'r') as file:
            year_key = self.years[year_idx]

            upper_sequence = file[year_key]['upper']['data'][within_year_idx: within_year_idx + self.sequence_length]
            surface_sequence = file[year_key]['surface']['data'][within_year_idx: within_year_idx + self.sequence_length]
            upper_label = file[year_key]['upper']['data'][within_year_idx + self.sequence_length]
            surface_label = file[year_key]['surface']['data'][within_year_idx + self.sequence_length]

        # Convert to torch tensors
        data = {
            'upper': torch.from_numpy(upper_sequence).float(),
            'surface': torch.from_numpy(surface_sequence).float(),
            'upper_label': torch.from_numpy(upper_label).float(),
            'surface_label': torch.from_numpy(surface_label).float()
        }
        return data
