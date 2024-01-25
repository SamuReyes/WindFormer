import numpy as np
import os

def calculate_split_indices(total_length, train_pct, val_pct):
    """
    Calculate indices for splitting the dataset into train, validation, and test sets.
    """
    train_idx = int(total_length * train_pct)
    val_idx = train_idx + int(total_length * val_pct)
    return train_idx, val_idx

def split(data, train_idx, val_idx):
    """
    Split the data into train, validation, and test sets.
    """
    train_data = data[:train_idx]
    val_data = data[train_idx:val_idx]
    test_data = data[val_idx:]
    return train_data, val_data, test_data

def split_data(config: dict):
    """
    Load and split the data into train, validation, and test sets.
    """
    # Load data
    processed_data_path = os.path.join(config['global']['path'], config['global']['processed_data_path'])
    upper = np.load(os.path.join(processed_data_path,'upper.npy'))
    surface = np.load(os.path.join(processed_data_path,'surface.npy'))
    labels = np.load(os.path.join(processed_data_path,'labels.npy'))

    # Calculate split indices
    total_length = len(upper)
    train_idx, val_idx = calculate_split_indices(total_length, 0.65, 0.2)

    # Split data
    upper_train, upper_val, upper_test = split(upper, train_idx, val_idx)
    surface_train, surface_val, surface_test = split(surface, train_idx, val_idx)
    labels_train, labels_val, labels_test = split(labels, train_idx, val_idx)

    # Create directory if not exist
    train_data_path = os.path.join(config['global']['path'], config['global']['train_data_path'])
    os.makedirs(train_data_path, exist_ok=True)

    # Save data
    np.save(os.path.join(train_data_path,'upper_train.npy'), upper_train)
    np.save(os.path.join(train_data_path,'upper_val.npy'), upper_val)
    np.save(os.path.join(train_data_path,'upper_test.npy'), upper_test)

    np.save(os.path.join(train_data_path,'surface_train.npy'), surface_train)
    np.save(os.path.join(train_data_path,'surface_val.npy'), surface_val)
    np.save(os.path.join(train_data_path,'surface_test.npy'), surface_test)

    np.save(os.path.join(train_data_path,'labels_train.npy'), labels_train)
    np.save(os.path.join(train_data_path,'labels_val.npy'), labels_val)
    np.save(os.path.join(train_data_path,'labels_test.npy'), labels_test)