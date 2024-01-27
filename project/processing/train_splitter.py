import numpy as np
import os


def calculate_split_indices(total_length, train_pct, val_pct):
    """
    Calculates indices to split the dataset into training, validation, and test sets.

    Parameters:
    - total_length (int): The total number of samples in the dataset.
    - train_pct (float): The percentage of the dataset to allocate to the training set.
    - val_pct (float): The percentage of the dataset to allocate to the validation set.

    Returns:
    - tuple: Indices indicating where to split the dataset for training and validation sets.
    """
    train_idx = int(total_length * train_pct)
    val_idx = train_idx + int(total_length * val_pct)
    return train_idx, val_idx


def split(data, train_idx, val_idx):
    """
    Splits the data into training, validation, and test sets based on provided indices.

    Parameters:
    - data (numpy.ndarray): The dataset to split.
    - train_idx (int): The index where the training data ends and validation begins.
    - val_idx (int): The index where the validation data ends and test data begins.

    Returns:
    - tuple: Three numpy arrays representing the training, validation, and test datasets.
    """
    train_data = data[:train_idx]
    val_data = data[train_idx:val_idx]
    test_data = data[val_idx:]
    return train_data, val_data, test_data


def split_data(config: dict):
    """
    Split and save data into training, validation, and test sets.

    Parameters:
    - config (dict): Configuration dictionary.
    """
    # Load data
    processed_data_path = os.path.join(
        config['global']['path'], config['global']['processed_data_path'])
    upper = np.load(os.path.join(processed_data_path, 'upper.npy'))
    surface = np.load(os.path.join(processed_data_path, 'surface.npy'))
    labels = np.load(os.path.join(processed_data_path, 'labels.npy'))

    # Calculate indices for splitting the data
    total_length = len(upper)
    train_idx, val_idx = calculate_split_indices(total_length, 0.65, 0.2)

    # Split data into training, validation, and test sets
    upper_train, upper_val, upper_test = split(upper, train_idx, val_idx)
    surface_train, surface_val, surface_test = split(
        surface, train_idx, val_idx)
    labels_train, labels_val, labels_test = split(labels, train_idx, val_idx)

    # Ensure the directory exists
    train_data_path = os.path.join(
        config['global']['path'], config['global']['train_data_path'])
    os.makedirs(train_data_path, exist_ok=True)

    # Save data
    np.save(os.path.join(train_data_path, 'upper_train.npy'), upper_train)
    np.save(os.path.join(train_data_path, 'upper_val.npy'), upper_val)
    np.save(os.path.join(train_data_path, 'upper_test.npy'), upper_test)

    np.save(os.path.join(train_data_path, 'surface_train.npy'), surface_train)
    np.save(os.path.join(train_data_path, 'surface_val.npy'), surface_val)
    np.save(os.path.join(train_data_path, 'surface_test.npy'), surface_test)

    np.save(os.path.join(train_data_path, 'labels_train.npy'), labels_train)
    np.save(os.path.join(train_data_path, 'labels_val.npy'), labels_val)
    np.save(os.path.join(train_data_path, 'labels_test.npy'), labels_test)
