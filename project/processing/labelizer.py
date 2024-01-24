import numpy as np
import os

def get_labels(config:dict):
    """ Transforms the data into labels """

    # Load data
    surface_data = np.load(os.path.join(config['global']['path'], config['global']['normalized_data_path'], 'surface_normalized.npy'))
    upper_data = np.load(os.path.join(config['global']['path'], config['global']['normalized_data_path'], 'upper_normalized.npy'))

    upper_var_names = config['preprocessing']['upper_var_names']
    surface_var_names = config['preprocessing']['surface_var_names']

    # Get indices
    surface_vars_indices = [surface_var_names.index(var) for var in ['u10', 'v10']]
    upper_vars_indices = [upper_var_names.index(var) for var in ['u', 'v']]

    # Extract out variables
    surface_data = surface_data[:, :, :, surface_vars_indices]
    upper_data = upper_data[:, :, :, :, upper_vars_indices]

    # Combine data
    surface_data = np.expand_dims(surface_data, axis=1)
    combined_labels = np.concatenate((surface_data, upper_data), axis=1)

    # Save labels
    np.save(os.path.join(config['global']['path'], config['global']['normalized_data_path'], 'labels_normalized.npy'), combined_labels)

    # Free up memory
    del surface_data
    del upper_data
    del combined_labels