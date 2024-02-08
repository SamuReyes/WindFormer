import numpy as np
import os
import torch
import wandb
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from model.dataset import CustomDataset
from model.model_instance import init_model, device


def validate_model(model, val_loader, loss_fn):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                upper, surface, labels = data['upper'].to(
                    device), data['surface'].to(device), data['label'].to(device)
                outputs = model(upper, surface)
                val_loss += loss_fn(outputs, labels).item()
    return val_loss / len(val_loader)


def train_model(config: dict):
    """
    Trains the ViViT model based on the provided configuration.

    This function initializes the ViViT model with the given parameters, loads
    the training and validation datasets, and performs training for a specified
    number of epochs. It also handles validation and model saving.

    :param config: Dictionary containing the configuration settings.
    """

    # Extract training parameters from the config
    sequence_length = config['train']['sequence_length']
    delay = config['model']['delay']
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

    # Initialize the model
    model = init_model(config)

    # Initialize the gradient scaler for mixed precision training
    scaler = GradScaler()

    # Loss function and optimizer setup
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # TODO scheduler: linear decay (LINEARLR torch)

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        # Training iteration over the dataset
        total_batches = len(train_loader)
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            # Runs the forward pass under autocast
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                upper, surface, labels = data['upper'].to(
                    device), data['surface'].to(device), data['label'].to(device)
                outputs = model(upper, surface)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Monitor validation loss at the midpoint of the epoch
            if (i == total_batches // 2 and config['train']['log_mid_loss']):
                mid_epoch_val_loss = validate_model(model, val_loader, loss_fn)
                print(
                    f"Epoch [{epoch+0.5}/{epochs}], Val Loss: {mid_epoch_val_loss}")
                wandb.log(
                    {"Epoch": epoch + 0.5, "Val loss": mid_epoch_val_loss})

        # End-of-epoch validation
        val_loss = validate_model(model, val_loader, loss_fn)
        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}, Val Loss: {val_loss}")
        wandb.log(
            {"Epoch": epoch+1, "Train loss": loss.item(), "Val loss": val_loss})

        # Save intermediate model
        if config['train']['save_intermediate_model']:
            torch.save(model.state_dict(), os.path.join(
                config['global']['path'], config['global']['checkpoints_path'], config['train']['model_name'] + f'_epoch_{epoch+1}.pth'))

    # Save final model
    if config['train']['save_model']:
        torch.save(model.state_dict(), os.path.join(
            config['global']['path'], config['global']['checkpoints_path'], config['train']['model_name'] + '.pth'))
