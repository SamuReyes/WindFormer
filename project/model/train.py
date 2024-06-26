import numpy as np
import os
import torch
import wandb
import time
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from model.dataset import HDF5CustomDataset
from model.model_instance import init_model, device

def validate_model(model, val_loader, loss_fn):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                upper, surface, upper_label, surface_label = data['upper'].to(device), data['surface'].to(
                    device), data['upper_label'].to(device), data['surface_label'].to(device)
                upper_output, surface_output = model(upper, surface)
                upper_loss = loss_fn(upper_output, upper_label).item()
                surface_loss = loss_fn(surface_output, surface_label).item()
                val_loss += (upper_loss + surface_loss)
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
    batch_size = config['train']['batch_size']
    learning_rate = config['train']['learning_rate']
    epochs = config['train']['epochs']
    data_path = os.path.join(
        config['global']['path'], config['global']['processed_data_path'], config['global']['data_file'])
    train_split = config['train']['train_split']
    val_split = config['train']['val_split']
    prefetch_factor = config['train']['prefetch']
    workers = config['train']['workers']

     # Convert the split years into strings
    if isinstance(train_split[0], list):
        train_split = [str(year) for group in train_split for year in range(group[0], group[-1] + 1)]
    else:
        train_split = [str(year) for year in range(train_split[0], train_split[-1] + 1)]
    val_split = [str(year) for year in range(val_split[0], val_split[-1] + 1)] if len(val_split) > 1 else [str(val_split[0])]

    # Create dataset and dataloader for training and validation
    train_dataset = HDF5CustomDataset(
        hdf5_path=data_path, sequence_length=sequence_length, years=train_split)

    val_dataset = HDF5CustomDataset(
        hdf5_path=data_path, sequence_length=sequence_length, years=val_split)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True  # ! Check
    )

    # Initialize the model
    model = init_model(config)

    # Load the model weights if a pretrained model is provided
    if config['model']['pretrained']:
        pretrained_model = torch.load(os.path.join(config['global']['path'], config['global']['checkpoints_path'], config['model']['pretrained'] + '.pth'))
        model.load_state_dict(pretrained_model)

    # Initialize the gradient scaler for mixed precision training and gradient accumulation
    scaler = GradScaler()
    iters_to_accumulate = config['train']['iters_to_accumulate']

    # Loss function and optimizer setup
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler setup
    total_batches = len(train_loader)
    steps_per_epoch = total_batches//iters_to_accumulate
    final_steps = steps_per_epoch * epochs
    warmup_ratio = config['train']['warmup_ratio']/iters_to_accumulate
    warmup_steps = int(final_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=final_steps)
    best_val_loss = np.float16(np.inf)

    # Training loop
    for epoch in range(epochs):
        # Training iteration over the dataset
        model.train()
        t1 = time.time()
        total_loss = 0

        for i, data in enumerate(train_loader):
            # Runs the forward pass under autocast
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                upper, surface, upper_label, surface_label = data['upper'].to(device), data['surface'].to(
                    device), data['upper_label'].to(device), data['surface_label'].to(device)
                upper_output, surface_output = model(upper, surface)
                upper_loss = loss_fn(upper_output, upper_label)
                surface_loss = loss_fn(surface_output, surface_label)
                loss = (upper_loss + surface_loss) / iters_to_accumulate

            total_loss += (upper_loss.item() + surface_loss.item())
            scaler.scale(loss).backward()

            if (i + 1) % iters_to_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Update learning rate

                optimizer.zero_grad()

            if i % 100 == 0:
                wandb.log({"Step": i, "LR": optimizer.param_groups[0]['lr']})

            # Monitor validation loss at the midpoint of the epoch
            if (i == total_batches // 2 and config['train']['log_mid_loss']):
                mid_epoch_val_loss = validate_model(model, val_loader, loss_fn)
                print(f"Epoch [{epoch+0.5}/{epochs}], Val Loss: {mid_epoch_val_loss}")
                wandb.log({"Epoch": epoch + 0.5, "Val loss": mid_epoch_val_loss})

        # End-of-epoch validation
        average_loss = total_loss / len(train_loader)
        val_loss = validate_model(model, val_loader, loss_fn)
        t2 = time.time()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss}, Val Loss: {val_loss}, Time: {round((t2-t1)/60, 2)}")
        wandb.log({"Epoch": epoch+1, "Train loss": average_loss, "Val loss": val_loss, "Time": round((t2-t1)/60, 2)})

        # Save intermediate model
        if config['train']['save_model'] and best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(
                config['global']['path'], config['global']['checkpoints_path'], config['train']['model_name'] + '.pth'))
