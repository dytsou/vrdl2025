""""
Train a PyTorch model with validation and checkpointing.
This script defines functions to train a model for a specified number of epochs,
evaluate it on a validation set, and save the best model based on validation loss.
It also includes functions to plot training and validation loss curves.
The training process includes:
- Loading data using DataLoader
- Training the model for a specified number of epochs
- Evaluating the model on a validation set
- Saving the best model based on validation loss
- Saving checkpoints at each epoch
- Plotting training and validation loss curves
The script uses PyTorch and assumes that the model, optimizer, and data loaders
are already defined.
"""

import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import config

def train_one_epoch(model, optimizer, data_loader, device):
    """
    Train the model for one epoch.
    Args:
        model: The model to be trained.
        optimizer: The optimizer for the model.
        data_loader: DataLoader for the training data.
        device: Device to run the model on (CPU or GPU).
    Returns:
        Average loss for the epoch.
    """
    # Set the model to training mode
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        running_loss += losses.item()
    return running_loss / len(data_loader)

def evaluate(model, data_loader, device, logger):
    """
    Evaluate the model on the validation set.
    Args:
        model: The model to be evaluated.
        data_loader: DataLoader for the validation data.
        device: Device to run the model on (CPU or GPU).
    Returns:
        Average loss for the validation set.
    """
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Compute the loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item()
            logger.info("Validation Loss: %.4f", losses.item())
    # Average loss over the validation set
    return running_loss / len(data_loader)

def train_model(
    model, train_loader, val_loader, optimizer,
    num_epochs=None, device=None, save_dir=None, logger=None
):
    """
    Train the model with validation and checkpointing.
    Args:
        model: The model to be trained.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        optimizer: The optimizer for the model.
        num_epochs (int, optional): Number of epochs to train the model.
        device: Device to run the model on (CPU or GPU).
        save_dir (str, optional): Directory to save checkpoints.
    Returns:
        The trained model.
    """
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
    if device is None:
        device = torch.device(config.DEVICE)
    if save_dir is None:
        save_dir = config.CHECKPOINT_DIR
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Record training and validation losses
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        logger.info("Epoch %d/%d", epoch + 1, num_epochs)
        # Training
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        train_losses.append(train_loss)
        # Validation
        val_loss = evaluate(model, val_loader, device, logger)
        val_losses.append(val_loss)
        logger.info("Train Loss: %.4f, Val Loss: %.4f", train_loss, val_loss)
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            logger.info("Saved best model!")
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    # Plot training and validation loss
    logger.info("Plotting training and validation loss...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()
    logger.info("Training complete!")
    return model
