import os
import torch
import numpy as np
import math


def calculate_psnr(img1, img2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) between two images"""
    # img1 and img2 have range [0, 1]
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def save_checkpoint(state, is_best, output_dir='output'):
    """Save model checkpoint"""
    os.makedirs(output_dir, exist_ok=True)

    # Save latest checkpoint
    checkpoint_path = os.path.join(output_dir, 'checkpoint_latest.pth.tar')
    torch.save(state, checkpoint_path)

    # Save best checkpoint
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoint_best.pth.tar')
        torch.save(state, best_path)
        print(f"Saved best checkpoint to {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint"""
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return 0, 0

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Return epoch and best PSNR
    epoch = checkpoint.get('epoch', 0)
    best_psnr = checkpoint.get('best_psnr', 0)

    return epoch, best_psnr


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize tensor to display image"""
    # Clone the tensor to avoid modifying the original
    tensor = tensor.clone().detach()

    # Reshape mean and std for broadcasting
    mean = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1)

    # Denormalize
    tensor = tensor * std + mean

    # Clamp to ensure values are in [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    return tensor


def save_predictions_to_npz(model, test_loader, output_path='pred.npz', device='cuda'):
    """Generate and save predictions for test images"""
    model.eval()
    predictions = {}

    with torch.no_grad():
        for batch in test_loader:
            # Get data
            degraded = batch['degraded'].to(device)
            filenames = batch['filename']

            # Determine degradation type from filename
            degradation_types = ['rain' if 'rain' in filename else 'snow'
                                 for filename in filenames]

            # Process one by one since batch size is 1
            for i, (img, filename, deg_type) in enumerate(zip(degraded, filenames, degradation_types)):
                # Forward pass
                output = model(img.unsqueeze(0), deg_type)

                # Denormalize output
                output = denormalize(output)

                # Convert to numpy and transpose to (C, H, W)
                output_np = output.squeeze(0).cpu().numpy()

                # Store prediction
                predictions[filename] = output_np

    # Save predictions to npz file
    np.savez(output_path, **predictions)
    print(f"Saved {len(predictions)} predictions to {output_path}")

    return predictions
