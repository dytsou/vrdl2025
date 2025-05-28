from utils import save_checkpoint, load_checkpoint, calculate_psnr
from data import get_data_loaders
from model import PromptIR
from pathlib import Path
import json
import csv
from pytorch_msssim import MS_SSIM
import numpy as np
import time
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch
import argparse
import os
# Set PyTorch CUDA allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


def train_one_epoch(model, loader, criterion_l1, criterion_msssim, optimizer, device, args):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_psnr = 0

    with tqdm(loader, desc="Training") as pbar:
        for batch in pbar:
            # Get data
            degraded = batch['degraded'].to(device)
            clean = batch['clean'].to(device)
            filenames = batch['filename']

            # Determine degradation type from filename
            # Assuming filenames contain 'rain' or 'snow'
            degradation_types = []
            for filename in filenames:
                if 'rain' in filename:
                    degradation_types.append('rain')
                elif 'snow' in filename:
                    degradation_types.append('snow')
                else:
                    raise ValueError(
                        f"Invalid filename {filename}: must contain 'rain' or 'snow'")

            # Forward pass
            optimizer.zero_grad()
            outputs = []
            for i, img in enumerate(degraded):
                # Process each image with its corresponding degradation type
                # Passing as positional argument (model expects inp_img, noise_emb=None)
                output = model(img.unsqueeze(0), degradation_types[i])
                outputs.append(output)
            outputs = torch.cat(outputs)

            # Calculate loss
            l1_loss_val = criterion_l1(outputs, clean)
            msssim_loss_val = 1 - criterion_msssim(outputs, clean)
            loss = args.loss_alpha * l1_loss_val + \
                (1 - args.loss_alpha) * msssim_loss_val

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            with torch.no_grad():
                # Ensure outputs are in [0,1] before PSNR calculation
                clamped_outputs = torch.clamp(outputs, 0, 1)
                psnr = calculate_psnr(clamped_outputs, clean)
                epoch_psnr += psnr

            # Update progress bar
            pbar.set_postfix(loss=loss.item(), psnr=psnr)

    return epoch_loss / len(loader), epoch_psnr / len(loader)


def validate(model, loader, criterion_l1, criterion_msssim, device, args):
    """Evaluate model on validation set"""
    model.eval()
    val_loss = 0
    val_psnr = 0

    with torch.no_grad():
        with tqdm(loader, desc="Validating") as pbar:
            for batch in pbar:
                # Get data
                degraded = batch['degraded'].to(device)
                clean = batch['clean'].to(device)
                filenames = batch['filename']

                # Determine degradation type from filename
                degradation_types = []
                for filename in filenames:
                    if 'rain' in filename:
                        degradation_types.append('rain')
                    elif 'snow' in filename:
                        degradation_types.append('snow')
                    else:
                        raise ValueError(
                            f"Invalid filename {filename}: must contain 'rain' or 'snow'")

                # Forward pass
                outputs = []
                for i, img in enumerate(degraded):
                    # Passing as positional argument (model expects inp_img, noise_emb=None)
                    output = model(img.unsqueeze(0), degradation_types[i])
                    outputs.append(output)
                outputs = torch.cat(outputs)

                # Calculate loss
                l1_loss_val = criterion_l1(outputs, clean)
                msssim_loss_val = 1 - criterion_msssim(outputs, clean)
                loss = args.loss_alpha * l1_loss_val + \
                    (1 - args.loss_alpha) * msssim_loss_val

                # Track metrics
                val_loss += loss.item()
                # Ensure outputs are in [0,1] before PSNR calculation
                clamped_outputs = torch.clamp(outputs, 0, 1)
                psnr = calculate_psnr(clamped_outputs, clean)
                val_psnr += psnr

                # Update progress bar
                pbar.set_postfix(loss=loss.item(), psnr=psnr)

    return val_loss / len(loader), val_psnr / len(loader)


def main(args):
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Setup CSV logging
    log_file = os.path.join(logs_dir, 'training_log.csv')
    log_exists = os.path.exists(log_file)
    log_file_handle = open(log_file, 'a' if log_exists else 'w', newline='')
    log_writer = csv.writer(log_file_handle)

    # Write header if creating a new log file
    if not log_exists:
        log_writer.writerow(
            ['epoch', 'train_loss', 'train_psnr', 'val_loss', 'val_psnr', 'lr'])

    # Save training arguments
    with open(os.path.join(logs_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers
    )
    print(f"Train: {len(train_loader.dataset)} images")
    print(f"Validation: {len(val_loader.dataset)} images")
    print(f"Test: {len(test_loader.dataset)} images")

    # Create model
    # Convert num_blocks from int to list if it's an integer
    if isinstance(args.num_blocks, int):
        n = args.num_blocks
        # Distribute blocks across 4 levels: [n//4, n//4, n//4, n-3*(n//4)]
        blocks = [n//4, n//4, n//4, n-3*(n//4)]
    else:
        blocks = args.num_blocks

    model = PromptIR(
        inp_channels=3,
        out_channels=3,
        dim=args.base_channels,
        num_blocks=blocks,
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        decoder=True  # Enable prompt functionality
    )
    model = model.to(device)

    # Define loss and optimizer
    criterion_l1 = nn.L1Loss().to(device)
    criterion_msssim = MS_SSIM(
        data_range=1.0, size_average=True, channel=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Load checkpoint if resuming
    start_epoch = 0
    best_psnr = 0
    if args.resume:
        start_epoch, best_psnr = load_checkpoint(args.resume, model, optimizer)
        print(f"Resuming from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_psnr = train_one_epoch(
            model, train_loader, criterion_l1, criterion_msssim, optimizer, device, args
        )
        print(f"Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f}")

        # Validate
        val_loss, val_psnr = validate(
            model, val_loader, criterion_l1, criterion_msssim, device, args)
        print(f"Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}")

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        log_writer.writerow(
            [epoch+1, train_loss, train_psnr, val_loss, val_psnr, current_lr])
        log_file_handle.flush()

        # Update scheduler
        scheduler.step(val_psnr)

        # Save checkpoint
        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
            print(f"New best PSNR: {best_psnr:.2f}")

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_psnr': best_psnr
        }, is_best, args.output_dir)

    # Close log file
    log_file_handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PromptIR model for image restoration")

    # Data arguments
    parser.add_argument('--data-dir', type=str,
                        default='data', help='Path to data directory containing rain and snow images')
    parser.add_argument('--output-dir', type=str,
                        default='output', help='Output directory for checkpoints and logs')
    parser.add_argument('--batch-size', type=int,
                        default=8, help='Batch size for training')
    parser.add_argument('--val-ratio', type=float,
                        default=0.1, help='Validation set ratio')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')

    # Model arguments
    parser.add_argument('--base-channels', type=int,
                        default=64, help='Base number of channels (dim) in the model')
    parser.add_argument('--num-blocks', type=int,
                        default=9, help='Number of transformer blocks (will be distributed across 4 levels)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--loss-alpha', type=float, default=0.84,
                        help='Weight for L1 loss in combined L1+MS-SSIM loss (1-alpha for MS-SSIM)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
