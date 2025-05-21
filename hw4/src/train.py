import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import numpy as np
from pytorch_msssim import MS_SSIM

from model import PromptIR
from data import get_data_loaders
from utils import save_checkpoint, load_checkpoint, calculate_psnr


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
            degradation_types = ['rain' if 'rain' in filename else 'snow'
                                 for filename in filenames]

            # Forward pass
            optimizer.zero_grad()
            outputs = []
            for i, img in enumerate(degraded):
                # Process each image with its corresponding degradation type
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
                degradation_types = ['rain' if 'rain' in filename else 'snow'
                                     for filename in filenames]

                # Forward pass
                outputs = []
                for i, img in enumerate(degraded):
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
    model = PromptIR(base_channels=args.base_channels,
                     prompt_dim=args.prompt_dim,
                     num_blocks=args.num_blocks)
    model = model.to(device)

    # Define loss and optimizer
    criterion_l1 = nn.L1Loss().to(device)
    criterion_msssim = MS_SSIM(
        data_range=1.0, size_average=True, channel=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PromptIR model for image restoration")

    # Data arguments
    parser.add_argument('--data-dir', type=str,
                        default='data', help='Path to data directory')
    parser.add_argument('--output-dir', type=str,
                        default='output', help='Output directory')
    parser.add_argument('--batch-size', type=int,
                        default=8, help='Batch size')
    parser.add_argument('--val-ratio', type=float,
                        default=0.1, help='Validation ratio')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')

    # Model arguments
    parser.add_argument('--base-channels', type=int,
                        default=64, help='Base number of channels')
    parser.add_argument('--prompt-dim', type=int,
                        default=64, help='Prompt dimension')
    parser.add_argument('--num-blocks', type=int,
                        default=9, help='Number of residual blocks in FeatureExtractor')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--loss-alpha', type=float, default=0.84,
                        help='Alpha for L1 loss in combined L1+MS-SSIM loss')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
