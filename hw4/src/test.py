import os
import argparse
import torch
import numpy as np

from model import PromptIR
from data import get_data_loaders
from utils import load_checkpoint, save_predictions_to_npz


def main(args):
    """Generate predictions on test set"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders (we only need the test loader)
    _, _, test_loader = get_data_loaders(
        root_dir=args.data_dir,
        batch_size=1,  # Process one image at a time for test
        val_ratio=0.1,
        num_workers=1
    )
    print(f"Test: {len(test_loader.dataset)} images")

    # Create model
    model = PromptIR(base_channels=args.base_channels,
                     prompt_dim=args.prompt_dim)
    model = model.to(device)

    # Load checkpoint
    if not args.checkpoint:
        raise ValueError("Checkpoint path must be provided for testing")

    epoch, best_psnr = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint from epoch {epoch}, best PSNR: {best_psnr:.2f}")

    # Set model to evaluation mode
    model.eval()

    # Generate and save predictions
    save_predictions_to_npz(
        model=model,
        test_loader=test_loader,
        output_path=args.output,
        device=device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test PromptIR model on test set")

    # Data arguments
    parser.add_argument('--data-dir', type=str,
                        default='data', help='Path to data directory')
    parser.add_argument('--checkpoint', type=str,
                        required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str,
                        default='pred.npz', help='Output npz file path')

    # Model arguments
    parser.add_argument('--base-channels', type=int,
                        default=64, help='Base number of channels')
    parser.add_argument('--prompt-dim', type=int,
                        default=64, help='Prompt dimension')

    args = parser.parse_args()
    main(args)
