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

    # Load checkpoint
    if not args.checkpoint:
        raise ValueError("Checkpoint path must be provided for testing")

    epoch, best_psnr = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint from epoch {epoch}, best PSNR: {best_psnr:.2f}")

    # Set model to evaluation mode
    model.eval()

    # Generate and save predictions
    with torch.no_grad():
        predictions = []
        filenames = []
        for batch in test_loader:
            degraded = batch['degraded'].to(device)
            filename = batch['filename'][0]  # Get filename for this image

            # Determine degradation type from filename
            degradation_type = 'rain' if 'rain' in filename else 'snow'

            # Forward pass with degradation type as positional argument
            # Model expects (inp_img, noise_emb=None)
            output = model(degraded, degradation_type)

            # Store prediction and filename
            predictions.append(output.cpu().numpy())
            filenames.append(filename)

        # Save predictions
        np.savez(args.output,
                 predictions=np.array(predictions),
                 filenames=np.array(filenames))


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
                        default=64, help='Base number of channels (dim) in the model')
    parser.add_argument('--num-blocks', type=int,
                        default=9, help='Number of transformer blocks (will be distributed across 4 levels)')

    args = parser.parse_args()
    main(args)
