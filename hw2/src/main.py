"""
Digit Recognition with Faster R-CNN
This script implements a digit recognition system using the Faster R-CNN model.
It includes data loading, training, and inference functionalities.
The script is designed to be run from the command line and accepts various
arguments for configuration.
"""
import argparse
import random
import logging
import os
import datetime
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import config
from data_preprocessing import get_data_loaders
from model import get_improved_faster_rcnn_model
from train import train_model
from inference import TestDataset
from inference import inference, recognize_numbers, save_predictions

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    """
    Main function to run the digit recognition system
    """
    parser = argparse.ArgumentParser(description='Digit Recognition with Faster R-CNN')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR,
                        help='Path to data directory')
    parser.add_argument('--train_dir', type=str, default=config.TRAIN_DIR,
                        help='Path to train directory')
    parser.add_argument('--val_dir', type=str, default=config.VAL_DIR,
                        help='Path to validation directory')
    parser.add_argument('--train_annotation', type=str, default=config.TRAIN_ANNOTATION,
                        help='Path to train annotation file')
    parser.add_argument('--val_annotation', type=str, default=config.VAL_ANNOTATION,
                        help='Path to validation annotation file')
    parser.add_argument('--test_dir', type=str, default=config.TEST_DIR,
                        help='Path to test directory')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR,
                        help='Path to output directory')
    parser.add_argument('--checkpoint_dir', type=str, default=config.CHECKPOINT_DIR,
                        help='Path to checkpoint directory')
    parser.add_argument('--num_classes', type=int, default=config.NUM_CLASSES,
                        help='Number of classes (10 digits + background)')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=config.NUM_EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=config.MOMENTUM,
                        help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=config.WEIGHT_DECAY,
                        help='Weight decay')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both',
                        help='Mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    parser.add_argument('--log_dir', type=str, default=config.LOG_DIR,
                        help='Path to log directory')
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=config.NUM_WORKERS,
                        help='Number of workers for data loading')
    args = parser.parse_args()
    # set standard seed for reproducibility
    set_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    # Set up logging to file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"log_{timestamp}.log"
    log_file = os.path.join(args.log_dir, log_filename)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Log the arguments
    logger.info("Arguments: %s", args)
    # Log the configuration
    logger.info("Starting digit recognition system")
    # Check if CUDA is available
    logger.info("Using device: %s", device)
    model = get_improved_faster_rcnn_model(
        num_classes=args.num_classes
    )
    logger.info("Using improved Faster R-CNN model")
    model.to(device)
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Loaded checkpoint from %s", args.checkpoint)
    # Train mode
    if args.mode in ['train', 'both']:
        # Define training and validation datasets and loaders
        train_loader, val_loader = get_data_loaders(
            args.train_dir,
            args.val_dir,
            args.train_annotation,
            args.val_annotation,
            args.batch_size,
            num_workers=args.num_workers
        )
        # Define optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        # Train the model
        model = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            args.num_epochs,
            device,
            args.checkpoint_dir,
            logger
        )
    # Test mode
    if args.mode in ['test', 'both']:
        # Load the best model for testing
        best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            logger.info("Loaded best model for testing from %s", best_model_path)
        else:
            logger.warning("Best model not found at %s, using current model", best_model_path)
            
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
        ])
        test_dataset = TestDataset(args.test_dir, transform=transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda batch: tuple(zip(*batch))
        )
        # Predict on test data
        predictions, image_ids = inference(model, test_loader, device)
        # Recognize numbers
        number_predictions = recognize_numbers(predictions, image_ids)
        # Save predictions
        save_predictions(predictions, number_predictions, args.output_dir)
        logger.info("Predictions saved to %s", args.output_dir)

if __name__ == "__main__":
    main()
