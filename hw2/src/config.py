"""
This file contains all the necessary configurations for training,
validation, and testing the model.
It includes paths for data directories, model parameters,
training parameters, image parameters, data augmentation parameters,
and device configuration.
"""
import os
import torch

# Set Path
DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')
TRAIN_ANNOTATION = os.path.join(DATA_DIR, 'train.json')
VAL_ANNOTATION = os.path.join(DATA_DIR, 'valid.json')
OUTPUT_DIR = 'output'
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'

# Model Arguments
NUM_CLASSES = 11  # 10 classes + 1 for background
USE_IMPROVED_MODEL = True  # Use improved model or not

# Training Parameters
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
SCORE_THRESHOLD = 0.5  # Score threshold for predictions

# Image Parameters
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
IMG_HEIGHT = 800
IMG_WIDTH = 800

# Data Augmentation Parameters
FLIP_PROB = 0.5  # Probability of flipping the image
ROTATE_ANGLE = 15  # Angle for random rotation
ROTATE_ANGLE_PROB = 0.5  # Probability of rotating the image
SCALE = (0.8, 1.2)  # Scale range for random scaling
SCALE_PROB = 0.5  # Probability of scaling the image
HUE_PROB = 0.5  # Probability of changing hue
HUE_DELTA = 0.1  # Delta for hue change
BLUR_PROB = 0.5  # Probability of applying Gaussian blur
BLUR_KERNEL_SIZE = 5  # Kernel size for Gaussian blur
BLUR_SIGMA = 1.0  # Sigma for Gaussian blur
# Probability of changing brightness and contrast
RANDOM_BRIGHTNESS_CONTRAST_PROB = 0.5  

# RPN Parameters for Anchor Generation
ANCHOR_SIZES = ((16,), (32,), (64,), (128,), (256,))
ANCHOR_RATIOS = ((0.5, 1.0, 2.0),) * 5

# Training Parameters
NUM_WORKERS = 4  # Number of workers for data loading
# Device Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42
