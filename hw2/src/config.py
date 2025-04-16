"""
This file contains all the configuration parameters for training, validating,
and testing the model. It includes data directory paths, model parameters, 
training parameters, image parameters, data augmentation parameters, and
device configuration.
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
RESULTS_DIR = 'results'

# Model Arguments
NUM_CLASSES = 11  # 10 classes + 1 for background

# Training Parameters
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
LR_SCHEDULER_STEP_SIZE = 10 # Reduce learning rate every 10 epochs
LR_SCHEDULER_GAMMA = 0.1 # Learning rate reduction factor

# Inference Parameters
SCORE_THRESHOLD = 0.5 # Score threshold for predictions

# Image Parameters
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
IMG_HEIGHT = 800
IMG_WIDTH = 800

# Data Augmentation Parameters
FLIP_PROB = 0.5  # Probability of flipping the image
ROTATE_ANGLE = 20  # Angle for random rotation
ROTATE_ANGLE_PROB = 0.7  # Probability of rotating the image
SCALE = (0.7, 1.3)  # Scale range for random scaling
SCALE_PROB = 0.7  # Probability of scaling the image
HUE_PROB = 0.5  # Probability of changing hue
HUE_DELTA = 0.1  # Delta for hue change
BLUR_PROB = 0.3  # Probability of applying Gaussian blur
BLUR_KERNEL_SIZE = 5  # Kernel size for Gaussian blur
BLUR_SIGMA = 1.0  # Sigma for Gaussian blur
# Probability of changing brightness and contrast
RANDOM_BRIGHTNESS_CONTRAST_PROB = 0.7
# Additional data augmentation parameters
SHIFT_SCALE_ROTATE_PROB = 0.5  # Probability of shift scale rotate
SHIFT_SCALE_LIMIT = 0.1  # Scale limit for shift scale rotate
SHIFT_ROTATE_LIMIT = 15  # Rotate limit for shift scale rotate
SHIFT_LIMIT = 0.1  # Limit for shift scale rotate
CUTOUT_PROB = 0.3  # Probability of cutout
CUTOUT_MAX_H_SIZE = 40
CUTOUT_MAX_W_SIZE = 40
GRID_DISTORTION_PROB = 0.3  # Probability of grid distortion
BBOX_PARAMS_MIN_VISIBILITY = 0.3  # Minimum visibility of bounding box

# RPN Parameters for Anchor Generation
ANCHOR_SIZES = ((8,), (16,), (32,), (64,), (128,), (256,))
ANCHOR_RATIOS = ((0.5, 0.8, 1.0, 1.25, 2.0),) * 6

# Model specific parameters
ROI_OUTPUT_SIZE = 7  # ROI pooling output size
ROI_NMS_THRESH = 0.4  # ROI non-maximum suppression threshold
ROI_SCORE_THRESH = 0.05  # ROI score threshold (inference)
RPN_NMS_THRESH = 0.6  # RPN non-maximum suppression threshold
RPN_FG_IOU_THRESH = 0.6  # RPN foreground IoU threshold
RPN_BG_IOU_THRESH = 0.4  # RPN background IoU threshold
FPN_MAX_LEVEL = 6  # FPN maximum level

# Early Stopping Parameters
PATIENCE = 5  # Number of epochs to wait before stopping
MIN_DELTA = 0.0005  # Minimum change in validation loss to qualify as an improvement
EARLY_STOPPING_EPOCH = 10  # Number of epochs to wait before stopping
# Training settings
NUM_WORKERS = 8  # Number of data loading workers
PIN_MEMORY = True  # Use fixed memory (for GPU training)

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42  # Random seed
