"""
Configuration file for PromptIR image restoration model.
Contains all hyperparameters used across training, testing, model architecture, and data loading.
"""

# Data parameters
DATA_CONFIG = {
    # Path to data directory containing rain and snow images
    'data_dir': 'data',
    'output_dir': 'output',            # Output directory for checkpoints and logs
    'batch_size': 8,                   # Batch size for training
    'val_ratio': 0.1,                  # Validation set ratio
    'num_workers': 4,                  # Number of workers for data loading
}

# Model architecture parameters
MODEL_CONFIG = {
    'inp_channels': 3,                 # Number of input channels
    'out_channels': 3,                 # Number of output channels
    # Base number of channels (dim) in the model
    'base_channels': 64,
    # Number of transformer blocks (will be distributed across 4 levels)
    'num_blocks': 9,
    'num_refinement_blocks': 4,        # Number of refinement blocks
    # Number of attention heads at each level
    'heads': [1, 2, 4, 8],
    'ffn_expansion_factor': 2.66,      # Expansion factor for feed-forward network
    'bias': False,                     # Whether to use bias in convolutions
    'LayerNorm_type': 'WithBias',      # LayerNorm type: 'WithBias' or 'BiasFree'
    'decoder': True,                   # Enable prompt functionality
    'prompt_dim': 64,                  # Base dimension for prompt features
}

# PromptGenBlock parameters
PROMPT_CONFIG = {
    'prompt_len': 5,                   # Length of prompts
    'prompt_sizes': [64, 32, 16],      # Sizes of prompts at different levels
}

# Training parameters
TRAIN_CONFIG = {
    'epochs': 100,                     # Number of training epochs
    'lr': 1e-4,                        # Learning rate
    # Weight for L1 loss in combined L1+MS-SSIM loss (1-alpha for MS-SSIM)
    'loss_alpha': 0.84,
    'scheduler_patience': 5,           # Patience for learning rate scheduler
    'scheduler_factor': 0.5,           # Factor for learning rate scheduler
}

# FastHOGAwareAttention parameters
HOG_CONFIG = {
    'orientations': 9,                 # Number of orientation bins for HOG
    'eps': 1e-6,                       # Epsilon for numerical stability
}

# Environment parameters
ENV_CONFIG = {
    # PyTorch CUDA allocation configuration
    'cuda_alloc_conf': 'expandable_segments:True,max_split_size_mb:128',
    'random_seed': 42,                 # Random seed for reproducibility
}
