# Homework 1

This homework contains the implementation of a deep learning model for multi-class image classification.

## Overview

The project implements a ResNet50-based classifier with the following key features:
- Transfer learning from ImageNet pre-trained weights
- Simplified CBAM attention mechanism
- Focal Loss for handling class imbalance
- Comprehensive data augmentation pipeline
- Memory-efficient architecture design

## Project Structure

```
.
├── data/                   # Dataset directory
│   ├── train/             # Training images
│   ├── val/               # Validation images
│   └── test/              # Test images
├── models/                # Saved model checkpoints
├── train.py              # Training script
├── predict.py            # Inference script
├── report.md             # Detailed project report
└── README.md             # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- pandas
- tqdm
- PIL (Python Imaging Library)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vrdl2025.git
cd vrdl2025/hw1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model:
```bash
python train.py
```

The script will:
- Load and preprocess the dataset
- Train the model with the specified configuration
- Save checkpoints in the `models/` directory
- Display training progress and metrics

### Inference

To generate predictions on test data:
```bash
python predict.py
```

This will:
- Load the best model checkpoint
- Generate predictions for test images
- Save results to `prediction.csv`
- Create `solution.zip` for submission

## Model Architecture

- Base model: ResNet50 (pre-trained on ImageNet)
- Attention: Simplified CBAM
- Loss function: Focal Loss (gamma=2.0)
- Final layer: Custom classification head for 100 classes

## Training Configuration

- Batch size: 16
- Training duration: 50 epochs
- Learning rate: 1e-3 with CosineAnnealingWarmRestarts
- Weight decay: 1e-4
- Optimizer: AdamW

## Performance

- Model size: 90.99 MB
- Training accuracy: 95.88%
- Validation accuracy: 80.67%
- Training time: 2.11 minutes per epoch
- Inference speed: 0.04 seconds per batch

## Data Augmentation

- Random resized crop (scale: 0.8-1.0)
- Random horizontal flip (p=0.5)
- Color jittering
- Random affine transformations

## Results

The model achieves:
- Superior validation accuracy (80.67%)
- Efficient training time (2.11 min/epoch)
- Compact model size (90.99 MB)
- Stable convergence characteristics