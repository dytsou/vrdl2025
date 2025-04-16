# Digit Recognition with Faster R-CNN

A deep learning project that implements a digit recognition system using the Faster R-CNN model. This project is part of the Deep Learning course homework (HW2).

## Overview

This project implements a digit detection and recognition system using Faster R-CNN with a MobileNetV3 Large backbone and Feature Pyramid Network (FPN). The system can detect and classify digits in images, making it useful for applications such as number plate recognition, digit extraction from documents, and more.

## Repository Structure

```
.
├── src/
│   ├── config.py               # Configuration parameters
│   ├── data_preprocessing.py   # Dataset handling and loading
│   ├── inference.py            # Inference and prediction functions
│   ├── main.py                 # Main script to run training and inference
│   ├── model.py                # Model architecture definition
│   ├── train.py                # Training and validation functions
│   ├── transforms.py           # Data augmentation transforms
│   └── visualizations.py       # Visualization utilities
├── results/
│   ├── loss_curve.png
│   └── visualizations/         # Visualization results
├── README.md
└── report.md                   # Detailed project report
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- CUDA (for GPU acceleration)

### Dependencies
Install the required packages:

```bash
pip install torch torchvision
pip install albumentations
pip install pycocotools
pip install matplotlib
```

## Usage

### Training

To train the model with default parameters:

```bash
python src/main.py --mode train
```

Key training parameters:
- Batch size: `8`
- Number of epochs: `30`
- Initial learning rate: `0.005`
- Learning rate scheduler: `StepLR` (step size `10`, gamma `0.1`)
- Optimizer: SGD with momentum `0.9` and weight decay `0.0001`

### Inference

To run inference on test images:

```bash
python src/main.py --mode inference --checkpoint path/to/checkpoint.pth
```

## Model Architecture

The architecture uses an improved version of Faster R-CNN with the following components:

- **Backbone**: `MobileNetV3 Large with FPN`
- **Region Proposal Network (RPN)**:
  - Custom anchor sizes: `(8, 16, 32, 64, 128, 256)`
  - Custom anchor ratios: `(0.5, 0.8, 1.0, 1.25, 2.0)`
- **Head**: Custom FastRCNNPredictor

## Data Preprocessing

The data preprocessing pipeline handles images and annotations in COCO format:
- Images are resized to 800×800 pixels
- Extensive data augmentation techniques implemented with albumentations library

## Results

The model achieves competitive results:

| Metric | Public Score | Private Score |
|--------|--------------|---------------|
| mAP    | 0.943        | 0.937         |
| F1-Score | 0.951      | 0.945         |
| Accuracy | 0.928      | 0.921         |

For detailed results and analysis, please see the [full report](report.md).
