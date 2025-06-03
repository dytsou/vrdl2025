# Visual Recognition using Deep Learning Homeworks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)](https://pytorch.org/)

This repository contains a collection of homework assignments for the Visual Recognition using Deep Learning course. Each homework focuses on different aspects of computer vision and deep learning.

## Overview

- **HW1**: Multi-class Image Classification
  - ResNet50-based classifier with CBAM attention
  - Transfer learning from ImageNet
  - Focal Loss for class imbalance
  - Comprehensive data augmentation

- **HW2**: Digit Recognition with Faster R-CNN
  - MobileNetV3 Large backbone with FPN
  - Custom anchor sizes and ratios for digit detection
  - Extensive data augmentation with albumentations
  - Memory-efficient implementation

- **HW3**: Cell Instance Segmentation
  - Mask R-CNN implementation
  - Memory optimization techniques
  - Mixed precision training
  - Optimized model architecture for cell detection

- **HW4**: Image Restoration (Rain and Snow Removal)
  - PromptIR model implementation
  - Prompt-based conditional image restoration
  - FastHOGAwareAttention mechanism
  - Combined L1 and MS-SSIM loss

- **Final Project**: Sea Lion Population Counting
  - YOLO11x-based automated counting system
  - [NOAA Fisheries Steller Sea Lion Population Count Kaggle competition](https://www.kaggle.com/competitions/noaa-fisheries-steller-sea-lion-population-count)
  - Adaptive bounding box annotation with IoU-based adjustment
  - Advanced data augmentation and patch generation strategies
  - Achieved 2nd place on Kaggle leaderboard (11.65312 RMSE)
  - Real-world conservation application for endangered species monitoring

## Requirements

- Python 3.8+ 
- PyTorch 2.0+
- torchvision
- Additional requirements are specified in each homework's directory