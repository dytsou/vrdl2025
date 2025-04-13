""" transforms.py
This module contains the transformation functions for the dataset.
It uses the albumentations library to apply various augmentations
to the images and bounding boxes.
The transformations include resizing, random brightness/contrast,
horizontal flipping, rotation, scaling, and normalization.
The module also includes a function to get the transform for
training and validation datasets.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import config

def get_transform(train=True):
    """Get the transform for the dataset"""
    if train:
        return A.Compose([
            A.Resize(height=config.IMG_HEIGHT, width=config.IMG_WIDTH),
            A.RandomBrightnessContrast(p=config.RANDOM_BRIGHTNESS_CONTRAST_PROB),
            A.HorizontalFlip(p=config.FLIP_PROB),
            A.Rotate(limit=config.ROTATE_ANGLE, p=config.ROTATE_ANGLE_PROB),
            A.RandomScale(scale_limit=config.SCALE, p=config.SCALE_PROB),
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=config.SHIFT_SCALE_ROTATE_PROB),
            A.GridDistortion(p=config.GRID_DISTORTION_PROB),
            A.GaussianBlur(blur_limit=config.BLUR_KERNEL_SIZE, p=config.BLUR_PROB),
            A.HueSaturationValue(hue_shift_limit=config.HUE_DELTA*255, p=config.HUE_PROB),
            A.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels'], min_visibility=0.3))
    else:
        return A.Compose([
            A.Resize(height=config.IMG_HEIGHT, width=config.IMG_WIDTH),
            A.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
