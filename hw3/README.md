# Cell Instance Segmentation with Mask R-CNN

This repository contains a Jupyter notebook implementation of Mask R-CNN for cell instance segmentation. The implementation includes memory optimization techniques, mixed precision training, and optimized model architecture for cell detection.

## Requirements

- PyTorch 2.0+
- torchvision
- torchmetrics
- pycocotools
- tqdm
- PIL
- numpy
- skimage

## Dataset Structure

The dataset should be organized as follows:

```
/kaggle/input/dataset/
├── train/
│   ├── sample1/
│   │   ├── image.tif
│   │   ├── class1.tif
│   │   ├── class2.tif
│   │   └── ...
│   └── ...
├── test_release/
│   ├── test_image1.tif
│   ├── test_image2.tif
│   └── ...
└── test_image_name_to_ids.json
```

## Usage

### Running the Notebook

1. Open the Jupyter notebook `hw3.ipynb` in Kaggle.
2. Run all cells in sequence

The notebook contains the following key sections:

1. Imports and utility functions
2. Dataset and transforms implementation
3. Model architecture
4. Training functions with mixed precision
5. Inference and submission

### Training

The model training is handled by the `main_train()` function. Key parameters:

- Model type: `resnet50_v2` (default)
- Batch size: 2 (optimized to prevent OOM errors)
- Learning rate: 1e-4
- Number of epochs: 50
- Mixed precision: Enabled

The trained model will be saved to `/kaggle/working/maskrcnn_model.pth`

### Inference

The model inference is handled by the `main_test()` function. Key parameters:

- Confidence threshold: 0.5
- Mask threshold: 0.6

The inference results will be saved to `test-results.json` in COCO format.

### Memory Optimization

The implementation includes several memory optimization techniques:

- Mixed precision training with `torch.amp`
- Explicit CUDA cache clearing with `torch.cuda.empty_cache()`
- Reduced batch size (2)
- Gradient clipping

### Model Architecture

The implementation supports two backbone architectures:

1. `resnet50` - Standard ResNet-50 backbone
2. `resnet50_v2` - Improved ResNet-50 backbone with better performance

Both models are optimized for cell detection with:

- Smaller anchor sizes: (8, 16, 32, 64, 128)
- More aspect ratios: (0.5, 1.0, 2.0)
- Increased detections per image: 200
- Lower NMS threshold: 0.3

## Performance

The model is evaluated using Mean Average Precision (mAP) metrics:

- mAP: Overall mean average precision
- mAP@50: Mean average precision at IoU threshold of 0.5

## Acknowledgements

This implementation is based on the PyTorch implementation of Mask R-CNN and includes optimizations for cell instance segmentation. 