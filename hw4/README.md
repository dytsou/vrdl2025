# Image Restoration for Rain and Snow Removal

This project implements a unified deep learning model (PromptIR) for restoring images degraded by rain and snow.

## Project Structure

```
.
├── data/
│   ├── train/
│   │   ├── clean/       # Clean ground truth images
│   │   └── degraded/    # Degraded images (rain or snow)
│   └── test/
│       └── degraded/    # Test set degraded images
├── src/
│   ├── config.py        # Centralized configuration parameters
│   ├── data.py          # Data loading and preprocessing
│   ├── model.py         # PromptIR implementation
│   ├── train.py         # Training script
│   ├── test.py          # Testing script
│   └── utils.py         # Utility functions
└── requirements.txt     # Dependencies
```

## Environment Setup

1. Create a Python virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure

The dataset should be organized as follows:
- `data/train/clean/`: Clean ground truth images
- `data/train/degraded/`: Degraded images with rain or snow
- `data/test/degraded/`: Test set degraded images (no ground truth)

Degraded images should have filenames that indicate their degradation type (containing "rain" or "snow").

## Configuration

All hyperparameters are centralized in `src/config.py` and organized into logical groups:
- `DATA_CONFIG`: Data loading parameters (paths, batch sizes, etc.)
- `MODEL_CONFIG`: Model architecture parameters (channels, blocks, etc.)
- `PROMPT_CONFIG`: Prompt-specific parameters
- `TRAIN_CONFIG`: Training hyperparameters (learning rate, epochs, etc.)
- `HOG_CONFIG`: HOG attention parameters
- `ENV_CONFIG`: Environment settings

## Training

To train the model:

```bash
python src/train.py --data-dir data --output-dir output --epochs 100 --batch-size 4 --lr 1e-4
```

If you encounter CUDA Out Of Memory errors, try reducing the `--batch-size` (e.g., to 4 or 2).

Additional arguments:
- `--val-ratio`: Ratio of validation set (default: 0.1)
- `--base-channels`: Base number of channels in model (default: 64)
- `--num-blocks`: Number of transformer blocks (default: 9)
- `--resume`: Path to checkpoint to resume training

## Testing

To generate predictions for the test set:

```bash
python src/test.py --data-dir data --checkpoint output/checkpoint_best.pth.tar --output pred.npz
```

This will produce a `pred.npz` file containing the restored images, ready for submission.

## Model Architecture

The PromptIR model consists of several key components:
1. **Feature Extractor**: Extracts features from the degraded image
2. **Prompt Encoder**: Encodes degradation type (rain or snow) into a feature space
3. **Fusion Module**: Merges image features with degradation prompt
4. **Decoder**: Reconstructs the restored image

The model uses a unified approach where a single model can handle both rain and snow removal by leveraging prompt-based conditioning.

## Evaluation

The model is evaluated using PSNR (Peak Signal-to-Noise Ratio) on the validation set during training. The test set predictions are saved in the required `pred.npz` format for submission. 

## Sample Results

![Sample Results](visualizations/sample_results.png)
