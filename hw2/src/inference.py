"""
This script is used for inference on a test dataset using a trained model.
It loads the test images, applies the model to get predictions,
and saves the results in a specified format.
The script includes the following functionalities:
- Loading test images from a specified directory
- Applying the trained model to get predictions
- Filtering predictions based on a confidence score threshold
- Converting predictions to COCO format
- Saving predictions to a JSON file
- Recognizing numbers from the predictions
- Saving recognized numbers to a CSV file
- The script uses PyTorch for model inference and assumes that the model
  is already trained and saved.
- The script is designed to be run from the command line and accepts
  various arguments for configuration.
- The script is designed to work with a specific dataset format and
  assumes that the test images are stored in a directory with a specific
  naming convention.
- The script also includes functions to handle the loading of images,
  applying transformations, and converting predictions to the required format.
- The script is modular and can be easily integrated into a larger
  pipeline for digit recognition tasks.
"""    
import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import config

class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Custom dataset for loading test images.
        Args:
            data_dir (str): Directory containing the test images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')])
    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.image_files)
    def __getitem__(self, idx):
        """
        Load an image and its ID.
        Args:
            idx (int): Index of the image to load.
        Returns:
            image (PIL Image): Loaded image.
            image_id (int): Image ID.
        """
        # Get image file name
        image_file = self.image_files[idx]
        image_path = os.path.join(self.data_dir, image_file)
        image_id = int(image_file.split('.')[0])  # Assuming image file name is the ID
        image = Image.open(image_path).convert("RGB")
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        return image, image_id

def inference(model, test_loader, device=None, score_threshold=None):
    """
    Perform inference on the test dataset using the trained model.
    Args:
        model: Trained model for inference.
        test_loader: DataLoader for the test dataset.
        device: Device to run the model on (CPU or GPU).
        score_threshold: Confidence score threshold for filtering predictions.
    Returns:
        predictions: List of predictions in COCO format.
        image_ids: List of image IDs for which predictions were made.
    """
    if device is None:
        device = torch.device(config.DEVICE)
    if score_threshold is None:
        score_threshold = config.SCORE_THRESHOLD
    # Set the model to evaluation mode 
    model.eval()
    predictions = []
    image_ids = []
    # Iterate through the test dataset
    with torch.no_grad():
        for images, ids in test_loader:
            images = list(image.to(device) for image in images)
            outputs = model(images)
            for i, output in enumerate(outputs):
                image_id = ids[i]
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                # Filter predictions based on score threshold
                keep = scores > score_threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                # Convert [x_min, y_min, x_max, y_max] to [x_min, y_min, width, height]
                # COCO format: [x_min, y_min, width, height]
                coco_boxes = []
                for box in boxes:
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    coco_boxes.append([
                        float(x_min), 
                        float(y_min),
                        float(width),
                        float(height)
                    ])
                # Append predictions to the list
                for box, score, label in zip(coco_boxes, scores, labels):
                    predictions.append({
                        'image_id': int(image_id),
                        'bbox': box,
                        'score': float(score),
                        'category_id': int(label)
                    })
                image_ids.append(int(image_id))
    return predictions, list(set(image_ids))

def recognize_numbers(predictions, image_ids):
    """
    Recognize numbers from the predictions.
    Args:
        predictions: List of predictions in COCO format.
        image_ids: List of image IDs for which predictions were made.
    Returns:
        number_predictions: List of recognized numbers for each image.
    """
    # Group predictions by image ID
    predictions_by_image = {}
    for pred in predictions:
        image_id = pred['image_id']
        if image_id not in predictions_by_image:
            predictions_by_image[image_id] = []
        predictions_by_image[image_id].append(pred)
    
    # Recognize numbers
    number_predictions = []
    for image_id in image_ids:
        if image_id not in predictions_by_image or len(predictions_by_image[image_id]) == 0:
            number_predictions.append({'image_id': image_id, 'pred_label': -1})
            continue
        # Sort predictions by x_min (leftmost digit)
        digits = sorted(predictions_by_image[image_id], key=lambda x: x['bbox'][0])
        # Extract digits from predictions
        number = ''.join([str(digit['category_id'] - 1) for digit in digits])  # 類別ID從1開始，數字從0開始
        number_predictions.append({
            'image_id': image_id,
            'pred_label': int(number)
        })
    return number_predictions

def save_predictions(predictions, number_predictions, output_dir=None):
    """
    Save predictions to JSON and CSV files.
    Args:
        predictions: List of predictions in COCO format.
        number_predictions: List of recognized numbers for each image.
        output_dir (str, optional): Directory to save the predictions.
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    # Save predictions in COCO format to JSON file
    with open(os.path.join(output_dir, 'pred.json'), 'w', encoding='utf-8') as f:
        json.dump(predictions, f)
    # Save recognized numbers to CSV file
    with open(os.path.join(output_dir, 'pred.csv'), 'w', encoding='utf-8') as f:
        f.write('image_id,pred_label\n')
        for pred in number_predictions:
            f.write(f"{pred['image_id']},{pred['pred_label']}\n")
