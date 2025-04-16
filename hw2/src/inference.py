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
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from visualizations import visualize_detections, visualize_failure_cases

class TestDataset(Dataset):
    """
    Custom dataset for loading test images.
    Args:
        data_dir (str): Directory containing the test images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
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
        self.image_files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ])
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

def visualize_test_predictions(model, test_loader, device, output_dir, num_samples=10):
    """
    Visualize the model's predictions on test data.
    
    Args:
        model: The trained model
        test_loader: DataLoader for test dataset
        device: Device to run inference on
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    # Create directory for visualization
    vis_dir = os.path.join(output_dir, 'test_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    model.eval()
    count = 0
    
    # Set up color map for different classes (excluding background)
    colors = cm.get_cmap('hsv')(np.linspace(0, 1, config.NUM_CLASSES - 1))
    colors = (colors[:, :3] * 255).astype(int)
    
    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Visualizing test predictions"):
            if count >= num_samples:
                break
                
            images = [img.to(device) for img in images]
            predictions = model(images)
            
            for i, (image, image_id, prediction) in enumerate(zip(images, image_ids, predictions)):
                if count >= num_samples:
                    break
                
                # Convert image from tensor to numpy for visualization
                image_np = image.cpu().permute(1, 2, 0).numpy()
                
                # Un-normalize the image
                image_np = image_np * np.array(config.NORMALIZE_STD) + np.array(config.NORMALIZE_MEAN)
                image_np = np.clip(image_np, 0, 1)
                
                # Create figure for plotting
                fig, ax = plt.subplots(1, figsize=(10, 8))
                ax.imshow(image_np)
                
                # Get predictions
                boxes = prediction['boxes'].cpu().numpy()
                labels = prediction['labels'].cpu().numpy()
                scores = prediction['scores'].cpu().numpy()
                
                # Filter by confidence score
                keep = scores > config.SCORE_THRESHOLD
                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]
                
                # Plot bounding boxes
                for box, label, score in zip(boxes, labels, scores):
                    if label == 0:  # Skip background class
                        continue
                        
                    x1, y1, x2, y2 = box
                    
                    # Get color for this digit
                    digit = label - 1  # Convert from class ID (starting at 1) to actual digit
                    color = colors[digit]
                    
                    # Draw the box
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1, 
                        linewidth=2, 
                        edgecolor=color/255, 
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add label and score
                    ax.text(
                        x1, y1-5, 
                        f'{digit}: {score:.2f}', 
                        color='white', 
                        fontsize=12, 
                        backgroundcolor=color/255
                    )
                
                # Sort predictions by x position to recognize the full number
                if len(boxes) > 0:
                    sorted_idx = np.argsort([box[0] for box in boxes])
                    sorted_labels = labels[sorted_idx]
                    recognized_digits = [str(l-1) for l in sorted_labels if l > 0]
                    recognized_number = ''.join(recognized_digits)
                    ax.set_title(f"Image ID: {image_id}, Recognized Number: {recognized_number}")
                else:
                    ax.set_title(f"Image ID: {image_id}, No digits detected")
                
                plt.axis('off')
                plt.tight_layout()
                
                # Save the figure
                plt.savefig(os.path.join(vis_dir, f'test_prediction_{image_id}.png'), bbox_inches='tight', dpi=300)
                plt.close(fig)
                
                count += 1
                
                if count >= num_samples:
                    break

def generate_comparative_visualizations(base_model, improved_model, test_loader, device, output_dir):
    """
    Generate side-by-side comparisons of base model vs improved model predictions.
    
    Args:
        base_model: The baseline model
        improved_model: The improved model
        test_loader: DataLoader for test dataset
        device: Device to run inference on
        output_dir: Directory to save visualizations
    """
    # Create directory for visualization
    vis_dir = os.path.join(output_dir, 'model_comparison')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set models to evaluation mode
    base_model.eval()
    improved_model.eval()
    
    # Set up color map for different classes
    colors = cm.get_cmap('hsv')(np.linspace(0, 1, config.NUM_CLASSES - 1))
    colors = (colors[:, :3] * 255).astype(int)
    
    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Comparing model predictions"):
            images = [img.to(device) for img in images]
            
            # Get predictions from both models
            base_predictions = base_model(images)
            improved_predictions = improved_model(images)
            
            for i, (image, image_id, base_pred, improved_pred) in enumerate(
                zip(images, image_ids, base_predictions, improved_predictions)
            ):
                # Convert image from tensor to numpy for visualization
                image_np = image.cpu().permute(1, 2, 0).numpy()
                
                # Un-normalize the image
                image_np = image_np * np.array(config.NORMALIZE_STD) + np.array(config.NORMALIZE_MEAN)
                image_np = np.clip(image_np, 0, 1)
                
                # Create figure for side-by-side comparison
                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                axes[0].imshow(image_np)
                axes[1].imshow(image_np)
                
                # Process base model predictions
                base_boxes = base_pred['boxes'].cpu().numpy()
                base_labels = base_pred['labels'].cpu().numpy()
                base_scores = base_pred['scores'].cpu().numpy()
                
                # Filter by confidence score
                base_keep = base_scores > config.SCORE_THRESHOLD
                base_boxes = base_boxes[base_keep]
                base_labels = base_labels[base_keep]
                base_scores = base_scores[base_keep]
                
                # Process improved model predictions
                imp_boxes = improved_pred['boxes'].cpu().numpy()
                imp_labels = improved_pred['labels'].cpu().numpy()
                imp_scores = improved_pred['scores'].cpu().numpy()
                
                # Filter by confidence score
                imp_keep = imp_scores > config.SCORE_THRESHOLD
                imp_boxes = imp_boxes[imp_keep]
                imp_labels = imp_labels[imp_keep]
                imp_scores = imp_scores[imp_keep]
                
                # Plot base model predictions
                for box, label, score in zip(base_boxes, base_labels, base_scores):
                    if label == 0:  # Skip background class
                        continue
                        
                    x1, y1, x2, y2 = box
                    digit = label - 1
                    color = colors[digit]
                    
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1, 
                        linewidth=2, 
                        edgecolor=color/255, 
                        facecolor='none'
                    )
                    axes[0].add_patch(rect)
                    axes[0].text(
                        x1, y1-5, 
                        f'{digit}: {score:.2f}', 
                        color='white', 
                        fontsize=12, 
                        backgroundcolor=color/255
                    )
                
                # Plot improved model predictions
                for box, label, score in zip(imp_boxes, imp_labels, imp_scores):
                    if label == 0:  # Skip background class
                        continue
                        
                    x1, y1, x2, y2 = box
                    digit = label - 1
                    color = colors[digit]
                    
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1, 
                        linewidth=2, 
                        edgecolor=color/255, 
                        facecolor='none'
                    )
                    axes[1].add_patch(rect)
                    axes[1].text(
                        x1, y1-5, 
                        f'{digit}: {score:.2f}', 
                        color='white', 
                        fontsize=12, 
                        backgroundcolor=color/255
                    )
                
                # Sort predictions by x position to recognize the full number
                if len(base_boxes) > 0:
                    base_sorted_idx = np.argsort([box[0] for box in base_boxes])
                    base_sorted_labels = base_labels[base_sorted_idx]
                    base_recognized_digits = [str(l-1) for l in base_sorted_labels if l > 0]
                    base_recognized_number = ''.join(base_recognized_digits)
                    axes[0].set_title(f"Baseline Model: {base_recognized_number}")
                else:
                    axes[0].set_title("Baseline Model: No digits detected")
                
                if len(imp_boxes) > 0:
                    imp_sorted_idx = np.argsort([box[0] for box in imp_boxes])
                    imp_sorted_labels = imp_labels[imp_sorted_idx]
                    imp_recognized_digits = [str(l-1) for l in imp_sorted_labels if l > 0]
                    imp_recognized_number = ''.join(imp_recognized_digits)
                    axes[1].set_title(f"Improved Model: {imp_recognized_number}")
                else:
                    axes[1].set_title("Improved Model: No digits detected")
                
                for ax in axes:
                    ax.axis('off')
                
                plt.suptitle(f"Image ID: {image_id}", fontsize=16)
                plt.tight_layout()
                
                # Save the figure
                plt.savefig(
                    os.path.join(vis_dir, f'model_comparison_{image_id}.png'), 
                    bbox_inches='tight', 
                    dpi=300
                )
                plt.close(fig)
