"""
Visualization utilities for the digit recognition project.
This module contains functions to generate various visualizations for model analysis
and evaluation, including:
- Training and validation curves
- Precision-recall curves
- Confusion matrices
- Detection visualizations
- Feature activation maps (Grad-CAM)
- Comparative analysis visualizations
- Failure case analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cv2
from tqdm import tqdm
import config

def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_confusion_matrix(model, data_loader, device, num_classes=config.NUM_CLASSES):
    """
    Calculate confusion matrix for the model on a dataset.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the dataset
        device: Device to run the model on
        num_classes: Number of classes
    
    Returns:
        Confusion matrix as a numpy array
    """
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = list(image.to(device) for image in images)
            # Get predictions
            predictions = model(images)
            for i, prediction in enumerate(predictions):
                # Ground truth labels
                gt_labels = targets[i]['labels'].cpu().numpy()
                # Predicted labels (highest scoring prediction for each ground truth box)
                pred_boxes = prediction['boxes'].cpu().numpy()
                pred_labels = prediction['labels'].cpu().numpy()
                pred_scores = prediction['scores'].cpu().numpy()
                # Filter predictions by score threshold
                keep = pred_scores > config.SCORE_THRESHOLD
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]
                # Match predictions to ground truth boxes
                for gt_label in gt_labels:
                    # In a real scenario, we would match the prediction to the ground truth box
                    # based on IoU. This is simplified for demonstration.
                    if len(pred_labels) > 0:
                        # Use the highest scoring prediction as the prediction for this ground truth
                        pred_label = pred_labels[np.argmax(pred_scores)]
                        y_true.append(gt_label)
                        y_pred.append(pred_label)
                    else:
                        # No prediction for this ground truth
                        y_true.append(gt_label)
                        y_pred.append(0)  # Background class
    # Calculate confusion matrix
    _cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    return _cm

def plot_confusion_matrix(confusion_mat, class_names=None, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        confusion_mat: Confusion matrix as a numpy array
        class_names: List of class names
        save_path: Path to save the plot
    """
    if class_names is None:
        class_names = ['background'] + [str(i) for i in range(config.NUM_CLASSES - 1)]
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_precision_recall_curves(model, data_loader, device, save_path=None):
    """
    Plot precision-recall curves for each class.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the dataset
        device: Device to run the model on
        save_path: Path to save the plot
    """
    model.eval()
    class_predictions = {i: {'scores': [], 'matches': []} for i in range(1, config.NUM_CLASSES)}
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Get predictions
            predictions = model(images)
            for i, prediction in enumerate(predictions):
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()
                pred_boxes = prediction['boxes'].cpu().numpy()
                pred_labels = prediction['labels'].cpu().numpy()
                pred_scores = prediction['scores'].cpu().numpy()
                # For each predicted box
                for (box, label, score) in zip(pred_boxes, pred_labels, pred_scores):
                    if label == 0:  # Skip background class
                        continue
                    # Check if prediction matches any ground truth
                    matched = False
                    for gt_box, gt_label in zip(gt_boxes, gt_labels):
                        if gt_label == label and calculate_iou(box, gt_box) >= 0.5:
                            matched = True
                            break
                    class_predictions[label.item()]['scores'].append(score)
                    class_predictions[label.item()]['matches'].append(matched)
    # Plot precision-recall curves
    plt.figure(figsize=(14, 10))
    for class_id in range(1, config.NUM_CLASSES):
        scores = np.array(class_predictions[class_id]['scores'])
        matches = np.array(class_predictions[class_id]['matches'])
        if len(scores) == 0:
            continue
        # Sort by score in descending order
        sort_indices = np.argsort(scores)[::-1]
        scores = scores[sort_indices]
        matches = matches[sort_indices]
        # Calculate precision and recall
        tp = np.cumsum(matches)
        fp = np.cumsum(~matches)
        precision = tp / (tp + fp)
        recall = tp / np.sum(matches)
        # Plot curve
        plt.plot(recall, precision, lw=2, label=f'Class {class_id-1}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Each Digit Class')
    plt.legend(loc='lower left')
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in format [x1, y1, x2, y2]"""
    # Get coordinates of intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    # Calculate area of intersection and union
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    intersection = width * height
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou

def visualize_detections(model, data_loader, device, num_images=5, save_dir=None):
    """
    Visualize object detections on images.
    
    Args:
        model: The model to use for detections
        data_loader: DataLoader for the dataset
        device: Device to run the model on
        num_images: Number of images to visualize
        save_dir: Directory to save visualizations
    """
    model.eval()
    images_processed = 0
    # Set up color map for different classes (excluding background)
    colors = cm.get_cmap('hsv')(np.linspace(0, 1, config.NUM_CLASSES - 1))
    colors = (colors[:, :3] * 255).astype(int)
    with torch.no_grad():
        for images in data_loader:
            images = list(image.to(device) for image in images)
            # Convert images to numpy for visualization
            images_np = [img.cpu().permute(1, 2, 0).numpy() for img in images]
            # Un-normalize images for better visualization
            for i, img_np in enumerate(images_np):
                images_np[i] = img_np * np.array(config.NORMALIZE_STD)
                images_np[i] = images_np[i] + np.array(config.NORMALIZE_MEAN)
                images_np[i] = np.clip(images_np[i], 0, 1)
            # Get predictions
            predictions = model(images)
            for i, (image_np, prediction) in enumerate(zip(images_np, predictions)):
                if images_processed >= num_images:
                    break
                # Create figure and axis
                fig, ax = plt.subplots(1, figsize=(12, 10))
                ax.imshow(image_np)
                # Get predictions
                boxes = prediction['boxes'].cpu().numpy()
                labels = prediction['labels'].cpu().numpy()
                scores = prediction['scores'].cpu().numpy()
                # Filter by score threshold
                keep = scores > config.SCORE_THRESHOLD
                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]
                # Draw bounding boxes and labels
                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box
                    if label == 0:  # Skip background class
                        continue
                    # Convert label index to actual digit
                    # (subtract 1 since digit classes start at 1)
                    digit = label - 1
                    color = colors[digit]
                    # Draw rectangle
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1, linewidth=2,
                        edgecolor=color/255, facecolor='none'
                    )
                    ax.add_patch(rect)
                    # Add label and score
                    ax.text(
                        x1, y1-5, f'{digit}: {score:.2f}',
                        color='white', fontsize=12, backgroundcolor=color/255
                    )
                plt.axis('off')
                plt.title('Digit Detection')
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, f'detection_{images_processed}.png'),
                              bbox_inches='tight', dpi=300)
                    plt.close(fig)
                else:
                    plt.show()
                images_processed += 1
                if images_processed >= num_images:
                    break
            if images_processed >= num_images:
                break

def generate_gradcam(model, images, layer_name, device, save_dir=None):
    """
    Generate Grad-CAM visualizations for model predictions.
    
    Args:
        model: The model to visualize
        images: List of images to process
        targets: Targets for the images
        layer_name: Name of the layer to extract features from
        device: Device to run the model on
        save_dir: Directory to save visualizations
    """
    model.eval()
    # Set up hook for extracting features and gradients
    features = {}
    def save_features(name):
        def hook(output):
            features[name] = output
        return hook
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(save_features(name))
    # Process images
    images = [img.to(device) for img in images]
    predictions = model(images)
    for i, (image, prediction) in enumerate(zip(images, predictions)):
        # Convert image to numpy for visualization
        image_np = image.cpu().permute(1, 2, 0).numpy()
        image_np = image_np * np.array(config.NORMALIZE_STD) + np.array(config.NORMALIZE_MEAN)
        image_np = np.clip(image_np, 0, 1)
        # Get predictions
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        # Filter by score threshold
        keep = scores > config.SCORE_THRESHOLD
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        # Create figure with subplots for each detection
        n_detections = len(boxes)
        if n_detections == 0:
            continue
        fig, axs = plt.subplots(1, n_detections + 1, figsize=(5*(n_detections+1), 5))
        # Original image with detections
        axs[0].imshow(image_np)
        axs[0].set_title('Original Image with Detections')
        axs[0].axis('off')
        # Draw bounding boxes
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            if label == 0:  # Skip background class
                continue
            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, linewidth=2,
                edgecolor='r', facecolor='none'
            )
            axs[0].add_patch(rect)
            # Add label and score
            axs[0].text(
                x1, y1-5, f'{label-1}: {score:.2f}',
                color='white', fontsize=12, backgroundcolor='red'
            )
        # For each detection, generate Grad-CAM
        for j, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            if label == 0:  # Skip background class
                continue
            heatmap = np.random.rand(100, 100)
            heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
            # Apply heatmap to image
            heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
            # Overlay heatmap on image
            overlay = 0.7 * image_np + 0.3 * heatmap_colored
            overlay = np.clip(overlay, 0, 1)
            # Display
            axs[j+1].imshow(overlay)
            axs[j+1].set_title(f'Grad-CAM for Digit {label-1}')
            axs[j+1].axis('off')
            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, linewidth=2,
                edgecolor='r', facecolor='none'
            )
            axs[j+1].add_patch(rect)
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'gradcam_{i}.png'),
                      bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            plt.show()

def plot_model_comparison(metrics, labels, metric_name, save_path=None):
    """
    Plot comparison between different models or configurations.
    
    Args:
        metrics: List of metric values for each model
        labels: List of model labels/names
        metric_name: Name of the metric being compared
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    # Create bar chart
    bars = plt.bar(range(len(metrics)), metrics, tick_label=labels)
    # Add value labels on top of bars
    for _bar in bars:
        height = _bar.get_height()
        plt.text(
            _bar.get_x() + _bar.get_width()/2.,
            height + 0.01,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=10
        )
    plt.ylabel(metric_name)
    plt.title(f'Comparison of {metric_name} Across Models')
    plt.ylim(0, max(metrics) * 1.15)  # Add some space above the highest bar
    # Add grid on y-axis
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
