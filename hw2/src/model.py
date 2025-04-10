"""
This module contains the model definition for the object detection task.
It uses the Faster R-CNN model from torchvision with a ResNet50 backbone.
The model is designed to detect digits in images and is trained on a custom dataset.
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import config

def get_faster_rcnn_model(num_classes=config.NUM_CLASSES):
    """
    Get the Faster R-CNN model with a ResNet50 backbone.
    Args:
        num_classes (int): Number of classes for the model.
            If None, it will use the default number of classes from config.
    Returns:
        model (torchvision.models.detection.FasterRCNN): The Faster R-CNN model.
    """
    # Load the pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one (for our dataset)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_improved_faster_rcnn_model(num_classes=config.NUM_CLASSES):
    """
    Get an improved Faster R-CNN model with a ResNet50 backbone.
    Args:
        num_classes (int): Number of classes for the model.
            Default is the number of classes from config.
    Returns:
        model (torchvision.models.detection.FasterRCNN): The improved Faster R-CNN model.
    """
    # Load the pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one (for our dataset)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Set the anchor generator sizes and aspect ratios
    model.rpn.anchor_generator.sizes = config.ANCHOR_SIZES
    model.rpn.anchor_generator.aspect_ratios = config.ANCHOR_RATIOS
    return model
