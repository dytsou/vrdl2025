"""
This module contains the model definition for the object detection task.
It uses the Faster R-CNN model from torchvision with a MobileNetV3 Large backbone.
The model is designed to detect digits in images and is trained on a custom dataset.
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import config

def get_improved_faster_rcnn_model(num_classes=config.NUM_CLASSES):
    """
    Get an improved Faster R-CNN model with a MobileNetV3 Large backbone.
    Args:
        num_classes (int): Number of classes for the model.
            Default is the number of classes from config.
    Returns:
        model (torchvision.models.detection.FasterRCNN): The improved Faster R-CNN model.
    """
    # Use the lightweight MobileNetV3 Large as the backbone
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one (for our dataset)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Set RPN parameters
    model.rpn.anchor_generator.sizes = config.ANCHOR_SIZES
    model.rpn.anchor_generator.aspect_ratios = config.ANCHOR_RATIOS
    # Set NMS threshold
    model.roi_heads.nms_thresh = config.ROI_NMS_THRESH
    # Set detection threshold
    model.roi_heads.score_thresh = config.ROI_SCORE_THRESH
    # Set RPN NMS threshold
    model.rpn.nms_thresh = config.RPN_NMS_THRESH
    # Set RPN foreground and background thresholds
    model.rpn.fg_iou_thresh = config.RPN_FG_IOU_THRESH
    model.rpn.bg_iou_thresh = config.RPN_BG_IOU_THRESH
    return model
