"""
This module contains the DigitDataset class for loading and processing
the digit dataset.
It uses the COCO format for annotations and applies transformations to
the images.
The DigitDataset class inherits from PyTorch's Dataset class and implements
the necessary methods for loading images and annotations.
It also includes a function to get the data loaders for training and
validation datasets.
The dataset is expected to be in the COCO format, with images and annotations
stored in separate directories.
The module also includes a function to get the data loaders for training and
validation datasets.
The DigitDataset class is designed to work with the COCO dataset format.
It includes methods for loading images and annotations, applying
transformations, and returning the data in a format suitable for training.
"""
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import config

class DigitDataset(Dataset):
    """
    Custom dataset for loading digit images and annotations in COCO format.
    Args:
        data_dir (str): Directory containing the images.
        annotation_file (str): Path to the COCO annotation file.
        transform (callable, optional): Optional transform to be applied
            on a sample.
        is_train (bool): If True, the dataset is used for training.
    """
    def __init__(self, data_dir, annotation_file, transform=None, is_train=True):
        """
        Custom dataset for loading digit images and annotations in COCO format.
        Args:
            data_dir (str): Directory containing the images.
            annotation_file (str): Path to the COCO annotation file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            is_train (bool): If True, the dataset is used for training.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        # Read COCO annotations
        self.coco = COCO(annotation_file)
        self.image_ids = list(sorted(self.coco.imgs.keys()))
        # Get image ids
        self.categories = {
            cat['id']: cat['name']
            for cat in self.coco.loadCats(self.coco.getCatIds())
        }
    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.image_ids)
    def __getitem__(self, idx):
        """Load an image and its annotations."""
        # Get image id
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.data_dir, image_info['file_name'])
        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if not self.is_train:
            return image, image_id
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        for ann in annotations:
            bbox = ann['bbox']  # [x_min, y_min, width, height]
            # Convert to [x_min, y_min, x_max, y_max]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            boxes.append(bbox)
            labels.append(ann['category_id'])
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([image_id])
        return image, target

def get_data_loaders(
        train_dir=config.TRAIN_DIR,
        val_dir=config.VAL_DIR,
        train_annotation=config.TRAIN_ANNOTATION,
        val_annotation=config.VAL_ANNOTATION,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    ):
    """
    Get data loaders for training and validation datasets.
    Args:
        train_dir (str): Directory containing training images.
        val_dir (str): Directory containing validation images.
        train_annotation (str): Path to training annotation file.
        val_annotation (str): Path to validation annotation file.
        batch_size (int, optional): Batch size for data loaders.
        num_workers (int, optional): Number of workers for data loading.
    Returns:
        train_loader (DataLoader): Data loader for training dataset.
        val_loader (DataLoader): Data loader for validation dataset.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
    ])
    # Create datasets
    train_dataset = DigitDataset(
        data_dir=train_dir,
        annotation_file=train_annotation,
        transform=transform,
        is_train=True
    )
    val_dataset = DigitDataset(
        data_dir=val_dir,
        annotation_file=val_annotation,
        transform=transform,
        is_train=True
    )
    # Create data loaders
    def collate_fn(batch):
        return tuple(zip(*batch))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == 'cuda' else False,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == 'cuda' else False,
        collate_fn=collate_fn
    )
    return train_loader, val_loader
