"""
Data loading and processing for the image restoration task.
"""

import os
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from config import DATA_CONFIG, ENV_CONFIG


class RestorationDataset(Dataset):
    """Dataset for image restoration task (rain and snow removal)"""

    def __init__(self,
                 root_dir,
                 split='train',
                 val_ratio=DATA_CONFIG['val_ratio'],
                 transform=None,
                 augment=False):
        """
        Args:
            root_dir: Root directory of the dataset
            split: 'train', 'val', or 'test'
            val_ratio: Ratio of validation set when split is 'train' or 'val'
            transform: Optional ToTensor and normalization transforms
            augment: Whether to apply data augmentation (for train split)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.augment = augment and self.split == 'train'

        if split == 'test':
            self.img_dir = os.path.join(root_dir, 'test', 'degraded')
            self.files = sorted([f for f in os.listdir(self.img_dir)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            self.clean_dir = None
        else:
            self.img_dir = os.path.join(root_dir, 'train', 'degraded')
            self.clean_dir = os.path.join(root_dir, 'train', 'clean')

            # Get all files in the degraded directory
            degraded_files_all = sorted([f for f in os.listdir(self.img_dir)
                                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            # Get all files in the clean directory as a set for efficient lookup
            clean_files_set = set([f for f in os.listdir(self.clean_dir)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            paired_files = []
            for degraded_fname in degraded_files_all:
                name_part, ext = os.path.splitext(degraded_fname)
                # Split only on the first hyphen
                parts = name_part.split('-', 1)

                if len(parts) == 2:
                    img_type = parts[0]
                    img_id = parts[1]

                    # Construct expected clean filename, e.g., rain_clean-123.png
                    expected_clean_fname = f"{img_type}_clean-{img_id}{ext}"

                    if expected_clean_fname in clean_files_set:
                        paired_files.append(degraded_fname)

            # Split into train and validation
            train_files, val_files = train_test_split(
                paired_files, test_size=val_ratio, random_state=ENV_CONFIG['random_seed'])

            # Select appropriate files for the split
            if split == 'train':
                self.files = train_files
            elif split == 'val':
                self.files = val_files
            else:
                self.files = degraded_files_all

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # This is a degraded image filename, e.g., rain-123.png
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        degraded_img = Image.open(img_path).convert('RGB')

        if self.split == 'test':
            # Apply transforms (ToTensor, Normalization) for test images
            if self.transform:
                degraded_img = self.transform(degraded_img)
            else:
                # Default transform to [0, 1] tensor if no transform provided
                to_tensor_transform = transforms.Compose(
                    [transforms.ToTensor()])
                degraded_img = to_tensor_transform(degraded_img)

            return {
                'degraded': degraded_img,
                'filename': img_name
            }

        # Construct the corresponding clean image filename
        name_part, ext = os.path.splitext(img_name)
        parts = name_part.split('-', 1)
        img_type = parts[0]
        img_id = parts[1]
        clean_img_name = f"{img_type}_clean-{img_id}{ext}"
        clean_img_path = os.path.join(self.clean_dir, clean_img_name)

        clean_img = Image.open(clean_img_path).convert('RGB')

        # Apply synchronized augmentations for training set
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                degraded_img = TF.hflip(degraded_img)
                clean_img = TF.hflip(clean_img)

            # Random vertical flip
            if random.random() > 0.5:
                degraded_img = TF.vflip(degraded_img)
                clean_img = TF.vflip(clean_img)

            # Random rotation
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                degraded_img = TF.rotate(degraded_img, angle)
                clean_img = TF.rotate(clean_img, angle)
            # Or small angle rotations:
            # angle = random.uniform(-10, 10)
            # degraded_img = TF.rotate(degraded_img, angle)
            # clean_img = TF.rotate(clean_img, angle)

        # Apply transforms (ToTensor, Normalization) to both images
        if self.transform:
            # Apply transform to degraded_img
            degraded_img = self.transform(degraded_img)
            clean_img = self.transform(clean_img)
        else:
            # Default transform to [0, 1] tensor if no transform provided
            to_tensor_transform = transforms.Compose([transforms.ToTensor()])
            degraded_img = to_tensor_transform(degraded_img)
            clean_img = to_tensor_transform(clean_img)

        return {
            'degraded': degraded_img,
            'clean': clean_img,
            'filename': img_name
        }


def get_data_loaders(root_dir,
                     batch_size=DATA_CONFIG['batch_size'],
                     val_ratio=DATA_CONFIG['val_ratio'],
                     num_workers=DATA_CONFIG['num_workers']):
    """Create data loaders for training, validation and testing"""

    # Define transformations
    # Geometric augmentations are now handled inside Dataset's __getitem__
    # So train_transform will only contain ToTensor and potentially normalization
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create datasets
    train_dataset = RestorationDataset(
        root_dir, split='train', val_ratio=val_ratio, transform=train_transform, augment=True)
    val_dataset = RestorationDataset(
        root_dir, split='val', val_ratio=val_ratio, transform=val_test_transform, augment=False)
    test_dataset = RestorationDataset(
        root_dir, split='test', transform=val_test_transform, augment=False)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
