"""Script for training the model."""

import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torchvision

# Configuration
CONFIG = {
    'MODEL_NAME': 'resnet50',
    'NUM_CLASSES': 100,
    'DROPOUT_RATE': 0.3,
    'BATCH_SIZE': 16, 
    'NUM_EPOCHS': 50,
    'LEARNING_RATE': 1e-3,
    'WEIGHT_DECAY': 1e-4,
    'NUM_WORKERS': 4,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'DATA_DIR': 'data',
    'IMG_SIZE': 224, 
    'TRAIN_AUGS': {
        'RANDOM_RESIZED_CROP': (0.8, 1.0),
        'HORIZONTAL_FLIP_PROB': 0.5,
        'COLOR_JITTER': {
            'BRIGHTNESS': 0.2,
            'CONTRAST': 0.2,
            'SATURATION': 0.2,
            'HUE': 0.1
        },
        'RANDOM_AFFINE': {
            'DEGREES': 15,
            'TRANSLATE': (0.1, 0.1),
            'SCALE': (0.9, 1.1)
        }
    },
    'MAX_MODEL_SIZE_MB': 100,
    # New configurations for enhanced training
    'USE_FOCAL_LOSS': True,
    'FOCAL_LOSS_GAMMA': 2.0,
    'USE_CBAM': True,
    'USE_ENSEMBLE': False,  # Keep ensemble disabled to reduce model size
    'CBAM_REDUCTION_RATIO': 128,  # Further increased reduction ratio for ResNet50
    'USE_SIMPLIFIED_CBAM': True  # Keep simplified CBAM
}

def seed_everything(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImageDataset(Dataset):
    """Dataset for loading and preprocessing the image data."""
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_test = is_test
        if not is_test:
            # For training/validation data
            self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            self.class_to_idx = {cls_name: int(cls_name) for cls_name in self.classes}
            self.samples = []
            for class_name in self.classes:
                class_dir = self.root_dir / class_name
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((
                            str(class_dir / img_name),
                            self.class_to_idx[class_name]
                        ))
        else:
            # For test data
            self.samples = [(str(self.root_dir / f), f) for f in os.listdir(root_dir)
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            # For test data, return image and filename
            return image, os.path.basename(img_path)
        return image, target

def get_transforms(is_train=True):
    """Get data transforms for training or validation/testing."""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                CONFIG['IMG_SIZE'],
                scale=CONFIG['TRAIN_AUGS']['RANDOM_RESIZED_CROP']
            ),
            transforms.RandomHorizontalFlip(
                p=CONFIG['TRAIN_AUGS']['HORIZONTAL_FLIP_PROB']
            ),
            transforms.ColorJitter(
                brightness=CONFIG['TRAIN_AUGS']['COLOR_JITTER']['BRIGHTNESS'],
                contrast=CONFIG['TRAIN_AUGS']['COLOR_JITTER']['CONTRAST'],
                saturation=CONFIG['TRAIN_AUGS']['COLOR_JITTER']['SATURATION'],
                hue=CONFIG['TRAIN_AUGS']['COLOR_JITTER']['HUE']
            ),
            transforms.RandomAffine(
                degrees=CONFIG['TRAIN_AUGS']['RANDOM_AFFINE']['DEGREES'],
                translate=CONFIG['TRAIN_AUGS']['RANDOM_AFFINE']['TRANSLATE'],
                scale=CONFIG['TRAIN_AUGS']['RANDOM_AFFINE']['SCALE']
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(CONFIG['IMG_SIZE'] * 1.14)),
            transforms.CenterCrop(CONFIG['IMG_SIZE']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class ChannelAttention(nn.Module):
    """Channel Attention Module with reduced parameters."""
    def __init__(self, in_channels, reduction_ratio=32):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

class SpatialAttention(nn.Module):
    """Spatial Attention Module."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, in_channels, reduction_ratio=32, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class SimplifiedChannelAttention(nn.Module):
    """Simplified Channel Attention Module with minimal parameters."""
    def __init__(self, in_channels, reduction_ratio=64):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

class SimplifiedSpatialAttention(nn.Module):
    """Simplified Spatial Attention Module with minimal parameters."""
    def __init__(self, kernel_size=3):  # Reduced kernel size
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = self.conv(avg_out)
        return self.sigmoid(x)

class SimplifiedCBAM(nn.Module):
    """Simplified CBAM with reduced parameters."""
    def __init__(self, in_channels, reduction_ratio=64, kernel_size=3):
        super().__init__()
        self.channel_attention = SimplifiedChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SimplifiedSpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class ImageClassifier(nn.Module):
    """Image classification model using ResNet50 with optional simplified CBAM."""
    def __init__(self, model_name=CONFIG['MODEL_NAME'], num_classes=CONFIG['NUM_CLASSES']):
        super().__init__()
        # Load pre-trained ResNet50
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        
        # Add CBAM if enabled
        if CONFIG['USE_CBAM']:
            if CONFIG['USE_SIMPLIFIED_CBAM']:
                # Apply CBAM after each bottleneck block
                self.cbam1 = SimplifiedCBAM(256, reduction_ratio=CONFIG['CBAM_REDUCTION_RATIO'])
                self.cbam2 = SimplifiedCBAM(512, reduction_ratio=CONFIG['CBAM_REDUCTION_RATIO'])
                self.cbam3 = SimplifiedCBAM(1024, reduction_ratio=CONFIG['CBAM_REDUCTION_RATIO'])
                self.cbam4 = SimplifiedCBAM(2048, reduction_ratio=CONFIG['CBAM_REDUCTION_RATIO'])
            else:
                self.cbam1 = CBAM(256, reduction_ratio=CONFIG['CBAM_REDUCTION_RATIO'])
                self.cbam2 = CBAM(512, reduction_ratio=CONFIG['CBAM_REDUCTION_RATIO'])
                self.cbam3 = CBAM(1024, reduction_ratio=CONFIG['CBAM_REDUCTION_RATIO'])
                self.cbam4 = CBAM(2048, reduction_ratio=CONFIG['CBAM_REDUCTION_RATIO'])
            
            # Modify forward pass to include CBAM
            self.original_forward = self.model.forward
            self.model.forward = self.forward_with_cbam
        
        # Modify the final layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(CONFIG['DROPOUT_RATE']),
            nn.Linear(num_features, num_classes)
        )

    def forward_with_cbam(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        # Apply CBAM after each bottleneck block
        x = self.model.layer1(x)
        x = self.cbam1(x)
        
        x = self.model.layer2(x)
        x = self.cbam2(x)
        
        x = self.model.layer3(x)
        x = self.cbam3(x)
        
        x = self.model.layer4(x)
        x = self.cbam4(x)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x

    def forward(self, x):
        if CONFIG['USE_CBAM']:
            return self.forward_with_cbam(x)
        return self.model(x)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    return running_loss/len(train_loader), correct/total

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    return running_loss/len(val_loader), correct/total

def count_parameters(model):
    """Count number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model):
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def main():
    """Main training function."""
    # Set random seeds for reproducibility
    seed_everything(42)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create single model (ensemble disabled)
    model = ImageClassifier().to(CONFIG['DEVICE'])
    
    # Print number of trainable parameters and model size
    num_params = count_parameters(model)
    model_size = get_model_size(model)
    print(f"Number of trainable parameters: {num_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    
    # Check if model size exceeds the maximum allowed size
    if model_size > CONFIG['MAX_MODEL_SIZE_MB']:
        raise ValueError(f"Model size ({model_size:.2f} MB) exceeds the maximum allowed size ({CONFIG['MAX_MODEL_SIZE_MB']} MB)")
    
    # Memory optimization for CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Create data loaders
    train_transform = get_transforms(is_train=True)
    val_transform = get_transforms(is_train=False)
    
    train_dataset = ImageDataset(
        os.path.join(CONFIG['DATA_DIR'], 'train'),
        transform=train_transform
    )
    val_dataset = ImageDataset(
        os.path.join(CONFIG['DATA_DIR'], 'val'),
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True
    )
    
    # Create criterion based on configuration
    if CONFIG['USE_FOCAL_LOSS']:
        criterion = FocalLoss(gamma=CONFIG['FOCAL_LOSS_GAMMA'])
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['LEARNING_RATE'],
        weight_decay=CONFIG['WEIGHT_DECAY']
    )
    
    # Use CosineAnnealingWarmRestarts scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # First restart at epoch 10
        T_mult=2,  # Double the restart interval after each restart
        eta_min=CONFIG['LEARNING_RATE'] * 0.01  # Minimum learning rate
    )
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(CONFIG['NUM_EPOCHS']):
        print(f'\nEpoch {epoch+1}/{CONFIG["NUM_EPOCHS"]}')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, CONFIG['DEVICE']
        )
        
        # Step the scheduler after each epoch
        scheduler.step()
        
        # Clear cache after training epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, CONFIG['DEVICE']
        )
        
        # Save current epoch model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, f'models/{CONFIG["MODEL_NAME"]}_epoch_{epoch+1}.pth')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, f'models/{CONFIG["MODEL_NAME"]}_best_model.pth')
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%')
        print(f'Best Val Acc: {best_val_acc*100:.2f}%')
        print(f'Model saved: models/{CONFIG["MODEL_NAME"]}_epoch_{epoch+1}.pth')
        if val_acc == best_val_acc:
            print(f'New best model saved: models/{CONFIG["MODEL_NAME"]}_best_model.pth!')
        
        # Clear cache after validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main() 