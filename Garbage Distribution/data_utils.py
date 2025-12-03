"""
Data loading and analysis utilities for Garbage Classification System
"""

# Fix OpenMP library conflict on Windows
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from config import CLASS_NAMES, DATA_DIR, TRAIN_DIR, DISTRIBUTION_PLOT_PATH


def analyze_distribution(data_dir):
    """
    Analyze class distribution in the dataset.
    
    Args:
        data_dir: Path to the data directory containing class subfolders
        
    Returns:
        class_counts: Dictionary mapping class names to image counts
        class_weights: Tensor of class weights for handling imbalance
    """
    class_counts = {}
    
    # Count images in each class folder
    for class_name in CLASS_NAMES:
        class_path = os.path.join(data_dir, class_name)
        if os.path.exists(class_path):
            # Count all image files
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            class_counts[class_name] = len(image_files)
        else:
            class_counts[class_name] = 0
            print(f"Warning: Class folder '{class_name}' not found in {data_dir}")
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = plt.bar(classes, counts, color='steelblue', alpha=0.7)
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=12, fontweight='bold')
    plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(DISTRIBUTION_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to {DISTRIBUTION_PLOT_PATH}")
    plt.close()
    
    # Calculate class weights (inverse frequency) to handle imbalance
    total_samples = sum(class_counts.values())
    if total_samples == 0:
        raise ValueError("No images found in the dataset!")
    
    # Calculate weights: weight = total_samples / (num_classes * class_count)
    # This gives higher weight to underrepresented classes
    class_weights = []
    for class_name in CLASS_NAMES:
        count = class_counts[class_name]
        if count > 0:
            weight = total_samples / (len(CLASS_NAMES) * count)
        else:
            weight = 0.0
        class_weights.append(weight)
    
    # Normalize weights
    class_weights = np.array(class_weights)
    class_weights = class_weights / class_weights.sum() * len(CLASS_NAMES)
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    print("\nClass Distribution Summary:")
    print("-" * 50)
    for class_name, count in class_counts.items():
        weight = class_weights[CLASS_NAMES.index(class_name)]
        print(f"{class_name:20s}: {count:5d} images (weight: {weight:.4f})")
    print("-" * 50)
    print(f"Total images: {total_samples}")
    
    return class_counts, class_weights_tensor


def get_transforms(use_strong_augmentation=True):
    """
    Define data transforms for training and validation.
    
    Args:
        use_strong_augmentation: Whether to use strong augmentation (default: True)
    
    Returns:
        train_transform: Transformations for training (with augmentation)
        val_transform: Transformations for validation (no augmentation)
    """
    if use_strong_augmentation:
        # Enhanced training transforms with aggressive data augmentation
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize larger first
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),  # Add vertical flip
            transforms.RandomRotation(degrees=30),  # Increased rotation
            transforms.ColorJitter(
                brightness=0.3, 
                contrast=0.3, 
                saturation=0.3, 
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.1, 0.1), 
                scale=(0.9, 1.1), 
                shear=10
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    else:
        # Basic training transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def get_class_mapping(data_dir):
    """
    Get class to index mapping from ImageFolder dataset.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        class_to_idx: Dictionary mapping class names to indices
        idx_to_class: Dictionary mapping indices to class names
        dataset_classes: Sorted list of class names as used by ImageFolder
    """
    # Load dataset just to get class mapping
    temp_dataset = datasets.ImageFolder(root=data_dir, transform=None)
    class_to_idx = temp_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    dataset_classes = sorted([os.path.basename(d) for d in temp_dataset.classes])
    
    return class_to_idx, idx_to_class, dataset_classes


def get_dataloaders(data_dir=None, batch_size=32, train_split=0.8, num_workers=None):
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Path to data directory (defaults to DATA_DIR from config)
        batch_size: Batch size for dataloaders
        train_split: Proportion of data to use for training (default 0.8 = 80%)
        num_workers: Number of worker processes for data loading (None = auto-detect)
        
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        class_weights: Tensor of class weights for loss function
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    # Set num_workers for data loading
    # On Windows, use 0 to avoid multiprocessing issues
    # On Linux/Mac, use more workers for faster data loading
    if num_workers is None:
        import platform
        if platform.system() == 'Windows':
            num_workers = 0  # Windows multiprocessing issues
        else:
            # Use 4-8 workers for faster data loading on Linux/Mac
            num_workers = min(8, os.cpu_count() or 4)
    
    # Get transforms with strong augmentation
    train_transform, val_transform = get_transforms(use_strong_augmentation=True)
    
    # Load full dataset without transform (we'll apply transforms separately)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=None)
    
    # Get the actual class names and indices from ImageFolder
    # ImageFolder sorts classes alphabetically
    dataset_classes = sorted([os.path.basename(d) for d in full_dataset.classes])
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Verify class names match
    if dataset_classes != CLASS_NAMES:
        print(f"Warning: Dataset classes {dataset_classes} don't match config classes {CLASS_NAMES}")
        print(f"Using dataset classes: {dataset_classes}")
        print(f"Class to index mapping: {class_to_idx}")
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    # Split dataset into subsets
    train_subset, val_subset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create wrapper to apply transforms
    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, idx):
            # Get image and label from subset (PIL image, no transform yet)
            image, label = self.subset[idx]
            # Apply transform if provided
            if self.transform:
                image = self.transform(image)
            return image, label
        
        def __len__(self):
            return len(self.subset)
    
    # Create train and val datasets with appropriate transforms
    train_dataset = TransformDataset(train_subset, transform=train_transform)
    val_dataset = TransformDataset(val_subset, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Calculate class weights
    _, class_weights = analyze_distribution(data_dir)
    
    print(f"\nDataLoader Summary:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Class mapping: {class_to_idx}")
    
    return train_loader, val_loader, class_weights, class_to_idx, idx_to_class, dataset_classes


if __name__ == "__main__":
    # Test the functions
    print("Testing data utilities...")
    
    # Analyze distribution
    class_counts, class_weights = analyze_distribution(DATA_DIR)
    
    # Get dataloaders
    train_loader, val_loader, weights, class_to_idx, idx_to_class, dataset_classes = get_dataloaders(batch_size=32)
    
    print("\nData utilities test completed successfully!")

