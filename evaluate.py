"""
Evaluation script for Garbage Classification System
Generates confusion matrix and detailed metrics
"""

# Fix OpenMP library conflict on Windows
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from model import create_model
from data_utils import get_dataloaders
from config import (
    NUM_CLASSES, BATCH_SIZE, MODEL_SAVE_PATH, DATA_DIR,
    CONFUSION_MATRIX_PATH, CONFUSION_MATRIX_DATA_PATH, CLASS_NAMES
)


def evaluate_model(model, val_loader, device, class_names):
    """
    Evaluate the model and generate metrics.
    
    Args:
        model: Trained model
        val_loader: Validation DataLoader
        device: Device to run on
        class_names: List of class names
        
    Returns:
        Dictionary containing all metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Running evaluation on validation set...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches...")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate overall accuracy
    accuracy = (all_preds == all_labels).sum() / len(all_labels) * 100
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Calculate macro and weighted averages
    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_macro = f1.mean()
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }
    
    return metrics


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(14, 12))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Frequency'}
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {save_path}")
    plt.close()


def print_metrics(metrics, class_names):
    """
    Print detailed evaluation metrics.
    
    Args:
        metrics: Dictionary containing metrics
        class_names: List of class names
    """
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.2f}%")
    print(f"\nMacro-Averaged Metrics:")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall: {metrics['recall_macro']:.4f}")
    print(f"  F1-Score: {metrics['f1_macro']:.4f}")
    
    print(f"\nWeighted-Averaged Metrics:")
    print(f"  Precision: {metrics['precision_weighted']:.4f}")
    print(f"  Recall: {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score: {metrics['f1_weighted']:.4f}")
    
    print("\n" + "-"*80)
    print("Per-Class Metrics:")
    print("-"*80)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*80)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20} {metrics['precision'][i]:<12.4f} "
              f"{metrics['recall'][i]:<12.4f} {metrics['f1'][i]:<12.4f} "
              f"{int(metrics['support'][i]):<10}")
    
    print("\n" + "="*80)
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print(classification_report(
        metrics['labels'],
        metrics['predictions'],
        target_names=class_names,
        digits=4
    ))


def main():
    """
    Main evaluation function.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "="*60)
    print("Loading validation data...")
    print("="*60)
    _, val_loader, _, _, _, _ = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE
    )
    
    # Create model
    print("\n" + "="*60)
    print("Loading model...")
    print("="*60)
    model = create_model(num_classes=NUM_CLASSES, pretrained=False, device=device)
    
    # Load trained weights
    if os.path.exists(MODEL_SAVE_PATH):
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {MODEL_SAVE_PATH}")
        if 'val_acc' in checkpoint:
            print(f"Model validation accuracy: {checkpoint['val_acc']:.2f}%")
    else:
        print(f"Warning: Model file not found at {MODEL_SAVE_PATH}")
        print("Using untrained model for evaluation.")
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluating model...")
    print("="*60)
    metrics = evaluate_model(model, val_loader, device, CLASS_NAMES)
    
    # Print metrics
    print_metrics(metrics, CLASS_NAMES)
    
    # Plot confusion matrix
    print("\n" + "="*60)
    print("Generating confusion matrix...")
    print("="*60)
    plot_confusion_matrix(metrics['confusion_matrix'], CLASS_NAMES, CONFUSION_MATRIX_PATH)
    
    # Save confusion matrix data for use in app
    np.savez(
        CONFUSION_MATRIX_DATA_PATH,
        confusion_matrix=metrics['confusion_matrix'],
        class_names=np.array(CLASS_NAMES, dtype=object)
    )
    print(f"Confusion matrix data saved to {CONFUSION_MATRIX_DATA_PATH}")
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)


if __name__ == "__main__":
    main()

