"""
Model definition for Garbage Classification System
Uses MobileNetV3 for lightweight, real-time inference
"""

# Fix OpenMP library conflict on Windows
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, MODEL_NAME


class GarbageClassifier(nn.Module):
    """
    Garbage classification model using MobileNetV3 as backbone.
    Lightweight architecture suitable for real-time inference.
    """
    
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, dropout=0.2):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of output classes (default: 12)
            pretrained: Whether to use pretrained weights (default: True)
            dropout: Dropout probability for regularization (default: 0.2)
        """
        super(GarbageClassifier, self).__init__()
        
        # Load pretrained MobileNetV3 Small
        if MODEL_NAME == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            # Get the number of input features for the classifier
            in_features = self.backbone.classifier[-1].in_features
            # Replace the final classifier layer with dropout for regularization
            self.backbone.classifier = nn.Sequential(
                self.backbone.classifier[0],  # Keep the first layer
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        elif MODEL_NAME == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        elif MODEL_NAME == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            raise ValueError(f"Unsupported model: {MODEL_NAME}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        if MODEL_NAME.startswith('resnet'):
            return self.backbone(x)
        else:
            return self.backbone(x)
    
    def get_feature_extractor(self):
        """
        Get the feature extractor (backbone without final classifier).
        Useful for feature visualization or transfer learning.
        
        Returns:
            Feature extractor model
        """
        feature_extractor = nn.Sequential(*list(self.backbone.features))
        return feature_extractor


def create_model(num_classes=NUM_CLASSES, pretrained=True, device='cpu', dropout=0.2):
    """
    Factory function to create and initialize the model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        device: Device to load the model on ('cpu' or 'cuda')
        dropout: Dropout probability for regularization
        
    Returns:
        Initialized model
    """
    model = GarbageClassifier(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Summary:")
    print(f"Architecture: {MODEL_NAME}")
    print(f"Number of classes: {num_classes}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Dropout: {dropout}")
    print(f"Device: {device}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device=device)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output[0][:5]}...")  # Show first 5 logits
    print("\nModel test completed successfully!")

