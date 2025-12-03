"""
Advanced loss functions for improved training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    
    Paper: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=None, gamma=2.0, weight=None, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for rare class (can be tensor of size n_classes)
            gamma: Focusing parameter (higher = more focus on hard examples)
            weight: Manual class weights (if None, uses alpha)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits), shape (N, C)
            targets: Ground truth labels, shape (N,)
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    
    Helps prevent overconfidence and improves generalization.
    """
    def __init__(self, smoothing=0.1, weight=None):
        """
        Args:
            smoothing: Smoothing factor (0 = no smoothing, 1 = uniform distribution)
            weight: Class weights
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits), shape (N, C)
            targets: Ground truth labels, shape (N,)
        """
        log_probs = F.log_softmax(inputs, dim=1)
        num_classes = inputs.size(1)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            if self.weight is not None:
                true_dist = true_dist * self.weight.unsqueeze(0)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

