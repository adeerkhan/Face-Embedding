#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunction(nn.Module):
    def __init__(self, num_classes, q=0.9):
        """
        Generalized Cross Entropy (GCE) Loss
        :param num_classes: Number of classes in the dataset
        :param q: Hyperparameter to adjust the robustness to noise (0 < q <= 1).
                  Smaller values of q reduce sensitivity to noisy labels.
        """
        super(LossFunction, self).__init__()
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels, reduction='mean'):
        """
        Compute the GCE loss.
        :param pred: Predicted logits (batch_size, num_classes)
        :param labels: Ground-truth labels (batch_size)
        :param reduction: Loss reduction method
        :return: GCE loss value
        """
        # Convert logits to probabilities using softmax
        pred = F.softmax(pred, dim=1)
        # Clamp predictions to avoid numerical instability
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        # Create one-hot encoded ground truth labels
        label_one_hot = F.one_hot(labels, self.num_classes).float()
        # Compute the GCE loss
        gce = (1.0 - torch.sum(label_one_hot * pred, dim=1).pow(self.q)) / self.q
        
        if reduction == 'mean':
            return gce.mean()
        elif reduction == 'none':
            return gce
        else:
            return gce.sum()