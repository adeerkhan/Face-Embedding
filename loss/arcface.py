# loss/arcface.py
#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LossFunction(nn.Module):
    """
    ArcFace Loss Function.
    Enhances the discriminative power of softmax loss by adding an angular margin.
    """
    def __init__(self, nOut=1024, nClasses=1230, scale=30.0, margin=0.1, **kwargs):
        super(LossFunction, self).__init__()
        
        self.in_features = nOut
        self.out_features = nClasses
        self.s = scale
        self.m = margin
        self.weight = nn.Parameter(torch.FloatTensor(nClasses, nOut))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.criterion = torch.nn.CrossEntropyLoss()
        
        print(f'Initialized ArcFace Loss with scale={scale}, margin={margin}')
        
    def forward(self, input, label):
        """
        Forward pass for the ArcFace loss.
        
        Args:
            input (Tensor): Embedding vectors of shape [batch_size, nOut]
            label (Tensor): Ground truth labels of shape [batch_size]
        
        Returns:
            Tensor: Computed ArcFace loss
        """
        # Normalize input and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # Shape: [batch_size, nClasses]
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + 1e-12)        # Added epsilon for numerical stability
        phi = cosine * self.cos_m - sine * self.sin_m                # Shape: [batch_size, nClasses]

        # If cosine > theta, use phi; else, use cosine - mm
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot encoding for labels
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Add the margin to the target logits
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        # Compute cross-entropy loss
        loss = self.criterion(output, label)

        return loss
