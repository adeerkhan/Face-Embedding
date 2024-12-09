# loss/softmax.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunction(nn.Module):
    """
    Softmax Loss Function.
    """
    def __init__(self, nOut, nClasses):
        super(LossFunction, self).__init__()
        # Ensure that the input features match the embedding size (nOut=1024)
        self.fc = nn.Linear(nOut, nClasses, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, embeddings, labels):
        """
        Forward pass for the softmax loss.
        
        Args:
            embeddings (Tensor): Embedding vectors of shape [batch_size, nOut]
            labels (Tensor): Ground truth labels of shape [batch_size]
        
        Returns:
            Tensor: Computed cross-entropy loss
        """
        logits = self.fc(embeddings)  # Shape: [batch_size, nClasses]
        loss = F.cross_entropy(logits, labels)
        return loss
