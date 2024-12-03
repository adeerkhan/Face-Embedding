# loss/gce.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunction(nn.Module):
    def __init__(self, nOut, q=0.7, **kwargs):
        super(LossFunction, self).__init__()
        self.q = q
        self.nOut = nOut
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        # x: [batch_size, num_classes]
        # label: [batch_size]
        probs = F.softmax(x, dim=1)
        probs = probs.gather(1, label.view(-1, 1)).squeeze()
        loss = (1 - probs ** self.q) / self.q
        return loss.mean()
