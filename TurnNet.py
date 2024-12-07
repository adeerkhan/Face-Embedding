import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import importlib

class TURNNet(nn.Module):
    def __init__(self, model, trainfunc, **kwargs):
        super(TURNNet, self).__init__()

        # Embedding model
        Model = importlib.import_module('models.' + model).__getattribute__('MainModel')
        self.__E__ = Model(**kwargs)

        # Loss function
        if isinstance(trainfunc, nn.Module):
            self.__C__ = trainfunc
        else:
            LossFunction = importlib.import_module('loss.' + trainfunc).__getattribute__('LossFunction')
            self.__C__ = LossFunction(**kwargs)

    def forward(self, data, labels=None):
        features = self.__E__.forward(data)
        if labels is None:
            return features
        return self.__C__.forward(features, labels)

class TURNTrainer:
    def __init__(self, turn_model, optimizer, scheduler, mixedprec=False, test_interval=1, max_epoch=1, **kwargs):
        self.__model__ = turn_model
        Optimizer = importlib.import_module('optimizer.' + optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)
        Scheduler = importlib.import_module('scheduler.' + scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, test_interval, max_epoch, **kwargs)
        self.mixedprec = mixedprec
        self.scaler = GradScaler()

    def train_network(self, loader, loss_function):
        self.__model__.train()
        total_loss, counter = 0, 0

        for data, labels in tqdm(loader, desc="Training"):
            self.__model__.zero_grad()

            if self.mixedprec:
                with autocast():
                    loss = loss_function(self.__model__(data.cuda()), labels.cuda())
                self.scaler.scale(loss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                loss = loss_function(self.__model__(data.cuda()), labels.cuda())
                loss.backward()
                self.__optimizer__.step()

            total_loss += loss.item()
            counter += 1

        if self.lr_step == 'epoch':
            self.__scheduler__.step()

        return total_loss / counter

    def compute_per_sample_losses(self, loader, loss_function):
        self.__model__.eval()
        per_sample_losses = []

        with torch.no_grad():
            for data, labels in tqdm(loader, desc="Computing Losses"):
                logits = self.__model__(data.cuda())
                losses = loss_function(logits, labels.cuda(), reduction='none')
                per_sample_losses.extend(losses.cpu().numpy())

        return per_sample_losses

    def validate(self, loader, loss_function):
        self.__model__.eval()
        total_loss, total_correct, total_samples = 0, 0, 0

        # Use a binary loss function for validation
        bce_loss_func = torch.nn.BCEWithLogitsLoss()

        with torch.no_grad():
            for img1, img2, labels in tqdm(loader, desc="Validating"):
                # Pass both images through the model and compute the similarity
                feat1 = self.__model__(img1.cuda())
                feat2 = self.__model__(img2.cuda())
                logits = F.cosine_similarity(feat1, feat2)

                # Compute binary loss using BCEWithLogitsLoss
                # Reshape to (B, 1) for BCE input, and labels to float
                loss = bce_loss_func(logits.unsqueeze(1), labels.float().cuda().unsqueeze(1))
                total_loss += loss.item()

                # Predictions: threshold the similarity at 0.5
                predictions = (logits > 0.5).long()
                total_correct += (predictions == labels.cuda()).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * total_correct / total_samples
        return avg_loss, accuracy

    def saveParameters(self, path):
        torch.save(self.__model__.state_dict(), path)

    def loadParameters(self, path):
        print(f"Loading model from {path}")
        checkpoint = torch.load(path)
        model_dict = self.__model__.state_dict()

        # Filter out unnecessary keys and ensure shapes match
        filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
        mismatched_keys = [k for k in checkpoint.keys() if k not in filtered_dict]

        if mismatched_keys:
            print(f"Warning: The following keys were skipped due to mismatched shapes or being unexpected: {mismatched_keys}")

        model_dict.update(filtered_dict)
        self.__model__.load_state_dict(model_dict)
