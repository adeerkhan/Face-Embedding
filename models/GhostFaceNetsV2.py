#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Import the custom GhostFaceNetsV2 model
import torch
import torch.nn as nn
from models.ghostfacenets import GhostFaceNetsV2, ClassifierModule

def MainModel(nOut=512, **kwargs):
    # Adjust parameters like image_size, width, dropout, etc., to match your setup
    feature_extractor = GhostFaceNetsV2(image_size=256, num_classes=0, width=1.0, dropout=0.2, nOut=nOut)
    classifier = ClassifierModule(feature_extractor.output_channel, nOut=nOut, dropout=0.2, image_size=256)
    return nn.Sequential(
        feature_extractor,
        classifier
    )