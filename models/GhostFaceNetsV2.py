#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Import the custom GhostFaceNetsV2 model
from models.ghostfacenets import GhostFaceNetsV2

def MainModel(nOut=256, **kwargs):
    # Adjust parameters like image_size, width, dropout, etc., to match your setup
    return GhostFaceNetsV2(image_size=256, num_classes=nOut, width=1.0, dropout=0.2)