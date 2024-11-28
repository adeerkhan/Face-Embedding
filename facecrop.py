#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
import time
import os
import argparse
import pdb
import glob
import datetime
import numpy as np
import logging
from EmbedNet import *
from DatasetLoader import get_data_loader
from sklearn import metrics
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from PIL import Image

temp_path = 'data/vgg/train'
crop_path = 'data/vgg_crop/train'

# initialise face detector on GPU
mtcnn = MTCNN(device='cuda:0')

## Read Training Files
files = glob.glob(temp_path+'/*/*.jpg')
files.sort()

for fname in tqdm(files):

  # read image and convert to RGB
  image = Image.open(fname)

  # check if the image is RGB
  if image.mode != 'RGB':
    print('[INFO] Skipping Non-RGB image')
    continue;

  # run forward pass on face detector
  bboxes = mtcnn.detect(image)

  ## this removes all images with no face detection or two or more face detections
  if bboxes[0] is not None and len(bboxes[0]) == 1:

    bbox = bboxes[0][0]

    # get the confidence of face detection
    conf = bboxes[1][0]

    # find the center and the box size of the bounding box
    sx = (bbox[0] + bbox[2]) / 2
    sy = (bbox[1] + bbox[3]) / 2
    ss = int(max((bbox[3]-bbox[1]),(bbox[2]-bbox[0]))/1.5)

    # crop the face area using pillow image crop
    face = image.crop((int(sx-ss), int(sy-ss), int(sx+ss), int(sy+ss)))

    # check that the cropped image is (1) square, (2) at least 50 pixels wide, (3) confidence of at least 0.9
    if face.size[0] == face.size[1] and face.size[0] > 50 and conf > 0.9:

      # specify the path to write to
      outname = fname.replace(temp_path, crop_path).replace('.png','.jpg')
      os.makedirs(os.path.dirname(outname),exist_ok=True)

      # resize image to (256,256)
      face = face.resize((256, 256))

      # save the image to 'outname' with pillow image save
      face = face.save(outname)

    else:
      # if it does not satify the above 3 criteria
      print('[INFO] Invalid image {}'.format(fname))