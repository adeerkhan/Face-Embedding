#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
import math

class LossFunction(nn.Module):
	def __init__(self, nOut, nClasses, scale, margin, **kwargs):
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

		self.criterion  = torch.nn.CrossEntropyLoss()
		
		print(f'Initialised ArcFace Loss with scale={scale}, margin={margin}')
		
	def forward(self, input, label):
		cosine = F.linear(F.normalize(input), F.normalize(self.weight))
		sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
		phi = cosine * self.cos_m - sine * self.sin_m
		# if self.easy_margin:
		#     phi = torch.where(cosine > 0, phi, cosine)
		# else:
		#     phi = torch.where(cosine > self.th, phi, cosine - self.mm)
		phi = torch.where(cosine > self.th, phi, cosine - self.mm)

		one_hot = torch.zeros(cosine.size(), device=input.device)
		one_hot.scatter_(1, label.view(-1, 1).long(), 1)
		output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
		output *= self.s

		nloss   = self.criterion(output, label)

		return nloss