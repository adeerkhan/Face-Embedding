#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):

	sche_fn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-5)

	lr_step = 'epoch'

	print('Initialised cosine LR scheduler')

	return sche_fn, lr_step