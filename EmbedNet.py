#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys
import time, importlib
from DatasetLoader import test_dataset_loader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

class EmbedNet(nn.Module):

    def __init__(self, model, optimizer, trainfunc, nPerClass, **kwargs):
        super(EmbedNet, self).__init__()

        self.trainfunc = trainfunc  # Store trainfunc as an attribute

        ## __E__ is the embedding model
        EmbedNetModel = importlib.import_module('models.' + model).__getattribute__('MainModel')
        self.__E__ = EmbedNetModel(**kwargs)

        # Remove the classifier layers if present to ensure embeddings are correctly output
        if hasattr(self.__E__, 'classifier'):
            if hasattr(self.__E__.classifier, 'linear'):
                self.__E__.classifier.linear = nn.Identity()
            if hasattr(self.__E__.classifier, 'conv_dw'):
                self.__E__.classifier.conv_dw = nn.Identity()

        ## __C__ is the classifier plus the loss function
        LossFunctionModule = importlib.import_module('loss.' + trainfunc)
        LossFunction = LossFunctionModule.__getattribute__('LossFunction')

        if trainfunc == 'triplet':
            # Extract triplet-specific arguments
            triplet_kwargs = {
                'hard_rank': kwargs.get('hard_rank', 0),
                'hard_prob': kwargs.get('hard_prob', 0),
                'margin': kwargs.get('margin', 0.05),
                # Add other triplet-specific arguments here if necessary
            }
            print(f"Initializing TripletLoss with args: {triplet_kwargs}")
            self.__C__ = LossFunction(**triplet_kwargs)

        elif trainfunc == 'softmax':
            # Extract softmax-specific arguments
            required_args = ['nOut', 'nClasses']
            for arg in required_args:
                if kwargs.get(arg) is None:
                    raise ValueError(f"Missing required argument '{arg}' for Softmax loss.")
            softmax_kwargs = {
                'nOut': kwargs['nOut'],
                'nClasses': kwargs['nClasses'],
            }
            print(f"Initializing SoftmaxLoss with args: {softmax_kwargs}")
            self.__C__ = LossFunction(**softmax_kwargs)

        elif trainfunc == 'arcface':
            # Extract arcface-specific arguments
            required_args = ['nOut', 'nClasses', 'scale', 'margin']
            for arg in required_args:
                if kwargs.get(arg) is None:
                    raise ValueError(f"Missing required argument '{arg}' for ArcFace loss.")
            arcface_kwargs = {
                'nOut': kwargs['nOut'],
                'nClasses': kwargs['nClasses'],
                'scale': kwargs['scale'],
                'margin': kwargs['margin'],
            }
            print(f"Initializing ArcFace Loss with args: {arcface_kwargs}")
            self.__C__ = LossFunction(**arcface_kwargs)

        else:
            raise ValueError(f"Unsupported train function: {trainfunc}")

        ## Number of examples per identity per batch
        self.nPerClass = nPerClass

    def forward(self, data, label=None):

        data = data.reshape(-1, data.size()[-3], data.size()[-2], data.size()[-1])
        outp = self.__E__.forward(data)

        if label is None:
            return outp

        else:
            if self.trainfunc in ['triplet']:
                # Reshape only for triplet loss
                outp = outp.reshape(self.nPerClass, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)
            # For softmax and arcface, no reshaping is needed

            nloss = self.__C__.forward(outp, label)
            return nloss

class ModelTrainer(object):

    def __init__(self, embed_model, optimizer, scheduler, mixedprec, **kwargs):

        self.__model__  = embed_model

        ## Optimizer (e.g. Adam or SGD)
        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        ## Learning rate scheduler
        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        ## For mixed precision training
        self.scaler = GradScaler(device='cuda')  # Updated to address FutureWarning
        self.mixedprec = mixedprec

        assert self.lr_step in ['epoch', 'iteration']

    # ====== Train network ======

    def train_network(self, loader):

        self.__model__.train()

        stepsize = loader.batch_size

        counter = 0
        index   = 0
        loss    = 0

        with tqdm(loader, unit="batch") as tepoch:
        
            for data, label in tepoch:

                tepoch.total = tepoch.__len__()

                data    = data.transpose(1,0)

                ## Reset gradients
                self.__model__.zero_grad()

                ## Forward and backward passes
                if self.mixedprec:
                    with autocast():
                        nloss = self.__model__(data.cuda(), label.cuda())
                    self.scaler.scale(nloss).backward()
                    self.scaler.step(self.__optimizer__)
                    self.scaler.update()       
                else:
                    nloss = self.__model__(data.cuda(), label.cuda())
                    nloss.backward()
                    self.__optimizer__.step()

                loss    += nloss.detach().cpu().item()
                counter += 1
                index   += stepsize

                # Print statistics to progress bar
                tepoch.set_postfix(loss=loss/counter)

                if self.lr_step == 'iteration': self.__scheduler__.step()

            if self.lr_step == 'epoch': self.__scheduler__.step()
        
        return (loss/counter)

    # ====== Evaluate from list ======

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, transform, print_interval=100, num_eval=10, **kwargs):
        
        self.__model__.eval()
        
        feats       = {}

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = sum([x.strip().split(',')[-2:] for x in lines],[])
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, transform=transform, num_eval=num_eval, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
        )

        print('Generating embeddings')

        ## Extract features for every image
        for data in tqdm(test_loader):
            inp1                = data[0][0].cuda()
            ref_feat            = self.__model__(inp1).detach().cpu()
            feats[data[1][0]]   = ref_feat

        all_scores = []
        all_labels = []
        all_trials = []

        print('Computing similarities')

        ## Read files and compute all scores
        for line in tqdm(lines):

            data = line.strip().split(',')

            ref_feat = feats[data[1]]
            com_feat = feats[data[2]]

            score = F.cosine_similarity(ref_feat, com_feat)

            all_scores.append(score.item())  
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + "," + data[2])

        return (all_scores, all_labels, all_trials)


    # ====== Save parameters ======

    def saveParameters(self, path):
        
        torch.save(self.__model__.state_dict(), path)

    # ====== Load parameters ======

    def loadParameters(self, path):
        self_state = self.__model__.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            if name not in self_state:
                print(f"{name} is not in the model.")
                continue

            if self_state[name].size() != param.size():
                print(f"Skipping {name}: model size {self_state[name].size()} vs loaded size {param.size()}")
                continue

            self_state[name].copy_(param)

