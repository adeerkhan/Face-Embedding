#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys
import time, importlib
from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

class EmbedNet(nn.Module):

    def __init__(self, model, optimizer, trainfunc, nPerClass, **kwargs):
        super(EmbedNet, self).__init__();

        ## __E__ is the embedding model
        EmbedNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__E__ = EmbedNetModel(**kwargs);

        ## __C__ is the classifier plus the loss function
        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__C__ = LossFunction(**kwargs);

        ## Number of examples per identity per batch
        self.nPerClass = nPerClass

    def forward(self, data, label=None):

        data    = data.reshape(-1,data.size()[-3],data.size()[-2],data.size()[-1])
        outp    = self.__E__.forward(data)

        if label == None:
            return outp

        else:
            outp    = outp.reshape(self.nPerClass,-1,outp.size()[-1]).transpose(1,0).squeeze(1)
            nloss = self.__C__.forward(outp,label)
            return nloss


class ModelTrainer(object):

    def __init__(self, embed_model, optimizer, scheduler, **kwargs):

        self.__model__  = embed_model

        ## Optimizer (e.g. Adam or SGD)
        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)
        
        ## Learning rate scheduler
        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        assert self.lr_step in ['epoch', 'iteration']

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader):

        self.__model__.train()

        stepsize = loader.batch_size

        counter = 0
        loss = 0

        with tqdm(loader, unit="batch") as tepoch:

            for data, label in tepoch:

                tepoch.total = tepoch.__len__()

                data = data.transpose(1, 0)

                ## Reset gradients
                self.__optimizer__.zero_grad()

                ## Forward pass and compute loss
                nloss = self.__model__(data.cuda(), label.cuda())

                ## Backward pass
                nloss.backward()

                ## Optimizer step
                self.__optimizer__.step()

                ## Keep cumulative statistics
                loss += nloss.item()
                counter += 1

                # Print statistics to progress bar
                tepoch.set_postfix(loss=loss / counter)

                if self.lr_step == 'iteration':
                    self.__scheduler__.step()

            if self.lr_step == 'epoch':
                self.__scheduler__.step()

        return loss / counter


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, transform, print_interval=100, num_eval=10, **kwargs):
        self.__model__.eval()
        feats = {}

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = sum([x.strip().split(',')[-2:] for x in lines], [])
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
            inp1 = data[0][0].cuda()  # Shape: [num_eval, C, H, W]
            with torch.no_grad():
                ref_feat = self.__model__(inp1).detach().cpu()  # Shape: [num_eval, embedding_size]
            ref_feat = torch.mean(ref_feat, dim=0)  # Shape: [embedding_size]
            feats[data[1][0]] = ref_feat

        all_scores = []
        all_labels = []
        all_trials = []

        print('Computing similarities')

        ## Read files and compute all scores
        for line in tqdm(lines):
            data = line.strip().split(',')

            ref_feat = feats[data[1]]  # Shape: [embedding_size]
            com_feat = feats[data[2]]  # Shape: [embedding_size]

            ## Find cosine similarity score
            score = F.cosine_similarity(ref_feat, com_feat, dim=0)
            all_scores.append(score.item())
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + "," + data[2])

        return (all_scores, all_labels, all_trials)


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        
        torch.save(self.__model__.state_dict(), path);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.state_dict();
        loaded_state = torch.load(path);
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                if name not in self_state:
                    print("{} is not in the model.".format(origname));
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);

