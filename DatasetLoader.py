#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy
import random
import math
import pdb
import glob
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def round_down(num, divisor):
    return num - (num%divisor)

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)

class meta_loader(Dataset):
    def __init__(self, train_path, train_ext, transform):
        
        ## Read Training Files
        files = glob.glob('%s/*/*.%s'%(train_path,train_ext))

        ## Make a mapping from Class Name to Class Number
        dictkeys = list(set([x.split('/')[-2] for x in files]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

        self.transform  = transform

        self.label_dict = {}
        self.data_list  = []
        self.data_label = []
        
        for lidx, file in enumerate(files):
            speaker_name = file.split('/')[-2]
            speaker_label = dictkeys[speaker_name];

            if not (speaker_label in self.label_dict):
                self.label_dict[speaker_label] = [];

            self.label_dict[speaker_label].append(lidx);
            
            self.data_label.append(speaker_label)
            self.data_list.append(file)

        print('{:d} files from {:d} classes found.'.format(len(self.data_list),len(self.label_dict)))

    def __getitem__(self, indices):

        feat = []
        for index in indices:
            feat.append(self.transform(Image.open(self.data_list[index])));
        feat = numpy.stack(feat, axis=0)

        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):

        return len(self.data_list)

class test_dataset_loader(Dataset):
    def __init__(self, test_list, test_path, transform, **kwargs):
        self.test_path  = test_path
        self.data_list  = test_list
        self.transform  = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.test_path, self.data_list[index]))
        return self.transform(img), self.data_list[index]

    def __len__(self):
        return len(self.data_list)


class meta_sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerClass, max_img_per_cls, batch_size):

        self.label_dict         = data_source.label_dict
        self.nPerClass          = nPerClass
        self.max_img_per_cls    = max_img_per_cls;
        self.batch_size         = batch_size;

        self.num_iters          = 0
        
    def __iter__(self):
        
        ## Get a list of identities
        dictkeys = list(self.label_dict.keys());
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        ## Data for each class
        for findex, key in enumerate(dictkeys):
            data    = self.label_dict[key]
            numSeg  = round_down(min(len(data),self.max_img_per_cls),self.nPerClass)
            
            rp      = lol(numpy.random.permutation(len(data))[:numSeg],self.nPerClass)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Data in random order
        mixid           = numpy.random.permutation(len(flattened_label))
        mixlabel        = []
        mixmap          = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        batch_indices = [flattened_list[i] for i in mixmap]

        self.num_iters = len(batch_indices)

        return iter(batch_indices)
    
    def __len__(self):
        return self.num_iters

def get_data_loader(batch_size, max_img_per_cls, nDataLoaderThread, nPerClass, train_path, train_ext, transform, **kwargs):
    
    train_dataset = meta_loader(train_path, train_ext, transform)

    train_sampler = meta_sampler(train_dataset, nPerClass, max_img_per_cls, batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    
    return train_loader


