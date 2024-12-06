#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import glob
import shutil
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class meta_loader(Dataset):
    def __init__(self, train_path, train_ext, transform, selected_indices=None):
        files = glob.glob(f'{train_path}/*/*.{train_ext}')
        self.transform = transform
        self.data_list = []
        self.data_label = []

        # Map class name to class number
        class_mapping = {key: idx for idx, key in enumerate(sorted(set([x.split('/')[-2] for x in files])))}

        for file in files:
            class_name = file.split('/')[-2]
            self.data_list.append(file)
            self.data_label.append(class_mapping[class_name])

        # Filter samples if selected_indices is provided
        if selected_indices is not None:
            self.data_list = [self.data_list[i] for i in selected_indices]
            self.data_label = [self.data_label[i] for i in selected_indices]

        print(f"{len(self.data_list)} files from {len(set(self.data_label))} classes loaded.")

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.data_list[idx]))
        label = self.data_label[idx]
        return img, label

    def __len__(self):
        return len(self.data_list)

class TURNLoader:
    def __init__(self, train_path, train_ext, transform, batch_size, max_img_per_cls, nDataLoaderThread, nPerClass):
        self.train_path = train_path
        self.train_ext = train_ext
        self.transform = transform
        self.batch_size = batch_size
        self.max_img_per_cls = max_img_per_cls
        self.nDataLoaderThread = nDataLoaderThread
        self.nPerClass = nPerClass

        self.meta_dataset = meta_loader(train_path, train_ext, transform)

    def get_linear_probing_loader(self):
        return DataLoader(
            self.meta_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.nDataLoaderThread,
            worker_init_fn=worker_init_fn,
            drop_last=True
        )

    def save_clean_samples(self, clean_indices, save_dir):
        """Save clean samples to structured directories."""
        os.makedirs(save_dir, exist_ok=True)
        for idx in clean_indices:
            file_path = self.meta_dataset.data_list[idx]
            label = self.meta_dataset.data_label[idx]
            label_dir = os.path.join(save_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            shutil.copy(file_path, label_dir)
        print(f"Saved {len(clean_indices)} clean samples to {save_dir}")

    def get_cleansed_loader(self, clean_indices):
        cleansed_dataset = meta_loader(
            train_path=self.train_path,
            train_ext=self.train_ext,
            transform=self.transform,
            selected_indices=clean_indices
        )
        return DataLoader(
            cleansed_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.nDataLoaderThread,
            worker_init_fn=worker_init_fn,
            drop_last=True
        )

    def get_validation_loader(self, val_path, val_list):
        files1, files2, labels = [], [], []
        with open(val_list, 'r') as f:
            for line in f:
                label, file1, file2 = line.strip().split(',')
                labels.append(int(label))
                files1.append(os.path.join(val_path, file1))
                files2.append(os.path.join(val_path, file2))

        class ValDataset(Dataset):
            def __init__(self, files1, files2, labels, transform):
                self.files1 = files1
                self.files2 = files2
                self.labels = labels
                self.transform = transform

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                img1 = self.transform(Image.open(self.files1[idx]))
                img2 = self.transform(Image.open(self.files2[idx]))
                label = self.labels[idx]
                return img1, img2, label

        val_dataset = ValDataset(files1, files2, labels, self.transform)

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.nDataLoaderThread,
            worker_init_fn=worker_init_fn,
            drop_last=False
        )