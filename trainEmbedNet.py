#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import os
import argparse
import pdb
import glob
import datetime
import numpy
import logging
from EmbedNet import *
from DatasetLoader import get_data_loader, meta_loader
from sklearn import metrics
import torchvision.transforms as transforms

from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Parse arguments
# ## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description="Face Recognition Training")

## Data loader
parser.add_argument('--batch_size', type=int, default=100, help='Batch size, defined as the number of classes per batch')
parser.add_argument('--max_img_per_cls', type=int, default=500, help='Maximum number of images per class per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=5, help='Number of data loader threads')

## Training details
parser.add_argument('--test_interval', type=int, default=5, help='Test and save every [test_interval] epochs')
# We'll set max_epoch dynamically based on elp_epochs and efft_epochs
parser.add_argument('--max_epoch', type=int, default=50, help='Maximum number of epochs')
parser.add_argument('--trainfunc', type=str, default="gce", help='Loss function to use (gce for linear probing)')

## Optimizer
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer')
parser.add_argument('--scheduler', type=str, default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate')
parser.add_argument("--lr_decay", type=float, default=0.90, help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay in the optimizer')

## Loss functions
parser.add_argument('--margin', type=float, default=0.05, help='Loss margin, only for some loss functions')
parser.add_argument('--scale', type=float, default=15, help='Loss scale, only for some loss functions')
parser.add_argument('--nPerClass', type=int, default=1, help='Number of images per class per batch, only for metric learning based losses')
parser.add_argument('--nClasses', type=int, default=1230, help='Number of classes in the softmax layer (Train2 has 1230 classes)')

## Load and save
parser.add_argument('--initial_model', type=str, default="", help='Initial model weights trained on Train1')
parser.add_argument('--save_path', type=str, default="exps/exp1", help='Path for model and logs')

## Training and evaluation data
parser.add_argument('--train_path', type=str, default="data/train2", help='Absolute path to the train set (Train2)')
parser.add_argument('--train_ext', type=str, default="jpg", help='Training files extension')
parser.add_argument('--test_path', type=str, default="data/val", help='Absolute path to the test set')
parser.add_argument('--test_list', type=str, default="data/val_pairs.csv", help='Evaluation list')

## Model definition
parser.add_argument('--model', type=str, default="GhostFaceNetsV2", help='Name of model definition')
parser.add_argument('--nOut', type=int, default=512, help='Embedding size in the last FC layer')

parser.add_argument('--width',         type=int,   default=1)
parser.add_argument('--dropout',       type=float, default=0.0)
parser.add_argument('--image_size',    type=int,   default=256)

## For test only
parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')
parser.add_argument('--output', type=str, default="", help='Save a log of output to this file name')

## Training
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')
parser.add_argument('--gpu', type=int, default=0, help='GPU index')

## TURN algorithm parameters
parser.add_argument('--elp_epochs', type=int, default=5, help='Number of epochs for linear probing')
parser.add_argument('--efft_epochs', type=int, default=10, help='Number of epochs for full fine-tuning')

args = parser.parse_args()

## ===== ===== ===== ===== ===== ===== ===== =====
## Script to compute EER
## ===== ===== ===== ===== ===== ===== ===== =====

def compute_eer(all_labels, all_scores):

    # Compute ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)

    # Calculate False Negative Rate (FNR)
    fnr = 1 - tpr

    # Calculate EER where FNR and FPR are closest
    eer_threshold = thresholds[numpy.nanargmin(numpy.absolute(fnr - fpr))]
    EER = fpr[numpy.nanargmin(numpy.absolute(fnr - fpr))]

    return EER

## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu)

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.save_path + "/scores.txt", mode="a+"),
        ],
        level=logging.DEBUG,
        format='[%(levelname)s] :: %(asctime)s :: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ## Input transformations for training (you can change if you like)
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(args.image_size),
            transforms.RandomCrop([args.image_size - 32, args.image_size - 32]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ## Input transformations for evaluation (you can change if you like)
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(args.image_size),
            transforms.CenterCrop([args.image_size - 32, args.image_size - 32]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ## Load models
    model = EmbedNet(**vars(args)).cuda()
    trainer = ModelTrainer(model, **vars(args))

    ## Load initial model trained on Train1
    if args.initial_model != "":
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))
    else:
        print("Please provide the initial_model trained on Train1.")
        exit()

    ## Print total number of model parameters
    pytorch_total_params = sum(p.numel() for p in model.__E__.parameters())
    print('Total model parameters: {:,}'.format(pytorch_total_params))

    ## Evaluation code
    if args.eval == True:

        sc, lab, trials = trainer.evaluateFromList(transform=test_transform, **vars(args))

        EER = compute_eer(lab, sc)

        print('EER {:.2f}%'.format(EER * 100))

        if args.output != '':
            with open(args.output, 'w') as f:
                for ii in range(len(sc)):
                    f.write('{:4f},{:d},{}\n'.format(sc[ii], lab[ii], trials[ii]))

        quit()

    ## Log arguments
    logger.info('{}'.format(args))

    #########################
    # Step 1: Linear Probing
    #########################

    # Freeze the feature extractor
    trainer.__model__.freeze_feature_extractor()

    # Use GCE loss function
    args.trainfunc = 'gce'  # Ensure that GCE loss is used

    # Modify optimizer to update only classifier parameters
    trainer.__optimizer__ = importlib.import_module('optimizer.'+args.optimizer).__getattribute__('Optimizer')(
        filter(lambda p: p.requires_grad, trainer.__model__.parameters()), **vars(args)
    )

    # Modify scheduler to use the new optimizer
    trainer.__scheduler__, trainer.lr_step = importlib.import_module('scheduler.'+args.scheduler).__getattribute__('Scheduler')(
        trainer.__optimizer__, **vars(args)
    )

    # Set max_epoch to elp_epochs
    args.max_epoch = args.elp_epochs

    # Initialise trainer and data loader for linear probing
    trainLoader = get_data_loader(transform=train_transform, **vars(args))

    print("Starting Linear Probing for {} epochs...".format(args.elp_epochs))

    for ep in range(1, args.max_epoch + 1):

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        logger.info("Linear Probing Epoch {:04d} started with LR {:.5f} ".format(ep, max(clr)))
        loss = trainer.train_network(trainLoader)
        logger.info("Linear Probing Epoch {:04d} completed with TLOSS {:.5f}".format(ep, loss))

        if trainer.lr_step == 'epoch':
            trainer.__scheduler__.step()

        # No evaluation during linear probing

    #########################
    # Step 2: Cleansing and Fine-Tuning
    #########################

    # Compute per-sample losses
    print("Computing per-sample losses...")
    per_sample_losses, per_sample_labels = trainer.compute_per_sample_losses(trainLoader)

    # Fit GMM to per-sample losses
    print("Fitting GMM to per-sample losses...")
    losses = per_sample_losses.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type='full').fit(losses)
    probs = gmm.predict_proba(losses)

    # Identify the component with the lower mean loss (assumed to be clean)
    clean_component = numpy.argmin(gmm.means_)
    threshold = 0.9  # Adjust this threshold as needed
    clean_indices = numpy.where(probs[:, clean_component] > threshold)[0]

    print("Selected {} clean samples out of {}".format(len(clean_indices), len(losses)))

    # Create cleansed dataset and data loader
    print("Creating cleansed dataset...")
    cleansed_dataset = meta_loader(
        train_path=args.train_path,
        train_ext=args.train_ext,
        transform=train_transform,
        selected_indices=clean_indices
    )

    cleansed_loader = torch.utils.data.DataLoader(
        cleansed_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        worker_init_fn=worker_init_fn
    )

    # Unfreeze the feature extractor
    trainer.__model__.unfreeze_feature_extractor()

    # Reinitialize the optimizer to update all parameters
    trainer.__optimizer__ = importlib.import_module('optimizer.'+args.optimizer).__getattribute__('Optimizer')(
        trainer.__model__.parameters(), **vars(args)
    )

    # Reinitialize the scheduler
    trainer.__scheduler__, trainer.lr_step = importlib.import_module('scheduler.'+args.scheduler).__getattribute__('Scheduler')(
        trainer.__optimizer__, **vars(args)
    )

    # Set max_epoch to efft_epochs
    args.max_epoch = args.efft_epochs

    # Use standard loss function for fine-tuning
    args.trainfunc = 'softmax'  # Or 'arcface', depending on your choice
    trainer.__model__.__C__ = importlib.import_module('loss.' + args.trainfunc).__getattribute__('LossFunction')(
        nOut=args.nOut, nClasses=args.nClasses, **vars(args)
    )

    print("Starting Fine-Tuning for {} epochs...".format(args.efft_epochs))

    for ep in range(1, args.max_epoch + 1):

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        logger.info("Fine-Tuning Epoch {:04d} started with LR {:.5f} ".format(ep, max(clr)))
        loss = trainer.train_network(cleansed_loader)
        logger.info("Fine-Tuning Epoch {:04d} completed with TLOSS {:.5f}".format(ep, loss))

        if ep % args.test_interval == 0:

            sc, lab, trials = trainer.evaluateFromList(transform=test_transform, **vars(args))
            EER = compute_eer(lab, sc)

            logger.info("Fine-Tuning Epoch {:04d}, Val EER {:.2f}%".format(ep, EER * 100))
            trainer.saveParameters(args.save_path + "/epoch{:04d}.model".format(ep))

        if trainer.lr_step == 'epoch':
            trainer.__scheduler__.step()

## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====

def main():

    if not (os.path.exists(args.save_path)):
        os.makedirs(args.save_path)

    main_worker(args)

if __name__ == '__main__':
    main()
