# Face embedding trainer

This repository contains the framework for training deep embeddings for face recognition. The trainer is intended for the face recognition exercise of the EE488 Deep Learning for Visual Understanding course.

### Dependencies
```
pip install -r requirements.txt
```

### Training examples

- Softmax:
```
python ./trainEmbedNet.py --model ResNet18 --trainfunc softmax --save_path exps/exp1 --nClasses 2000 --batch_size 200 --gpu 8
```

GPU ID must be specified using `--gpu` flag.

Use `--mixedprec` flag to enable mixed precision training. This is recommended for Tesla V100, GeForce RTX 20 series or later models.

### Implemented loss functions
```
Softmax (softmax)
Triplet (triplet)
```

For softmax-based losses, `nPerClass` should be 1, and `nClasses` must be specified. For metric-based losses, `nPerClass` should be 2 or more. 

### Implemented models
```
ResNet18
```

### Adding new models and loss functions

You can add new models and loss functions to `models` and `loss` directories respectively. See the existing definitions for examples.

### Data

The test list should contain labels and image pairs, one line per pair, as follows. `1` is a target and `0` is an imposter.
```
1,id10001/00001.jpg,id10001/00002.jpg
0,id10001/00003.jpg,id10002/00001.jpg
```

The folders in the training set should contain images for each identity (i.e. `identity/image.jpg`).

The input transformations can be changed in the code.

### Inference

In order to save pairwise similarity scores to file, use `--output` flag.

## Train1
python ./trainEmbedNet.py \
  --gpu 0 \
  --train_path data/ee488_24_data/train1 \
  --test_path data/ee488_24_data/val \
  --test_list data/ee488_24_data/val_pairs.csv \
  --nClasses 2882 \
  --save_path ./exps/train1/exp03 \
  --optimizer adamw \
  --scheduler cosinelr \
  --batch_size 330 \
  --trainfunc arcface \
  --image_size 256 \
  --lr 0.0001 \
  --max_epoch 50 \
  --nOut 1024 \
  --margin 0.1 \
  --scale 30

## Better results
python ./trainEmbedNet.py \
  --gpu 0 \
  --train_path data/ee488_24_data/train1 \
  --test_path data/ee488_24_data/val \
  --test_list data/ee488_24_data/val_pairs.csv \
  --nClasses 2882 \
  --save_path ./exps/train1/exp05 \
  --optimizer adopt \
  --scheduler steplr \
  --batch_size 150 \
  --trainfunc arcface \
  --image_size 256 \
  --lr 0.001 \
  --lr_decay 0.9 \
  --max_epoch 100 \
  --nOut 1024 \
  --margin 0.1 \
  --scale 30
  --width 1
  --model GhostFaceNetsV2

## TURN Train1
python ./trainEmbedNet.py \
  --gpu 0 \
  --train_path data/ee488_24_data/train1 \
  --test_path data/ee488_24_data/val \
  --test_list data/ee488_24_data/val_pairs.csv \
  --nClasses 2882 \
  --save_path ./exps/train1/exp11 \
  --optimizer adopt \
  --scheduler steplr \
  --batch_size 150 \
  --trainfunc arcface \
  --image_size 256 \
  --lr 0.001 \
  --lr_decay 0.9 \
  --max_epoch 30 \
  --nOut 1024 \
  --margin 0.1 \
  --scale 30 \
  --width 1 \
  --model GhostFaceNetsV2

## New Try:
python ./trainEmbedNet.py \
  --gpu 0 \
  --train_path data/ee488_24_data/train1 \
  --test_path data/ee488_24_data/val \
  --test_list data/ee488_24_data/val_pairs.csv \
  --nClasses 2882 \
  --save_path ./exps/train1/exp09 \
  --optimizer adopt \
  --scheduler steplr \
  --batch_size 150 \
  --trainfunc softmax \
  --image_size 256 \
  --lr 0.001 \
  --lr_decay 0.9 \
  --max_epoch 50 \
  --margin 0.1 \
  --scale 30 \
  --nOut 1024 \
  --dropout 0.2 \
  --model GhostFaceNetsV2

## New TURN TRAIN2
python ./trainEmbedNet.py \
    --gpu 0 \
    --train_path data/ee488_24_data/train2 \
    --test_path data/ee488_24_data/val \
    --test_list data/ee488_24_data/val_pairs.csv \
    --nClasses 1230 \
    --save_path ./exps/train2/exp03 \
    --initial_model ./exps/train1/exp08/epoch0050.model \
    --optimizer adopt \
    --scheduler steplr \
    --batch_size 100 \
    --trainfunc softmax \
    --image_size 256 \
    --lr 0.0005 \
    --lr_decay 0.85 \
    --max_epoch 30 \
    --margin 0.1 \
    --scale 30 \
    --nOut 1024 \
    --dropout 0.2 \
    --elp_epochs 10 \
    --efft_epochs 20 \
    --model GhostFaceNetsV2 \
    --nPerClass 1

## TurnTrain.py
python trainTurn.py \
    --batch_size 128 \
    --max_img_per_cls 500 \
    --nDataLoaderThread 4 \
    --elp_epochs 5 \
    --efft_epochs 10 \
    --threshold 0.9 \
    --val_path data/ee488_24_data/val \
    --val_list data/ee488_24_data/val_pairs.csv \
    --initial_model ./exps/train1/exp05/epoch0030.model \
    --save_path ./exps/turn \
    --train_path ./data/ee488_24_data/train2 \
    --train_ext jpg \
    --image_size 256 \
    --gpu 0 \
    --lr 0.0005 \
    --lr_decay 0.85 \
    --weight_decay 1e-4

## Train 2
python ./trainEmbedNet.py \
  --gpu 0 \
  --train_path data/ee488_24_data/train2 \
  --test_path data/ee488_24_data/val \
  --test_list data/ee488_24_data/val_pairs.csv \
  --nClasses 1230 \
  --save_path ./exps/train2/turn_exp01 \
  --initial_model ./exps/train1/exp06/epoch0030.model \
  --optimizer adopt \
  --scheduler steplr \
  --batch_size 150 \
  --trainfunc gce \
  --image_size 256 \
  --lr 0.001 \
  --lr_decay 0.9 \
  --elp_epochs 5 \
  --efft_epochs 10 \
  --model GhostFaceNetsV2


python ./trainEmbedNet.py --gpu 0 --train_path data/ee488_24_data/train2 --test_path data/ee488_24_data/val --test_list data/ee488_24_data/val_pairs.csv --save_path ./exps/train2/exp01 --optimizer adopt --batch_size 250 --model GhostFaceNetV2 --trainfunc arcface 

Notes:
-Train1: --nClasses 2882
-Train2: --nClasses 1230




## for Turn new
python ./trainEmbedNet.py \
  --gpu 0 \
  --train_path data/ee488_24_data/train1 \
  --test_path data/ee488_24_data/val \
  --test_list data/ee488_24_data/val_pairs.csv \
  --nClasses 2882 \
  --save_path ./exps/train1/exp09 \
  --optimizer adopt \
  --scheduler steplr \
  --batch_size 150 \
  --trainfunc softmax \
  --image_size 256 \
  --lr 0.001 \
  --lr_decay 0.9 \
  --max_epoch 50 \
  --margin 0.1 \
  --scale 30 \
  --nOut 1024 \
  --dropout 0.2 \
  --model GhostFaceNetsV2

## TurnTrain.py
python trainTurn.py \
    --batch_size 128 \
    --max_img_per_cls 500 \
    --nDataLoaderThread 4 \
    --elp_epochs 5 \
    --efft_epochs 10 \
    --threshold 0.9 \
    --val_path data/ee488_24_data/val \
    --val_list data/ee488_24_data/val_pairs.csv \
    --initial_model ./exps/train1/exp11/epoch0005.model \
    --save_path ./exps/turn \
    --train_path ./data/ee488_24_data/train2 \
    --train_ext jpg \
    --image_size 256 \
    --gpu 0 \
    --lr 0.0005 \
    --lr_decay 0.85 \
    --weight_decay 1e-4
