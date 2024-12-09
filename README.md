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

-----------------------------------------------------------------
## Train1
python ./trainEmbedNet.py \
  --gpu 0 \
  --train_path data/ee488_24_data/train1 \
  --test_path data/ee488_24_data/val \
  --test_list data/ee488_24_data/val_pairs.csv \
  --nClasses 2882 \
  --save_path ./exps/train1/exp20 \
  --optimizer adopt \
  --scheduler steplr \
  --batch_size 330 \
  --trainfunc arcface \
  --image_size 256 \
  --lr 0.0001 \
  --nOut 512 \
  --dropout 0.2 \
  --model ResNet18 \
  --nPerClass 1
-----------------------------------------------------------------

-----------------------------------------------------------------
###### Train 2
python ./trainEmbedNet.py \
  --gpu 0 \
  --train_path data/ee488_24_data/train2 \
  --test_path data/ee488_24_data/val \
  --test_list data/ee488_24_data/val_pairs.csv \
  --nClasses 1230 \
  --save_path ./exps/train2/exp05 \
  --initial_model ./exps/train1/exp11/epoch0020.model \
  --optimizer adopt \
  --scheduler steplr \
  --batch_size 150 \
  --trainfunc arcface \
  --image_size 256 \
  --margin 0.1 \
  --scale 30 \
  --dropout 0.0 \
  --lr 0.001 \
  --lr_decay 0.9 \
  --nPerClass 1 \
  --model GhostFaceNetsV2 \
  --max_epoch 50
----------------------------------------------------------------------

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
    --batch_size 250 \
    --max_img_per_cls 500 \
    --nDataLoaderThread 4 \
    --elp_epochs 20 \
    --efft_epochs 40 \
    --threshold 0.8 \
    --val_path data/ee488_24_data/val \
    --val_list data/ee488_24_data/val_pairs.csv \
    --initial_model ./exps/train1/exp11/epoch0020.model \
    --save_path ./exps/turn3 \
    --train_path ./data/ee488_24_data/train2 \
    --train_ext jpg \
    --image_size 256 \
    --gpu 0 \
    --lr 0.0001 \
    --lr_decay 0.85 \
    --weight_decay 1e-4


python trainTurn.py \
    --batch_size 250 \
    --max_img_per_cls 500 \
    --nDataLoaderThread 4 \
    --elp_epochs 20 \
    --efft_epochs 30 \
    --threshold 0.65 \
    --val_path ./data/ee488_24_data/val \
    --val_list ./data/ee488_24_data/val_pairs.csv \
    --initial_model ./exps/train1/exp08/epoch0050.model \
    --save_path ./exps/turn24 \
    --train_path ./data/ee488_24_data/train2 \
    --image_size 256 \
    --gpu 0 \
    --lr 0.0001 \
    --lr_decay 0.65 \
    --weight_decay 1e-4 \
    --gce_q 0.4 \
    --optimizer adam \
    --scheduler steplr \
    --fine_tune_loss generalizedcrossentropy


## Test Run:
python ./trainEmbedNet.py \
    --gpu 0 \
    --test_path data/ee488_24_data/ee488_24_test/test \
    --test_list data/ee488_24_data/ee488_24_test/test_pairs.csv \
    --initial_model ./exps/turn23/fine_tune_epoch0020.model \
    --save_path ./exps/eval4 \
    --output test_output_3.csv \
    --trainfunc arcface \
    --batch_size 150 \
    --optimizer adopt \
    --eval




1. Please upload your paper presentation in PPT or PDF to the following link at least 30 minutes before the class.

Make sure that your file name starts with your team number.

Dropbox upload: https://www.dropbox.com/request/dHJUQko9nSISuLLrqC3F




2. Submitting your results on the test set (by 9pm on 12/9)

The test set can be downloaded from the project page: https://mm.kaist.ac.kr/share/ee488_24/ee488_24_test.zip (password is dlvu2024)

Ground truth labels are randomized. Therefore, you will get an EER of around 50% when you run your code, which is normal.

You should save the results to file using < --eval --output $path_to_save_output > flags in the baseline code. This will be required to submit your results.

In addition, you need to import a trained model using < --initial_model $path_to_model > in order to evaluate.

For example, < python ./trainEmbedNet.py --gpu 9 --test_path ee488_24_test/test --test_list ee488_24_test/test_pairs.csv --initial_model exps/exp7/epoch0025.model --eval --save_path exps/eval1 --output test_output_1.csv >

You can submit your results up to 5 times, of which the best will be counted.

Submit to the following link: http://mm.kaist.ac.kr:9050/scores (within KAIST network only).

The EER will be shown to you immediately.

Your student number and passphrase are 20244046 and vyt0vd

If you find any bugs in the system, please report to KLMS Q&A.



3. Submitting the report and the code (by 9pm on 12/9)

You should submit the report (in PDF file) and the code (in ZIP file).

Maximum upload size is 2MB, so please do not include model checkpoints or data in your ZIP file.

Submit to the following link: http://mm.kaist.ac.kr:9050/report (within KAIST network or KVPN).

In your code, you should provide the bash script that you used to start training. You should also include pre-processing code if you used any.

The report should be up to 2 pages, single-column, font size between 10 and 12. The following is a non-exhaustive list of things that you might want to include in your report:

- Key design choices such as loss functions, models, etc.
- Hyperparameters
- Ablations you have tried
- Validation / test set performance
- Equipment (GPU) used to train your model

Important: please write your student number on EVERY PAGE of your report.

Submission EER:
-EER is 22.092 pct with TURN
-EER is 22.081 pct With TURN
-EER is 44 pct with Resnet Model