import sys
import os
import argparse
import numpy
import logging
from sklearn.mixture import GaussianMixture
import torchvision.transforms as transforms
from TurnDataloader import TURNLoader
from TurnNet import TURNTrainer, TURNNet
from models.GhostFaceNetsV2 import GhostFaceNetsV2
from loss.GeneralizedCrossEntropy import LossFunction
from sklearn import metrics
import numpy as np
import torch
import torch.nn.functional as F
import importlib


# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Parse arguments
# ## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description="TURN Algorithm Training")

## Data loader
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--max_img_per_cls', type=int, default=500, help='Max images per class per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=5, help='Number of data loader threads')

## TURN-specific parameters
parser.add_argument('--elp_epochs', type=int, default=5, help='Number of epochs for linear probing')
parser.add_argument('--efft_epochs', type=int, default=10, help='Number of epochs for full fine-tuning')
parser.add_argument('--threshold', type=float, default=0.9, help='Threshold for clean sample selection')

## Model and training
parser.add_argument('--initial_model', type=str, default="", help='Path to the pre-trained model')
parser.add_argument('--save_path', type=str, default="exps/turn", help='Path to save the model and logs')
parser.add_argument('--train_path', type=str, default="data/train2", help='Path to training data')
parser.add_argument('--train_ext', type=str, default="jpg", help='Training data file extension')
parser.add_argument('--val_path', type=str, default="data/val", help='Path to validation data')
parser.add_argument('--val_list', type=str, default="data/val_pairs.csv", help='Path to validation file list')
parser.add_argument('--image_size', type=int, default=256, help='Image size')
parser.add_argument('--gpu', type=int, default=0, help='GPU index')

## Optimizer
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer')
parser.add_argument('--scheduler', type=str, default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Learning rate decay')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for the model')

## Parameters for losses
# Parameters for TURN Training Loss
parser.add_argument('--train_loss', type=str, default="GeneralizedCrossEntropy", help='Loss function for training phase')
parser.add_argument('--train_loss_q', type=float, default=0.7, help='Hyperparameter q for training loss, if applicable')

# Parameters for Fine-Tuning Loss
parser.add_argument('--fine_tune_loss', type=str, default="arcface", help='Loss function for fine-tuning phase')
parser.add_argument('--fine_tune_loss_nOut', type=int, default=1024, help='Output dimension for fine-tuning loss')
parser.add_argument('--fine_tune_loss_nClasses', type=int, default=1230, help='Number of classes for fine-tuning loss')
parser.add_argument('--fine_tune_loss_scale', type=float, default=64.0, help='Scale for fine-tuning loss')
parser.add_argument('--fine_tune_loss_margin', type=float, default=0.5, help='Margin for fine-tuning loss')

args = parser.parse_args()

def compute_eer(labels, scores):
    # Convert to numpy if not already
    labels = np.array(labels)
    scores = np.array(scores)

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    # Find the threshold where fpr and fnr cross
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    EER = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    return EER, eer_threshold

def evaluate_and_compute_eer(trainer, loader):
    trainer.__model__.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for img1, img2, labels in loader:
            feat1 = trainer.__model__(img1.cuda())
            feat2 = trainer.__model__(img2.cuda())
            # Cosine similarity scores
            scores = F.cosine_similarity(feat1, feat2).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())

    EER, eer_threshold = compute_eer(all_labels, all_scores)
    return EER

def load_loss_function(loss_name, **kwargs):
    try:
        # Dynamically import the loss module and get the LossFunction class
        loss_module = importlib.import_module(f'loss.{loss_name}')
        LossFunction = getattr(loss_module, 'LossFunction')
        
        # Define required parameters for specific loss functions
        loss_specific_params = {
            'arcface': ['nOut', 'nClasses', 'scale', 'margin'],
            'generalizedcrossentropy': ['num_classes', 'q'],
            # Add other loss functions and their required parameters here
        }
        
        # Get the required parameters for the selected loss
        required_params = loss_specific_params.get(loss_name.lower(), [])
        
        # Check if all required parameters are provided
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter '{param}' for {loss_name} loss.")
        
        return LossFunction(**kwargs)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(f"Loss function '{loss_name}' could not be loaded. Ensure it exists in the 'loss' folder. Error: {e}")


# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## TURN Training Script
# ## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('[%(levelname)s] :: %(asctime)s :: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    # Create StreamHandler (console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Create FileHandler (file) for live updates
    file_handler = logging.FileHandler(os.path.join(args.save_path, "turn_scores.txt"), mode="a+")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info('{}'.format(args))
    logger.info("Starting TURN Training for Linear Probing...")

    # Data transformations
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.image_size),
        transforms.RandomCrop([args.image_size - 32, args.image_size - 32]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.image_size),
        transforms.CenterCrop([args.image_size - 32, args.image_size - 32]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize TURNLoader
    turn_loader = TURNLoader(
        train_path=args.train_path,
        train_ext=args.train_ext,
        transform=train_transform,
        batch_size=args.batch_size,
        max_img_per_cls=args.max_img_per_cls,
        nDataLoaderThread=args.nDataLoaderThread,
        nPerClass=1
    )

    # Validation Loader
    val_loader = turn_loader.get_validation_loader(
        val_path=args.val_path,
        val_list=args.val_list
    )

    # Initialize Training Loss
    training_loss = load_loss_function(
        args.train_loss,
        num_classes=args.fine_tune_loss_nClasses,
        q=args.train_loss_q
    ).cuda()
    

    # Initialize Model with Training Loss
    model = TURNNet(
        model="GhostFaceNetsV2", 
        trainfunc=training_loss, 
        num_classes=args.fine_tune_loss_nClasses, 
        nOut=args.fine_tune_loss_nOut,
        dropout=args.dropout
    ).cuda()

    # Initialize Trainer
    trainer = TURNTrainer(
        turn_model=model,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        lr=args.lr,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay
    )

    # Load pre-trained model
    if args.initial_model:
        trainer.loadParameters(args.initial_model)
        logger.info(f"Loaded initial model from {args.initial_model}")

    # Initialize Fine-Tuning Loss
    fine_tune_loss = load_loss_function(
        args.fine_tune_loss,
        nOut=args.fine_tune_loss_nOut,
        nClasses=args.fine_tune_loss_nClasses,
        scale=args.fine_tune_loss_scale,
        margin=args.fine_tune_loss_margin
    ).cuda()

    logger.info(f"Using TURN loss: {args.train_loss}")

    # Step 1: Linear Probing
    logger.info("Linear Probing...")
    trainLoader = turn_loader.get_linear_probing_loader()
    for ep in range(1, args.elp_epochs + 1):
        logger.info(f"Epoch {ep}/{args.elp_epochs} for Linear Probing")
        trainer.train_network(trainLoader, training_loss)
        val_loss, val_acc = trainer.validate(val_loader, training_loss)
        logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")


    # Step 2: Cleanse and Fine-Tune
    logger.info("Cleansing and Fine-Tuning...")

    # Compute per-sample losses
    losses = trainer.compute_per_sample_losses(trainLoader, training_loss)
    losses = numpy.array(losses).reshape(-1, 1)

    # Fit GMM to identify clean samples
    gmm = GaussianMixture(n_components=2).fit(losses)
    probs = gmm.predict_proba(losses)
    clean_component = numpy.argmin(gmm.means_)
    clean_indices = numpy.where(probs[:, clean_component] > args.threshold)[0]

    logger.info(f"Selected {len(clean_indices)} clean samples")

    # Save clean samples to disk
    clean_sample_dir = os.path.join(args.save_path, "cleaned_samples")
    os.makedirs(clean_sample_dir, exist_ok=True)
    turn_loader.save_clean_samples(clean_indices, clean_sample_dir)

    logger.info(f"Using Fine-tune loss: {args.fine_tune_loss}")
    # Fine-Tune on clean samples
    cleansed_loader = turn_loader.get_cleansed_loader(clean_indices)
    for ep in range(1, args.efft_epochs + 1):
        logger.info(f"Epoch {ep}/{args.efft_epochs} for Fine-Tuning")
        trainer.train_network(cleansed_loader, fine_tune_loss)

        # Validate after each epoch
        logger.info("Validating on validation set...")
        val_loss, val_accuracy = trainer.validate(val_loader, fine_tune_loss)
        logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Compute EER after validation (if desired every epoch or at intervals)
        EER = evaluate_and_compute_eer(trainer, val_loader)
        logger.info(f"Epoch {ep}, Val EER: {EER*100:.2f}%")

        # Save checkpoint periodically
        if ep % 5 == 0 or ep == args.efft_epochs:
            checkpoint_path = os.path.join(args.save_path, f"fine_tune_epoch{ep:04d}.model")
            trainer.saveParameters(checkpoint_path)
            logger.info(f"Checkpoint saved at {checkpoint_path}")

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Main function
# ## ===== ===== ===== ===== ===== ===== ===== =====

def main():
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    main_worker(args)

if __name__ == '__main__':
    main()