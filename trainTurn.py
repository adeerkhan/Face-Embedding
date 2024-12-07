import sys
import os
import argparse
import numpy
import logging
from sklearn.mixture import GaussianMixture
import torchvision.transforms as transforms
from TurnDataloader import TURNLoader
from TurnNet import TURNTrainer, TURNNet
from tqdm import tqdm
from models.GhostFaceNetsV2 import GhostFaceNetsV2
from loss.GeneralizedCrossEntropy import LossFunction
import importlib

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Parse arguments
# ## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description="TURN Algorithm Training")

# Data loader
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--max_img_per_cls', type=int, default=500, help='Max images per class per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=5, help='Number of data loader threads')

# TURN-specific parameters
parser.add_argument('--elp_epochs', type=int, default=5, help='Number of epochs for linear probing')
parser.add_argument('--efft_epochs', type=int, default=10, help='Number of epochs for full fine-tuning')
parser.add_argument('--threshold', type=float, default=0.9, help='Threshold for clean sample selection')

# Model and training
parser.add_argument('--initial_model', type=str, default="", help='Path to the pre-trained model')
parser.add_argument('--save_path', type=str, default="exps/turn", help='Path to save the model and logs')
parser.add_argument('--train_path', type=str, default="data/train2", help='Path to training data')
parser.add_argument('--train_ext', type=str, default="jpg", help='Training data file extension')
parser.add_argument('--val_path', type=str, default="data/val", help='Path to validation data')
parser.add_argument('--val_list', type=str, default="data/val_pairs.csv", help='Path to validation file list')
parser.add_argument('--image_size', type=int, default=256, help='Image size')
parser.add_argument('--gpu', type=int, default=0, help='GPU index')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Learning rate decay')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')

args = parser.parse_args()

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## TURN Training Script
# ## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Set up logging to both terminal and file
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.save_path, "turn_scores.txt"), mode="a+")
        ],
        level=logging.DEBUG,
        format='[%(levelname)s] :: %(asctime)s :: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting TURN Training...")

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

    # Initialize Model
    model = TURNNet(model="GhostFaceNetsV2", 
                    trainfunc=LossFunction(num_classes=1230, q=0.7), 
                    num_classes=1230, 
                    nOut=1024).cuda()

    # Initialize Trainer
    trainer = TURNTrainer(
        turn_model=model,
        optimizer="adam",
        scheduler="steplr",
        mixedprec=True,
        lr=args.lr,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay
    )

    # Load pre-trained model
    if args.initial_model:
        trainer.loadParameters(args.initial_model)
        print(f"Loaded initial model from {args.initial_model}")

    # Initialize GCE Loss
    gce_loss = LossFunction(num_classes=1230, q=0.7).cuda()

    # Step 1: Linear Probing
    print("Starting Linear Probing...")
    trainLoader = turn_loader.get_linear_probing_loader()
    for ep in range(1, args.elp_epochs + 1):
        print(f"Epoch {ep}/{args.elp_epochs} for Linear Probing")
        trainer.train_network(trainLoader, gce_loss)
        val_loss, val_acc = trainer.validate(val_loader, gce_loss)
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

    # Step 2: Cleanse and Fine-Tune
    print("Cleansing and Fine-Tuning...")

    # Compute per-sample losses
    losses = trainer.compute_per_sample_losses(trainLoader, gce_loss)
    losses = numpy.array(losses).reshape(-1, 1)

    # Fit GMM to identify clean samples
    gmm = GaussianMixture(n_components=2).fit(losses)
    probs = gmm.predict_proba(losses)
    clean_component = numpy.argmin(gmm.means_)
    clean_indices = numpy.where(probs[:, clean_component] > args.threshold)[0]

    print(f"Selected {len(clean_indices)} clean samples")

    # Save clean samples to disk (optional)
    clean_sample_dir = os.path.join(args.save_path, "cleaned_samples")
    os.makedirs(clean_sample_dir, exist_ok=True)
    turn_loader.save_clean_samples(clean_indices, clean_sample_dir)

    # Fine-Tune on clean samples
    cleansed_loader = turn_loader.get_cleansed_loader(clean_indices)
    for ep in range(1, args.efft_epochs + 1):
        print(f"Epoch {ep}/{args.efft_epochs} for Fine-Tuning")
        trainer.train_network(cleansed_loader, gce_loss)

        # Validate after each epoch
        print("Validating on validation set...")
        val_loss, val_accuracy = trainer.validate(val_loader, gce_loss)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Save checkpoint periodically
        if ep % 5 == 0 or ep == args.efft_epochs:
            checkpoint_path = os.path.join(args.save_path, f"fine_tune_epoch{ep:04d}.model")
            trainer.saveParameters(checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Main function
# ## ===== ===== ===== ===== ===== ===== ===== =====

def main():
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    main_worker(args)

if __name__ == '__main__':
    main()