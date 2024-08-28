import os
import yaml

import wandb

import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributed import init_process_group


# Imports from other files
from utils.losses import build_loss_fn
from build_dataset import build_dataset, build_dataloader
from model import build_segformer3d_model
from argument_parser import parser
from utils.metrics import build_metric_fn
from utils.augmentations import build_augmentations
from trainer import Brats2018_Trainer, ddp_setup

# global variables

world_size = torch.cuda.device_count()


# Parse arguments
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#wandb

if args.wandb:
    wandb.init(project="3d-segmentation")
    wandb.config.update(parser.parse_args())



torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {"num_workers": 8, "pin_memory": True} if args.cuda else {}

# Define path for saving model
os.makedirs("models", exist_ok=True)
args.save = os.path.join("models", args.save)

# Define data paths
file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(file_path, "preproc_data")
train_path = os.path.join(data_path, "train_data")
valid_path = os.path.join(data_path, "val_data")

# Define dataset and dataloader
train_dataset = build_dataset(
    dataset_type=args.dataset, root_dir=train_path, is_train=True
)
valid_dataset = build_dataset(
    dataset_type=args.dataset, root_dir=valid_path, is_train=False
)

ddp_setup(rank=0, world_size=1, backend="nccl")

train_loader = build_dataloader(train_dataset)
valid_loader = build_dataloader(valid_dataset)

# Load the configuration file
config_path = os.path.join(file_path, "configs", "config.yaml")
config_dict = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

# Define the model
model = build_segformer3d_model(config=config_dict)
model.to(device)


# Model parameters
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criteria = build_loss_fn(args.loss)
warmup_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
training_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)


def main(rank, world_size, epochs, log_interval):
    # Initialize the distributed process group
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    # Set up the model and trainer
    trainer_lindo = Brats2018_Trainer(
        config=config_dict,
        model=model,
        optimizer=optimizer,
        criterion=criteria,
        train_loader=train_loader,
        valid_loader=valid_loader,
        warmup_scheduler=warmup_scheduler,
        training_scheduler=training_scheduler,
        gpu_id=rank,  # Each process uses a different GPU
        print_every=args.log_interval,
        wandb=True
    )
    
    # Start training
    trainer_lindo.train()

if __name__ == "__main__":
    mp.spawn(main, args=(world_size, args.epochs, args.log_interval), nprocs=world_size, join=True)
