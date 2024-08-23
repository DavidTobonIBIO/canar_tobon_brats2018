import os
import yaml

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

#Imports from other files
from utils.losses import build_loss_fn
from build_dataset import build_dataset, build_dataloader
from model import build_segformer3d_model
from argument_parser import parser


# Parse arguments

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {"num_workers": 8, "pin_memory": True} if args.cuda else {}

#Define path for saving model

os.makedirs("models", exist_ok=True)
args.save = os.path.join("models", args.save)


# Define data paths

file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(file_path, "preproc_data")

train_path = os.path.join(data_path, "train_data")
valid_path = os.path.join(data_path, "val_data")

#Define dataset and dataloader

train_dataset = build_dataset(dataset_type=args.dataset, root_dir=train_path, is_train=True)
valid_dataset = build_dataset(dataset_type=args.dataset, root_dir=valid_path, is_train=False)

train_loader = build_dataloader(train_dataset)
valid_loader = build_dataloader(valid_dataset)


# Load the configuration file

config_path = os.path.join(file_path, 'configs', 'config.yaml')
config_dict = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

# Define the model

model = build_segformer3d_model(config=config_dict)
model.to(device) #Model sent to device, is loaded correctly :)

