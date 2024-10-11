import os
import re
import json
import shutil
import tempfile
import time

from swin_monai_split import split_data
from argument_parser import parser

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai import data
from monai.data import decollate_batch
from functools import partial

import torch
import wandb


def save_checkpoint(model, epoch, model_name, dir_add):

    best_acc = 0
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "model_state_dict": state_dict, "best_acc": best_acc}
    file_name = os.path.join(dir_add, model_name)
    torch.save(save_dict, file_name)
    print(f"Model saved as {file_name}")


def transforms_swin(roi, split):
    if split == "train":
        split_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.CropForegroundd(
                    keys=["image", "label"], source_key="image", k_divisible=[roi[0], roi[1], roi[2]]
                ),  # Adjusted to match roi's length
                transforms.RandSpatialCropd(
                    keys=["image", "label"], roi_size=roi, random_size=False
                ),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                transforms.NormalizeIntensityd(
                    keys="image", nonzero=True, channel_wise=True
                ),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
    elif split == "valid":
        split_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.NormalizeIntensityd(
                    keys="image", nonzero=True, channel_wise=True
                ),
            ]
        )

    return split_transform


def get_data_list(data_dir):
    """Traverse subfolders and get image and segmentation pairs based on filename patterns."""
    data_list = []
    # Regular expressions to match the image and segmentation files
    image_pattern = re.compile(r"Brats18_[a-zA-Z0-9]+_[a-zA-Z0-9]+_1_(flair|t1|t1ce|t2)\.nii$", re.IGNORECASE)
    label_pattern = re.compile(r"Brats18_[a-zA-Z0-9]+_[a-zA-Z0-9]+_1_seg\.nii$", re.IGNORECASE)

    for root, dirs, files in os.walk(data_dir):
        # Dictionary to store found images by ID
        found_images = {}

        # First pass: collect all flair images
        for file in files:
            match = image_pattern.search(file)
            if match:
                match_list = match.group(0).split("_")  # Extract the ID
                id_ = "_".join(match_list[:3])
                if id_ in found_images:
                    found_images[id_].get("image", []).append(os.path.join(root, file))
                else:
                    found_images[id_] = {"image": [os.path.join(root, file)], "label": None}

        # Second pass: collect all segmentation images
        for file in files:
            match = label_pattern.search(file)
            if match:
                match_list = match.group(0).split("_")  # Extract the ID
                id_ = "_".join(match_list[:3])
                if id_ in found_images:
                    found_images[id_]["label"] = os.path.join(root, file)

        # Add to data_list only if both image and label are found
        for item in found_images.values():
            if item["label"] is not None:
                data_list.append(item)
                
    print(f"Found {len(data_list)} image-label pairs in {data_dir}")
    print(f"Example: {data_list[0]}")
    return data_list


def get_dataloader(data_dir, batch_size, roi, split):

    print(f"Loading data from {data_dir} for {split} split.")

    data_list = get_data_list(data_dir)  # Use the function to get the data paths
    if split == "train":
        train_transforms = transforms_swin(roi, split)
        dataset = data.Dataset(data=data_list, transform=train_transforms)
        dataloader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
        )
    elif split == "valid":
        valid_transforms = transforms_swin(roi, split)
        dataset = data.Dataset(data=data_list, transform=valid_transforms)
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

    print(f"{split.capitalize()} dataloader created with batch size {batch_size}.")
    return dataloader


if __name__ == "__main__":

    args = parser.parse_args()

    print_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set paths
    this_file = os.path.abspath(__file__)
    parent_dir = os.path.dirname(this_file)
    data3d_dir = os.path.join(parent_dir, "data3D")
    train_data_dir = os.path.join(data3d_dir, "train_data")
    save_path_monai_data = os.path.join(parent_dir, "swin_monai_data")

    if not os.path.exists(save_path_monai_data):
        split_data(train_data_dir, 0.8, save_path_monai_data)

    train_data_dir = os.path.join(save_path_monai_data, "train_data")
    valid_data_dir = os.path.join(save_path_monai_data, "val_data")

    # Hyperparameters
    # roi = (224, 224, 144)
    roi = (128, 128, 128)
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    # Create dataloaders
    print(f"Creating dataloaders with ROI: {roi}, Batch size: {batch_size}")
    train_dataloader = get_dataloader(train_data_dir, batch_size, roi, "train")
    valid_dataloader = get_dataloader(valid_data_dir, batch_size, roi, "valid")

    print("Data loaders created!")

    # Debug the dataloader
    print("Inspecting the first batch from train dataloader...")
    first_batch = next(iter(train_dataloader))
    print(f"First batch keys: {first_batch.keys()}")
    print(f"First image shape: {first_batch['image'].shape}")
    print(f"First label shape: {first_batch['label'].shape}")
    # xd = first_batch["label"][1:].squeeze()
    # print(f"Edited label shape: {xd.shape}")
