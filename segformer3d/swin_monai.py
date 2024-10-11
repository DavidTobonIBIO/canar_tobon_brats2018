import os
import json
import shutil
import tempfile
import time

from utils.swin_monai_split import split_data

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
    save_dict = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'best_acc': best_acc
    }
    file_name = os.path.join(dir_add, model_name)
    torch.save(save_dict, file_name)
    print(f"Model saved as {file_name}")


def transforms(roi, split):
    
    if split == 'train':
        split_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.CropForegroundd(keys=["image", "label"],
                                        source_key="image",
                                        k_divisible=[roi[0], roi[1], roi[2]]),
                transforms.RandSpatialCropd(keys=["image", "label"],
                                            roi_size=[roi[0], roi[1], roi[2]],
                                            random_size=False),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0)                      
            ]
        )
    elif split == 'valid':
        split_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )
    
    return split_transform

def get_dataloader(data_dir, batch_size, roi, split):
    
    if split == 'train':
        dataset = data.Dataset(data=data_dir, transform=transforms(roi, split))
        
        dataloader = data.DataLoader(dataset, 
                                       batch_size=batch_size, 
                                       shuffle=True,
                                       num_workers=1,
                                       pin_memory=True
                                       )
        



if __name__ == '__main__':
    
    print_config()

    #MONAI working directory

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    if directory is not None:
        os.makedirs(directory, exist_ok=True)   
        
    root_dir = tempfile.mkstemp() if directory is None else directory


    # Set paths 

    this_file = os.path.abspath(__file__)
    parent_dir = os.path.dirname(this_file)
    data3d_dir = os.path.join(parent_dir, "data3D")
    train_data_dir = os.path.join(data3d_dir, "train_data")    
    save_path_monai_data = os.path.join(parent_dir, 'swin_monai_data')

    if os.path.exists(save_path_monai_data):
        pass
    else:
        split_data(train_data_dir, 0.8, save_path_monai_data)
    
    train_data_dir = None
    