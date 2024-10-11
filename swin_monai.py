import os
import re
import json
import shutil
import tempfile
import time
from tqdm import tqdm

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


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def save_checkpoint(model, epoch, best_acc, save_dir, model_name):

    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "model_state_dict": state_dict, "best_acc": best_acc}
    file_name = os.path.join(save_dir, model_name)
    torch.save(save_dict, file_name)
    print(f"Model saved as {file_name}")


def transforms_swin(roi, split):
    if split == "train":
        split_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.CropForegroundd(
                    keys=["image", "label"],
                    source_key="image",
                    k_divisible=[roi[0], roi[1], roi[2]],
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
    image_pattern = re.compile(
        r"Brats18_[a-zA-Z0-9]+_[a-zA-Z0-9]+_1_(flair|t1|t1ce|t2)\.nii$", re.IGNORECASE
    )
    label_pattern = re.compile(
        r"Brats18_[a-zA-Z0-9]+_[a-zA-Z0-9]+_1_seg\.nii$", re.IGNORECASE
    )

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
                    found_images[id_] = {
                        "image": [os.path.join(root, file)],
                        "label": None,
                    }

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

    # print(f"Found {len(data_list)} image-label pairs in {data_dir}")
    # print(f"Example: {data_list[0]}")
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


def train_epoch(model, loader, optimizer, epoch, loss_func):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        run_loss.update(loss.item(), n=batch_size)
        print(
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()
    return run_loss.avg


def val_epoch(
    model,
    loader,
    epoch,
    acc_func,
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(
                device
            )
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [
                post_pred(post_sigmoid(val_pred_tensor))
                for val_pred_tensor in val_outputs_list
            ]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            dice_tc = run_acc.avg[0]
            dice_wt = run_acc.avg[1]
            dice_et = run_acc.avg[2]
            print(
                "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

    return run_acc.avg


def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    model_inferer=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
):
    val_acc_max = 0.0
    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []
    loss_epochs = []
    trains_epoch = []
    
    with tqdm(range(max_epochs), desc="Epochs") as pbar:
        for epoch in range(start_epoch, max_epochs):
            print(time.ctime(), "Epoch:", epoch)
            epoch_time = time.time()
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                epoch=epoch,
                loss_func=loss_func,
            )
            print(
                "Final training  {}/{}".format(epoch, max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )

            if (epoch + 1) % val_every == 0 or epoch == 0:
                loss_epochs.append(train_loss)
                trains_epoch.append(int(epoch))
                epoch_time = time.time()
                val_acc = val_epoch(
                    model,
                    val_loader,
                    epoch=epoch,
                    acc_func=acc_func,
                    model_inferer=model_inferer,
                    post_sigmoid=post_sigmoid,
                    post_pred=post_pred,
                )
                dice_tc = val_acc[0]
                dice_wt = val_acc[1]
                dice_et = val_acc[2]
                val_avg_acc = np.mean(val_acc)
                print(
                    "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                    ", dice_tc:",
                    dice_tc,
                    ", dice_wt:",
                    dice_wt,
                    ", dice_et:",
                    dice_et,
                    ", Dice_Avg:",
                    val_avg_acc,
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )
                dices_tc.append(dice_tc)
                dices_wt.append(dice_wt)
                dices_et.append(dice_et)
                dices_avg.append(val_avg_acc)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    save_checkpoint(
                        model,
                        epoch,
                        val_acc_max,
                        parent_dir,
                        args.save
                    )
                scheduler.step()
                if args.wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_dice_tc": dice_tc,
                            "val_dice_wt": dice_wt,
                            "val_dice_et": dice_et,
                            "val_dice_avg": val_avg_acc,
                        }
                    )
                
            pbar.update(1)
        print("Training Finished !, Best Accuracy: ", val_acc_max)
    return (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )


if __name__ == "__main__":

    args = parser.parse_args()
    
    if args.wandb:
        wandb.init(project="3d-segmentation", name=args.run_name)
        wandb.config.update(args)

    print_config()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
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
    max_epochs = args.epochs
    lr = args.lr
    sw_batch_size = 1
    infer_overlap = 0.5
    val_every = 1

    # Create dataloaders
    print()
    print(f"Creating dataloaders with ROI: {roi}, Batch size: {batch_size}")
    train_dataloader = get_dataloader(train_data_dir, batch_size, roi, "train")
    valid_dataloader = get_dataloader(valid_data_dir, batch_size, roi, "valid")

    print("Data loaders created!")
    print()
    # Debug the dataloader
    print("Inspecting the first batch from train dataloader...")
    first_batch = next(iter(train_dataloader))
    print(f"First batch keys: {first_batch.keys()}")
    print(f"First image shape: {first_batch['image'].shape}")
    print(f"First label shape: {first_batch['label'].shape}")
    print()

    model = SwinUNETR(
        img_size=roi,
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(device)

    torch.backends.cudnn.benchmark = True
    dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_acc = DiceMetric(
        include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True
    )
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    start_epoch = 0

    (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    ) = trainer(
        model=model,
        train_loader=train_dataloader,
        val_loader=valid_dataloader,
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        scheduler=scheduler,
        model_inferer=model_inferer,
        start_epoch=start_epoch,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
    )
    
    print(f"train completed, best average dice: {val_acc_max:.4f} ")
    
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, loss_epochs, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_avg, color="green")
    plt.show()
    plt.figure("train", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Val Mean Dice TC")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_tc, color="blue")
    plt.subplot(1, 3, 2)
    plt.title("Val Mean Dice WT")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_wt, color="brown")
    plt.subplot(1, 3, 3)
    plt.title("Val Mean Dice ET")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_et, color="purple")
    plt.imsave("train_loss.png")