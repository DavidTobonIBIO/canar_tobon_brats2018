"""
This module contains functions for volume visualization in the context of the BRATS2018 dataset.

Functions:
- animate: Animates pairs of image sequences on two conjugate axes.
- viz: Visualizes pairs of volume and label segmentation.

Note: This code is based on the segformer3d implementation from the official paper repository: https://github.com/OSUPCVLab/SegFormer3D/blob/main/data/brats2017_seg/brats2021_raw_data/brats2021_seg_preprocess.py
"""

import os
import random
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
from tqdm import tqdm


def animate(input_1, input_2):
    """animate pairs of image sequences of the same length on two conjugate axis"""
    assert len(input_1) == len(
        input_2
    ), f"two inputs should have the same number of frame but first input had {len(input_1)} and the second one {len(input_2)}"
    # set the figure and axis
    fig, axis = plt.subplots(1, 2, figsize=(8, 4))
    axis[0].set_axis_off()
    axis[1].set_axis_off()
    sequence_length = input_1.__len__()
    sequence = []
    for i in range(sequence_length):
        im_1 = axis[0].imshow(input_1[i], cmap="gray", animated=True)
        im_2 = axis[1].imshow(input_2[i], cmap="gray", animated=True)
        if i == 0:
            axis[0].imshow(input_1[i], cmap="gray")
            axis[1].imshow(input_2[i], cmap="gray")

        sequence.append([im_1, im_2])
    return animation.ArtistAnimation(
        fig,
        sequence,
        interval=25,
        blit=True,
        repeat_delay=100,
    )


def viz_animation(volume_indx: int, label_indx: int, volume, label) -> None:
    """
    pair visualization of the volume and label
    volume_indx: index for the volume. ["flair", "t1", "t1ce", "t2"]
    label_indx: index for the label segmentation ["TC" (Tumor core), "WT" (Whole tumor), "ET" (Enhancing tumor)]
    """
    assert volume_indx in [0, 1, 2]
    assert label_indx in [0, 1, 2]
    x = volume[volume_indx, ...]
    y = label[label_indx, ...]
    ani = animate(input_1=x, input_2=y)
    writervideo = animation.FFMpegWriter(fps=60)
    writergif = animation.PillowWriter(fps=15)
    ani.save(
        os.path.join(
            visualizations_dir, f"brats_{idx}_vol{volume_indx}_label{label_indx}.mp4"
        ),
        writer=writervideo,
    )
    ani.save(
        os.path.join(
            visualizations_dir, f"brats_{idx}_vol{volume_indx}_label{label_indx}.gif"
        ),
        writer=writergif,
    )


def get_max_slice(label) -> int:
    """get the slice with the maximum number of non-zero pixels"""
    max_slice_idx = 0
    max_area = 0
    for i in range(label.shape[0]):
        area = np.count_nonzero(label[i])
        if area > max_area:
            max_area = area
            max_slice_idx = i
    return max_slice_idx


def viz_figure(volume_indx: int, label_indx: int, volume, label) -> None:
    """
    pair visualization of the volume and label
    volume_indx: index for the volume. ["flair", "t1", "t1ce", "t2"]
    label_indx: index for the label segmentation ["TC" (Tumor core), "WT" (Whole tumor), "ET" (Enhancing tumor)]
    """
    assert volume_indx in [0, 1, 2]
    assert label_indx in [0, 1, 2]

    x = volume[volume_indx, ...]
    y = label[label_indx, ...]
    slice_idx = get_max_slice(y)
    fig, axis = plt.subplots(1, 2, figsize=(8, 4))
    axis[0].imshow(x[slice_idx], cmap="gray")
    axis[1].imshow(y[slice_idx], cmap="gray")
    axis[0].axis("off")
    axis[1].axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig(
        os.path.join(
            visualizations_dir,
            f"brats_{idx}_slice_{slice_idx}_vol{volume_indx}_label{label_indx}.png",
        )
    )


def viz_figure_all_labels(volume_indx: int, labels_vol) -> None:
    """
    pair visualization of the volume and all labels
    volume_indx: index for the volume. ["flair", "t1", "t1ce", "t2"]
    label_indx: index for the label segmentation ["TC" (Tumor core), "WT" (Whole tumor), "ET" (Enhancing tumor)]
    """
    assert volume_indx in [0, 1, 2]

    x = volume[volume_indx, ...]
    y = labels_vol[1, ...]
    slice_idx = get_max_slice(y)

    fig, axis = plt.subplots(1, 2, figsize=(8, 4))

    img = np.copy(x[slice_idx])
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    colors = [(0, 255, 255), (255, 255, 0), (255, 0, 0)]

    label_order = [1, 0, 2]
    for i in label_order:
        color = colors[i]
        label = labels_vol[i, ...]
        img[label[slice_idx] == True] = color

    axis[0].imshow(x[slice_idx], cmap="gray")
    axis[1].imshow(img)
    axis[0].axis("off")
    axis[1].axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig(
        os.path.join(
            visualizations_dir,
            f"brats_{idx}_slice_{slice_idx}_vol{volume_indx}.png",
        )
    )


def load_volume(vol_path: str):
    """Load the volume from the given path"""
    vol_file = os.path.join(vol_path, f"{vol_path.split('/')[-1]}_modalities.pt")
    volume = torch.load(vol_file)
    return volume


def load_label(label_path: str):
    """Load the label from the given path"""
    label_file = os.path.join(label_path, f"{label_path.split('/')[-1]}_label.pt")
    label = torch.load(label_file)
    return label


if __name__ == "__main__":

    file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(file_path)
    grandparent_dir = os.path.dirname(parent_dir)
    data_dir = os.path.join(grandparent_dir, "preproc_data", "val_data")
    cases = os.listdir(data_dir)

    for idx in tqdm(range(len(cases))):

        # idx = random.randint(0, len(cases))
        volume = load_volume(os.path.join(data_dir, cases[idx]))
        label = load_label(os.path.join(data_dir, cases[idx]))

        for i in range(3):
            visualizations_dir = os.path.join(
                grandparent_dir,
                "visualizations",
                "data",
                f"{idx}_{cases[idx]}",
                f"label_{i}",
            )
            os.makedirs(visualizations_dir, exist_ok=True)
            viz_animation(volume_indx=1, label_indx=i, volume=volume, label=label)
            viz_figure(volume_indx=1, label_indx=i, volume=volume, label=label)

        visualizations_dir = os.path.join(
            grandparent_dir, "visualizations", "data", f"{idx}_{cases[idx]}"
        )
        viz_figure_all_labels(volume_indx=1, labels_vol=label)
