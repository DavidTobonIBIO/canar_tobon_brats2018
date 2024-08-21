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
import torch

def animate(input_1, input_2):
    """animate pairs of image sequences of the same length on two conjugate axis"""
    assert len(input_1) == len(
        input_2
    ), f"two inputs should have the same number of frame but first input had {len(input_1)} and the second one {len(input_2)}"
    # set the figure and axis
    fig, axis = plt.subplots(1, 2, figsize=(8, 8))
    axis[0].set_axis_off()
    axis[1].set_axis_off()
    sequence_length = input_1.__len__()
    sequence = []
    for i in range(sequence_length):
        im_1 = axis[0].imshow(input_1[i], cmap="gray", animated=True)
        im_2 = axis[1].imshow(input_2[i], cmap="gray", animated=True)
        if i == 0:
            axis[0].imshow(input_1[i], cmap="gray")  # show an initial one first
            axis[1].imshow(input_2[i], cmap="gray")  # show an initial one first

        sequence.append([im_1, im_2])
    return animation.ArtistAnimation(
        fig,
        sequence,
        interval=25,
        blit=True,
        repeat_delay=100,
    )


def viz_animation(volume_indx: int = 1, label_indx: int = 1) -> None:
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
    ani.save(os.path.join(animations_dir, f'brats_{idx}.mp4'), writer=writervideo)
    
    
def viz_figure(volume_indx: int = 1, label_indx: int = 1) -> None:
    """
    pair visualization of the volume and label
    volume_indx: index for the volume. ["flair", "t1", "t1ce", "t2"]
    label_indx: index for the label segmentation ["TC" (Tumor core), "WT" (Whole tumor), "ET" (Enhancing tumor)]
    """
    assert volume_indx in [0, 1, 2]
    assert label_indx in [0, 1, 2]
    x = volume[volume_indx, ...]
    y = label[label_indx, ...]
    fig, axis = plt.subplots(1, 2, figsize=(8, 4))
    slice_indx = 64
    axis[0].imshow(x[slice_indx], cmap="gray")
    axis[1].imshow(y[slice_indx], cmap="gray")
    axis[0].axis("off")
    axis[1].axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(figures_dir, f'brats_{idx}.png'))


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
    data_dir = os.path.join(grandparent_dir, "preproc_data", "train_data")
    cases = os.listdir(data_dir)
    animations_dir = os.path.join(grandparent_dir, "animations")
    figures_dir = os.path.join(grandparent_dir, "figures")
    os.makedirs(animations_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    random.seed(42)

    idx = random.randint(0, len(cases))
    print(idx)
    volume = load_volume(os.path.join(data_dir, cases[idx]))
    label = load_label(os.path.join(data_dir, cases[idx]))
    print(volume.shape, label.shape)
    
    viz_animation(volume_indx=0, label_indx=0)
    
    viz_figure(volume_indx=0, label_indx=0)
    
