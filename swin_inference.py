import os
import torch

from swin_monai import transforms_swin
from monai.networks.nets import SwinUNETR
from functools import partial
from monai.inferers import sliding_window_inference
from monai.transforms import (
    LoadImaged, 
    ConvertToMultiChannelBasedOnBratsClassesd, 
    NormalizeIntensityd, 
    ToTensord
)
from monai.data import decollate_batch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Define paths
this_file = os.path.abspath(__file__)
parent_dir = os.path.dirname(this_file)
data3d_dir = os.path.join(parent_dir, "data3D", "train_data")
flair_path = '/home/scanar/BCV_project/canar_tobon_brats2018/data3D/train_data/Brats18_2013_0_1/Brats18_2013_0_1_flair.nii'
t1_path = '/home/scanar/BCV_project/canar_tobon_brats2018/data3D/train_data/Brats18_2013_0_1/Brats18_2013_0_1_t1.nii'
t1_ce_path = '/home/scanar/BCV_project/canar_tobon_brats2018/data3D/train_data/Brats18_2013_0_1/Brats18_2013_0_1_t1ce.nii'
t2_path = '/home/scanar/BCV_project/canar_tobon_brats2018/data3D/train_data/Brats18_2013_0_1/Brats18_2013_0_1_t2.nii'
seg_path = '/home/scanar/BCV_project/canar_tobon_brats2018/data3D/train_data/Brats18_2013_0_1/Brats18_2013_0_1_seg.nii'

vols_path_lt = [flair_path, t1_path, t1_ce_path, t2_path]


# Load model weights
this_file_dir = os.path.dirname(os.path.abspath(__file__))
weight_path = os.path.join(this_file_dir, 'first_run_swin.pt')
model_weights = torch.load(weight_path)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
roi = (128, 128, 128)
batch_size = 1

# Load model
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

model.load_state_dict(model_weights['model_state_dict'])
model.to(device)
model.eval()

# Define transforms
transforms = [
    LoadImaged(keys=['image', 'label']),
    ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
    NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
    ToTensord(keys=['image', 'label'])
]

# Load and apply transforms to volume and label
data_dict = {'image': vols_path_lt, 'label': seg_path}
for transform in transforms:
    data_dict = transform(data_dict)

# Extract preprocessed image and label
image = data_dict['image'].unsqueeze(0).to(device)
gt = data_dict['label'].unsqueeze(0).to(device)

# Inference
model_infer = partial(
    sliding_window_inference,
    roi_size=[roi[0], roi[1], roi[2]],
    sw_batch_size=1,
    predictor=model,
    overlap=0.6
)

with torch.no_grad():
    image_np = image.detach().cpu().numpy()
    gt = gt.squeeze(0)
    gt_np = gt.detach().cpu().numpy()
    prob = torch.sigmoid(model_infer(image))
    seg = prob[0].detach().cpu().numpy()
    seg = (seg > 0.5).astype(np.uint8)
    seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
    seg_out[seg[1] == 1] = 2
    seg_out[seg[0] == 1] = 1
    seg_out[seg[2] == 1] = 4
    
    print(seg_out.shape)

# Save image
print('Saving image...')


# Define custom colormap for the labels
cmap = mcolors.ListedColormap(['black', 'cyan', 'yellow', 'red'])

# Define normalization boundaries (0 for background, 1 for label 1, 2 for label 2, 4 for label 4)
bounds = [0, 1, 2, 4, 5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Save image
print('Saving image...')

first_chanel_vol = image_np[0, 0, :, :]
first_chanel_vol = (first_chanel_vol - np.min(first_chanel_vol)) / (np.max(first_chanel_vol) - np.min(first_chanel_vol))
first_chanel_vol = (first_chanel_vol * 255).astype(np.uint8)

label_back2og = np.zeros((gt_np.shape[1], gt_np.shape[2], gt_np.shape[3]))
label_back2og[gt_np[1] == 1] = 2
label_back2og[gt_np[0] == 1] = 1
label_back2og[gt_np[2] == 1] = 4
label_back2og = label_back2og.astype(np.uint8)

assert seg_out.shape == label_back2og.shape == first_chanel_vol.shape

_, _, z = seg_out.shape

max_area = 0
for i in range(z):
    area = np.sum(seg_out[:, :, i] != 0)
    if area > max_area:
        max_area = area
        z_max = i

print(f'Max area: {max_area} at z = {z_max}')

# Plot the image with overlap of brain and segmentation masks
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Input image
ax[0].imshow(first_chanel_vol[:, :, z_max], cmap='gray')
ax[0].set_title('Input image')
ax[0].axis('off')


# Ground truth with overlap
ax[1].imshow(first_chanel_vol[:, :, z_max], cmap='gray')
ax[1].imshow(label_back2og[:, :, z_max], cmap=cmap, norm=norm, alpha=0.5)  # alpha for transparency
ax[1].set_title('Ground truth')
ax[1].axis('off')


# Prediction with overlap
ax[2].imshow(first_chanel_vol[:, :, z_max], cmap='gray')
ax[2].imshow(seg_out[:, :, z_max], cmap=cmap, norm=norm, alpha=0.5)  # alpha for transparency
ax[2].set_title('Prediction')
ax[2].axis('off')

plt.savefig('swin_inference_overlap.png')

print('Inference done!')

