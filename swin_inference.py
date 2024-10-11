import os 
import torch

from swin_monai import get_dataloader
from monai.networks.nets import SwinUNETR
from functools import partial
from monai.inferers import sliding_window_inference

import numpy as np
import matplotlib.pyplot as plt


# Define paths
this_file = os.path.abspath(__file__)
parent_dir = os.path.dirname(this_file)
data3d_dir = os.path.join(parent_dir, "data3D")
train_data_dir = os.path.join(data3d_dir, "train_data")
save_path_monai_data = os.path.join(parent_dir, "swin_monai_data")
valid_data_dir = os.path.join(save_path_monai_data, "val_data")

# Load model weights
this_file_dir = os.path.dirname(os.path.abspath(__file__))
weight_path = os.path.join(this_file_dir, 'first_run_swin.pt')
model_weights = torch.load(weight_path)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define huyperparameters
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

# Load data
valid_dataloader = get_dataloader(valid_data_dir, batch_size, roi, 'valid')

# Inference
model_infer = partial(
    sliding_window_inference,
    roi_size=[roi[0], roi[1], roi[2]],
    sw_batch_size=1,
    predictor=model,
    overlap=0.6
)

with torch.no_grad():
    for batch_data in valid_dataloader:
        image = batch_data['image'].to(device)
        gt = batch_data['label'].to(device)
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

        break

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


fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(first_chanel_vol[:, :, z_max], cmap='gray')
ax[0].set_title('Input image')
ax[1].imshow(label_back2og[:, :, z_max], cmap='gray')
ax[1].set_title('Ground truth')
ax[2].imshow(seg_out[:, :, z_max], cmap='gray')
ax[2].set_title('Prediction')
plt.savefig('swin_inference.png')


print('Inference done!')
