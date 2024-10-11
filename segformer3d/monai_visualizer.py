import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import imageio
from PIL import Image
import matplotlib.pyplot as plt

def get_vol_data(vol_path):
    """
    Load and return 3D volume data from a NIfTI file.

    Parameters:
    - vol_path: Path to the NIfTI file.

    Returns:
    - vol_data: 3D numpy array of volume data.
    """
    vol = nib.load(vol_path)
    vol_data = vol.get_fdata()
    return vol_data

def check_data(pred_paths_dict):
    """
    Check that all ground truth and prediction volumes have the same shape.

    Parameters:
    - pred_paths_dict: Dictionary with paths to ground truth and prediction volumes.
    """
    with tqdm(total=len(pred_paths_dict)) as pbar:
        for folder in pred_paths_dict:
            gt_path = pred_paths_dict[folder][0]
            pred_path = pred_paths_dict[folder][1]
            if not os.path.exists(gt_path):
                print(f"Ground truth file {gt_path} not found.")
                continue
            if not os.path.exists(pred_path):
                print(f"Prediction file {pred_path} not found.")
                continue
            gt_data = get_vol_data(gt_path)
            pred_data = get_vol_data(pred_path)
            assert gt_data.shape == pred_data.shape, f'Error: {folder} has different shape'
            pbar.update(1)
    print('All data has the same shape!')

def dice_score_per_label(gt_data, pred_data, label):
    """
    Calculate the Dice Score for a specific label.

    Parameters:
    - gt_data: Ground truth volume data.
    - pred_data: Predicted volume data.
    - label: Label to calculate the Dice Score for.

    Returns:
    - dice: Dice Score for the given label.
    """
    gt_label = (gt_data == label).astype(np.float32)
    pred_label = (pred_data == label).astype(np.float32)
    
    intersection = np.sum(gt_label * pred_label)
    union = np.sum(gt_label) + np.sum(pred_label)
    
    if union == 0:
        return 1.0  # Handle case where no instances are present
    dice = 2 * intersection / union
    return dice

def mean_iou_score_per_label(gt_data, pred_data, label):
    """
    Calculate the mean Intersection over Union (IoU) for a specific label.

    Parameters:
    - gt_data: Ground truth volume data.
    - pred_data: Predicted volume data.
    - label: Label to calculate the IoU Score for.

    Returns:
    - iou: IoU Score for the given label.
    """
    gt_label = (gt_data == label).astype(np.float32)
    pred_label = (pred_data == label).astype(np.float32)
    
    intersection = np.sum(gt_label * pred_label)
    union = np.sum(gt_label) + np.sum(pred_label) - intersection
    
    if union == 0:
        return 1.0  # Handle case where no instances are present
    iou = intersection / union
    return iou

def get_dice_score(pred_paths_dict, labels):
    """
    Compute Dice Scores for each label across all volumes.

    Parameters:
    - pred_paths_dict: Dictionary with paths to ground truth and prediction volumes.
    - labels: List of labels to calculate Dice Scores for.
    """
    total_dice = {label: 0 for label in labels}
    with tqdm(total=len(pred_paths_dict)) as pbar:
        for folder in pred_paths_dict:
            gt_path = pred_paths_dict[folder][0]
            pred_path = pred_paths_dict[folder][1]
            if not os.path.exists(gt_path):
                print(f"Ground truth file {gt_path} not found.")
                continue
            if not os.path.exists(pred_path):
                print(f"Prediction file {pred_path} not found.")
                continue
            gt_data = get_vol_data(gt_path)
            pred_data = get_vol_data(pred_path)
            
            for label in labels:
                dice = dice_score_per_label(gt_data, pred_data, label)
                total_dice[label] += dice
            
            pbar.update(1)
    
    mean_dice_score = {label: total_dice[label] / len(pred_paths_dict) for label in labels}
    total_mean_dice_score = np.mean(list(mean_dice_score.values()))    
    print(f'Average Dice Score: {total_mean_dice_score}')
    print(f'Dice Score per label: {mean_dice_score}')

def get_miou_score(pred_paths_dict, labels):
    """
    Compute mean IoU Scores for each label across all volumes.

    Parameters:
    - pred_paths_dict: Dictionary with paths to ground truth and prediction volumes.
    - labels: List of labels to calculate IoU Scores for.
    """
    total_miou = {label: 0 for label in labels}
    with tqdm(total=len(pred_paths_dict)) as pbar:
        for folder in pred_paths_dict:
            gt_path = pred_paths_dict[folder][0]
            pred_path = pred_paths_dict[folder][1]
            if not os.path.exists(gt_path):
                print(f"Ground truth file {gt_path} not found.")
                continue
            if not os.path.exists(pred_path):
                print(f"Prediction file {pred_path} not found.")
                continue
            gt_data = get_vol_data(gt_path)
            pred_data = get_vol_data(pred_path)
            
            for label in labels:
                miou = mean_iou_score_per_label(gt_data, pred_data, label)
                total_miou[label] += miou
            
            pbar.update(1)
    mean_miou_score = {label: total_miou[label] / len(pred_paths_dict) for label in labels}
    total_mean_miou_score = np.mean(list(mean_miou_score.values()))
    print(f'Average IoU Score: {total_mean_miou_score}')
    print(f'IoU Score per label: {mean_miou_score}')

def save_visualization(flair_data, gt_data, pred_data, folder):
    """
    Save visualizations of ground truth and prediction volumes with FLAIR volume as the base.

    Parameters:
    - flair_data: FLAIR volume data.
    - gt_data: Ground truth volume data.
    - pred_data: Predicted volume data.
    - folder: Path to save visualizations.
    """
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Initialize lists to hold frames for GIFs
    gt_frames = []
    pred_frames = []
    
    # Find the slice with the highest GT sum (most tumor data)
    max_gt_slice_idx = np.argmax([np.sum(gt_slice) for gt_slice in gt_data])
    
    for i in range(gt_data.shape[0]):
        # Create a 3-channel image from FLAIR data
        flair_image = np.stack([flair_data[i]] * 3, axis=-1)  # Convert to 3-channel image
        
        # Initialize images for ground truth and prediction, using FLAIR as the base
        gt_image = flair_image.copy()
        pred_image = flair_image.copy()
        
        # Apply colors for labels with transparency
        gt_overlay = np.zeros_like(flair_image)
        pred_overlay = np.zeros_like(flair_image)
        
        # Define colors for labels
        colors = {
            1: [0, 255, 255], # Cyan for label 1
            2: [255, 255, 0], # Yellow for label 2
            4: [255, 0, 0] # Red for label 3
        }
        
        # Apply colors for ground truth
        for label, color in colors.items():
            gt_overlay[gt_data[i] == label] = color
        
        # Apply colors for prediction
        for label, color in colors.items():
            pred_overlay[pred_data[i] == label] = color
        
        # Blend the overlay with the FLAIR image
        alpha = 0.5  # Transparency factor
        gt_image = np.clip(flair_image * (1 - alpha) + gt_overlay * alpha, 0, 255).astype(np.uint8)
        pred_image = np.clip(flair_image * (1 - alpha) + pred_overlay * alpha, 0, 255).astype(np.uint8)
        
        # Save the image for the slice with the highest GT value
        if i == max_gt_slice_idx:
            gt_img = Image.fromarray(gt_image)
            pred_img = Image.fromarray(pred_image)
            gt_img.save(os.path.join(folder, 'gt_image.png'))
            pred_img.save(os.path.join(folder, 'pred_image.png'))
        
        gt_frames.append(gt_image)
        pred_frames.append(pred_image)
    
    # Save the GIFs
    imageio.mimsave(os.path.join(folder, 'gt_volume.gif'), gt_frames, duration=0.2)
    imageio.mimsave(os.path.join(folder, 'pred_volume.gif'), pred_frames, duration=0.2)
    
    # Load the saved static images
    gt_static = Image.open(os.path.join(folder, 'gt_image.png'))
    pred_static = Image.open(os.path.join(folder, 'pred_image.png'))

    # Convert to arrays for plotting
    gt_static = np.array(gt_static)
    pred_static = np.array(pred_static)

    # Plot subplots for the static images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gt_static)
    axes[0].set_title('GT - higher mask area slice')
    axes[0].axis('off')

    axes[1].imshow(pred_static)
    axes[1].set_title('Prediction - higher mask area slice')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'visual_comparison.png'))
    plt.show()


if __name__ == '__main__':
    # Define paths to volumes and labels
    this_path = os.path.abspath(__file__)
    parent_path = os.path.dirname(this_path)
    data_path = os.path.join(parent_path, 'data3D', 'train_data')
    pred_path = os.path.join(parent_path, 'eval')
    
    data_folders = os.listdir(data_path)
    pred_folders = os.listdir(pred_path)

    vol_folder_lt = []
    for folder in pred_folders:
        split = folder.split('_')
        final_folder_name = split[0] + '_' + split[1] + '_' + split[2] + '_' + split[3]
        vol_folder_lt.append(final_folder_name)

    pred_paths_dict = {}

    for folder in data_folders:
        if folder in vol_folder_lt:
            flair_path = os.path.join(data_path, folder, f'{folder}_flair.nii')
            gt_path = os.path.join(data_path, folder, f'{folder}_seg.nii')
            if os.path.exists(gt_path):
                pred_paths = f'eval/{folder}_t1ce/{folder}_t1ce_seg.nii.gz'
                if os.path.exists(pred_paths):
                    pred_paths_dict[folder] = [gt_path, pred_paths, flair_path]

    labels = [0, 1, 2, 4]  # 0: background, 1: TC tumor subregion, 2: WT tumor subregion, 4: ET tumor subregion

    # Check data consistency
    check_data(pred_paths_dict)
    
    # Compute and print Dice and IoU scores
    get_dice_score(pred_paths_dict, labels)
    get_miou_score(pred_paths_dict, labels)
    
    # Save visualizations
    visualization_path = os.path.join(parent_path, 'visualizations')
    monai_path = os.path.join(visualization_path, 'monai')
    if not os.path.exists(monai_path):
        os.makedirs(monai_path)

    with tqdm(total=len(pred_paths_dict)) as pbar:
        for folder in pred_paths_dict:
            gt_path = pred_paths_dict[folder][0]
            pred_path = pred_paths_dict[folder][1]
            flair_path = pred_paths_dict[folder][2]
            gt_data = get_vol_data(gt_path)
            pred_data = get_vol_data(pred_path)
            flair_data = get_vol_data(flair_path)
            save_visualization(flair_data, gt_data, pred_data, os.path.join(monai_path, folder))
            pbar.update(1)
