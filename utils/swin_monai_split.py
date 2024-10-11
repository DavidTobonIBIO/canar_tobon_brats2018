import os 
import shutil

def split_data(vol_paths, train_rate, save_dir):
    vol_dirs = sorted(os.listdir(vol_paths))  # Get a sorted list of directories/files
    total_vols = len(vol_dirs)
    
    print(f"Total volumes found: {total_vols}")  # Debug info
    train_vols = int(total_vols * train_rate)
    train_files = vol_dirs[:train_vols]
    val_files = vol_dirs[train_vols:]
    
    print(f"Training files: {train_files}")  # Debug info
    print(f"Validation files: {val_files}")  # Debug info
    
    train_data_dir = os.path.join(save_dir, "train_data")
    val_data_dir = os.path.join(save_dir, "val_data")
    
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(val_data_dir, exist_ok=True)
    
    # Copy training data
    for file in train_files:
        src_path = os.path.join(vol_paths, file)
        dst_path = os.path.join(train_data_dir, file)
        print(f"Copying {src_path} to {dst_path}")  # Debug info
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy(src_path, dst_path)
    
    # Copy validation data
    for file in val_files:
        src_path = os.path.join(vol_paths, file)
        dst_path = os.path.join(val_data_dir, file)
        print(f"Copying {src_path} to {dst_path}")  # Debug info
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy(src_path, dst_path)
    
    print(f"Data split into {train_rate * 100}% train and {(1-train_rate) * 100}% val")
