import os
import zipfile
import shutil
from tqdm import tqdm

file_path = os.path.abspath(__file__)
parent_path = os.path.dirname(file_path)
grand_parent_path = os.path.dirname(parent_path)
data_path = os.path.join(grand_parent_path, "BraTS_2018.zip")

def unzip3D(data_path: str, output_path: str):
    """
    Unzip the 3D volume data
    data_path: path to the zip file
    output_path: path to the output folder
    """
    print(f"Unzipping data from {data_path} to {output_path}")
    with zipfile.ZipFile(data_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)

def move_files_to_train_data(temp_path: str, train_data_path: str, val_data_path: str):
    """
    Move and rename the files from the HGG and LGG folders to a single folder (train_data).
    temp_path: path where the data is temporarily unzipped
    train_data_path: path to the unified 'train_data' folder
    val_data_path: path to the 'val_data' folder
    """
    print(f"Moving files from {temp_path} to {train_data_path} and {val_data_path}")
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    if not os.path.exists(val_data_path):
        os.makedirs(val_data_path)

    file_count = sum([len(files) for r, d, files in os.walk(temp_path)])
    with tqdm(total=file_count, desc="Processing files") as pbar:
        for root, dirs, files in os.walk(temp_path):
            for file in files:
                if file.endswith(".nii") or file.endswith(".nii.gz"):
                    folder_name = os.path.basename(root)
                    file_path = os.path.join(root, file)

                    # Debugging information
                    print(f"Processing file: {file_path}")

                    train_or_val = file_path.split(os.sep)[-3]
                    if 'Validation' in train_or_val:
                        new_val_data_path = os.path.join(val_data_path, folder_name)
                        if not os.path.exists(new_val_data_path):
                            os.makedirs(new_val_data_path)
                        new_file_path = os.path.join(new_val_data_path, file)
                        print(f"Moving to validation: {new_file_path}")
                        shutil.move(file_path, new_file_path)

                    elif 'Training' in file_path:
                        if 'HGG' in file_path:
                            new_file_name = file.replace("BraTS18_", "BraTS18_HGG_")
                        elif 'LGG' in file_path:
                            new_file_name = file.replace("BraTS18_", "BraTS18_LGG_")
                        
                        
                        new_train_path_name = os.path.join(train_data_path, folder_name)
                        if not os.path.exists(new_train_path_name):
                            os.makedirs(new_train_path_name)
                        new_file_path = os.path.join(new_train_path_name, new_file_name)

                        # Debugging information
                        print(f"Moving to training: {new_file_path}")
                        
                        shutil.move(file_path, new_file_path)
                    pbar.update(1)

if __name__ == "__main__":
    temp_path = os.path.join(grand_parent_path, "temp_data")  
    train_data_path = os.path.join(grand_parent_path, "data3D", "train_data")
    val_data_path = os.path.join(grand_parent_path, "data3D", "val_data")  
    
    # Debugging information
    print(f"Train data path: {train_data_path}")
    print(f"Temp data path: {temp_path}")
    
    if not os.path.exists(train_data_path):
        print(f"Train data path does not exist. Processing data...")

        if not os.path.exists(temp_path):
            print(f"Temp data path does not exist. Unzipping...")
            os.makedirs(temp_path)
            unzip3D(data_path, temp_path)
        else:
            print(f"Temp data path already exists. Skipping unzip...")

        move_files_to_train_data(temp_path, train_data_path, val_data_path)
        
        # Uncomment this line after verifying functionality to remove temporary files
        shutil.rmtree(temp_path)
    else:
        print("Data already processed and stored in train_data.")
