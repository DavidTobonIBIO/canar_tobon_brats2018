import os
import torch
import numpy as np
import os
import torch
import nibabel
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib import animation
from monai.data import MetaTensor
from multiprocessing import Process, Pool
from sklearn.preprocessing import MinMaxScaler
from monai.transforms import (
    Orientation,
    EnsureType,
)


"""
data3D 
│
├───train_data
│      ├──Brats18_2013_0_1
│      │      └──Brats18_2013_0_1_flair.nii.gz
|      |      └──Brats18_2013_0_1_seg.nii.gz
|      |      └──Brats18_2013_0_1_t1.nii.gz
|      |      └──Brats18_2013_0_1_t1ce.nii.gz
|      |      └──Brats18_2013_0_1_t2.nii.gz
│      ├──Brats18_2013_1_1
|      │      └──Brats18_2013_1_1_flair.nii.gz
|      |      └──Brats18_2013_1_1_seg.nii.gz
|      |      └──Brats18_2013_1_1_t1.nii.gz
|      |      └──Brats18_2013_1_1_t1ce.nii.gz
|      |      └──Brats18_2013_1_1_t2.nii.gz
│      └──...
├───val_data
|      ├──Brats18_CBICA_ABT_1
|      │      └──Brats18_CBICA_ABT_1_flair.nii.gz
|      |      └──Brats18_CBICA_ABT_1_seg.nii.gz
|      |      └──Brats18_CBICA_ABT_1_t1.nii.gz
|      |      └──Brats18_CBICA_ABT_1_t1ce.nii.gz
|      |      └──Brats18_CBICA_ABT_1_t2.nii.gz
│      ├──Brats18_CBICA_ABY_1
|      │      └──Brats18_CBICA_ABY_1_flair.nii.gz
|      |      └──Brats18_CBICA_ABY_1_seg.nii.gz
|      |      └──Brats18_CBICA_ABY_1_t1.nii.gz
|      |      └──Brats18_CBICA_ABY_1_t1ce.nii.gz
|      |      └──Brats18_CBICA_ABY_1_t2.nii.gz
│      └──...

"""


class ConvertToMultiChannelBasedOnBrats2017Classes(object):
    """
    Convert labels to multi channels based on brats17 classes:
    "0": "background",
    "1": "edema",
    "2": "non-enhancing tumor",
    "3": "enhancing tumour"
    Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2),
    and the necrotic and non-enhancing tumor (NCR/NET — label 1)
    """

    def __call__(self, img):
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [
            (img == 2) | (img == 3),
            (img == 2) | (img == 3) | (img == 1),
            img == 3,
        ]
        # merge labels 1 (tumor non-enh) and 3 (tumor enh) and 1 (large edema) to WT
        # label 3 is ET
        return (
            torch.stack(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0)
        )


class Brats2018Preprocess:
    def __init__(self, data_dir, folder_name, save_dir_vol, save_dir_label):
        """
        data_dir: Path to data folder where the raw data is.
        folder_name: Name of the folder of the training/validation data.
        save_dir_vol: Path to directory where each case (vol_name) will be saved as a single file containing four modalities.
        save_dir_label: Path to directory where the segmentations are going to be saved.
        """
        self.data_dir = data_dir

        self.save_dir_vol = save_dir_vol
        self.save_dir_label = save_dir_label
        self.case_name = next(os.walk(self.data_dir), (None, None, []))[
            2
        ]  # get all the cases in the folder

        print(self.case_name)

        self.MRI_CODE = {
            "Flair": "0000",
            "T1w": "0001",
            "T1gd": "0002",
            "T2w": "0003",
            "label": None,
        }

    def __len__(self):
        return self.case_name.__len__()

    def normalize(self, x: np.ndarray) -> np.ndarray:
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_array = scaler.fit_transform(x.reshape(-1, x.shape[-1]))
        normalized_array = normalized_array.reshape(x.shape)
        return normalized_array

    def detach_meta(self, x: MetaTensor) -> np.ndarray:
        assert type(x) == MetaTensor
        return EnsureType(data_type="numpy", track_meta=False)(x)

    def get_vol_name(self, vol_path: str) -> str:
        vol_name = vol_path.split(".")[0]
        return vol_name

    def get_modality_fp(self, case_name: str, folder: str, mri_code: str = None):
        if mri_code:
            f_name = f"{case_name}_{mri_code}.nii.gz"
        else:
            f_name = f"{case_name}.nii.gz"

        modality_fp = os.path.join(self.train_folder_dir, folder, f_name)
        return modality_fp

    def load_nii(self, path):
        try:
            vol_data = nibabel.load(path)
            vol_data = vol_data.get_fdata()
            affine = vol_data.affine
            print(f"Loaded {path} successfully with shape {vol_data.shape}")
            return vol_data, affine
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None, None

    def save_as_pt(self, idx):
        try:
            modalities, label, case_name = self.__getitem__(idx)
            vol_save_path = os.path.join(self.save_dir_vol, f"{case_name}_vol.pt")
            label_save_path = os.path.join(self.save_dir_label, f"{case_name}_label.pt")

            torch.save(modalities, vol_save_path)
            torch.save(label, label_save_path)
            print(
                f"Saved {case_name} volume and label to {vol_save_path} and {label_save_path}"
            )
        except Exception as e:
            print(f"Error saving case {self.case_name[idx]}: {e}")

    def crop_brats_2018(self, volume: np.ndarray) -> np.ndarray:
        vol = volume[40:200, 24:216, :]
        return vol

    def _2meta_tensor(self, data, affine):
        scan = MetaTensor(data, affine=affine)
        D, H, W = scan.shape
        scan = scan.view(1, D, H, W)
        return scan

    def prepocess_brats_modality(
        self, data_fp: str, label_flag: bool = False
    ) -> np.ndarray:
        data, affine = self.load_nii(data_fp)

        if label_flag:
            data = data.astype(np.uint8)
            data = ConvertToMultiChannelBasedOnBrats2017Classes()(data)
        else:
            data = self.normalize(x=data)
            data = data[np.newaxis, ...]

        data = MetaTensor(x=data, affine=affine)
        data = self.detach_meta(data)
        data = self.crop_brats_2018(data)

        return data

    def __getitem__(self, idx):
        case_name = self.case_name[idx]
        case_name = self.get_vol_name(case_name)

        code = self.MRI_CODE["Flair"]
        flair = self.get_modality_fp(case_name, folder="imagesTr", mri_code=code)
        flair = self.prepocess_brats_modality(flair)

        code = self.MRI_CODE["T1w"]
        t1w = self.get_modality_fp(case_name, folder="imagesTr", mri_code=code)
        t1w = self.prepocess_brats_modality(t1w)

        code = self.MRI_CODE["T1gd"]
        t1gd = self.get_modality_fp(case_name, folder="imagesTr", mri_code=code)
        t1gd = self.prepocess_brats_modality(t1gd)

        code = self.MRI_CODE["T2w"]
        t2w = self.get_modality_fp(case_name, folder="imagesTr", mri_code=code)
        t2w = self.prepocess_brats_modality(t2w)

        code = self.MRI_CODE["label"]
        label = self.get_modality_fp(case_name, folder="labelsTr", mri_code=code)
        label = self.prepocess_brats_modality(label, label_flag=True)

        modalities = torch.cat([flair, t1w, t1gd, t2w], dim=0)

        return modalities, label, case_name


if __name__ == "__main__":
    file_path = os.path.abspath(__file__)
    parent_path = os.path.dirname(file_path)
    grand_parent_path = os.path.dirname(parent_path)

    train_path = os.path.join(
        grand_parent_path, "data3D", "train_data"
    )
    validation_path = os.path.join(
        grand_parent_path, "data3D", "val_data"
    )

    save_path_train_vol = os.path.join(train_path, "volumes")
    save_path_val_vol = os.path.join(validation_path, "volumes")

    save_path_train_label = os.path.join(train_path, "segmentations")
    save_path_val_label = os.path.join(validation_path, "segmentations")

    if not os.path.exists(save_path_train_vol):
        os.makedirs(save_path_train_vol)

    if not os.path.exists(save_path_val_vol):
        os.makedirs(save_path_val_vol)

    if not os.path.exists(save_path_train_label):
        os.makedirs(save_path_train_label)

    if not os.path.exists(save_path_val_label):
        os.makedirs(save_path_val_label)


    brats2018_train = Brats2018Preprocess(
        data_dir=train_path,
        folder_name="train",
        save_dir_vol=save_path_train_vol,
        save_dir_label=save_path_train_label,
    )

    brats2018_val = Brats2018Preprocess(
        data_dir=validation_path,
        folder_name="val",
        save_dir_vol=save_path_val_vol,
        save_dir_label=save_path_val_label,
    )

    # Loop through all training cases and save them as .pt files
    for idx in range(len(brats2018_train)):
        brats2018_train.save_as_pt(idx)

    # Loop through all validation cases and save them as .pt files
    for idx in range(len(brats2018_val)):
        brats2018_val.save_as_pt(idx)
