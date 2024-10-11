"""
This module contains classes and functions for volume preprocessing in the context of the BRATS2018 dataset.

Classes:
- ConvertToMultiChannelBasedOnBrats2018Classes: Converts labels to multi channels based on the BRATS2018 classes.
- Brats2018Task1Preprocess: Preprocesses the BRATS2018 dataset for Task 1.

Functions:
- animate: Animates pairs of image sequences on two conjugate axes.

Note: This code is based on the segformer3d implementation from the official paper repository: https://github.com/OSUPCVLab/SegFormer3D/blob/main/data/brats2017_seg/brats2021_raw_data/brats2021_seg_preprocess.py
"""

import os
import torch
import nibabel
import numpy as np
from monai.data import MetaTensor
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler
from monai.transforms import (
    Orientation,
    EnsureType,
    ConvertToMultiChannelBasedOnBratsClasses,
)
import pandas as pd

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


class Brats2018Preprocessor:
    def __init__(
        self,
        data_dir: str,
        split_folder_name: str,
        save_dir: str,
    ):
        """
        root_dir: path to the data folder where the raw train folder is
        train_folder_name: name of the folder of the training data
        save_dir: path to directory where each case is going to be saved as a single file containing four modalities
        """
        self.split_folder_dir = os.path.join(data_dir, split_folder_name)
        assert os.path.exists(self.split_folder_dir)
        # walking through the raw training data and list all the folder names, i.e. case name
        self.case_name = next(os.walk(self.split_folder_dir), (None, None, []))[1]

        # MRI type
        self.MRI_TYPE = ["flair", "t1", "t1ce", "t2", "seg"]
        self.save_dir = os.path.join(save_dir, split_folder_name)

    def __len__(self):
        return self.case_name.__len__()

    def get_modality_fp(self, case_name: str, mri_type: str) -> str:
        """
        return the modality file path
        case_name: patient ID
        mri_type: any of the ["flair", "t1", "t1ce", "t2", "seg"]
        """
        modality_fp = os.path.join(
            self.split_folder_dir,
            case_name,
            case_name + f"_{mri_type}.nii",
        )
        return modality_fp

    def load_nifti(self, fp) -> list:
        """
        load a nifti file
        fp: path to the nifti file with (nii or nii.gz) extension
        """
        nifti_data = nibabel.load(fp)
        # get the floating point array
        nifti_scan = nifti_data.get_fdata()
        # get affine matrix
        affine = nifti_data.affine
        return nifti_scan, affine

    def normalize(self, x: np.ndarray) -> np.ndarray:
        # Transform features by scaling each feature to a given range.
        scaler = MinMaxScaler(feature_range=(0, 1))
        # (H, W, D) -> (H * W, D)
        normalized_1D_array = scaler.fit_transform(x.reshape(-1, x.shape[-1]))
        normalized_data = normalized_1D_array.reshape(x.shape)
        return normalized_data

    def orient(self, x: MetaTensor) -> MetaTensor:
        # orient the array to be in (Right, Anterior, Superior) scanner coordinate systems
        assert type(x) == MetaTensor
        return Orientation(axcodes="RAS")(x)

    def detach_meta(self, x: MetaTensor) -> np.ndarray:
        assert type(x) == MetaTensor
        return EnsureType(data_type="numpy", track_meta=False)(x)

    def crop_brats2018_zero_pixels(self, x: np.ndarray) -> np.ndarray:
        # get rid of the zero pixels around mri scan and cut it so that the region is useful
        # crop (1, 240, 240, 155) to (1, 128, 128, 128)
        return x[:, 56:184, 56:184, 13:141]

    def preprocess_brats_modality(
        self, data_fp: str, is_label: bool = False
    ) -> np.ndarray:
        """
        apply preprocess stage to the modality
        data_fp: directory to the modality
        """
        data, affine = self.load_nifti(data_fp)
        # label do not the be normalized
        if is_label:
            # Binary mask does not need to be float64! For saving storage purposes!
            data = data.astype(np.uint8)
            # categorical -> one-hot-encoded
            # (240, 240, 155) -> (3, 240, 240, 155)
            data = ConvertToMultiChannelBasedOnBratsClasses()(data)
        else:
            data = self.normalize(x=data)
            # (240, 240, 155) -> (1, 240, 240, 155)
            data = data[np.newaxis, ...]

        data = MetaTensor(x=data, affine=affine)
        # for oreinting the coordinate system we need the affine matrix
        data = self.orient(data)
        # detaching the meta values from the oriented array
        data = self.detach_meta(data)
        # (240, 240, 155) -> (128, 128, 128)
        data = self.crop_brats2018_zero_pixels(data)
        return data

    def __getitem__(self, idx):
        case_name = self.case_name[idx]
        # e.g: train/BraTS2018_00000/BraTS2018_00000_flair.nii.gz

        # preprocess Flair modality
        FLAIR = self.get_modality_fp(case_name, self.MRI_TYPE[0])
        flair = self.preprocess_brats_modality(data_fp=FLAIR, is_label=False)
        flair_transv = flair.swapaxes(1, 3)  # transverse plane

        # # preprocess T1 modality
        T1 = self.get_modality_fp(case_name, self.MRI_TYPE[1])
        t1 = self.preprocess_brats_modality(data_fp=T1, is_label=False)
        t1_transv = t1.swapaxes(1, 3)  # transverse plane

        # preprocess T1ce modality
        T1ce = self.get_modality_fp(case_name, self.MRI_TYPE[2])
        t1ce = self.preprocess_brats_modality(data_fp=T1ce, is_label=False)
        t1ce_transv = t1ce.swapaxes(1, 3)  # transverse plane

        # preprocess T2
        T2 = self.get_modality_fp(case_name, self.MRI_TYPE[3])
        t2 = self.preprocess_brats_modality(data_fp=T2, is_label=False)
        t2_transv = t2.swapaxes(1, 3)  # transverse plane

        # preprocess segmentation label
        Label = self.get_modality_fp(case_name, self.MRI_TYPE[4])
        label = self.preprocess_brats_modality(data_fp=Label, is_label=True)
        label_transv = label.swapaxes(1, 3)  # transverse plane

        # stack modalities along the first dimension
        modalities = np.concatenate(
            (flair_transv, t1_transv, t1ce_transv, t2_transv),
            axis=0,
        )
        label = label_transv
        return modalities, label, case_name

    def __call__(self):
        print("started preprocessing brats2018...")
        with Pool(processes=os.cpu_count()) as multi_p:
            multi_p.map_async(func=self.process, iterable=range(self.__len__()))
            multi_p.close()
            multi_p.join()
        print("finished preprocessing brats2018...")

    def process(self, idx):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # get the 4D modalities along with the label
        modalities, label, case_name = self.__getitem__(idx)
        # creating the folder for the current case id
        data_save_path = os.path.join(self.save_dir, case_name)
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
        # saving the preprocessed 4D modalities containing all the modalities to save path
        modalities_fn = data_save_path + f"/{case_name}_modalities.pt"
        torch.save(modalities, modalities_fn)
        # saving the preprocessed segmentation label to save path
        label_fn = data_save_path + f"/{case_name}_label.pt"
        torch.save(label, label_fn)


def val_subset_from_train_data(train_dir, val_dir):
    """
    move some of the training data to validation data
    train_dir: path to the training data
    val_dir: path to the validation data
    """
    os.makedirs(val_dir, exist_ok=True)
    print(len(os.listdir(train_dir)))
    train_case_names = next(os.walk(train_dir), (None, None, []))[1]
    # move 15% of the training data to the validation data
    val_case_names = np.random.choice(
        train_case_names, int(0.15 * len(train_case_names)), replace=False
    )
    for case_name in val_case_names:
        os.rename(os.path.join(train_dir, case_name), os.path.join(val_dir, case_name))

    print(len(os.listdir(train_dir)))
    print(len(os.listdir(val_dir)))


def get_csv_data(data_dir, save_dir):
    """
    get the data from the preprocessed data and save it to a csv file
    data_dir: path to the preprocessed data
    save_dir: path to save the csv file
    """

    case_lt = sorted(next(os.walk(data_dir), (None, None, []))[1])
    paths_cases = [os.path.join(data_dir, case) for case in case_lt]

    df = pd.DataFrame({"data_path": paths_cases, "case_name": case_lt})

    df.to_csv(save_dir, index=False)

    return None


if __name__ == "__main__":
    file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(file_path)
    grandparent_dir = os.path.dirname(parent_dir)
    data_dir = os.path.join(grandparent_dir, "data3D")
    save_dir = os.path.join(grandparent_dir, "preproc_data")

    brats2018TrainDataPreprocessor = Brats2018Preprocessor(
        data_dir=data_dir,
        split_folder_name="train_data",
        save_dir=save_dir,
    )
    # run the preprocessing pipeline
    brats2018TrainDataPreprocessor()

    val_subset_from_train_data(
        os.path.join(save_dir, "train_data"), os.path.join(save_dir, "val_data")
    )

    print("Creating csv files...")

    train_save_dir = os.path.join(save_dir, "train_data")
    val_save_dir = os.path.join(save_dir, "val_data")

    if not os.path.exists(train_save_dir):
        os.makedirs(train_save_dir)
    if not os.path.exists(val_save_dir):
        os.makedirs(val_save_dir)

    get_csv_data(
        os.path.join(save_dir, "train_data"),
        os.path.join(train_save_dir, "train_data.csv"),
    )
    get_csv_data(
        os.path.join(save_dir, "val_data"), os.path.join(val_save_dir, "val_data.csv")
    )

    print("Finished creating csv files...")

    print("Preprocessing finished...")
