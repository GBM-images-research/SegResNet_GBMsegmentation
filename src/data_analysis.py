from src import config
import nibabel as nib
import os
import glob
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt


def load_MRI(data_path: str, idx: int, modality: str) -> nib.nifti1.Nifti1Image:
    """Load a MRI volume of modality 'Mod' from the case 'case'
    from the path directory dir. \n
    Arg:
         data_path(str): path directory to data
         idx(int): number of the pacient case
         modality(str): modality( '_T1', '_T1GD', '_T2', '_FLAIR', '_segm' ...)
     return:
         MRI(Nifti1Image): MRI volume + header
    """
    if modality != "_segm":  # Do for no labels MRI
        case = os.listdir(data_path)[idx]
        mypath = os.path.join(data_path, case)
        pattern = f"*{modality}*"
        MRI_file = glob.glob(os.path.join(mypath, pattern))[0]
    else:  # Do for label images
        file = os.listdir(data_path)
        MRI_file = os.path.join(data_path, file[idx])
    MRI = nib.load(MRI_file)
    return MRI


def show_mri_info(MRI, modality: str):
    """
    Show descriptive features of the MRI
    """
    MRI_data = MRI.get_fdata()
    print("Modality: ", modality)
    print("Size:", MRI_data.shape)
    print("Data type:", MRI_data.dtype)
    print("Min:", np.min(MRI_data))
    print("Max:", np.max(MRI_data))
    print("Mean:", np.mean(MRI_data))
    print("Std:", np.std(MRI_data))
