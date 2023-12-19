import os
from glob import glob
from typing import List, Dict

import os
from monai.data import Dataset, DataLoader
from monai.transforms import LoadImaged, Compose
import json

from monai.transforms import (
    Compose,
    Randomizable,
    RandomizableTrait,
    Transform,
    LoadImaged,
    apply_transform,
    convert_to_contiguous,
    reset_ops_id,
)


# todas las subcarpetas finales dentro data_path
def get_final_subfolders(directory):
    final_subfolders = []
    for root, dirs, files in os.walk(directory):
        if not dirs:  # Si no hay subcarpetas, es una carpeta final
            final_subfolders.append(root)
    return final_subfolders


def get_file_data_label(
    data_path: str, label_path: str, modalities: List[str] = []
) -> List[Dict]:
    """Get the list of patient cases path as
    List[Dict]->[{'image':[paths of images], 'label':[paths of labels]}
    Arg:
         data_path(str): path directory, it has a folder by patient with MRIs
         label_path(str): path directory to labels a .nni.gz by patient
         modalities(List[str]): List of modalities to include example ['T1.', 'T1GD.', ...].
         if you want select all just setup modalities = [] or omit it
     return:
         dict_files(List[Dict]):  [{'image':[paths of images], 'label':[paths of labels]}]
    """
    list_data_ = sorted(os.listdir(data_path))
    list_lab_ = sorted(os.listdir(label_path))

    list_data_files = []
    if modalities == []:
        for folder in list_data_:
            data_mri = sorted(glob(os.path.join(data_path, folder, "*.nii.gz")))
            list_data_files.append(data_mri)
    else:
        files = [f"*{mri}*nii.gz" for mri in modalities]
        for folder in list_data_:
            data_mri = []
            for file in files:
                data_mri += glob(os.path.join(data_path, folder, file))
            list_data_files.append(data_mri)

    list_labels_files = []
    for file in list_lab_:
        label_mri = os.path.join(label_path, file)
        list_labels_files.append(label_mri)

    dict_files = []
    for i in range(len(list_labels_files)):
        diccionario = {"image": list_data_files[i], "label": list_labels_files[i]}
        dict_files.append(diccionario)

    return dict_files


##


def get_file_data_label2(
    data_path: str, split: str, modalities: List[str] = []
) -> List[Dict]:
    """Get the list of patient cases path as
    List[Dict]->[{'image':[paths of images], 'label':[paths of labels]}
    Arg:
         data_path(str): path directory, it has a folder by patient with MRIs
         label_path(str): path directory to labels a .nni.gz by patient
         split(str): Dataset split ("train" or "valid")
         modalities(List[str]): List of modalities to include example ['DSC', 'DTI', 'structural'].
         if you want to select all just set modalities = [] or omit it
     return:
         dict_files(List[Dict]):  [{'image':[paths of images], 'label':[paths of labels]}]
    """
    # obtener todas las subcarpetas dentro de data_path
    list_data_ = get_final_subfolders(os.path.join(data_path, split, "images"))

    # list_data_ = sorted(
    #     os.listdir(os.path.join(data_path, split, "images"))
    # )  # Assuming "train" or "valid" is the dataset split
    print(list_data_)

    list_data_files = []

    for modality in modalities:
        for folder in os.listdir(
            os.path.join(data_path, split, "images", f"images_{modality}")
        ):
            data_mri = sorted(
                glob(
                    os.path.join(
                        data_path,
                        split,
                        "images",
                        f"images_{modality}",
                        folder,
                        "*.nii.gz",
                    )
                )
            )
            list_data_files.append(data_mri)
    print(list_data_files)

    list_labels_files = []
    label_path = data_path

    for folder in list_data_:
        label_mri = os.path.join(label_path, split, "labels", f"{folder}.nii.gz")
        list_labels_files.append(label_mri)

    dict_files = []
    for i in range(len(list_labels_files)):
        diccionario = {"image": list_data_files[i], "label": list_labels_files[i]}
        dict_files.append(diccionario)

    return dict_files


###


class CustomDataset(Dataset):
    def __init__(self, root_dir, section="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.section = section
        self.image_files, self.label_files = self._load_files()

    def __len__(self):
        return len(self.image_files)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        image = self.image_files[index]
        label = self.label_files[index]

        if self.transform is not None:
            data = apply_transform(
                self.transform,
                data={"image": image, "label": label},
            )
            # label = apply_transform(self.transform, label)

        return data["image"], data["label"]

    def __getitem__(self, index):
        # image_path = self.image_files[index]
        # label_path = self.label_files[index]
        # image, label = self._load_data(image_path, label_path)
        if self.transform:
            image, label = self._transform(index)
            # print(image.shape, label.shape)
        return {"image": image, "label": label}

    def _load_files(self):
        image_files, label_files = [], []
        section_path = os.path.join(self.root_dir, self.section)

        modalities = ["images_DSC", "images_DTI", "images_structural"]

        for modality in modalities:
            modality_files = []
            modality_path = os.path.join(section_path, "images", modality)

            for n, case_folder in enumerate(os.listdir(modality_path)):
                case_path = os.path.join(modality_path, case_folder)

                # Obtener los archivos de imágenes para cada caso y modalidad
                case_files = {
                    n: [
                        os.path.join(case_path, file)
                        for file in os.listdir(case_path)
                        if file.endswith(".nii.gz")
                    ]
                }

                modality_files.append(case_files)

                # Obtener el archivo de etiqueta correspondiente
                label_file = os.path.join(
                    section_path,
                    "labels",
                    f"{case_folder}_automated_approx_segm.nii.gz",
                )

                # Verificar si el caso ya ha sido procesado
                if label_file not in label_files:
                    label_files.append(label_file)

            image_files.append(modality_files)

        # Lista de listas resultante

        converted_list = [[] for _ in range(len(image_files[0]))]

        for l in image_files:
            for key, values in enumerate(l):
                converted_list[key] += values[key]

        print(f"Found {len(converted_list)} images and {len(label_files)} labels.")
        # print(f"Image files: {converted_list}")
        # print(f"Label files: {label_files}")
        return converted_list, label_files
