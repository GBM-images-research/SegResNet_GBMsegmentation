import os
from pathlib import Path


DATASET = str(Path(__file__).parent.parent / "Dataset")
if not os.path.exists(DATASET):
    os.mkdir(DATASET)

WEIGHTS = str(Path(__file__).parent.parent / "Weights")
if not os.path.exists(WEIGHTS):
    os.mkdir(WEIGHTS)

# MODEL_PATH = str(Path(__file__).parent.parent / "Data/Model")

# DATA_TRAIN = str(Path(__file__).parent.parent / "Data/Data_train")
# LABEL_TRAIN = str(Path(__file__).parent.parent / "Data/Label_train")

# DATA_TEST = str(Path(__file__).parent.parent / "Data/Data_test")
# LABEL_TEST = str(Path(__file__).parent.parent / "Data/Label_test")

# RAW_DATA_SET = str(Path(__file__).parent.parent / "Data/Raw_data_set")

# CASE_TEST = str(Path(__file__).parent.parent / "Data/Case_test")
