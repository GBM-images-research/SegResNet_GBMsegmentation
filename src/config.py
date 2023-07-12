import os
from pathlib import Path


MODEL_PATH = str(Path(__file__).parent.parent / "Data/Model")

DATA_TRAIN = str(Path(__file__).parent.parent / "Data/Data_train")
LABEL_TRAIN = str(Path(__file__).parent.parent / "Data/Label_train")

DATA_TEST = str(Path(__file__).parent.parent / "Data/Data_test")
LABEL_TEST = str(Path(__file__).parent.parent / "Data/Label_test")
