from genericpath import exists
import json
import enum
import os
from pathlib import Path


class SourceType(enum.IntEnum):
    LOCAL = 0
    GOOGLESTORAGE = 1
    AWSSTORAGE = 2


class ModelType(enum.IntEnum):
    PT = 0
    TF = 1
    ONNX = 2


input_prefix = "./inputs"
output_prefix = "./outputs"
