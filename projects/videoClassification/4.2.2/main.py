import torch.nn as nnn
import pandas as pd
import matplotlib as plt
import numpy as np
from optical_model import OpticalStream
from temporal_model import TemporalStream

LEACKEAGE_DATASET_PATH = '/dtu/datasets1/02516/ucf101'
NO_LEACKEAGE_DATASET_PATH = '/dtu/datasets1/02516/ucf101_noleakage'


def fuse_prediction(optical_prediction, temporal_prediction):
    return optical_prediction+temporal_prediction / 2

def main():
    optical_model = OpticalStream(num_classes=101)
    sample = torch.randn(1, 2, 224, 224)
    out = optical_model(sample)
    print(out.shape)
if __name_ == "__main__":
    main() 