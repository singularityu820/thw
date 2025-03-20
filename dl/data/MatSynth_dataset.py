import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch
from torch.utils import data as data

from dl.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MatSynthDataset(data.Dataset):
    def __init__(self, opt):
        super(MatSynthDataset, self).__init__()
        self.opt = opt
        self.root = opt["svbrdf_root"]
        svbrdf_names = os.listdir(self.root)
        svbrdf_names.sort()
        self.svbrdf_paths = [os.path.join(self.root, svbrdf_name) for svbrdf_name in svbrdf_names]

    def __len__(self):
        return len(self.svbrdf_paths)

    def __getitem__(self, idx):
        svbrdf = cv2.imread(self.svbrdf_paths[idx])[:, 512:, ::-1] / 255.0
        svbrdf = torch.from_numpy(svbrdf).permute(2, 0, 1).float()
        return {
            "svbrdf": svbrdf,
            "name": os.path.basename(self.svbrdf_paths[idx]),
        }
