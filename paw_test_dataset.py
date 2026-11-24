import torch
from torch.utils.data import Dataset

import os
from PIL import Image
import pandas as pd
import numpy as np

class PawTestImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".jpg")
        image = Image.open(img_path).convert("RGB")
        # side_info = self.img_labels.iloc[idx, 1:].to_numpy(dtype="float32")
        side_info = self.img_labels.iloc[idx, 1:-1].to_numpy(dtype="float32")
        side_info = torch.tensor(side_info, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, side_info