import torch
from torch.utils.data import Dataset

import os
from PIL import Image
import pandas as pd
import numpy as np

class PawImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".jpg")
        image = Image.open(img_path).convert("RGB")
        side_info = self.img_labels.iloc[idx, 1:-1].to_numpy(dtype="float32")
        side_info = torch.tensor(side_info, dtype=torch.float32)
        score = self.img_labels.iloc[idx, -1] / 100
        score = torch.tensor(score, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            score = self.target_transform(score)
        return image, side_info, score