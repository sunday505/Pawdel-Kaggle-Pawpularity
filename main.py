import torch
from model import PAWDEL
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from paw_test_dataset import PawTestImageDataset

from tqdm import tqdm

import pandas as pd
import numpy as np
import os

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

pawdel = PAWDEL().to(device)
pawdel.load_state_dict(torch.load("model/pawdel.pth", weights_only=True, map_location=device))

current_path = os.getcwd()
csv_train_dataset_path = os.path.join(current_path, "data/raw/petfinder-pawpularity-score/test.csv")
image_train_dataset_path = os.path.join(current_path, "data/raw/petfinder-pawpularity-score/test")

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # )
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

paw_dataset = PawTestImageDataset(csv_train_dataset_path, image_train_dataset_path, transform=transform)

test_loader = DataLoader(paw_dataset, batch_size=64, shuffle=False)

pred_list = None

pawdel.eval()
with torch.no_grad():
    for X1, X2 in tqdm(test_loader, desc="Predicting"):
        X1, X2 = X1.to(device), X2.to(device)
        pred = pawdel(X2, X1).cpu().numpy()
        if pred_list is None:
            pred_list = pred.copy()
        else:
            pred_list = np.concatenate([pred_list, pred])
            
img_id_col = pd.read_csv(csv_train_dataset_path).iloc[:, 0].values.reshape((-1, 1))

submission = pd.DataFrame(np.concatenate([img_id_col, pred_list * 100], axis=1), columns=["Id", "Pawpularity"])
submission.to_csv("submission.csv", index=False)