import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from PIL import Image
import math

import os
import pandas as pd
from torchvision.io import decode_image

from model import PAWDEL
from paw_dataset import PawImageDataset

current_path = os.getcwd()
csv_train_dataset_path = os.path.join(current_path, "data/raw/petfinder-pawpularity-score/train.csv")
image_train_dataset_path = os.path.join(current_path, "data/raw/petfinder-pawpularity-score/train")

def compute_mean_std(dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    mean = 0.
    std = 0.
    n_samples = 0.

    for images, _, _ in loader:
        # images shape: [B, C, H, W], values in [0,1]
        b = images.size(0)
        images = images.view(b, images.size(1), -1)  # [B, C, H*W]
        
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_samples += b

    mean /= n_samples
    std /= n_samples

    print(mean, std)
    # tensor([0.5146, 0.4771, 0.4380]) tensor([0.2304, 0.2283, 0.2280])

    return mean.tolist(), std.tolist()

paw_dataset = PawImageDataset(csv_train_dataset_path, image_train_dataset_path)

train_size = int(len(paw_dataset) * 0.8)
test_size = len(paw_dataset) - train_size

train_dataset, test_dataset = random_split(paw_dataset, [train_size, test_size])


stats_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),  # if you use this in val/test
    transforms.ToTensor(),
])

# train_dataset.dataset.transform = stats_tf
# mean, std = compute_mean_std(train_dataset, 64)

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # )
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5146, 0.4771, 0.4380],
                std=[0.2304, 0.2283, 0.2280]),
])

transform_test = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # )
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5146, 0.4771, 0.4380],
                std=[0.2304, 0.2283, 0.2280]),
])


train_dataset.dataset.transform = transform
test_dataset.dataset.transform = transform_test

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
pawdel = PAWDEL(use_SI=True).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(pawdel.parameters())

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    for batch, (X1, X2, y) in enumerate(dataloader):
        X1, X2, y = X1.to(device), X2.to(device), y.to(device)
        # Compute prediction error
        pred = model(X2, X1)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 12 == 0:
            loss, current = loss.item(), (batch + 1) * len(X1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()

    total_loss = 0.0
    total_abs_error = 0.0
    total_samples = 0
    total_sq_error = 0.0

    with torch.no_grad():
        for X1, X2, y in dataloader:
            X1, X2, y = X1.to(device), X2.to(device), y.to(device)

            pred = model(X2, X1).squeeze(1)   # [B]
            y = y.float()

            loss = loss_fn(pred, y)
            total_loss += loss.item()

            total_abs_error += torch.sum(torch.abs(pred - y)).item()
            total_sq_error += torch.sum((pred - y) ** 2).item()
            total_samples += y.size(0)

    avg_loss = total_loss / num_batches
    mae = total_abs_error / total_samples
    rmse = math.sqrt(total_sq_error / total_samples)

    print(f"Test MAE: {mae:.3f} | Avg loss: {avg_loss:.6f}")
    print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f}")

    return avg_loss

mae_list = []
t = 0
mae_not_better = 0
train_threshold = 5
train_count = train_threshold
min_mae = float('inf')
best_pawdel = None
while True:
    train_count -= 1
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, pawdel, loss_fn, optimizer)
    mae = test(test_loader, pawdel, loss_fn)
    t += 1
    if mae < min_mae:
        train_count = train_threshold
        min_mae = mae
        best_pawdel = pawdel.state_dict()
    elif train_count == 0:
        break
print("Done!")

torch.save(best_pawdel, "model/pawdel_new_train_1.pth")
print("Saved PyTorch Model State to model.pth")