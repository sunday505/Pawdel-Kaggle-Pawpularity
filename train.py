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

from tqdm import tqdm

import matplotlib.pyplot as plt 

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
pawdel = PAWDEL(use_SI=False).to(device)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(pawdel.parameters())

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_log = []
    model.train()
    for X1, X2, y in tqdm(dataloader, desc="Training"):
        X1, X2, y = X1.to(device), X2.to(device), y.to(device)
        # Compute prediction error
        pred = model(X2, X1).squeeze(1)   # [B, 1] -> [B]
        y = y.float()
        loss = loss_fn(pred, y)

        loss_log.append(loss.item())

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    print(f"Training Loss: {(sum(loss_log) / num_batches):.6f}")

    return sum(loss_log) / num_batches

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)

    total_loss = 0.0
    total_abs_error = 0.0
    total_samples = 0
    total_sq_error = 0.0

    model.eval()
    with torch.no_grad():
        for X1, X2, y in tqdm(dataloader, desc="Testing"):
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
    print(f"RMSE: {rmse}")

    return avg_loss, rmse

file_name = "no_SI"

loss_list = []
test_loss_list = []
rmse_list = []
t = 0
train_threshold = 5
train_count = train_threshold
min_rmse = float('inf')

while True:
    train_count -= 1
    print(f"Epoch {t+1}\n-------------------------------")
    loss = train(train_loader, pawdel, loss_fn, optimizer)
    test_loss, rmse = test(test_loader, pawdel, loss_fn)

    loss_list.append(loss)
    test_loss_list.append(test_loss)
    rmse_list.append(rmse)

    print(f"Best Rmse: {min_rmse}")
    print(f"Threshold Left: {train_count}")
    
    if rmse < min_rmse:
        train_count = train_threshold
        min_rmse = rmse
        torch.save(pawdel.state_dict(), f"model/pawdel_{file_name}.pth")
    elif train_count == 0:
        break

    t += 1

result_df = pd.DataFrame({"Train_Loss": loss_list, "Test_Loss": test_loss_list, "RMSE": rmse_list})
result_df.to_csv(f"result/result_df_{file_name}.csv", index=False)

x_index = [i for i in range(len(loss_list))]
fig1, ax1 = plt.subplots()
ax1.plot(x_index, loss_list)
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
fig1.savefig(f"fig/training_loss_{file_name}.png")

fig2, ax2 = plt.subplots()
ax2.plot(x_index, test_loss_list)
ax2.set_title("Testing Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
fig2.savefig(f"fig/testing_loss_{file_name}.png")

fig3, ax3 = plt.subplots()
ax3.plot(x_index, rmse_list)
ax3.set_title("RMSE of Test")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("RMSE")
fig3.savefig(f"fig/test_rmse_{file_name}.png")

print("Done!")