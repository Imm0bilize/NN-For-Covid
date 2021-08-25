import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset
from model import AutoEncoder
from metrics import dice_coeff


IN_CHANNEL = 1
LR_RATE = 0.5e-5
BATCH_SIZE = 16
NUM_EPOCHS = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = Dataset(path_to_data="dataset/train.npy")
val_ds = Dataset(path_to_data="dataset/val.npy")

train_data_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

model = AutoEncoder()
model.to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)


def train_step():
    losses = []
    for batch_idx, data in enumerate(tqdm(train_data_loader)):
        data = data.to(device=device)

        predictions = model(data)

        loss = criterion(predictions, data)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"TRAIN MEAN LOSS: {np.mean(losses)}")


def val_step(epoch):
    losses = []
    with torch.no_grad():
        for batch_idx, data in enumerate(val_data_loader):
            data = data.to(device=device)
            predictions = model(data)

            loss = criterion(predictions, data)
            losses.append(loss.item())

    print(f"VAL LOSS FOR EPOCH {epoch}: {np.mean(losses)}, DICE: {dice_coeff(predictions, data)}")


for epoch in range(NUM_EPOCHS):
    train_step()
    val_step(epoch)
torch.save(model, "autoencoder.pth")
