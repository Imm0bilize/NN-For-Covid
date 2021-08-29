import torch
import numpy as np
import torchvision.models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset
from model import Classifier


IN_CHANNEL = 1
LR_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.transpose(1,2,0)),
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.transpose(1,2,0)),
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

train_ds = Dataset(path_to_data="train.npz", transform=train_transform)
val_ds = Dataset(path_to_data="val.npz")

train_data_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

model = Classifier()
model.to(device)
state_dict = torch.load('autoencoder.pth', map_location=device)
model.load_state_dict(state_dict, strict=False)


weight = torch.Tensor([2.3964, 1.0]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weight)
criterion = criterion.to(device)
optimizer = torch.optim.Adamax(model.parameters(), lr=LR_RATE, weight_decay=5e-2)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)


def train_step():
    losses = []
    acc = 0
    model.train()
    for batch_idx, (data, label) in enumerate(tqdm(train_data_loader)):
        data = data.to(device=device)
        label = label.to(device).squeeze()
        predictions = model(data)
        loss = criterion(predictions, label)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc += (torch.argmax(predictions, 1) == label).float().sum()
    print(f"TRAIN MEAN LOSS: {np.mean(losses):.4f}, ACC: {acc / len(train_ds):.4f}")


def save_model(last_metric, metric):
    print(f"VAL ACC: {last_metric:.5f} -> {metric:.5f}, save model...")
    torch.save(model.state_dict(), "classifier_test.pth")


def val_step():
    losses = []
    acc = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_data_loader):
            data = data.to(device=device)
            label = label.to(device).squeeze()
            predictions = model(data)
            loss = criterion(predictions, label)
            acc += (torch.argmax(predictions, 1) == label).float().sum()
            losses.append(loss.item())

    print(f"VAL MEAN LOSS: {np.mean(losses):.4f}, ACC: {acc / len(val_ds):.4f}")
    return acc / len(val_ds)


last_acc = 0

for epoch in range(NUM_EPOCHS):
    print(f"EPOCH: {epoch+1}/{NUM_EPOCHS}, LR: {scheduler.get_last_lr()[0]}")
    train_step()
    acc = val_step()
    if acc > last_acc:
        save_model(last_acc, acc)
        last_acc = acc
    scheduler.step()
    print("\n")
print(f"BEST VAL ACC: {last_acc*100:0.2f}%")