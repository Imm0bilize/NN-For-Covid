from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary

from dataset import Dataset
from model import UNet
from test_model import AttU_Net
from metrics import dice_coef


IN_CHANNEL = 1
LR_RATE = 1e-3
BATCH_SIZE = 8
NUM_EPOCHS = 60


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_ds = Dataset(paths_to_x="COVID19_1110/train/studies",
                   paths_to_y="COVID19_1110/train/masks",
                   use_transform=True)
val_ds = Dataset(paths_to_x="COVID19_1110/val/studies",
                 paths_to_y="COVID19_1110/val/masks",
                 use_transform=False)

train_data_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_data_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

# model = UNet()
model = AttU_Net()
model.to(device)

# summary(model, (1, 512, 512))



# def apply_pretrained_weights(model, device, path='classifier_88.pth'):
#     new_state = OrderedDict()
#     state_dict = torch.load(path, map_location=device)
#     for k1, k2 in zip(model.state_dict().keys(), state_dict.keys()):
#         if k2.find("encoder") != -1:
#             new_state[k1] = state_dict[k2]
#         else:
#             break
#     model.load_state_dict(new_state, strict=False)
#
#
# apply_pretrained_weights(model, device)


criterion = torch.nn.BCELoss()
criterion = criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


def train_step():
    losses = 0
    dice = 0
    model.train()
    for batch_idx, (data, label) in enumerate(tqdm(train_data_loader, unit=" batch")):
        data = data.to(device=device)
        label = label.to(device)
        predictions = model(data)
        loss = criterion(predictions, label)
        losses += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dice += dice_coef(predictions, label)
    print(f"TRAIN MEAN LOSS: {losses / len(train_ds):.4f}, ACC: {dice / len(train_ds):.4f}")


def save_model(last_metric, metric):
    print(f"VAL DICE: {last_metric:.5f} -> {metric:.5f}, save model...")
    torch.save(model.state_dict(), "classifier_test.pth")


def val_step():
    losses = 0
    dice = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_data_loader):
            data = data.to(device=device)
            label = label.to(device)
            predictions = model(data)
            loss = criterion(predictions, label)
            losses += loss.item()
            dice += dice_coef(predictions, label)
    print(f"VAL MEAN LOSS: {losses / len(val_ds):.4f}, DICE: {dice / len(val_ds):.4f}")
    return dice / len(val_ds)


last_acc = 0

for epoch in range(NUM_EPOCHS):
    print(f"EPOCH: {epoch+1}/{NUM_EPOCHS}, LR: {scheduler.get_last_lr()[0]}")
    train_step()
    acc = val_step()
    if acc > last_acc:
        save_model(last_acc, acc)
        last_acc = acc
    # scheduler.step()
    print("\n")
print(f"BEST VAL ACC: {last_acc*100:0.2f}%")