import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path_to_data, transform=None):
        self.inputs = np.load(path_to_data)['x']
        self.outputs = np.load(path_to_data)['y']
        self.length = self.inputs.shape[0]
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.inputs[index]
        class_label = self.outputs[index]
        y = torch.LongTensor([class_label])
        if self.transform:
            x = self.transform(x)
        return x, y
