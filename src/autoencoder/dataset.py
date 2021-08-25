import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path_to_data, transform=None):
        self.data = np.load(path_to_data)['arr_0']
        self.length = self.data.shape[0]
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = self.data[index]

        if self.transform:
            image = self.transform(image)
        return image
