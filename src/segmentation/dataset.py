import torch
import torchvision
import numpy as np
import pydicom as dicom
from glob import glob


class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths_to_x, paths_to_y, use_transform):
        self.paths_to_x = sorted(glob(f"{paths_to_x}/*/*.dcm"))
        self.paths_to_y = sorted(glob(f"{paths_to_y}/*/*.png"))

        self.length = len(self.paths_to_x)
        self.use_transform = use_transform

        self.min_bound = -1000
        self.max_bound = 400

    def __len__(self):
        return self.length

    def _prepare_data(self, data, is_dicom):
        data = data.float()
        if is_dicom:
            data[data < -1024.0] = 0.0
            data = (data - self.min_bound) / (self.max_bound - self.min_bound)
            data[data > 1.0] = 1.
            data[data < 0.0] = 0.
        else:
            data = data / 255.0
            data = torch.where(data > 0.05, 1.0, 0.0)
        return data

    def _download_file(self, path, is_dicom):
        if is_dicom:
            data = dicom.read_file(path).pixel_array
            data = torch.from_numpy(data)
            data = torch.unsqueeze(data, dim=0)
        else:
            data = torchvision.io.read_file(path)
            data = torchvision.io.decode_png(data)
            if data.shape[0] == 3:
                data = torchvision.transforms.functional.rgb_to_grayscale(data)
        return self._prepare_data(data, is_dicom)

    def _apply_transform(self, x, y):
        # if np.random.rand() > 0.5:
        #     x = torchvision.transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.5))
        if np.random.rand() > 0.5:
            angle = np.random.randint(-90, 90)
            x = torchvision.transforms.functional.rotate(x, angle)
            y = torchvision.transforms.functional.rotate(y, angle)
        return x, y

    def __getitem__(self, index):
        x = self._download_file(self.paths_to_x[index], is_dicom=True)
        y = self._download_file(self.paths_to_y[index], is_dicom=False)
        if self.use_transform:
            x, y = self._apply_transform(x, y)
        return x, y
