import torch
import torch.nn.functional as F
from torch import nn


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batch_norm: bool) -> None:
        super(ConvolutionalBlock, self).__init__()
        config: dict = dict(kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.use_batch_norm: bool = use_batch_norm

        self.conv_0 = nn.Conv2d(in_channels, out_channels, **config)
        self.conv_1 = nn.Conv2d(out_channels, out_channels, **config)
        self.batch_norm_0 = nn.BatchNorm2d(out_channels)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_0(x)
        if self.use_batch_norm:
            x = self.batch_norm_0(x)
        x = F.relu(x, inplace=True)

        x = self.conv_1(x)
        if self.use_batch_norm:
            x = self.batch_norm_1(x)
        x = F.relu(x, inplace=True)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1, start_num_filters: int = 32, use_batch_norm: bool = True) -> None:
        super(Encoder, self).__init__()

        filters = [start_num_filters * 2 ** i for i in range(4)]
        self.conv_block_0 = ConvolutionalBlock(in_channels=in_channels, out_channels=filters[0],
                                               use_batch_norm=use_batch_norm)
        self.conv_block_1 = ConvolutionalBlock(in_channels=filters[0], out_channels=filters[1],
                                               use_batch_norm=use_batch_norm)
        self.conv_block_2 = ConvolutionalBlock(in_channels=filters[1], out_channels=filters[2],
                                               use_batch_norm=use_batch_norm)
        self.conv_block_3 = ConvolutionalBlock(in_channels=filters[2], out_channels=filters[3],
                                               use_batch_norm=use_batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_0(x)
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = self.conv_block_1(x)
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = self.conv_block_2(x)
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = self.conv_block_3(x)
        x = F.max_pool2d(x, kernel_size=(2, 2))
        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.encoder = nn.Sequential(Encoder())

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_features=32 * 32 * 256, out_features=64),
            nn.ReLU(),
            nn.Dropout2d(),

            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Dropout2d(),

            nn.Linear(in_features=16, out_features=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
