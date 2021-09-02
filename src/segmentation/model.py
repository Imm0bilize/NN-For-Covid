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


class UpBlock(nn.Module):
    def __init__(self, n_channels_from_prev_layer, n_channels_from_decoder):
        super(UpBlock, self).__init__()

        self.t_conv = nn.ConvTranspose2d(in_channels=n_channels_from_prev_layer,
                                         out_channels=n_channels_from_decoder,
                                         kernel_size=(2, 2), stride=(2, 2))

        self.conv1 = nn.Conv2d(in_channels=n_channels_from_decoder * 2,
                               out_channels=n_channels_from_decoder,
                               kernel_size=(3, 3), padding=(1, 1))

        self.conv2 = nn.Conv2d(in_channels=n_channels_from_decoder,
                               out_channels=n_channels_from_decoder,
                               kernel_size=(3, 3), padding=(1, 1))

    def forward(self, value_from_prev_layer, value_from_decoder):
        up = self.t_conv(value_from_prev_layer)

        x = torch.cat([up, value_from_decoder], dim=1)

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, start_num_filters: int = 32):
        super(UNet, self).__init__()

        filters = [start_num_filters * 2 ** i for i in range(5)]
        self.encoder_0 = ConvolutionalBlock(in_channels=in_channels, out_channels=filters[0], use_batch_norm=True)  # 32
        self.encoder_1 = ConvolutionalBlock(in_channels=filters[0], out_channels=filters[1], use_batch_norm=True)   # 64
        self.encoder_2 = ConvolutionalBlock(in_channels=filters[1], out_channels=filters[2], use_batch_norm=True)   # 128
        self.encoder_3 = ConvolutionalBlock(in_channels=filters[2], out_channels=filters[3], use_batch_norm=True)   # 256

        self.encoder_4 = ConvolutionalBlock(in_channels=filters[3], out_channels=filters[4], use_batch_norm=True)  # 256

        self.decoder_3 = UpBlock(n_channels_from_prev_layer=filters[4],  # 128
                                 n_channels_from_decoder=filters[3])

        self.decoder_2 = UpBlock(n_channels_from_prev_layer=filters[3],                        # 128
                                 n_channels_from_decoder=filters[2])

        self.decoder_1 = UpBlock(n_channels_from_prev_layer=filters[2],                        # 64
                                 n_channels_from_decoder=filters[1])

        self.decoder_0 = UpBlock(n_channels_from_prev_layer=filters[1],                        # 32
                                 n_channels_from_decoder=filters[0])

        self.out_layer = nn.Conv2d(in_channels=filters[0], out_channels=1,
                                   kernel_size=(1, 1))

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        # enc_0 = self.encoder_0(x)
        # x = self.pool(enc_0)
        #
        # enc_1 = self.encoder_1(x)
        # x = self.pool(enc_1)
        #
        # enc_2 = self.encoder_2(x)
        # x = self.pool(enc_2)
        #
        # bn = self.encoder_3(x)
        #
        # dec_2 = self.decoder_2(bn, enc_2)
        #
        # dec_1 = self.decoder_1(dec_2, enc_1)
        #
        # dec_0 = self.decoder_0(dec_1, enc_0)
        #
        # x = self.out_layer(dec_0)

        enc_0 = self.encoder_0(x)
        x = self.pool(enc_0)

        enc_1 = self.encoder_1(x)
        x = self.pool(enc_1)

        enc_2 = self.encoder_2(x)
        x = self.pool(enc_2)

        enc_3 = self.encoder_3(x)
        x = self.pool(enc_3)

        bn = self.encoder_4(x)

        dec_3 = self.decoder_3(bn, enc_3)

        dec_2 = self.decoder_2(dec_3, enc_2)

        dec_1 = self.decoder_1(dec_2, enc_1)

        dec_0 = self.decoder_0(dec_1, enc_0)

        x = self.out_layer(dec_0)

        return torch.sigmoid(x)

#
# from torchsummary import summary
#
# model = UNet()
#
# summary(model, (1, 512, 512))



class AttentionUNet(nn.Module):
    def __init__(self):
        super(AttentionUNet, self).__init__()