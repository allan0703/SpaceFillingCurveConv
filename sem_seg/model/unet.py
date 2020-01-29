from collections import OrderedDict

import torch
import torch.nn as nn

from .weighted_conv import WeightedConv1D, WeightedConvTranspose1D


class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UnetBlock, self).__init__()

        self.stride = stride
        self.conv1 = WeightedConv1D(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = WeightedConv1D(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, coords, sigma):
        x = self.relu(self.bn1(self.conv1(x, coords, sigma)))

        coords = coords[:, :, ::self.stride]

        x = self.relu(self.bn2(self.conv2(x, coords, sigma)))

        return x, coords


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, kernel_size=9, sigma=1.0):
        super(UNet, self).__init__()

        self.sigma = sigma

        features = init_features
        padding = kernel_size // 2
        self.encoder1 = UnetBlock(in_channels, features, kernel_size, stride=1, padding=padding)
        self.encoder2 = UnetBlock(features, features * 2, kernel_size, stride=2, padding=padding)
        self.encoder3 = UnetBlock(features * 2, features * 4, kernel_size, stride=2, padding=padding)
        self.encoder4 = UnetBlock(features * 4, features * 8, kernel_size, stride=2, padding=padding)

        self.bottleneck = UnetBlock(features * 8, features * 16, kernel_size, stride=2, padding=padding)

        self.upconv4 = WeightedConvTranspose1D(features * 16, features * 8, kernel_size, stride=2, padding=padding)
        self.decoder4 = UnetBlock((features * 8) * 2, features * 8, kernel_size, stride=1, padding=padding)

        self.upconv3 = WeightedConvTranspose1D(features * 8, features * 4, kernel_size, stride=2, padding=padding)
        self.decoder3 = UnetBlock((features * 4) * 2, features * 4, kernel_size, stride=1, padding=padding)

        self.upconv2 = WeightedConvTranspose1D(features * 4, features * 2, kernel_size, stride=2, padding=padding)
        self.decoder2 = UnetBlock((features * 2) * 2, features * 2, kernel_size, stride=1, padding=padding)

        self.upconv1 = WeightedConvTranspose1D(features * 2, features, kernel_size, stride=2, padding=padding)
        self.decoder1 = UnetBlock(features * 2, features, kernel_size, stride=1, padding=padding)

        # self.upconv4 = nn.ConvTranspose1d(
        #     features * 16, features * 8, kernel_size=4, stride=2, padding=1
        # )
        # self.decoder4 = UNet._block((features * 8) * 2, features * 8, kernel_size, name="dec4")
        # self.upconv3 = nn.ConvTranspose1d(
        #     features * 8, features * 4, kernel_size=4, stride=2, padding=1
        # )
        # self.decoder3 = UNet._block((features * 4) * 2, features * 4, kernel_size, name="dec3")
        # self.upconv2 = nn.ConvTranspose1d(
        #     features * 4, features * 2, kernel_size=4, stride=2, padding=1
        # )
        # self.decoder2 = UNet._block((features * 2) * 2, features * 2, kernel_size, name="dec2")
        # self.upconv1 = nn.ConvTranspose1d(
        #     features * 2, features, kernel_size=4, stride=2, padding=1
        # )
        # self.decoder1 = UNet._block(features * 2, features, kernel_size, name="dec1")

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x, coords):
        enc1, coords1 = self.encoder1(x, coords, self.sigma)
        enc2, coords2 = self.encoder2(enc1, coords1, 2 * self.sigma)
        enc3, coords3 = self.encoder3(enc2, coords2, 4 * self.sigma)
        enc4, coords4 = self.encoder4(enc3, coords3, 8 * self.sigma)

        bottleneck, coordsb = self.bottleneck(enc4, coords4, 16 * self.sigma)

        dec4 = self.upconv4(bottleneck, coordsb, 16 * self.sigma)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4, _ = self.decoder4(dec4, coords4, 8 * self.sigma)

        dec3 = self.upconv3(dec4, coords4, 8 * self.sigma)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3, _ = self.decoder3(dec3, coords3, 4 * self.sigma)

        dec2 = self.upconv2(dec3, coords3, 4 * self.sigma)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2, _ = self.decoder2(dec2, coords2, 2 * self.sigma)

        dec1 = self.upconv1(dec2, coords2, 2 * self.sigma)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1, _ = self.decoder1(dec1, coords, self.sigma)

        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, k, name):
        padding = k // 2
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=k,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=k,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


def unet(input_size=3, num_classes=21, kernel_size=27):
    return UNet(in_channels=input_size, out_channels=num_classes, init_features=32, kernel_size=kernel_size)


if __name__ == '__main__':
    x = torch.rand((4, 4, 4096), dtype=torch.float)
    coords = torch.rand((4, 3, 4096), dtype=torch.float)
    sigma = 3.0
    print('Input size {}'.format(x.size()))
    net = UNet(in_channels=4, out_channels=13, init_features=32, sigma=sigma)
    out = net(x, coords)

    print('Output size {}'.format(out.size()))
