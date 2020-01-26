from collections import OrderedDict

import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, kernel_size=9):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, kernel_size, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
        self.encoder2 = UNet._block(features, features * 2, kernel_size, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
        self.encoder3 = UNet._block(features * 2, features * 4, kernel_size, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
        self.encoder4 = UNet._block(features * 4, features * 8, kernel_size, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.bottleneck = UNet._block(features * 8, features * 16, kernel_size, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=4, stride=2, padding=1
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, kernel_size, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=4, stride=2, padding=1
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, kernel_size, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=4, stride=2, padding=1
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, kernel_size, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=4, stride=2, padding=1
        )
        self.decoder1 = UNet._block(features * 2, features, kernel_size, name="dec1")

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x, coords):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
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
    coords = torch.rand((4, 5, 4096), dtype=torch.float)
    print('Input size {}'.format(x.size()))
    net = UNet(in_channels=4, out_channels=13, init_features=32)
    out = net(x, coords)

    print('Output size {}'.format(out.size()))
