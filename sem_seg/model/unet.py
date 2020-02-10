import torch
import torch.nn as nn
import time

from .weighted_conv import WeightedConv1D, WeightedConvTranspose1D, MultiOrderWeightedConv1D


class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UnetBlock, self).__init__()

        self.stride = stride
        self.conv1 = MultiOrderWeightedConv1D(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=kernel_size, dilation=1, padding=padding, stride=1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = MultiOrderWeightedConv1D(in_channels=out_channels, out_channels=out_channels,
                                              kernel_size=kernel_size, dilation=1,  padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, coords, rotations, distances, sigma):
        x = self.relu(self.bn1(self.conv1(x, coords, rotations, distances, sigma)))
        x = self.relu(self.bn2(self.conv2(x, coords, rotations, distances, sigma)))

        return x, coords[:, :, ::self.stride]


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

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, WeightedConv1D):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, WeightedConvTranspose1D):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, coords, rotations, distances):
        enc1, coords1 = self.encoder1(x, coords, rotations, distances, self.sigma)
        enc2, coords2 = self.encoder2(enc1, coords1, rotations, distances, 2 * self.sigma)
        enc3, coords3 = self.encoder3(enc2, coords2, rotations, distances, 4 * self.sigma)
        enc4, coords4 = self.encoder4(enc3, coords3, rotations, distances, 8 * self.sigma)

        bottleneck, coordsb = self.bottleneck(enc4, coords4, rotations, distances, 16 * self.sigma)

        dec4 = self.upconv4(bottleneck, coords4, 8 * self.sigma)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4, _ = self.decoder4(dec4, coords4, rotations, distances, 8 * self.sigma)

        dec3 = self.upconv3(dec4, coords3, 4 * self.sigma)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3, _ = self.decoder3(dec3, coords3, rotations, distances, 4 * self.sigma)

        dec2 = self.upconv2(dec3, coords2, 2 * self.sigma)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2, _ = self.decoder2(dec2, coords2, rotations, distances, 2 * self.sigma)

        dec1 = self.upconv1(dec2, coords, self.sigma)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1, _ = self.decoder1(dec1, coords, rotations, distances, self.sigma)

        return self.conv(dec1)


def unet(input_size=3, num_classes=21, kernel_size=27, sigma=1.0):
    return UNet(in_channels=input_size, out_channels=num_classes,
                init_features=64, kernel_size=kernel_size, sigma=sigma)


if __name__ == '__main__':
    device = torch.device('cuda:0')
    res = 128
    x = torch.rand((4, 9, 4096), dtype=torch.float).to(device)
    coords = torch.rand((4, 3, 4096), dtype=torch.float).to(device)

    rotation_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32).t()
    rotation_y = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32).t()
    rotation_z = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32).t()
    rotations = torch.stack((torch.eye(3), rotation_x, rotation_y, rotation_z), dim=0).to(device)

    distances = torch.randint(res ** 3, (res ** 3,)).to(device, dtype=torch.long)

    kernel_size = 3
    sigma = 0.05
    print('Input size {}'.format(x.size()))
    net = unet(input_size=9, num_classes=13, kernel_size=kernel_size, sigma=sigma).to(device)
    start_time = time.time()
    out = net(x, coords, rotations, distances)
    print('Took {}s'.format(time.time() - start_time))

    print('Output size {}'.format(out.size()))
