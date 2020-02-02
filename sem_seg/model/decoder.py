import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time

from model.weighted_conv import WeightedConv2d

__all__ = ['decoder']


class DecoderConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, drop, sigma=1.0):
        super(DecoderConv, self).__init__()
        self.sigma = sigma
        self.conv = WeightedConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=1, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x, coords):
        x = self.relu(self.bn(self.conv(x, coords, self.sigma)))
        x = self.drop(x)

        return x


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone='xception', kernel_size=9, sigma=1.0):
        super(Decoder, self).__init__()
        if backbone == 'resnet101':
            low_level_inplanes = 256
        elif backbone == 'resnet18':
            low_level_inplanes = 64
        elif backbone == 'xception':
            low_level_inplanes = 128
        else:
            raise NotImplementedError

        # 48
        emd_channels = 256
        self.conv1 = nn.Conv2d(low_level_inplanes, emd_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(emd_channels)
        self.relu = nn.ReLU()

        self.last_conv1 = DecoderConv(256+emd_channels, 256, kernel_size=kernel_size, drop=0.5, sigma=sigma)
        self.last_conv2 = DecoderConv(256, 256, kernel_size=kernel_size, drop=0.1, sigma=sigma)

        self.conv_out = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self._init_weight()

    def forward(self, x, low_level_feat, coords):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        # low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:4], mode='bicubic', align_corners=True)
        x = torch.cat((x+low_level_feat, low_level_feat), dim=1)
        # x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv1(x, coords)
        x = self.last_conv2(x, coords)
        x = self.conv_out(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, WeightedConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def decoder(num_classes, backbone, kernel_size, sigma):
    return Decoder(num_classes, backbone, kernel_size, sigma)