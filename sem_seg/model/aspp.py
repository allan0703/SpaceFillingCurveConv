import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time

from .weighted_conv import WeightedConv1D

__all__ = ['aspp']


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size=9, sigma=1.0):
        super(ASPPConv, self).__init__()
        self.sigma = dilation * sigma
        self.conv = WeightedConv1D(in_channels, out_channels, kernel_size, padding=dilation * (kernel_size // 2),
                                   dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, coords):
        x = self.conv(x, coords, self.sigma)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        size = x.shape[-1]
        # print('Before pool ', x.size())
        x = self.pool(x)
        # print('After pool ', x.size())
        x = self.conv(x)
        if x.shape[0] > 1:
            x = self.bn(x)
        x = self.relu(x)
        # print('Interpolating with size {}'.format(size))
        # print('Size of x before interpolate {}'.format(x.size()))
        x = F.interpolate(x, size=size, mode='linear', align_corners=False)
        # print('Size of x after interpolate {}'.format(x.size()))
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride, kernel_size, sigma):
        super(ASPP, self).__init__()
        modules = []

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())

        modules.append(ASPPConv(in_channels, out_channels, 16, kernel_size, sigma))

        self.convs = nn.ModuleList(modules)
        self.pool = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(
            nn.Conv1d(3 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, WeightedConv1D):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, coords):
        res = []
        res.append(self.conv1(x))
        for conv in self.convs:
            res.append(conv(x, coords))
            # print('Current size {}'.format(res[-1].size()))
        res.append(self.pool(x))
        res = torch.cat(res, dim=1)
        # print('Size before project {}'.format(res.size()))
        return self.project(res)


def aspp(in_channels, out_channels, output_stride, kernel_size=9, sigma=1.0):
    return ASPP(in_channels, out_channels, output_stride, kernel_size, sigma=sigma)


if __name__ == '__main__':
    numeric_level = getattr(logging, 'INFO', None)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        level=numeric_level)
    log_format = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)04d] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    device = torch.device('cpu')

    feats = torch.rand((4, 512, 128), dtype=torch.float).to(device)
    coords = torch.rand((4, 3, 128), dtype=torch.float).to(device)
    k = 21

    net = aspp(in_channels=512, out_channels=256, output_stride=16, kernel_size=k).to(device)

    start_time = time.time()
    out = net(feats, coords)
    logging.info('It took {:f}s'.format(time.time() - start_time))
    # out = out.mean(dim=1)
    logging.info('Output size {}'.format(out.size()))

    out.mean().backward()