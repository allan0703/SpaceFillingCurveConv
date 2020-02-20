import torch
import torch.nn as nn
import torch.nn.functional as F
from .weighted_conv import WeightedConv1D
from .common import MLP

__all__ = ['decoder']


class DecoderConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, drop=None, sigma=1.0):
        super(DecoderConv, self).__init__()
        self.sigma = sigma
        self.conv = WeightedConv1D(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=1, padding=kernel_size // 2)

        unlinear = [nn.BatchNorm1d(out_channels), nn.LeakyReLU()]
        if drop is not None:
            unlinear.append(nn.Dropout(drop))
        self.unlinear = nn.Sequential(*unlinear)

    def forward(self, x, coords):
        x = self.conv(x, coords, self.sigma)
        x = self.unlinear(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channles, num_classes, kernel_size=9, sigma=1.0):
        super(Decoder, self).__init__()
        self.classifier = nn.Sequential(MLP([in_channles * 2, 512], act='leakyrelu', norm='batch'),
                                        torch.nn.Dropout(0.5),
                                        MLP([512, 256], act='leakyrelu', norm='batch'),
                                        torch.nn.Dropout(0.5),
                                        MLP([256, num_classes], act=None, norm=None))
        self._init_weight()

    def forward(self, x, coords):
        # fusion = self.fusion_conv(x, coords)
        x1 = F.adaptive_max_pool1d(x, 1)
        x2 = F.adaptive_avg_pool1d(x, 1)
        logits = self.classifier(torch.cat((x1, x2), dim=1).squeeze(-1)).squeeze(-1)
        return logits

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, WeightedConv1D):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def decoder(num_classes, backbone, kernel_size, sigma):
    return Decoder(num_classes, backbone, kernel_size, sigma)


