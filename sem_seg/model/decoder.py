import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time

from model.weighted_conv import WeightedConv1D

__all__ = ['decoder']


class DecoderConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, drop, sigma=1.0):
        super(DecoderConv, self).__init__()
        self.sigma = sigma
        self.conv = WeightedConv1D(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=1, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x, coords):
        x = self.relu(self.bn(self.conv(x, coords, self.sigma)))
        x = self.drop(x)

        return x


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone='resnet18', kernel_size=9, sigma=1.0):
        super(Decoder, self).__init__()
        if backbone == 'resnet101':
            low_level_inplanes = 512
        elif backbone == 'resnet18':
            low_level_inplanes = 128
        elif backbone == 'xception':
            low_level_inplanes = 128
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv1d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(48)
        self.relu = nn.ReLU()

        self.last_conv1 = DecoderConv(304, 256, kernel_size=kernel_size, drop=0.5, sigma=sigma)
        self.last_conv2 = DecoderConv(256, 256, kernel_size=kernel_size, drop=0.1, sigma=sigma)
        self.conv_out = nn.Conv1d(256, num_classes, kernel_size=1, stride=1)
        self._init_weight()

    def forward(self, x, low_level_feat, coords):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)   # channels = 48

        low_level_coords = coords[:, :, ::2]
        x = weighted_interpolation(x, low_level_coords)  #
        x = torch.cat((x, low_level_feat), dim=1)  # along channels
        x = self.last_conv1(x, low_level_coords)
        x = self.last_conv2(x, low_level_coords)
        x = weighted_interpolation(x, coords)
        x = self.conv_out(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, WeightedConv1D):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def weighted_interpolation(in_feats, coords):
    """
    Perform weighted interpolation of features using given coordinates
    to compute weights. Interpolation is done accross the last dimension.
    If in_feats is of size BxFxN, then coords must be Bx3xM, where M > N,
    and M / N is an integer which defines the scaling factor. The output
    is of size BxFxM.
    :param in_feats: features to be upsampled [BxFxN]
    :param coords: coordinates in upsampling space [Bx3xM]
    :return: upsampled features [BxFxM]
    """
    scale_factor = coords.shape[-1] // in_feats.shape[-1]
    # get the before and after coordinates of each point in current space
    before = coords[:, :, ::scale_factor].repeat_interleave(scale_factor, dim=-1)
    after = torch.cat((coords[:, :, scale_factor::scale_factor], coords[:, :, -1].unsqueeze(-1)),
                      dim=-1).repeat_interleave(scale_factor, dim=-1)
    # compute weights with values inverse to distance
    before = torch.exp(-torch.sqrt(torch.sum((coords - before) ** 2, dim=1)))
    after = torch.exp(-torch.sqrt(torch.sum((coords - after) ** 2, dim=1)))
    # normalize weights for each point
    den = before + after
    before /= den
    after /= den
    # repeat features for easier computations
    feats_before = in_feats.repeat_interleave(scale_factor, dim=-1)
    feats_after = torch.cat((in_feats[:, :, 1:], in_feats[:, :, -1].unsqueeze(-1)),
                            dim=-1).repeat_interleave(scale_factor, dim=-1)
    # return weighted interpolation
    return before.unsqueeze(1) * feats_before + after.unsqueeze(1) * feats_after


def decoder(num_classes, backbone, kernel_size, sigma):
    return Decoder(num_classes, backbone, kernel_size, sigma)


if __name__ == '__main__':
    numeric_level = getattr(logging, 'INFO', None)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        level=numeric_level)
    log_format = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)04d] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    device = torch.device('cuda')
    feats = torch.rand((4, 256, 128), dtype=torch.float).to(device)
    lowlevel_feats = torch.rand((4, 64, 256), dtype=torch.float).to(device)
    coords = torch.rand((4, 3, 128*8), dtype=torch.float).to(device)
    k = 21

    net = Decoder(num_classes=20).to(device)

    start_time = time.time()
    out = net(feats, lowlevel_feats, coords)
    logging.info('It took {:f}s'.format(time.time() - start_time))
    # out = out.mean(dim=1)
    logging.info('Output size {}'.format(out.size()))

    out.mean().backward()

