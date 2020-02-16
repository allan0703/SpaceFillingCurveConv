import torch
import torch.nn as nn
import numpy as np
import logging
import time

from model.weighted_conv import WeightedConv1D

__all__ = ['decoder']


class DecoderConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, drop=0.5, sigma=1.0):
        super(DecoderConv, self).__init__()
        self.sigma = sigma
        self.conv = WeightedConv1D(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=1, padding=kernel_size // 2)

        nonlinear = [nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True)]
        if drop>0:
            nonlinear.append(nn.Dropout(drop))
        self.nonlinear = nn.Sequential(*nonlinear)

    def forward(self, x, coords):
        x = self.conv(x, coords, self.sigma)
        x = self.nonlinear(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone='resnet18', kernel_size=9, sigma=1.0):
        super(Decoder, self).__init__()
        channels = np.array([64, 128, 256])
        if backbone == 'resnet101':
            channels *= 4

        self.conv1 = DecoderConv(channels[-1]+256, 256, kernel_size=kernel_size, drop=0, sigma=sigma//2)
        self.conv2 = DecoderConv(channels[-2]+256, 256, kernel_size=kernel_size, drop=0, sigma=sigma//4)
        self.conv3 = DecoderConv(channels[-3]+256, 256, kernel_size=kernel_size, drop=0, sigma=sigma//8)
        # self.conv4 = DecoderConv(channels[-4]+256, 256, kernel_size=kernel_size, drop=0, sigma=sigma//8)

        self.last_conv1 = DecoderConv(256, 256, kernel_size=kernel_size, drop=0.5, sigma=sigma)
        self.last_conv2 = DecoderConv(256, 256, kernel_size=kernel_size, drop=0.1, sigma=sigma)
        self.conv_out = nn.Conv1d(256, num_classes, kernel_size=1, stride=1)
        self._init_weight()

    def forward(self, x, layer1_feat, layer2_feat, layer3_feat, coords1, coords2, coords3):
        #
        # low_level_feat = self.conv1(low_level_feat)
        # low_level_feat = self.bn1(low_level_feat)
        # low_level_feat = self.relu(low_level_feat)   # channels = 48

        x = weighted_interpolation(x, coords3)
        x = self.conv1(torch.cat((x, layer3_feat), dim=1), coords3)

        x = weighted_interpolation(x, coords2)
        x = self.conv2(torch.cat((x, layer2_feat), dim=1), coords2)

        x = weighted_interpolation(x, coords1)
        x = self.conv3(torch.cat((x, layer1_feat), dim=1), coords1)

        x = self.last_conv1(x, coords1)
        x = self.last_conv2(x, coords1)
        x = weighted_interpolation(x, coords1)
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
    net = Decoder(num_classes=20).to(device)


    # feats = torch.rand((4, 256, 128), dtype=torch.float).to(device)
    # lowlevel_feats = torch.rand((4, 64, 256), dtype=torch.float).to(device)
    # coords = torch.rand((4, 3, 128*8), dtype=torch.float).to(device)
    # k = 21
    # start_time = time.time()
    # out = net(feats, lowlevel_feats, coords)

    x = torch.rand((4, 256, 512), dtype=torch.float).to(device)
    layer1_feat = torch.rand((4, 64, 4096), dtype=torch.float).to(device)
    layer2_feat = torch.rand((4, 128, 2048), dtype=torch.float).to(device)
    layer3_feat = torch.rand((4, 256, 1024), dtype=torch.float).to(device)
    # layer4_feat = torch.rand((4, 512, 512), dtype=torch.float).to(device)

    coords1 = torch.rand((4, 3, 4096), dtype=torch.float).to(device)
    coords2 = torch.rand((4, 3, 2048), dtype=torch.float).to(device)
    coords3 = torch.rand((4, 3, 1024), dtype=torch.float).to(device)
    # coords4 = torch.rand((4, 3, 512), dtype=torch.float).to(device)

    start_time = time.time()
    out = net(x, layer1_feat, layer2_feat, layer3_feat, coords1, coords2, coords3)

    logging.info('It took {:f}s'.format(time.time() - start_time))
    # out = out.mean(dim=1)
    logging.info('Output size {}'.format(out.size()))

    out.mean().backward()

