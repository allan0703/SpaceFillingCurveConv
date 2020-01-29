import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
from model.weighted_conv import WeightedConv1D
from model.architecture_knn import BasicConv1d, MultiSeq

__all__ = ['decoder']


class DecoderConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, drop=0., sigma=1.0):
        super(DecoderConv, self).__init__()
        self.sigma = sigma
        self.conv = WeightedConv1D(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=1, padding=kernel_size // 2)

        unlinear = [nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True)]
        if drop > 0:
            unlinear.append(nn.Dropout(drop))
        self.unlinear = nn.Sequential(*unlinear)

    def forward(self, x, coords):
        x = self.conv(x, coords, self.sigma)
        x = self.unlinear(x)
        return x


class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sigma=1.0, drop=0.1, embed_channels=256):
        super(DeConvBlock, self).__init__()
        self.conv1 = BasicConv1d(in_channels, embed_channels, 1)
        self.conv2 = DecoderConv(embed_channels, embed_channels, kernel_size=kernel_size, drop=drop, sigma=sigma)
        self.conv3 = DecoderConv(embed_channels, embed_channels, kernel_size=kernel_size, drop=drop, sigma=sigma)
        self.conv4 = nn.Conv1d(embed_channels, out_channels, kernel_size=1, stride=1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, WeightedConv1D):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, coords):
        x = self.conv1(x)
        x = self.conv2(x, coords)
        x = self.conv3(x, coords)
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        x = self.conv4(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels=256, kernel_size=9, sigma=1.0, embed_channels=48):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, embed_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(embed_channels)
        self.relu = nn.ReLU()

        self.last_conv1 = DecoderConv(in_channels+embed_channels, 256, kernel_size=kernel_size, drop=0.5, sigma=sigma)
        self.last_conv2 = DecoderConv(256, 256, kernel_size=kernel_size, drop=0.1, sigma=sigma)

        self.conv_out = nn.Conv1d(256, out_channels, kernel_size=1, stride=1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, WeightedConv1D):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, low_level_feat, coords):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[-1], mode='linear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv1(x, coords)
        x = self.last_conv2(x, coords)
        x = self.conv_out(x)

        return x


class C2FDecoder(nn.Module):
    """
    coarse to fine decoder
    """
    def __init__(self, in_channels, out_channels=256, kernel_size=9, sigma=1.0, drop=0.1, embed_channels=256):
        super(C2FDecoder, self).__init__()
        self.conv1 = BasicConv1d(embed_channels, out_channels, 1)

        self.deconv1 = DeConvBlock(in_channels, embed_channels, kernel_size, sigma, drop, embed_channels)
        self.deconv2 = DeConvBlock(embed_channels, embed_channels, kernel_size, sigma/2, drop, embed_channels)
        self.deconv3 = DeConvBlock(embed_channels+embed_channels, embed_channels, kernel_size, sigma/4, drop, embed_channels)
        self.deconv4 = DeConvBlock(embed_channels, embed_channels, kernel_size, sigma/16, drop, embed_channels)
        self.conv_out = nn.Conv1d(embed_channels, out_channels, kernel_size=1, stride=1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, WeightedConv1D):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, low_level_feat, coords):
        low_level_feat = self.conv1(low_level_feat)
        x = self.deconv1(x, coords[:, :, ::16])
        x = self.deconv2(x, coords[:, :, ::8])
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.deconv3(x, coords[:, :, ::4])
        x = self.deconv4(x, coords[:, :, ::2])
        x = self.conv_out(x)
        return x


if __name__ == '__main__':
    numeric_level = getattr(logging, 'INFO', None)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        level=numeric_level)
    log_format = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)04d] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    device = torch.device('cuda')

    feats = torch.rand((4, 256, 128), dtype=torch.float).to(device)
    low_level_feat = torch.rand((4, 256, 128*4), dtype=torch.float).to(device)

    coords = torch.rand((4, 3, 128*32), dtype=torch.float).to(device)
    sigma = 1.0
    k =3

    net = C2FDecoder(in_channels=256, out_channels=128, kernel_size=k).to(device)

    start_time = time.time()
    out = net(feats, low_level_feat, coords)
    logging.info('It took {:f}s'.format(time.time() - start_time))
    # out = out.mean(dim=1)
    logging.info('Output size {}'.format(out.size()))

    out.mean().backward()




