import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from .xception import AlignedXception
from .resnet import resnet18, resnet101
from .aspp import aspp
from .decoder import decoder
from .weighted_conv import weighted_interpolation

__all__ = ['deeplab']


class DeepLab(nn.Module):
    def __init__(self, backbone='xception', input_size=3, output_stride=16,
                 num_classes=21, kernel_size=9, sigma=1.0):
        super(DeepLab, self).__init__()
        self.num_classes = num_classes
        if backbone == 'resnet18':
            self.backbone = resnet18(input_size=input_size, kernel_size=kernel_size, sigma=sigma)
            sigma *= 32
            self.aspp = aspp(in_channels=512, out_channels=256, output_stride=output_stride,
                             kernel_size=kernel_size, sigma=sigma)
        elif backbone == 'resnet101':
            self.backbone = resnet101(input_size=input_size, kernel_size=kernel_size, sigma=sigma)
            self.aspp = aspp(in_channels=2048, out_channels=256, output_stride=output_stride,
                             kernel_size=kernel_size, sigma=sigma)
        else:
            self.backbone = AlignedXception(output_stride=output_stride, input_size=input_size)
            self.aspp = aspp(in_channels=2048, out_channels=256, output_stride=output_stride, kernel_size=kernel_size)

        self.decoder = decoder(num_classes=num_classes, backbone=backbone, kernel_size=kernel_size, sigma=sigma)

    def forward(self, input, coords):
        original_coords = coords
        x, low_level_feat, coords = self.backbone(input, coords)
        # print('Backbone: Output size {} Feat size {}'.format(x.size(), low_level_feat.size()))
        x = self.aspp(x, coords)
        # print('ASPP: Output size {} - {}'.format(x.size(), coords.size()))
        x = self.decoder(x, low_level_feat, original_coords[:, :, ::4])
        # print('Decoder: Output size {}'.format(x.size()))
        # x = F.interpolate(x, size=input.size(-1), mode='linear', align_corners=True)
        x = weighted_interpolation(x, original_coords)

        return x


def deeplab(backbone='resnet101', input_size=3, output_stride=16, num_classes=21, kernel_size=27, sigma=1.0):
    return DeepLab(backbone, input_size, output_stride, num_classes, kernel_size, sigma)


if __name__ == '__main__':
    batch_size = 8
    groups = 4
    in_channels = 9
    out_channels = 16
    num_points = 4096
    T = 4
    kernel_size = 21
    padding = 10
    dilation = 2
    stride = 2
    sigma = 0.5
    res = 128
    device = torch.device('cuda:1')

    feats = torch.rand((batch_size, in_channels, num_points), dtype=torch.float).to(device, dtype=torch.float32)
    coords = torch.rand((batch_size, 3, num_points), dtype=torch.float).to(device, dtype=torch.float32)

    # rotation_x = np.transpose([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # rotation_y = np.transpose([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    # rotation_z = np.transpose([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    # rotations = np.stack((np.eye(3), rotation_x, rotation_y, rotation_z), axis=0)
    # rotations = torch.from_numpy(rotations).to(device, dtype=torch.float32)
    #
    # distances = torch.randint(res ** 3, (res ** 3,)).to(device, dtype=torch.long)

    # print(feats.shape, coords.shape, rotations.shape, distances.shape)
    # print(feats.dtype, coords.dtype, rotations.dtype, distances.dtype)
    print(feats.shape, coords.shape)
    net = deeplab(input_size=in_channels, num_classes=13, backbone='resnet18', kernel_size=kernel_size).to(device)

    start_time = time.time()
    out = net(feats, coords)
    print('Forward took {:f}s'.format(time.time() - start_time))
    print('Output size {}'.format(out.size()))

    start_time = time.time()
    out.mean().backward()
    print('Backward took {:f}s'.format(time.time() - start_time))

