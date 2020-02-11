import torch
import torch.nn as nn
import torch.nn.functional as F

from model.xception import AlignedXception
from model.resnet import resnet18, resnet101
from model.aspp import aspp
from model.decoder import decoder

__all__ = ['deeplab']


class DeepLab(nn.Module):
    def __init__(self, backbone='xception', input_size=3, output_stride=16, num_classes=21, kernel_size=9, sigma=1.0):
        super(DeepLab, self).__init__()

        if backbone == 'resnet18':
            self.backbone = resnet18(input_size=input_size, kernel_size=kernel_size, sigma=sigma)
            sigma *= 4
            self.aspp = aspp(in_channels=512, out_channels=256, output_stride=output_stride,
                             kernel_size=kernel_size, sigma=sigma)
        elif backbone == 'resnet101':
            self.backbone = resnet101(input_size=input_size, kernel_size=kernel_size, sigma=sigma)
            sigma *= 4
            self.aspp = aspp(in_channels=2048, out_channels=256, output_stride=output_stride,
                             kernel_size=kernel_size, sigma=sigma)
        else:
            self.backbone = AlignedXception(output_stride=output_stride, input_size=input_size)
            self.aspp = aspp(in_channels=2048, out_channels=256, output_stride=output_stride, kernel_size=kernel_size)

        self.decoder = decoder(num_classes=num_classes, backbone=backbone, kernel_size=kernel_size, sigma=sigma)

    def forward(self, input, coords, edge_index):
        x, low_level_feat, coords = self.backbone(input, coords,edge_index)

        x = self.aspp(x, coords)

        x = self.decoder(x, low_level_feat, coords)

        return x


def deeplab(backbone='resnet101', input_size=3, output_stride=16, num_classes=21, kernel_size=27, sigma=1.0):
    return DeepLab(backbone, input_size, output_stride, num_classes, kernel_size, sigma)


if __name__ == '__main__':
    x = torch.rand((4, 4, 4096), dtype=torch.float)
    coords = torch.rand((4, 3, 4096), dtype=torch.float)
    print('Input size {}'.format(x.size()))
    net = deeplab(input_size=4, num_classes=13, backbone='resnet18', kernel_size=9)
    out = net(x, coords)

    print('Output size {}'.format(out.size()))

