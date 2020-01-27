import torch
import torch.nn as nn
import torch.nn.functional as F
from xception import AlignedXception
import architecture_knn as architecture
from aspp import aspp
from decoder import decoder

__all__ = ['deeplab']


# todo: make channels changeable. When acc is up to 50.
class DeepLab(nn.Module):
    def __init__(self, backbone='resnet18', in_channels=3, num_classes=21, kernel_size=9, output_stride=16,
                 use_weighted_conv=True, sigma=1.0,
                 use_knn=True, knn=5):
        super(DeepLab, self).__init__()

        if backbone == 'resnet18':
            self.backbone = architecture.resnet18(in_channels=in_channels, kernel_size=kernel_size,
                                                  use_weighted_conv=use_weighted_conv, sigma=sigma,
                                                  use_knn=use_knn, knn=knn
                                                  )
            sigma *= 32
            self.aspp = aspp(in_channels=512, out_channels=256, output_stride=output_stride,
                             kernel_size=kernel_size, sigma=sigma)

        elif backbone == 'resnet50':
            self.backbone = architecture.resnet50(in_channels=in_channels, kernel_size=kernel_size,
                                                  use_weighted_conv=use_weighted_conv, sigma=sigma,
                                                  use_knn=use_knn, knn=knn)
            sigma *= 32
            self.aspp = aspp(in_channels=2048, out_channels=256, output_stride=output_stride,
                             kernel_size=kernel_size, sigma=sigma)
        elif backbone == 'resnet101':
            self.backbone = architecture.resnet101(in_channels=in_channels, kernel_size=kernel_size,
                                                   use_weighted_conv=use_weighted_conv, sigma=sigma,
                                                   use_knn=use_knn, knn=knn)
            self.aspp = aspp(in_channels=2048, out_channels=256, output_stride=output_stride,
                             kernel_size=kernel_size, sigma=sigma)
        else:
            raise NotImplementedError('{} has not been implemented yet'.format(backbone))
            # self.backbone = AlignedXception(output_stride=output_stride, in_channels=in_channels)
            # self.aspp = aspp(in_channels=2048, out_channels=256, output_stride=output_stride, kernel_size=kernel_size)

        self.decoder = decoder(num_classes=num_classes, backbone=backbone, kernel_size=kernel_size, sigma=sigma)

    def forward(self, input):
        decoder_coords = input.detach()[:, :, ::4]
        x, low_level_feat, coords = self.backbone(input)
        # print('Backbone: Output size {} Feat size {}'.format(x.size(), low_level_feat.size()))
        x = self.aspp(x, coords)
        # print('ASPP: Output size {} - {}'.format(x.size(), coords.size()))
        x = self.decoder(x, low_level_feat, decoder_coords)
        # print('Decoder: Output size {}'.format(x.size()))
        x = F.interpolate(x, size=input.size(-1), mode='linear', align_corners=True)
        # todo: coarse to fine fashion
        return x


if __name__ == '__main__':
    x = torch.rand((4, 3, 4096), dtype=torch.float)
    coords = torch.rand((4, 3, 4096), dtype=torch.float)
    print('Input size {}'.format(x.size()))
    net = DeepLab('resnet50', in_channels=3, num_classes=13, kernel_size=9, knn=9,
                  use_weighted_conv=False, sigma=1.0)
    out = net(x, coords)

    print('Output size {}'.format(out.size()))
