import torch
import torch.nn as nn
import torch.nn.functional as F

from model.xception import AlignedXception
from model.resnet import resnet18, resnet101
from model.aspp import aspp
from model.decoder import decoder

__all__ = ['deeplab']


class DeepLab(nn.Module):
    def __init__(self, backbone='xception', input_size=3, output_stride=16,
                 num_classes=21, kernel_size=9, sigma=1.0, T=4):
        super(DeepLab, self).__init__()

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
        self.fusion_multi_conv = nn.Sequential(nn.Conv1d(num_classes*T, 256, 3, padding=1),
                                               nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                               nn.Conv1d(256, 64, 3, padding=1),
                                               nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                                               nn.Conv1d(64, num_classes, 1)
                                               )

    def forward(self, multi_input, multi_coords, reindices):
        out = []
        for i in range(multi_input.shape[-1]):
            input = multi_input[:, :, :, i]
            coords = multi_coords[:, :, :, i]
            decoder_coords = coords[:, :, ::4]
            x, low_level_feat, coords = self.backbone(input, coords)
            # print('Backbone: Output size {} Feat size {}'.format(x.size(), low_level_feat.size()))
            x = self.aspp(x, coords)
            # print('ASPP: Output size {} - {}'.format(x.size(), coords.size()))
            x = self.decoder(x, low_level_feat, decoder_coords)
            # print('Decoder: Output size {}'.format(x.size()))
            x = F.interpolate(x, size=input.size(-1), mode='linear', align_corners=True)

            # reorder x
            x = torch.index_select(x, dim=-1, index=reindices[:, i])  # reindices[:,i]
            out.append(x)
        out = torch.cat(out, dim=1)  # out : B X CT X N
        out = self.fusion_multi_conv(out)
        return out


def deeplab(backbone='resnet101', input_size=3, output_stride=16, num_classes=21, kernel_size=27, sigma=1.0):
    return DeepLab(backbone, input_size, output_stride, num_classes, kernel_size, sigma)


if __name__ == '__main__':
    x = torch.rand((4, 4, 4096, 4), dtype=torch.float)
    coords = torch.rand((4, 3, 4096, 4), dtype=torch.float)
    reindices = torch.stack((torch.randperm(4096), torch.randperm(4096),
                             torch.randperm(4096),torch.randperm(4096)), dim=0)
    print('Input size {}'.format(x.size()))
    net = deeplab(input_size=4, num_classes=13, backbone='resnet18', kernel_size=9)
    out = net(x, coords, reindices)

    print('Output size {}'.format(out.size()))

