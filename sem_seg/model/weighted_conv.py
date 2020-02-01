import torch
from torch import nn
from torch.nn import functional as F
import time

__all__ = ['WeightedConv1D']


class WeightedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, padding=1, stride=1, group=1):
        super(WeightedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        # self.group = group

        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size**2))

    def forward(self, x, coords, sigma=0.08):
        # for x -> 4 x
        windows = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding,
                           dilation=self.dilation, stride=self.stride)
        dist_weights = F.unfold(coords, kernel_size=self.kernel_size, padding=self.padding,
                                dilation=self.dilation, stride=self.stride)
        dist_weights = dist_weights.view(x.shape[0], 3, self.kernel_size**2, -1)
        dist_weights = dist_weights - dist_weights[:, :, self.kernel_size**2 // 2, :].unsqueeze(-2)
        dist_weights = 1 - torch.sqrt(torch.sum(dist_weights ** 2, dim=1)) / sigma
        dist_weights[dist_weights < 0] = 0.0
        dist_weights = dist_weights.repeat(1, self.in_channels, 1)
        windows = windows * dist_weights

        out = windows.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)
        return out.view(x.shape[0], self.out_channels, x.shape[2]//self.stride, x.shape[3]//self.stride)


class WeightedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, stride):
        super(WeightedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size[0]))

    def forward(self, x, coords, sigma):
        windows = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding,
                           dilation=self.dilation, stride=self.stride)
        dist_weights = F.unfold(coords, kernel_size=self.kernel_size, padding=self.padding,
                                dilation=self.dilation, stride=self.stride)

        dist_weights = dist_weights.view(x.shape[0], 3, self.kernel_size[0], -1)
        dist_weights = dist_weights - dist_weights[:, :, self.kernel_size[0] // 2, :].unsqueeze(-2)
        dist_weights = 1 - torch.sqrt(torch.sum(dist_weights ** 2, dim=1)) / sigma
        dist_weights[dist_weights < 0] = 0.0
        dist_weights = dist_weights.repeat(1, self.in_channels, 1)

        windows = windows * dist_weights

        out = windows.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)

        return out


class SeparableWeightedConv(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, padding, stride):
        super(SeparableWeightedConv, self).__init__()
        self.in_channels = in_channels

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        self.weight = nn.Parameter(torch.Tensor(self.in_channels, 1, self.kernel_size[0]))

    def forward(self, x, coords, sigma):
        windows = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding,
                           dilation=self.dilation, stride=self.stride)
        dist_weights = F.unfold(coords, kernel_size=self.kernel_size, padding=self.padding,
                                dilation=self.dilation, stride=self.stride)

        dist_weights = dist_weights.view(x.shape[0], 3, self.kernel_size[0], -1)
        dist_weights = dist_weights - dist_weights[:, :, self.kernel_size[0] // 2, :].unsqueeze(-2)
        dist_weights = 1 - torch.sqrt(torch.sum(dist_weights ** 2, dim=1)) / sigma
        dist_weights[dist_weights < 0] = 0.0
        dist_weights = dist_weights.repeat(1, self.in_channels, 1)

        windows = windows * dist_weights
        windows = windows.reshape(x.shape[0], self.in_channels, self.kernel_size[0], -1)
        out = windows.transpose(3, 2).matmul(self.weight.transpose(2, 1)).squeeze(-1)

        return out


class WeightedConv1D(WeightedConv):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1):
        super(WeightedConv1D, self).__init__(in_channels, out_channels, (kernel_size, 1),
                                             (dilation, 1), (padding, 0), (stride, 1))

    def forward(self, x, coords=None, sigma=0.08):
        if coords is not None:
            coords = coords.unsqueeze(-1)
        out = super().forward(x.unsqueeze(-1), coords, sigma)

        return out.squeeze(-1)


class SeparableWeightedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, dilation=1, padding=0, stride=1):
        super(SeparableWeightedConv1D, self).__init__()

        self.convs = []
        self.bns = []
        for i in range(groups):
            self.convs.append(WeightedConv1D(in_channels, in_channels, kernel_size, dilation, padding, stride))
            self.bns.append(nn.BatchNorm1d(in_channels))

        self.pointwise = nn.Conv1d(groups * in_channels, out_channels, 1)

    def forward(self, x, coords, inv=None, sigma=0.08):
        outs = []
        for i, conv in enumerate(self.convs):
            outs.append(self.bns[i](conv(x[:, i, ...], coords[:, i, ...], sigma)))

        outs = torch.cat(outs, dim=1)

        out = self.pointwise(outs)

        return out


if __name__ == '__main__':
    batch_size = 2
    groups = 4
    in_channels = 3
    out_channels = 3
    num_points = 4096
    kernel_size = 1
    padding = kernel_size//2
    dilation = 1
    stride = 1

    # feats = torch.rand((batch_size, in_channels, int(num_points/256), 256), dtype=torch.float)
    # coords = torch.rand((batch_size, 3, int(num_points/256), 256), dtype=torch.float)
    feats = torch.rand((batch_size, in_channels, 4, 4), dtype=torch.float)
    coords = torch.rand((batch_size, 3, 4, 4), dtype=torch.float)

    conv = WeightedConv2d(in_channels, out_channels, kernel_size, dilation, padding=padding, stride=stride)

    # feats = torch.rand((batch_size, groups, in_channels, num_points), dtype=torch.float)
    # coords = torch.rand((batch_size, groups, 3, num_points), dtype=torch.float)
    # conv = SeparableWeightedConv1D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
    #                                groups=groups, dilation=dilation, padding=padding, stride=stride)

    start_time = time.time()
    out = conv(feats, coords)

    out.mean().backward()
    print('Time elapsed: {:f}s'.format(time.time() - start_time))
    print(out.shape)

