import torch
from torch import nn
from torch.nn import functional as F, Sequential as Seq
import time

__all__ = ['WeightedConv1D']


class MultiSeq(Seq):
    def __init__(self, *args):
        super(MultiSeq, self).__init__(*args)

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class WeightedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, stride, group=1, t=4):
        super(WeightedConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        # self.group = group
        self.conv = MultiSeq(*[WeightedConv1D(in_channels, out_channels, kernel_size, dilation, padding),
                               WeightedConv1D(in_channels, out_channels, kernel_size, dilation, padding),
                               WeightedConv1D(in_channels, out_channels, kernel_size, dilation, padding),
                               WeightedConv1D(in_channels, out_channels, kernel_size, dilation, padding)])

        self.fusion_multi_conv = nn.Sequential(nn.Conv1d(int(out_channels * t), out_channels, 1, padding=1, stride=stride),
                                               nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True)
                                               )

    def forward(self, input, multi_coords, indices, reindices, sigma=0.05):
        out = []
        for i in range(multi_coords.shape[-1]):
            input = torch.gather(input, dim=-1, index=indices[:, :, i].unsqueeze(1).repeat(1, self.in_channels, 1))  # reindices[:,i]
            coords = multi_coords[:, :, :, i]
            x = self.conv[i](input, coords)
            x = torch.gather(x, dim=-1, index=reindices[:, :, i].unsqueeze(1).repeat(1, self.out_channels, 1))  # reindices[:,i]
            out.append(x)
        out = torch.cat(out, dim=1)  # out : B X CT X N
        out = self.fusion_multi_conv(out)
        return out

class WeightedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, stride, group=1):
        super(WeightedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        # self.group = group

        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size[0]))

    def forward(self, x, coords, sigma):
        # for x -> 4 x
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
    in_channels = 4
    out_channels = 16
    num_points = 4096
    T = 4
    kernel_size = 9
    padding = 4
    dilation = 1
    stride = 1

    feats = torch.rand((batch_size, in_channels, num_points), dtype=torch.float)
    coords = torch.rand((batch_size, 3, num_points, T), dtype=torch.float)
    reindices1 = torch.stack([torch.randperm(num_points),torch.randperm(num_points),torch.randperm(num_points),torch.randperm(num_points)], dim=-1).long()
    reindices2 = torch.stack([torch.randperm(num_points),torch.randperm(num_points),torch.randperm(num_points),torch.randperm(num_points)], dim=-1).long()
    reindices = torch.rand((batch_size, num_points, T))
    reindices[0]= reindices1
    reindices[1]=reindices2
    reindices = reindices.long()
    indices1 = torch.stack([torch.randperm(num_points),torch.randperm(num_points),torch.randperm(num_points),torch.randperm(num_points)], dim=-1).long()
    indices2 = torch.stack([torch.randperm(num_points),torch.randperm(num_points),torch.randperm(num_points),torch.randperm(num_points)], dim=-1).long()
    indices = torch.rand((batch_size, num_points, T))
    indices[0]= indices1
    indices[1]= indices2
    indices = indices.long()
    conv = WeightedConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                          dilation=dilation, padding=padding, stride=stride)

    # feats = torch.rand((batch_size, groups, in_channels, num_points), dtype=torch.float)
    # coords = torch.rand((batch_size, groups, 3, num_points), dtype=torch.float)
    # conv = SeparableWeightedConv1D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
    #                                groups=groups, dilation=dilation, padding=padding, stride=stride)

    start_time = time.time()
    out = conv(feats, coords,indices,reindices)

    out.mean().backward()
    print('Time elapsed: {:f}s'.format(time.time() - start_time))
    print(out.shape)
