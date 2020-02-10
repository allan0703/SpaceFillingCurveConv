import torch
from torch import nn
from torch.nn import functional as F, ModuleList as ModList
import numpy as np
import time

__all__ = ['WeightedConv1D', 'MultiOrderWeightedConv1D']


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
                                             (dilation, 1), (padding * dilation, 0), (stride, 1))

    def forward(self, x, coords, rotations=None, distances=None, sigma=0.08):
        # if coords is not None:
        #     coords = coords.unsqueeze(-1)
        out = super().forward(x.unsqueeze(-1), coords.unsqueeze(-1), sigma)

        return out.squeeze(-1)


class SeparableWeightedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1, T=4):
        super(SeparableWeightedConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = (kernel_size, 1)
        self.dilation = (dilation, 1)
        self.padding = (padding * dilation, 0)
        self.stride = (stride, 1)

        self.weight = nn.Parameter(torch.Tensor(T, self.out_channels, self.in_channels, self.kernel_size[0]))

    def forward(self, x, coords, sigma):
        """
        :param x: multi-level point cloud features [B, T, C, N]
        :param coords: corresponding coordinates of features [B, T, 3, N]
        :param sigma: sigma value for density normalization of point weights
        :return: output point cloud features [B, T, F, N]
        """
        windows = F.unfold(x.reshape(x.shape[0] * x.shape[1], -1, x.shape[-1], 1),
                           kernel_size=self.kernel_size, padding=self.padding,
                           dilation=self.dilation, stride=self.stride)
        windows = windows.reshape(x.shape[0], x.shape[1], -1, x.shape[-1])

        dist_weights = F.unfold(coords.reshape(coords.shape[0] * coords.shape[1], -1, coords.shape[-1], 1),
                                kernel_size=self.kernel_size, padding=self.padding,
                                dilation=self.dilation, stride=self.stride)

        dist_weights = dist_weights.view(x.shape[0], x.shape[1], 3, self.kernel_size[0], -1)
        dist_weights = dist_weights - dist_weights[:, :, :, self.kernel_size[0] // 2, :].unsqueeze(-2)
        dist_weights = 1 - torch.sqrt(torch.sum(dist_weights ** 2, dim=2)) / sigma
        dist_weights[dist_weights < 0] = 0.0
        dist_weights = dist_weights.repeat(1, 1, self.in_channels, 1)

        windows = windows * dist_weights

        out = windows.transpose(2, 3).matmul(self.weight.view(self.weight.shape[0],
                                                              self.weight.shape[1], -1).transpose(2, 1)).transpose(3, 2)

        return out


class MultiOrderWeightedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, stride, T=4):
        super(MultiOrderWeightedConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.T = T

        # convs = []
        # bns = []
        # for i in range(T):
        #     convs.append(WeightedConv1D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        #                                 dilation=dilation, padding=padding, stride=1))
        #     bns.append(nn.BatchNorm1d(out_channels))
        #
        # self.convs = ModList(convs)
        # self.bns = ModList(bns)

        self.conv = SeparableWeightedConv1D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            dilation=dilation, padding=padding, stride=1, T=T)
        self.pointwise = nn.Conv1d(int(out_channels * T), out_channels, 1, stride=stride)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, coords, rotations, distances, sigma=0.05, res=128):
        """
        :param x: the point cloud in the original order [B, C, N]
        :param coords: corresponding 3D coordinates of features in x [B, 3, N]
        :param rotations: T rotation matrices to compute multi-level hilbert orders [T, 3, 3]
        :param distances: Hilbert distances for full space [M], M = (2 ^ p) ^ 3, where p is Hilbert order
        :param sigma: sigma value for weighted convolution
        :param res: grid resolution used in hilbert distances
        :return:
        """
        # # print(x.shape, indices.shape, indices.view(x.shape[0], -1).unsqueeze(1).repeat(1, x.shape[1], 1).shape)
        batch_size, in_channels, num_points = x.shape
        in_coords = coords.shape[1]
        # out = []
        # for i in range(rotations.shape[0]):
        #     rot_points = coords.transpose(2, 1).matmul(rotations[i, ...])
        #     rot_points = rot_points - rot_points.min(dim=1)[0].unsqueeze(1)
        #     rot_points = rot_points / (rot_points.max(dim=1)[0].unsqueeze(1) + 1e-23)
        #     rot_points = torch.floor(rot_points * (res - 1)).int()
        #
        #     idx = (res ** 2) * rot_points[:, :, 0] + res * rot_points[:, :, 1] + rot_points[:, :, 2]
        #     hilbert_dist = distances[idx.long()]
        #     idx = hilbert_dist.argsort(dim=1)
        #
        #     feats = torch.gather(x, dim=-1, index=idx.unsqueeze(1).repeat(1, in_channels, 1))
        #     xyz = torch.gather(coords, dim=-1, index=idx.unsqueeze(1).repeat(1, in_coords, 1))
        #
        #     result = self.bns[i](self.convs[i](feats, xyz, sigma))
        #     # result = self.bn(self.conv(feats, xyz, sigma))
        #     # result = self.conv(feats, xyz, sigma)
        #     # out.append(result)
        #
        #     reidx = torch.arange(num_points, device=x.device).repeat(batch_size, 1)
        #     reidx = torch.gather(reidx, dim=-1, index=idx.argsort())
        #
        #     out.append(torch.gather(result, dim=-1, index=reidx.unsqueeze(1).repeat(1, result.shape[1], 1)))
        #
        # out = torch.cat(out, dim=1)
        rot_points = coords.transpose(2, 1).unsqueeze(1).matmul(rotations)
        rot_points = rot_points - rot_points.min(dim=2)[0].unsqueeze(2)
        rot_points = rot_points / (rot_points.max(dim=2)[0].unsqueeze(2) + 1e-23)
        rot_points = torch.floor(rot_points * (res - 1)).int()

        idx = (res ** 2) * rot_points[:, :, :, 0] + res * rot_points[:, :, :, 1] + rot_points[:, :, :, 2]
        hilbert_dist = distances[idx.long()]
        idx = hilbert_dist.argsort(dim=2)

        x = torch.gather(x.unsqueeze(1).repeat(1, self.T, 1, 1),
                         dim=-1, index=idx.unsqueeze(-2).repeat(1, 1, in_channels, 1))
        coords = torch.gather(coords.unsqueeze(1).repeat(1, self.T, 1, 1), dim=-1,
                              index=idx.unsqueeze(-2).repeat(1, 1, in_coords, 1))

        reidx = torch.arange(num_points, device=x.device).repeat(batch_size, self.T, 1)
        reidx = torch.gather(reidx, dim=-1, index=idx.argsort())

        out = self.conv(x, coords, sigma)
        out = torch.gather(out, dim=-1, index=reidx.unsqueeze(-2).repeat(1, 1, out.shape[2], 1))
        out = out.reshape(batch_size, -1, out.shape[3])
        out = self.bn(self.pointwise(out))

        return out


class WeightedConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, stride):
        super(WeightedConvTranspose, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size[0]))

    def forward(self, x, coords, sigma):
        x_t = torch.zeros((x.shape[0], x.shape[1], self.stride[0] * x.shape[2], self.stride[1] * x.shape[3]),
                          dtype=x.dtype, device=x.device)
        x_t[:, :, ::self.stride[0], ::self.stride[1]] = x

        # coords_t = torch.zeros((coords.shape[0], coords.shape[1], self.stride[0] * coords.shape[2],
        #                         self.stride[1] * coords.shape[3]), dtype=coords.dtype, device=coords.device)
        # coords_t[:, :, ::self.stride[0], ::self.stride[1]] = coords

        windows = F.unfold(x_t, kernel_size=self.kernel_size, padding=self.padding,
                           dilation=self.dilation, stride=1)
        dist_weights = F.unfold(coords, kernel_size=self.kernel_size, padding=self.padding,
                                dilation=self.dilation, stride=1)

        dist_weights = dist_weights.view(x.shape[0], 3, self.kernel_size[0], -1)
        dist_weights = dist_weights - dist_weights[:, :, self.kernel_size[0] // 2, :].unsqueeze(-2)
        dist_weights = 1 - torch.sqrt(torch.sum(dist_weights ** 2, dim=1)) / sigma
        dist_weights[dist_weights < 0] = 0.0
        dist_weights = dist_weights.repeat(1, self.in_channels, 1)

        windows = windows * dist_weights

        out = windows.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)

        return out


class WeightedConvTranspose1D(WeightedConvTranspose):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1):
        super(WeightedConvTranspose1D, self).__init__(in_channels, out_channels, (kernel_size, 1),
                                                      (dilation, 1), (padding * dilation, 0), (stride, 1))

    def forward(self, x, coords, sigma=0.08):
        out = super().forward(x.unsqueeze(-1), coords.unsqueeze(-1), sigma)

        return out.squeeze(-1)


if __name__ == '__main__':
    batch_size = 16
    groups = 4
    in_channels = 9
    out_channels = 16
    num_points = 4096
    T = 4
    kernel_size = 9
    padding = 4
    dilation = 1
    stride = 1
    sigma = 0.5
    res = 128
    device = torch.device('cuda:0')

    feats = torch.rand((batch_size, in_channels, num_points), dtype=torch.float).to(device, dtype=torch.float32)
    coords = torch.rand((batch_size, 3, num_points), dtype=torch.float).to(device, dtype=torch.float32)

    rotation_x = np.transpose([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rotation_y = np.transpose([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    rotation_z = np.transpose([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rotations = np.stack((np.eye(3), rotation_x, rotation_y, rotation_z), axis=0)
    rotations = torch.from_numpy(rotations).to(device, dtype=torch.float32)

    distances = torch.randint(res ** 3, (res ** 3,)).to(device, dtype=torch.long)

    print(feats.shape, coords.shape, rotations.shape, distances.shape)
    print(feats.dtype, coords.dtype, rotations.dtype, distances.dtype)

    # reindices1 = torch.stack([torch.randperm(num_points), torch.randperm(num_points),
    #                           torch.randperm(num_points), torch.randperm(num_points)], dim=0).long()
    # reindices2 = torch.stack([torch.randperm(num_points), torch.randperm(num_points),
    #                           torch.randperm(num_points), torch.randperm(num_points)], dim=0).long()
    # reindices = torch.rand((batch_size, T, num_points))
    # reindices[0] = reindices1
    # reindices[1] = reindices2
    # reindices = reindices.long().to(device)
    #
    # indices1 = torch.stack([torch.randperm(num_points), torch.randperm(num_points),
    #                         torch.randperm(num_points), torch.randperm(num_points)], dim=0).long()
    # indices2 = torch.stack([torch.randperm(num_points), torch.randperm(num_points),
    #                         torch.randperm(num_points), torch.randperm(num_points)], dim=0).long()
    # indices = torch.rand((batch_size, T, num_points))
    # indices[0] = indices1
    # indices[1] = indices2
    # indices = indices.long().to(device)

    # print(feats.shape, coords.shape, indices.shape, reindices.shape)

    # feats = torch.gather(feats, dim=-1, index=indices.view(feats.shape[0], -1).unsqueeze(1).repeat(1, feats.shape[1], 1))
    # feats = feats.view(batch_size, in_channels, T, num_points).transpose(2, 1)
    #
    # coords = torch.gather(coords, dim=-1, index=indices.view(feats.shape[0], -1).unsqueeze(1).repeat(1, in_channels, 1))
    # coords = coords.view(batch_size, in_channels, T, num_points).transpose(2, 1)
    #
    # print(feats.shape, coords.shape)

    #
    # conv = MultiOrderWeightedConv1D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
    #                                 dilation=dilation, padding=padding, stride=stride).to(device)
    # conv = SeparableWeightedConv1D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
    #                                dilation=dilation, padding=padding, stride=stride, groups=T).to(device)
    conv = MultiOrderWeightedConv1D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    dilation=dilation, padding=padding, stride=stride, T=T).to(device)
    #
    # # feats = torch.rand((batch_size, groups, in_channels, num_points), dtype=torch.float)
    # # coords = torch.rand((batch_size, groups, 3, num_points), dtype=torch.float)
    # # conv = SeparableWeightedConv1D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
    # #                                groups=groups, dilation=dilation, padding=padding, stride=stride)

    start_time = time.time()
    # # N = 1
    # # for _ in range(N):
    # # out = conv(feats, coords, indices, reindices)
    out = conv(feats, coords, rotations, distances, sigma, res)
    # out.mean().backward()
    #
    print('Time elapsed: {:f}s'.format((time.time() - start_time)))
    print(out.shape)
    # out = torch.gather(out, dim=-1, index=reindices.unsqueeze(2).repeat(1, 1, out.shape[2], 1))
    # print(out.shape)
