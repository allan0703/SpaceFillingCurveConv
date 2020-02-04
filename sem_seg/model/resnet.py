import torch
import torch.nn as nn
import logging
import time

from model.weighted_conv import WeightedConv2D


__all__ = ['resnet18', 'resnet50', 'resnet101']


# def convKxK(in_planes, out_planes, stride=1, k=9, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     padding = k // 2
#     return nn.Conv1d(in_planes, out_planes, kernel_size=k, stride=stride,
#                      padding=dilation * padding, groups=groups, bias=False, dilation=dilation)


def convKxK(in_planes, out_planes, stride=1, k=9, dilation=1):
    """Kx1 weighted convolution"""
    padding = k // 2

    return WeightedConv2D(in_planes, out_planes, kernel_size=k, dilation=dilation, padding=padding, stride=stride)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, k=9, downsample=None, groups=1,
                 base_width=64, dilation=1, sigma=1.0):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        norm_layer = nn.BatchNorm1d
        self.sigma = sigma

        self.conv1 = convKxK(inplanes, planes, stride=stride, k=k)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convKxK(planes, planes, k=k)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs):
        x, multi_coords, indices, reindices= inputs #b*c*n
        # coords = multi_coords  # now is b*3*n*t indices is b* n*4 reindices is b * n *4
        identity = x

        out = self.conv1(x, coords, indices, reindices, self.sigma)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, coords, indices, reindices, self.sigma * self.stride)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, k=9, downsample=None, groups=1,
                 base_width=64, dilation=1, sigma=1.0):
        super(Bottleneck, self).__init__()

        self.sigma = sigma
        norm_layer = nn.BatchNorm1d

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = convKxK(width, width, stride, k, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs):
        x, multi_coords, indices, reindices= inputs #b*c*n

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, coords, indices, reindices, self.sigma)
        #coords = coords[:, :, ::self.stride, :] #b*c*(n/2)*t

        # selected_indices = torch.arange(0, x.shape[2], self.stride).expand(coords.shape[0], coords.shape[3],coords.shape[2] ).transpose(2,1) #b*(n/2)*t selected original points(indices)
        # indices = torch.gather(indices, 2, selected_indices)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, k, input_size=3, num_classes=40, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=[True, True, True], sigma=1.0):
        super(ResNet, self).__init__()

        norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        self.sigma = sigma

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = convKxK(input_size, self.inplanes, stride=1, k=9, dilation=1)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.sigma *= 4
        self.layer1 = self._make_layer(block, 64, layers[0], k=k, sigma=self.sigma)
        self.layer2 = self._make_layer(block, 128, layers[1], k=k, stride=2,
                                       dilate=replace_stride_with_dilation[0], sigma=self.sigma)
        self.sigma *= 2
        self.layer3 = self._make_layer(block, 256, layers[2], k=k, stride=2,
                                       dilate=replace_stride_with_dilation[1], sigma=self.sigma)
        self.sigma *= 2
        self.layer4 = self._make_layer(block, 512, layers[3], k=k, stride=2,
                                       dilate=replace_stride_with_dilation[2], sigma=self.sigma)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, WeightedConv2D):
            #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, k=9, stride=1, dilate=False, sigma=1.0):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, k, downsample, self.groups,
                            self.base_width, previous_dilation, sigma=sigma))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, k=k, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x, multi_coords, indices, reindices):
        # print('Size at input: {}'.format(x.size()))
        # x = self.conv1(x)
        x = self.conv1(x, coords, indices, reindices)
        x = self.bn1(x)
        x = self.relu(x)
        # print('Size after conv1: {}'.format(x.size()))
        # x = self.maxpool(x)
        # print('Size after maxpool: {}'.format(x.size()))
        # coords = coords[:, :, ::4]
        x= self.layer1((x, multi_coords),indices, reindices)
        # print('Size after layer1: {}'.format(x.size()))
        low_level_feats = x
        x = self.layer2(x,multi_coords,indices, reindices)
        # print('Size after layer2: {}'.format(x.size()))
        x = self.layer3(x, multi_coords,indices, reindices)
        # print('Size after layer3: {}'.format(x.size()))
        x= self.layer4(x, multi_coords,indices, reindices)
        # print('Size after layer4: {}'.format(x.size()))

        return x


def _resnet(block, layers, k, **kwargs):
    model = ResNet(block, layers, k, **kwargs)
    return model


def resnet18(kernel_size=9, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], kernel_size, **kwargs)


def resnet50(kernel_size=9, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], kernel_size, **kwargs)


def resnet101(kernel_size=9, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], kernel_size, **kwargs)


if __name__ == '__main__':
    numeric_level = getattr(logging, 'INFO', None)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        level=numeric_level)
    log_format = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)04d] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    device = torch.device('cpu')

    feats = torch.rand((4, 3, 4096), dtype=torch.float).to(device)
    coords = torch.rand((4, 3, 4096, 4), dtype=torch.float).to(device)
    k = 9
    batch_size, num_points, T = 4, 4096, 4
    reindices1 = torch.stack([torch.randperm(num_points), torch.randperm(num_points), torch.randperm(num_points),
                              torch.randperm(num_points)], dim=-1).long()
    reindices2 = torch.stack([torch.randperm(num_points), torch.randperm(num_points), torch.randperm(num_points),
                              torch.randperm(num_points)], dim=-1).long()
    reindices = torch.rand((batch_size, num_points, T))
    reindices[0] = reindices1
    reindices[1] = reindices2
    reindices = reindices.long()
    indices1 = torch.stack([torch.randperm(num_points), torch.randperm(num_points), torch.randperm(num_points),
                            torch.randperm(num_points)], dim=-1).long()
    indices2 = torch.stack([torch.randperm(num_points), torch.randperm(num_points), torch.randperm(num_points),
                            torch.randperm(num_points)], dim=-1).long()
    indices = torch.rand((batch_size, num_points, T))
    indices[0] = indices1
    indices[1] = indices2
    indices = reindices.long()
    net = resnet101(kernel_size=k, input_size=3, num_classes=40).to(device)

    start_time = time.time()
    out = net(feats, coords,indices,reindices)
    logging.info('It took {:f}s'.format(time.time() - start_time))
    # out = out.mean(dim=1)
    logging.info('Output size {}'.format(out.size()))
    # logging.info('Feats size {}'.format(low_level_feats.size()))
    # logging.info('Out coords size {}'.format(out_coords.size()))

    out.mean().backward()
