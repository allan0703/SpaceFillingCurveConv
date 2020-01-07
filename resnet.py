import torch
import torch.nn as nn
import logging


__all__ = ['resnet18', 'resnet50', 'resnet101']


def convKxK(in_planes, out_planes, stride=1, k=9, groups=1, dilation=1):
    """3x3 convolution with padding"""
    padding = k // 2
    return nn.Conv1d(in_planes, out_planes, kernel_size=k, stride=stride,
                     padding=dilation * padding, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, k=9, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        norm_layer = nn.BatchNorm1d

        self.conv1 = convKxK(inplanes, planes, stride=stride, k=k)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convKxK(planes, planes, k=k)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, k=9, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(Bottleneck, self).__init__()

        norm_layer = nn.BatchNorm1d

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = convKxK(width, width, stride, k, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
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
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(ResNet, self).__init__()

        norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.dropout = 0.5

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(input_size, self.inplanes, kernel_size=49, stride=2, padding=24,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool1d(kernel_size=9, stride=2, padding=4)
        self.layer1 = self._make_layer(block, 64, layers[0], k=k)
        self.layer2 = self._make_layer(block, 128, layers[1], k=k, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], k=k, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], k=k, stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)
        # expand fc layers.
        # self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        self.fc = nn.Sequential(nn.Linear(512 * block.expansion * 2, 512), norm_layer(512), nn.ReLU(inplace=True),
                                nn.Dropout(p=self.dropout),
                                nn.Linear(512, 128), norm_layer(128), nn.ReLU(inplace=True),
                                nn.Dropout(p=self.dropout),
                                nn.Linear(128, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                # if isinstance(m, Bottleneck):
                #     nn.init.constant_(m.bn3.weight, 0)
                # el
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, k=9, stride=1, dilate=False):
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
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, k=k, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # logging.info('Size at input: {}'.format(x.size()))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # logging.info('Size after conv1: {}'.format(x.size()))
        x = self.maxpool1(x)
        # logging.info('Size after maxpool: {}'.format(x.size()))

        x = self.layer1(x)
        # logging.info('Size after layer1: {}'.format(x.size()))
        x = self.layer2(x)
        # logging.info('Size after layer2: {}'.format(x.size()))
        x = self.layer3(x)
        # logging.info('Size after layer3: {}'.format(x.size()))
        x = self.layer4(x)
        # logging.info('Size after layer4: {}'.format(x.size()))

        avg_feat = self.avgpool(x)
        # logging.info('Size after avgpool: {}'.format(x.size()))
        # add max pooling
        max_feat = self.maxpool2(x)
        x = torch.cat((avg_feat, max_feat), dim=1)

        x = torch.flatten(x, 1)
        # logging.info('Size after flatten: {}'.format(x.size()))
        x = self.fc(x)

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

    x = torch.rand((4, 3, 1024), dtype=torch.float)
    k = 21

    net = resnet101(kernel_size=k, input_size=3, num_classes=40)

    out = net(x)
    # out = out.mean(dim=1)
    logging.info('Output size {}'.format(out.size()))
