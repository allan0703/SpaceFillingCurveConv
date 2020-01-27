import torch
import torch.nn as nn
import logging
from torch.nn import Sequential as Seq, Linear as Lin, Conv1d, Conv2d
from torch_edge import DenseDilatedKnnGraph, batched_index_select
from model.weighted_conv import WeightedConv1D


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


def act_layer(act, inplace=True, neg_slope=0.2, n_prelu=1):
    """
    helper selecting activation
    :param act:
    :param inplace:
    :param neg_slope:
    :param n_prelu:
    :return:
    """

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


class MLP(Seq):
    def __init__(self, channels, act='relu', norm=True, bias=False, dropout=False, drop_p=0.5):
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if dropout:
                m.append(nn.Dropout(p=drop_p))
            if norm:
                m.append(nn.BatchNorm1d(channels[i]))
            if act:
                m.append(act_layer(act))
        super(MLP, self).__init__(*m)


class BasicWeighedConv1d(Seq):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1,
                 act='relu', norm=True, bias=False, dropout=False, drop_p=0.5, sigma=0.08):
        super(BasicWeighedConv1d, self).__init__()
        padding = kernel_size // 2
        m = [WeightedConv1D(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=dilation * padding, dilation=dilation)]

        if dropout:
            m.append(nn.Dropout(p=drop_p))
        if norm:
            m.append(nn.BatchNorm1d(out_channels))
        if act:
            m.append(act_layer(act))
        self.body = MultiSeq(*m)
        self.sigma = sigma

    def forward(self, x, coords):
        x = self.body(x, coords, self.sigma)
        return x


class BasicConv1d(Seq):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1,
                 act='relu', norm=True, bias=False, dropout=False, drop_p=0.5, **kwargs):
        super(BasicConv1d, self).__init__()
        padding = kernel_size // 2
        m = []
        m.append(Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                        padding=dilation * padding, bias=bias, dilation=dilation, **kwargs))
        if dropout:
            m.append(nn.Dropout(p=drop_p))
        if norm:
            m.append(nn.BatchNorm1d(out_channels))
        if act:
            m.append(act_layer(act))
        self.body = nn.Sequential(*m)

    def forward(self, x, coords=None):
        x = self.body(x)
        return x


class BasicSeparableConv(Seq):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, dilation=1, knn=1,
                 act='relu', norm=True, bias=False, dropout=False, drop_p=0.5,
                 use_weighted_conv=True, sigma=0.08, **kwargs):
        super(BasicSeparableConv, self).__init__()
        self.conv1 = Seq(
            Conv2d(in_channels, out_channels, kernel_size=[1, knn], stride=1,
                   bias=bias, dilation=dilation, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if not use_weighted_conv:
            self.conv2 = BasicConv1d(out_channels, out_channels, kernel_size, stride, dilation, act, norm, bias,
                                     dropout, drop_p)
        else:
            self.conv2 = BasicWeighedConv1d(out_channels, out_channels, kernel_size, stride, dilation, act, norm, bias,
                                            dropout, drop_p, sigma)
        self.sigma = sigma

    def forward(self, x, edge_index=None, coords=None):
        if edge_index is not None:
            x = batched_index_select(x, edge_index)
        else:
            x = x.unsqueeze(-1)

        x = self.conv1(x).squeeze(-1)
        x = self.conv2(x, coords)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=5, knn=9, downsample=None, groups=1,
                 base_width=64, dilation=1, use_weighted_conv=True, sigma=1.0, knn_graph=None):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')  # zheli de groups he base_width shishenme yisi
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.sigma = sigma
        self.stride = stride

        self.conv1 = BasicSeparableConv(in_channels, out_channels, kernel_size, stride, knn=knn,
                                        use_weighted_conv=use_weighted_conv, sigma=sigma)
        self.conv2 = BasicSeparableConv(out_channels, out_channels, kernel_size, knn=knn, act=None,
                                        use_weighted_conv=use_weighted_conv, sigma=sigma)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.knn_graph = knn_graph

    def forward(self, inputs):
        x, edge_index, coords = inputs
        identity = x

        out = self.conv1(x, edge_index, coords)

        if self.stride > 1:
            coords = coords[:, :, ::self.stride]
            edge_index = self.knn_graph(coords)

        out = self.conv2(out, edge_index, coords)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, edge_index, coords


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=5, knn=9, downsample=None, groups=1,
                 base_width=64, dilation=1, use_weighted_conv=True, sigma=1.0, knn_graph=None):
        super(Bottleneck, self).__init__()

        self.sigma = sigma
        self.use_weighted_conv = use_weighted_conv
        hidden_channels = int(out_channels * (base_width / 64.)) * groups
        self.conv1 = BasicConv1d(in_channels, hidden_channels)
        self.conv2 = BasicSeparableConv(hidden_channels, hidden_channels, kernel_size, stride, knn=knn,
                                        use_weighted_conv=use_weighted_conv, sigma=sigma)
        self.conv3 = BasicConv1d(hidden_channels, out_channels*self.expansion, act=None)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.knn_graph = knn_graph

    def forward(self, inputs):
        x, edge_index, coords = inputs
        identity = x

        out = self.conv1(x)
        out = self.conv2(out, edge_index, coords)

        if self.stride > 1:
            coords = coords[:, :, ::self.stride]
            edge_index = self.knn_graph(coords)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, edge_index, coords


class ResNet(nn.Module):
    def __init__(self, block, layers, kernel_size=5, in_channels=3, num_classes=40, channels=64,
                 use_knn=True, knn=5,
                 use_weighted_conv=True, sigma=1.0,
                 use_tnet=False, n_points=1024,
                 replace_stride_with_dilation=None):
        super(ResNet, self).__init__()

        self.channels = channels
        self.dilation = 1
        self.dropout = 0.5
        self.n_points = n_points
        self.use_tnet = use_tnet
        self.use_weighted_conv = use_weighted_conv
        self.use_knn = use_knn
        self.knn_graph = DenseDilatedKnnGraph(knn, 1)
        self.sigma = sigma
        self.base_width = channels

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        # if self.use_tnet:
        #     self.tnet3 = TNet(3, self.n_points)
        self.conv1 = BasicSeparableConv(in_channels, channels, 49, stride=2, sigma=self.sigma, knn=knn,
                                        use_weighted_conv=self.use_weighted_conv)
        self.maxpool1 = nn.MaxPool1d(kernel_size=9, stride=2, padding=4)
        self.sigma *= 4
        self.layer1 = self._make_layer(block, 64, layers[0], kernel_size, knn=knn, sigma=self.sigma)
        self.layer2 = self._make_layer(block, 128, layers[1], kernel_size, stride=2, knn=knn, sigma=self.sigma,
                                       dilate=replace_stride_with_dilation[0], knn_graph=self.knn_graph)
        self.sigma *= 2
        self.layer3 = self._make_layer(block, 256, layers[2], kernel_size, stride=2, knn=knn, sigma=self.sigma,
                                       dilate=replace_stride_with_dilation[1], knn_graph=self.knn_graph)
        self.sigma *= 2
        self.layer4 = self._make_layer(block, 512, layers[3], kernel_size, stride=2, knn=knn, sigma=self.sigma,
                                       dilate=replace_stride_with_dilation[2], knn_graph=self.knn_graph)
        # expand fc layers.
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.pred = Seq(
            BasicConv1d(512 * block.expansion + 2, 512),
            BasicConv1d(512, 256, dropout=True),
            nn.Conv1d(256, num_classes, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, n_block, kernel_size=5, stride=1, dilate=False,
                    knn=9, knn_graph=None, sigma=1.0,
                    ):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.channels != channels * block.expansion:
            downsample = BasicConv1d(self.channels, channels * block.expansion, stride=stride, act=None)

        layers = [block(self.channels, channels, stride, kernel_size, knn, downsample,
                        self.base_width, previous_dilation, sigma=sigma, use_weighted_conv=self.use_weighted_conv,
                        knn_graph=knn_graph)]
        self.channels = channels * block.expansion
        for _ in range(1, n_block):
            layers.append(block(self.channels, channels, kernel_size=kernel_size, knn=knn,
                                base_width=self.base_width, dilation=self.dilation, sigma=sigma,
                                use_weighted_conv=self.use_weighted_conv, knn_graph=knn_graph))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.use_tnet:
            aligned_pos = self.tnet3(x)
            x = torch.cat((aligned_pos, x), dim=1)
        if self.use_knn:
            edge_index = self.knn_graph(x.detach())
        else:
            edge_index = None

        coords = x.detach()[:, :3, :]

        x = self.conv1(x, edge_index, coords)
        x = self.maxpool1(x)
        coords = coords[:, :, ::4]

        edge_index = self.knn_graph(coords.detach())
        x, edge_index, coords = self.layer1((x, edge_index, coords))
        low_level_feats = x
        x, edge_index, coords = self.layer2((x, edge_index, coords))
        x, edge_index, coords = self.layer3((x, edge_index, coords))
        x, edge_index, coords = self.layer4((x, edge_index, coords))
        # x = torch.cat((x, torch.max(x, dim=1, keepdim=True)[0], torch.mean(x, dim=1, keepdim=True)), dim=1)
        # x = self.pred(x)
        return x, low_level_feats, coords


def _resnet(block, layers, kernel_size=5, **kwargs):
    model = ResNet(block, layers, kernel_size, **kwargs)
    return model


def resnet_10(kernel_size=5, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(BasicBlock, [1, 1, 1, 1], kernel_size, **kwargs)


def resnet18(kernel_size=5, **kwargs):
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

    x = torch.rand((4, 3, 1024), dtype=torch.float).cuda()
    kernel_size = 5
    knn = 5
    sigma = 1.5

    net = resnet50(kernel_size=kernel_size, knn=knn, in_channels=3, num_classes=40, sigma=sigma,
                    use_weighted_conv=False)
    net = net.cuda()

    out, low_level_feature, coords = net(x)
    # out = out.mean(dim=1)
    logging.info('Output size {}, low_level_size {}'.format(out.size(), low_level_feature.size()))


