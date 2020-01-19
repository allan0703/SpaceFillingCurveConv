import torch
import torch.nn as nn
import logging
from torch.nn import Sequential as Seq, Linear as Lin, Conv1d, Conv2d
from torch_edge import DenseDilatedKnnGraph, batched_index_select


# __all__ = ['resnet18', 'resnet50', 'resnet101']


class MultiSeq(Seq):  # 这个是干嘛的？
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


class BasicConv2(Seq):
    def __init__(self, channels, k=1, stride=1, dilation=1, act='relu', norm=True, bias=False,
                 dropout=False, drop_p=0.5, **kwargs):
        super(BasicConv2, self).__init__()
        padding = k // 2
        m = []
        m.append(Conv1d(channels[0], channels[1], kernel_size=k, stride=stride,
                        padding=dilation * padding, bias=bias, dilation=dilation, **kwargs))
        if dropout:
            m.append(nn.Dropout(p=drop_p))
        if norm:
            m.append(nn.BatchNorm1d(channels[1]))
        if act:
            m.append(act_layer(act))
        self.body = nn.Sequential(*m)

    def forward(self, x, edge_index=None):
        x = self.body(x)
        return x


class BasicConv(Seq):
    def __init__(self, channels, kernel_size=5, knn=9, stride=1, dilation=1, act='relu', norm=True, bias=False,
                 dropout=False, drop_p=0.5, use_knn=False, **kwargs):
        super(BasicConv, self).__init__()
        # padding = knn // 2

        self.conv1 = Conv2d(channels[0], channels[0], kernel_size=[1, knn], stride=stride,
                            bias=bias, dilation=dilation, **kwargs)
        self.bn = nn.BatchNorm1d(channels[0])
        self.act = nn.ReLU(inplace=True)

        padding = kernel_size // 2
        conv2 = [Conv1d(channels[0], channels[1], kernel_size=kernel_size, stride=stride,
                        padding=dilation * padding, bias=bias, dilation=dilation, **kwargs)]
        if dropout:
            conv2.append(nn.Dropout(p=drop_p))
        if norm:
            conv2.append(nn.BatchNorm1d(channels[1]))
        if act:
            conv2.append(act_layer(act))
        self.conv2 = Seq(*conv2)

    def forward(self, x, edge_index=None):
        x_i = batched_index_select(x, edge_index)
        x = self.act(self.bn(self.conv1(x_i).squeeze(-1)))
        x = self.conv2(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=5, knn=9, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')  # zheli de groups he base_width shishenme yisi
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # self.knn = DenseDilatedKnnGraph(k, 1, stochastic, epsilon)
        self.conv1 = BasicConv([in_channels, out_channels], stride=stride, kernel_size=kernel_size, knn=knn)
        self.conv2 = BasicConv([out_channels, out_channels], kernel_size=kernel_size, knn=knn, act=None)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x, edge_index):

        # feats = [self.head(inputs, self.knn(inputs[:, 0:3]))]
        identity = x
        out = self.conv2(self.conv1(x, edge_index), edge_index)

        if self.downsample is not None:
            identity = self.downsample(x, edge_index)

        out += identity
        out = self.relu(out)  # zhegedifang hai xuyao ma

        return out


#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_channels, channels, stride=1, k=9, downsample=None, groups=1,
#                  base_width=64, dilation=1):
#         super(Bottleneck, self).__init__()
#
#         norm_layer = nn.BatchNorm1d
#
#         width = int(channels * (base_width / 64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(in_channels, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = convKxK(width, width, stride, k, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, channels * self.expansion)
#         self.bn3 = norm_layer(channels * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out

def generate_conv_indices(n_points=1048, kernel_size=3):
    idx = torch.range(-1, n_points)


def stn(x, transform_matrix=None):
    x = x.transpose(2, 1)
    x = torch.bmm(x, transform_matrix)
    x = x.transpose(2, 1)
    return x


class Transformation(nn.Module):
    def __init__(self, k=3, n_points=4096):
        super(Transformation, self).__init__()
        self.k = k
        self.convs = BasicConv((k, 64, 128, 1024))
        self.fcs = Seq(*[MLP((1024, 512, 256)),
                         MLP((256, k * k), None, False)])

    def forward(self, x):
        x = self.convs(x)
        x = torch.max(x, dim=2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.fcs(x)
        identity = torch.eye(self.k, device=x.device)
        x = x.view(-1, self.k, self.k) + identity[None]
        return x


class TNet(nn.Module):
    def __init__(self, k=3, n_points=4096):
        super(TNet, self).__init__()
        self.trans = Transformation(k, n_points)

    def forward(self, x):
        transform_matrix = self.trans(x)
        x = stn(x, transform_matrix)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, kernel_size=5, knn=9, in_channels=3, num_classes=40, channels=64,
                 zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, use_tnet=False, n_points=1024):
        super(ResNet, self).__init__()  # groups shi shenme ?

        self.channels = channels
        self.dilation = 1
        self.dropout = 0.5
        self.n_points = n_points
        self.use_tnet = use_tnet
        self.knn = DenseDilatedKnnGraph(knn, 1)
        # self.knn = DenseDilatedKnnGraph(k, 1)

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group  #
        if self.use_tnet:
            self.tnet3 = TNet(3, self.n_points)
        self.conv1 = BasicConv([in_channels, channels], kernel_size, knn)
        self.layer1 = self._make_layer(block, 128, layers[0], kernel_size, knn)
        self.layer2 = self._make_layer(block, 256, layers[1], kernel_size, knn, stride=1,
                                       dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(block, 256, layers[2], k=k, stride=1,
        #                                dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], k=k, stride=1,
        #                                dilate=replace_stride_with_dilation[2])
        # expand fc layers.
        # add a global feature
        self.maxpool1 = nn.AdaptiveMaxPool1d(1)
        self.maxpool4 = nn.MaxPool1d(kernel_size=9, stride=1, padding=4)
        self.avgpool4 = nn.AvgPool1d(kernel_size=9, stride=1, padding=4)
        self.pred1 = BasicConv([3 * (256 * block.expansion) + 128, 512], kernel_size=kernel_size, knn=knn)
        self.pred2 = BasicConv([512, 256], kernel_size=kernel_size, knn=knn, dropout=True, drop_p=self.dropout)
        self.pred3 = nn.Conv1d(256, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, n_block, kernel_size=5, knn=9, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.channels != channels * block.expansion:
            downsample = BasicConv2([self.channels, channels * block.expansion], stride=stride, act=None)

        layers = [block(self.channels, channels, stride, kernel_size, knn, downsample, self.groups,
                        self.base_width, previous_dilation)]
        self.channels = channels * block.expansion
        for _ in range(1, n_block):
            layers.append(block(self.channels, channels, kernel_size=kernel_size, knn=knn, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return MultiSeq(*layers)

    def forward(self, x):
        if self.use_tnet:
            aligned_pos = self.tnet3(x)
            x = torch.cat((aligned_pos, x), dim=1)
        edge_index = self.knn(x.detach())

        x = self.conv1(x, edge_index)
        # logging.info('Size after conv1: {}'.format(x.size()))

        x = self.layer1(x, edge_index)
        # logging.info('Size after layer1: {}'.format(x.size()))
        max_1 = self.maxpool1(x)
        x = self.layer2(x, edge_index)
        # logging.info('Size after layer2: {}'.format(x.size()))
        # x = self.layer3(x, edge_index)
        # logging.info('Size after layer3: {}'.format(x.size()))
        # x = self.layer4(x, edge_index)
        max_4 = self.maxpool4(x)
        avg_4 = self.avgpool4(x)
        # logging.info('Size after layer4: {}'.format(x.size()))

        x = torch.cat((x, max_1.repeat((1, 1, self.n_points)), max_4, avg_4), dim=1)
        # logging.info('Size after flatten: {}'.format(x.size()))
        # x = self.prediction(x)
        # x = self.prediction(x)
        x = self.pred3(self.pred2(self.pred1(x, edge_index), edge_index))
        return x


def _resnet(block, layers, kernel_size=5, knn=9, **kwargs):
    model = ResNet(block, layers, kernel_size, knn, **kwargs)
    return model


def sfc_resnet_8(kernel_size=5, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(BasicBlock, [1, 1, 1, 1], kernel_size, **kwargs)


def resnet18(kernel_size=5, knn=9, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], kernel_size, knn, **kwargs)


if __name__ == '__main__':
    numeric_level = getattr(logging, 'INFO', None)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        level=numeric_level)
    log_format = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)04d] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    x = torch.rand((4, 9, 1024), dtype=torch.float)
    kernel_size = 5
    knn = 9

    net = sfc_resnet_8(kernel_size=kernel_size, knn=knn, in_channels=9, num_classes=40, n_points=1024, use_tnet=False)

    out = net(x)
    # out = out.mean(dim=1)
    logging.info('Output size {}'.format(out.size()))

