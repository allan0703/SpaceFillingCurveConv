import torch
import torch.nn as nn
import logging
from torch.nn import Sequential as Seq, Linear as Lin, Conv1d, Conv2d


# from torch_edge import DenseDilatedKnnGraph, batched_index_select


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


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
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


class BasicConv1d(Seq):  # putongde juanjiceng
    def __init__(self, channels, kernel_size=1, stride=1, dilation=1, act='relu', norm=True, bias=False,
                 dropout=False, drop_p=0.5, **kwargs):
        super(BasicConv1d, self).__init__()

        conv1 = [Conv1d(channels[0], channels[1], kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                        bias=bias, dilation=dilation, **kwargs)]  # zheliyao keep b*c*4096*t baozheng houliangwei
        if norm:
            conv1.append(nn.BatchNorm1d(channels[1]))
        if act:
            conv1.append(nn.ReLU())
        self.conv1 = Seq(*conv1)

    def forward(self, x):
        x = self.conv1(x)
        return x


class BasicConv(Seq):
    def __init__(self, channels, kernel_size=3, multi_order=4, stride=1, dilation=1, act='relu', norm=True, bias=False,
                 dropout=False, drop_p=0.5, use_knn=False, **kwargs):
        super(BasicConv, self).__init__()

        conv1 = [Conv2d(channels[0], channels[1], kernel_size=[kernel_size, kernel_size], stride=stride,
                        padding=kernel_size // 2,
                        bias=bias, dilation=dilation, **kwargs)]  # zheliyao keep b*c*4096*t baozheng houliangwei
        if norm:
            conv1.append(nn.BatchNorm2d(channels[1]))
        if act:
            conv1.append(nn.ReLU())
        self.conv1 = Seq(*conv1)
        '''
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
        '''

    def forward(self, x):
        x = self.conv1(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, multi_order=4, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')  # zheli de groups he base_width shishenme yisi
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # self.knn = DenseDilatedKnnGraph(k, 1, stochastic, epsilon)

        self.conv1 = BasicConv([in_channels, out_channels], stride=stride, kernel_size=kernel_size)
        self.conv2 = BasicConv([out_channels, out_channels], kernel_size=kernel_size,
                               act=None)  # zhelishibushixiecheng in_channels
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):

        # feats = [self.head(inputs, self.knn(inputs[:, 0:3]))]
        identity = x
        out = self.conv2(self.conv1(x))

        if self.downsample is not None:
            identity = self.downsample(x)

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
    def __init__(self, block, layers, kernel_size=3, multi_order=4, in_channels=9, num_classes=13, channels=64,
                 zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, use_tnet=False,
                 n_points=4096):  # layers = [1,1,1,1]
        super(ResNet, self).__init__()  # groups shi shenme ?

        self.channels = channels
        self.dilation = 1
        self.dropout = 0.5
        self.n_points = n_points
        self.use_tnet = use_tnet
        self.multi_order = multi_order

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
        self.conv1 = BasicConv([in_channels, channels], kernel_size=kernel_size)
        self.layer1 = self._make_layer(block, 128, layers[0], kernel_size)
        self.layer2 = self._make_layer(block, 256, layers[1], kernel_size, stride=1,
                                       dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(block, 256, layers[2], k=k, stride=1,
        #                                dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], k=k, stride=1,
        #                                dilate=replace_stride_with_dilation[2])
        # expand fc layers.
        # add a global feature
        self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # baochi le weidu
        self.maxpoolpre = nn.MaxPool2d(kernel_size=[1, 4], stride=1)
        self.avgpool4 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.pred1 = BasicConv1d([3 * (256 * block.expansion) + 128, 512],
                                 kernel_size=kernel_size)  # zhegedifang de weidu
        self.pred2 = BasicConv1d([512, 256], kernel_size=kernel_size, dropout=True, drop_p=self.dropout)
        self.pred3 = nn.Conv1d(256, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, n_block, kernel_size=3, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.channels != channels * block.expansion:
            downsample = BasicConv([self.channels, channels * block.expansion], stride=stride, act=None)

        layers = [block(self.channels, channels, stride, kernel_size, self.multi_order, downsample, self.groups,
                        self.base_width, previous_dilation)]
        self.channels = channels * block.expansion
        for _ in range(1, n_block):
            layers.append(block(self.channels, channels, kernel_size=kernel_size, multi_order=self.multi_order,
                                groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return MultiSeq(*layers)

    def forward(self, x):  # now is b * 4 * c *n
        if self.use_tnet:
            aligned_pos = self.tnet3(x[:, :3, :])
            x = torch.cat((aligned_pos, x[:, 3:, :]), dim=1)
        # edge_index = self.knn(x[:, :, :3, :].detach()) #now is b * 4 * c *n

        x = self.conv1(x)  # b*64*n*t
        # logging.info('Size after conv1: {}'.format(x.size()))

        x = self.layer1(x)
        # logging.info('Size after layer1: {}'.format(x.size()))
        max_1 = self.maxpool1(x)
        x = self.layer2(x)
        # logging.info('Size after layer2: {}'.format(x.size()))
        # x = self.layer3(x, edge_index)
        # logging.info('Size after layer3: {}'.format(x.size()))
        # x = self.layer4(x, edge_index)
        max_4 = self.maxpool4(x)
        avg_4 = self.avgpool4(x)
        # logging.info('Size after layer4: {}'.format(x.size()))

        x = torch.cat((x, max_1.repeat((1, 1, self.n_points, self.multi_order)), max_4, avg_4), dim=1)
        x = self.maxpoolpre(x).squeeze(-1)

        x = self.pred1(x)
        x = self.pred2(x)
        x = self.pred3(x)
        return x


def _resnet(block, layers, kernel_size=3, **kwargs):  # layers shi duo shaoceng
    model = ResNet(block, layers, kernel_size, **kwargs)
    return model


def sfc_resnet_8(kernel_size=3, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(BasicBlock, [1, 1, 1, 1], kernel_size, **kwargs)


def resnet18(kernel_size=3, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], kernel_size, **kwargs)


if __name__ == '__main__':
    numeric_level = getattr(logging, 'INFO', None)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        level=numeric_level)
    log_format = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)04d] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    x = torch.rand((5, 9, 1024, 4), dtype=torch.float)
    label = torch.randint(40, (5, 1024), dtype=torch.long)
    kernel_size = 3
    net = sfc_resnet_8(kernel_size=kernel_size, in_channels=9, num_classes=40, n_points=1024, use_tnet=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    out = net(x)  # 5*40*1024  yebusuanshi one hot ,zhishi yige jiaocha yanzheng
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, label)  # label = 5 * 1024
    # out = out.mean(dim=1)
    optimizer.zero_grad()
    with torch.autograd.set_detect_anomaly(True):
        loss.backward()
    optimizer.step()
    logging.info('Output size {}'.format(out.size()))
    logging.info('Output size {}'.format(out.size()))
    logging.info('Output size {}'.format(out.size()))




