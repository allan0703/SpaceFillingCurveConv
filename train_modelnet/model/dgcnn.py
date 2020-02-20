import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
from gcn_lib.dense import GraphConv2d, DilatedKnn2d, BasicConv


class Edgeconv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Edgeconv, self).__init__()
        self.conv1 = GraphConv2d(in_planes, out_planes, conv='edge', act='leakyrelu', norm='batch', bias=False)

    def forward(self, x, edge_index=None):
        x = self.conv1(x, edge_index)
        return x


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone='xception', kernel_size=9, sigma=1.0):
        super(Decoder, self).__init__()
        low_level_inplanes = 512

        self.fusion_conv = BasicConv([low_level_inplanes, 1024], act='leakyrelu', norm='batch')
        self.classifier = nn.Sequential(BasicConv([1024 * 2, 512], act='leakyrelu', norm='batch'),
                                        torch.nn.Dropout(0.5),
                                        BasicConv([512, 256], act='leakyrelu', norm='batch'),
                                        torch.nn.Dropout(0.5),
                                        BasicConv([256, num_classes], act=None, norm=None))
        self._init_weight()

    def forward(self, x, coords):
        fusion = self.fusion_conv(torch.cat(x, dim=1).unsqueeze(-1))
        x1 = F.adaptive_max_pool2d(fusion, 1)
        x2 = F.adaptive_avg_pool2d(fusion, 1)
        logits = self.classifier(torch.cat((x1, x2), dim=1)).squeeze(-1).squeeze(-1)
        return logits

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DGCNN(nn.Module):
    def __init__(self, input_size=3, num_classes=40, kernel_size=9):
        super(DGCNN, self).__init__()
        self.knn = DilatedKnn2d(kernel_size, dilation=1, self_loop=False)
        self.conv1 = Edgeconv(input_size, 64)
        self.conv2 = Edgeconv(64, 64)
        self.conv3 = Edgeconv(64, 128)
        self.conv4 = Edgeconv(128, 256)
        self.decoder = Decoder(num_classes, backbone='resnet14', kernel_size=1, sigma=1.0)

    def forward(self, x, coords, edge_index=None):
        edge_index = self.knn(x)  # test our knn
        out1 = self.conv1(x.unsqueeze(-1), edge_index)

        edge_index = self.knn(out1)
        out2 = self.conv2(out1.unsqueeze(-1), edge_index)

        edge_index = self.knn(out2)
        out3 = self.conv3(out2.unsqueeze(-1), edge_index)

        edge_index = self.knn(out3)
        out4 = self.conv4(out3.unsqueeze(-1), edge_index)
        out = self.decoder((out1, out2, out3, out4), coords)
        return out


def dgcnn(input_size=3, num_classes=40):
    r"""ResNet-14 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return DGCNN(input_size, num_classes)


if __name__ == '__main__':
    numeric_level = getattr(logging, 'INFO', None)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        level=numeric_level)
    log_format = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)04d] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    device = torch.device('cpu')

    feats = torch.rand((4, 3, 2048), dtype=torch.float).to(device)
    coords = torch.rand((4, 3, 2048), dtype=torch.float).to(device)
    # edge_index = torch.randint((0, 2048, ()), dtype=torch.float).to(device)
    k = 40

    knn = DilatedKnn2d(9, dilation=1, self_loop=False)
    edge_index = knn(feats)

    net = dgcnn().to(device)

    start_time = time.time()
    out = net(feats, coords, edge_index)
    logging.info('It took {:f}s'.format(time.time() - start_time))
    # out = out.mean(dim=1)
    logging.info('Output size {}'.format(out.size()))

    out.mean().backward()
