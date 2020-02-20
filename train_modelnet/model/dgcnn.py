import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
from gcn_lib.dense import GraphConv2d, DilatedKnn2d


class Edgeconv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Edgeconv, self).__init__()
        self.conv1 = GraphConv2d(in_planes, out_planes, conv='edge', act='relu', norm='batch', bias=False)

    def forward(self, x, edge_index=None):
        x = self.conv1(x, edge_index)
        return x


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone='xception', kernel_size=9, sigma=1.0):
        super(Decoder, self).__init__()
        if backbone == 'resnet101':
            low_level_inplanes = 256
        elif backbone == 'resnet18' or "resnet14":
            low_level_inplanes = 64
        elif backbone == 'xception':
            low_level_inplanes = 128
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv1d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(48)
        self.relu = nn.ReLU()

        self.fusion_conv = nn.Conv1d(304, 256, kernel_size=1, bias=False)
        self.classifier = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, stride=1, bias=False),
                                       nn.BatchNorm1d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(256, num_classes, kernel_size=1, stride=1, bias=False)
                                        )
        # self.conv_out = nn.Conv1d(256, num_classes, kernel_size=1, stride=1)
        self._init_weight()

    def forward(self, x, low_level_feat, coords):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        # x = F.interpolate(x, size=low_level_feat.size()[-1], mode='linear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        fusion = self.fusion_conv(x)
        x1 = F.adaptive_max_pool1d(fusion, 1)
        x2 = F.adaptive_avg_pool1d(fusion, 1)
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
    def __init__(self, input_size=3, num_classes=40):
        super(DGCNN, self).__init__()
        self.knn = DilatedKnn2d(9, dilation=1, self_loop=False)
        self.conv1 = Edgeconv(input_size, 64)
        self.conv2 = Edgeconv(64, 128)
        self.conv3 = Edgeconv(128, 256)
        self.conv4 = Edgeconv(256, 256)
        self.decoder = Decoder(num_classes, backbone='resnet14', kernel_size=1, sigma=1.0)

    def forward(self, x, coords, edge_index=None):
        edge_index = self.knn(x)
        out = self.conv1(x.unsqueeze(-1), edge_index)
        low_level_feats = out
        out = self.conv2(out.unsqueeze(-1), edge_index)
        out = self.conv3(out.unsqueeze(-1), edge_index)
        out = self.conv4(out.unsqueeze(-1), edge_index)
        out = self.decoder(out, low_level_feats, coords)
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
