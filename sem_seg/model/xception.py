import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AlignedXception']


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    # padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    padded_inputs = F.pad(inputs, (pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=None):
        super(SeparableConv2d, self).__init__()

        # to account for long 1D sequence
        # dilation = int(dilation ** 2)
        # stride = int(stride ** 2)

        self.conv1 = nn.Conv1d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv1d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, norm_layer=None,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        # to account for long 1D sequence
        # dilation = int(dilation ** 2)
        # stride = int(stride ** 2)

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
            self.skipbn = norm_layer(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 9, 1, dilation, BatchNorm=norm_layer))
            rep.append(norm_layer(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 9, 1, dilation, BatchNorm=norm_layer))
            rep.append(norm_layer(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 9, 1, dilation, BatchNorm=norm_layer))
            rep.append(norm_layer(planes))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 9, 2, BatchNorm=norm_layer))
            rep.append(norm_layer(planes))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 9, 1, BatchNorm=norm_layer))
            rep.append(norm_layer(planes))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x


class AlignedXception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, output_stride, input_size, norm_layer=None):
        super(AlignedXception, self).__init__()

        # if output_stride == 16:
        #     entry_block3_stride = 2
        #     middle_block_dilation = 1
        #     exit_block_dilations = (1, 2)
        # elif output_stride == 8:
        #     entry_block3_stride = 1
        #     middle_block_dilation = 2
        #     exit_block_dilations = (2, 4)
        # else:
        #     raise NotImplementedError
        entry_block3_stride = 1
        middle_block_dilation = 2
        exit_block_dilations = (2, 4)
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        # Entry flow
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=9, stride=2, padding=4, bias=False)
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4, bias=False)
        self.bn2 = norm_layer(64)

        self.block1 = Block(64, 128, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False,
                            grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, norm_layer=norm_layer,
                            start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        self.block4 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                            norm_layer=norm_layer, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                            norm_layer=norm_layer, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                            norm_layer=norm_layer, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                            norm_layer=norm_layer, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                            norm_layer=norm_layer, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                            norm_layer=norm_layer, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             norm_layer=norm_layer, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             norm_layer=norm_layer, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             norm_layer=norm_layer, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             norm_layer=norm_layer, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             norm_layer=norm_layer, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             norm_layer=norm_layer, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             norm_layer=norm_layer, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             norm_layer=norm_layer, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             norm_layer=norm_layer, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             norm_layer=norm_layer, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                             norm_layer=norm_layer, start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d(1024, 1536, 9, stride=1, dilation=exit_block_dilations[1], BatchNorm=norm_layer)
        self.bn3 = norm_layer(1536)

        self.conv4 = SeparableConv2d(1536, 1536, 9, stride=1, dilation=exit_block_dilations[1], BatchNorm=norm_layer)
        self.bn4 = norm_layer(1536)

        self.conv5 = SeparableConv2d(1536, 2048, 9, stride=1, dilation=exit_block_dilations[1], BatchNorm=norm_layer)
        self.bn5 = norm_layer(2048)

        # Init weights
        self._init_weight()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print('After conv 1: ', x.size())

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print('After conv 2: ', x.size())

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        # print('After block 1: ', x.size())
        low_level_feat = x
        x = self.block2(x)
        # print('After block 2: ', x.size())
        x = self.block3(x)
        # print('After block 3: ', x.size())

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        # print('After block 12: ', x.size())
        x = self.block13(x)
        # print('After block 13: ', x.size())
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    import torch
    model = AlignedXception(norm_layer=nn.BatchNorm1d, output_stride=16, input_size=9)
    x = torch.rand(1, 9, 4096)
    output, low_level_feat = model(x)
    print(output.size())
    print(low_level_feat.size())
