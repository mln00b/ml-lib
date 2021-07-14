# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

import torch.nn as nn
import torch.nn.functional as F

from .norm import get_norm_layer

from typing import List

class BasicBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, norm_type: str, stride: int = 1, expansion: int = 1):
        super().__init__()
        assert norm_type in ["LN", "BN"], f"norm_type not supported: {norm_type}"
        self.expansion = expansion
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.norm1 = get_norm_layer(out_ch, norm_type)
        self.norm2 = get_norm_layer(out_ch, norm_type)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != expansion * out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, expansion * out_ch, kernel_size=1, stride=stride, bias=False),
                get_norm_layer(expansion * out_ch, norm_type)
            )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block: nn.Module, blocks_ls: List, norm_type: str, num_classes: int=10):
        super().__init__()
        self.in_ch = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = get_norm_layer(64, norm_type)
        self.layer1 = self._make_layer(block, 64, blocks_ls[0], norm_type, stride=1)
        self.layer2 = self._make_layer(block, 128, blocks_ls[1], norm_type, stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_ls[2], norm_type, stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_ls[3], norm_type, stride=1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: nn.Module, ch: int, num_blocks: int, norm_type: str, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_ in strides:
            layers.append(block(self.in_ch, ch, norm_type, stride_))
            self.in_ch = ch * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(norm_type="BN"):
    return ResNet(BasicBlock, [2, 2, 2, 2], norm_type)


def ResNet34(norm_type="BN"):
    return ResNet(BasicBlock, [3, 4, 6, 3], norm_type)