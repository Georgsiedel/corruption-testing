from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from experiments.models import ct_model

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class WideResNet(ct_model.CtModel):
    def __init__(self, depth, widen_factor, dataset, normalized, corruptions, dropout_rate=0.0, num_classes=10,
                 factor=1, block=WideBasic, mixup_alpha=0.0, mixup_manifold=False, cutmix_alpha=0.0,
                 noise_minibatchsize=1, concurrent_combinations = 1):
        super(WideResNet, self).__init__(dataset=dataset, normalized=normalized, num_classes=num_classes,
                                         mixup_alpha=mixup_alpha, mixup_manifold=mixup_manifold, cutmix_alpha=cutmix_alpha,
                                         corruptions=corruptions, noise_minibatchsize=noise_minibatchsize,
                                         concurrent_combinations=concurrent_combinations)
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (int)((depth-4)/6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0], stride=1)
        self.layer1 = self._wide_layer(block, nStages[1], n, dropout_rate, stride=factor)
        self.layer2 = self._wide_layer(block, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(block, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        self.blocks = [self.conv1, self.layer1, self.layer2, self.layer3]

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, targets):
        out = super(WideResNet, self).forward_normalize(x)
        out, mixed_targets = super(WideResNet, self).forward_noise_mixup(out, targets)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out, mixed_targets

def WideResNet_28_2(num_classes, factor, dataset, normalized, corruptions, block=WideBasic, dropout_rate=0.0,
                    mixup_alpha=0.0, mixup_manifold=False, cutmix_alpha=0.0, noise_minibatchsize=1, concurrent_combinations=1):
    return WideResNet(depth=28, widen_factor=2, dataset=dataset, normalized=normalized, dropout_rate=dropout_rate,
                      num_classes=num_classes, factor=factor, block=block, mixup_alpha=mixup_alpha,
                      mixup_manifold=mixup_manifold, cutmix_alpha=cutmix_alpha, corruptions=corruptions,
                      noise_minibatchsize=noise_minibatchsize, concurrent_combinations=concurrent_combinations)

def WideResNet_28_4(num_classes, factor, dataset, normalized, corruptions, block=WideBasic, dropout_rate=0.0,
                    mixup_alpha=0.0, mixup_manifold=False, cutmix_alpha=0.0, noise_minibatchsize=1, concurrent_combinations=1):
    return WideResNet(depth=28, widen_factor=4, dataset=dataset, normalized=normalized, dropout_rate=dropout_rate,
                      num_classes=num_classes, factor=factor, block=block, mixup_alpha=mixup_alpha,
                      mixup_manifold=mixup_manifold, cutmix_alpha=cutmix_alpha, corruptions=corruptions,
                      noise_minibatchsize=noise_minibatchsize, concurrent_combinations=concurrent_combinations)

def WideResNet_28_10(num_classes, factor, dataset, normalized, corruptions, block=WideBasic, dropout_rate=0.0,
                    mixup_alpha=0.0, mixup_manifold=False, cutmix_alpha=0.0, noise_minibatchsize=1, concurrent_combinations=1):
    return WideResNet(depth=28, widen_factor=10, dataset=dataset, normalized=normalized, dropout_rate=dropout_rate,
                      num_classes=num_classes, factor=factor, block=block, mixup_alpha=mixup_alpha,
                      mixup_manifold=mixup_manifold, cutmix_alpha=cutmix_alpha, corruptions=corruptions,
                      noise_minibatchsize=noise_minibatchsize, concurrent_combinations=concurrent_combinations)

def WideResNet_28_12(num_classes, factor, dataset, normalized, corruptions, block=WideBasic, dropout_rate=0.0,
                    mixup_alpha=0.0, mixup_manifold=False, cutmix_alpha=0.0, noise_minibatchsize=1, concurrent_combinations=1):
    return WideResNet(depth=28, widen_factor=12, dataset=dataset, normalized=normalized, dropout_rate=dropout_rate,
                      num_classes=num_classes, factor=factor, block=block, mixup_alpha=mixup_alpha,
                      mixup_manifold=mixup_manifold, cutmix_alpha=cutmix_alpha, corruptions=corruptions,
                      noise_minibatchsize=noise_minibatchsize, concurrent_combinations=concurrent_combinations)

def WideResNet_40_10(num_classes, factor, dataset, normalized, corruptions, block=WideBasic, dropout_rate=0.0,
                    mixup_alpha=0.0, mixup_manifold=False, cutmix_alpha=0.0, noise_minibatchsize=1, concurrent_combinations=1):
    return WideResNet(depth=40, widen_factor=10, dataset=dataset, normalized=normalized, dropout_rate=dropout_rate,
                      num_classes=num_classes, factor=factor, block=block, mixup_alpha=mixup_alpha,
                      mixup_manifold=mixup_manifold, cutmix_alpha=cutmix_alpha, corruptions=corruptions,
                      noise_minibatchsize=noise_minibatchsize, concurrent_combinations=concurrent_combinations)
