import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class CtModel(nn.Module):

    def __init__(self, dataset, normalized):
        super(CtModel, self).__init__()
        self.normalized = normalized

        if normalized:
            mean, std = self._normalize(dataset)
            self.register_buffer('mu', mean)
            self.register_buffer('sigma', std)

    def _normalize(self, dataset):
        if dataset == 'CIFAR10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            std = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1)
        elif dataset == 'CIFAR100':
            mean = torch.tensor([0.50707516, 0.48654887, 0.44091784]).view(1, 3, 1, 1)
            std = torch.tensor([0.26733429, 0.25643846, 0.27615047]).view(1, 3, 1, 1)
        elif (dataset == 'ImageNet' or dataset == 'TinyImageNet'):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        else:
            print('no normalization values set for this dataset')

        return mean, std

    def forward(self, x):
        if self.normalized:
            x = (x - self.mu) / self.sigma
        return self.forward_ctmodel(x)

    def forward_ctmodel(self, x):
        return None