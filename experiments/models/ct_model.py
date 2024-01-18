import copy
import torch
import torch.nn as nn
import numpy as np
from experiments.mixup import mixup_process
from experiments.noise import apply_lp_corruption
from experiments.data import normalization_values

class CtModel(nn.Module):

    def __init__(self, dataset, normalized, num_classes, mixup_alpha, mixup_manifold, cutmix_alpha,
                 noise_minibatchsize, corruptions, concurrent_combinations):
        super(CtModel, self).__init__()
        self.normalized = normalized
        self.mixup_alpha = mixup_alpha
        self.mixup_manifold = mixup_manifold
        self.cutmix_alpha = cutmix_alpha
        self.num_classes = num_classes
        self.noise_minibatchsize = noise_minibatchsize
        self.corruptions = corruptions
        self.concurrent_combinations = concurrent_combinations
        self.dataset = dataset
        if normalized:
            mean, std = normalization_values(dataset)
            self.register_buffer('mu', mean)
            self.register_buffer('sigma', std)

    def forward_normalize(self, x):
        if self.normalized:
            x = (x - self.mu) / self.sigma
        return x

    def forward_noise_mixup(self, out, targets):

        #define where mixup is applied. k=0 is in the input space, k>0 is in the embedding space (manifold mixup)
        if self.training == False: k = -2
        elif self.mixup_alpha > 0.0 and self.mixup_manifold == True: k = np.random.choice(range(3), 1)[0]
        elif self.mixup_alpha > 0.0 or self.cutmix_alpha > 0.0: k = 0
        else: k = -1
        if k == -1:
            out = apply_lp_corruption(out, self.noise_minibatchsize, self.corruptions, self.concurrent_combinations, self.normalized, self.dataset)
        if k == 0:  # Do input mixup if k is 0
            out, targets = mixup_process(out, targets, self.num_classes, self.cutmix_alpha, self.mixup_alpha, manifold=False)
            out = apply_lp_corruption(out, self.noise_minibatchsize, self.corruptions, self.concurrent_combinations, self.normalized, self.dataset)
        out = self.blocks[0](out)

        for i, ResidualBlock in enumerate(self.blocks[1:]):
            out = ResidualBlock(out)
            #print(out.max(), out.min(), out.mean(), out.std())
            if k == (i + 1):  # Do manifold mixup if k is greater 0
                with torch.no_grad():
                    out, targets = mixup_process(out, targets, self.num_classes, self.cutmix_alpha, self.mixup_alpha, manifold=True)
                    out = apply_lp_corruption(out, self.noise_minibatchsize, self.corruptions, self.concurrent_combinations, self.normalized, self.dataset)

        return out, targets

