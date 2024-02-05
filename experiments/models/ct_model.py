import copy
import torch
import torch.nn as nn
import numpy as np
from experiments.mixup import mixup_process
from experiments.noise import apply_lp_corruption
from experiments.data import normalization_values

class CtModel(nn.Module):

    def __init__(self, dataset, normalized, num_classes, mixup, manifold, cutmix,
                 random_erase_p, noise_minibatchsize, corruptions, concurrent_combinations):
        super(CtModel, self).__init__()
        self.normalized = normalized
        self.mixup = mixup
        self.manifold = manifold
        self.cutmix = cutmix
        self.random_erase_p = random_erase_p
        self.num_classes = num_classes
        self.noise_minibatchsize = noise_minibatchsize
        self.corruptions = corruptions
        self.concurrent_combinations = concurrent_combinations
        self.dataset = dataset
        if normalized:
            mean, std = normalization_values(batch=None, dataset=dataset, normalized=normalized, manifold=False, manifold_factor=1)
            self.register_buffer('mu', mean)
            self.register_buffer('sigma', std)

    def forward_normalize(self, x):
        if self.normalized:
            x = (x - self.mu) / self.sigma
        return x

    def forward_noise_mixup(self, out, targets):
        #define where mixup is applied. k=0 is in the input space, k>0 is in the embedding space (manifold mixup)
        if self.training == False: k = -1
        elif self.mixup['alpha'] > 0.0 and self.manifold['apply'] == True: k = np.random.choice(range(3), 1, p=[0.5, 0.25, 0.25])[0] #
        else: k = 0
        if k == 0:  # Do input mixup if k is 0
            with torch.no_grad():
                out, targets = mixup_process(out, targets, self.num_classes, self.cutmix,
                                            self.mixup, self.random_erase_p, self.normalized, manifold=False)
                out = apply_lp_corruption(out, self.noise_minibatchsize, self.corruptions, self.concurrent_combinations,
                                          self.normalized, self.dataset, manifold=False, manifold_factor=1)
        out = self.blocks[0](out)

        for i, ResidualBlock in enumerate(self.blocks[1:]):
            out = ResidualBlock(out)
            #print(out.std())
            if k == (i + 1):  # Do manifold mixup if k is greater 0

                with torch.no_grad():
                    out, targets = mixup_process(out, targets, self.num_classes, self.cutmix,
                                    self.mixup, self.random_erase_p, self.normalized, manifold=True)
                    out = apply_lp_corruption(out, self.noise_minibatchsize, self.corruptions, self.concurrent_combinations,
                                    self.normalized, self.dataset, manifold=True, manifold_factor=self.manifold['noise_factor'])
        return out, targets

