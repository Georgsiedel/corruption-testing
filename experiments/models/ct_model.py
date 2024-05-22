from experiments.utils import plot_images
import torch.nn as nn
import numpy as np
from experiments.mixup import mixup_process
from experiments.noise import apply_noise, noise_up, apply_noise_add_and_mult
from experiments.data import normalization_values

class CtModel(nn.Module):

    def __init__(self, dataset, normalized, num_classes):
        super(CtModel, self).__init__()
        self.normalized = normalized
        self.num_classes = num_classes
        self.dataset = dataset
        if normalized:
            mean, std = normalization_values(batch=None, dataset=dataset, normalized=normalized, manifold=False, manifold_factor=1)
            self.register_buffer('mu', mean)
            self.register_buffer('sigma', std)

    def forward_normalize(self, x):
        if self.normalized:
            x = (x - self.mu) / self.sigma
        return x

    def forward_noise_mixup(self, out, targets, robust_samples, corruptions, mixup_alpha, mixup_p, manifold,
                            manifold_noise_factor, cutmix_alpha, cutmix_p, noise_minibatchsize,
                            concurrent_combinations, noise_sparsity, noise_patch_lower_scale):

        #define where mixup is applied. k=0 is in the input space, k>0 is in the embedding space (manifold mixup)
        if self.training == False: k = -1
        elif mixup_alpha > 0.0 and manifold == True: k = np.random.choice(range(3), 1)[0] #, p=[0.5, 0.25, 0.25]
        else: k = 0

        original_batchsize = (out.size(0) // (robust_samples + 1))

        if k == 0:  # Do input mixup if k is 0
            mixed_out, targets = mixup_process(out, targets, robust_samples, self.num_classes, mixup_alpha, mixup_p,
                                         cutmix_alpha, cutmix_p, manifold=False, inplace=True)
            #mixed = mixed_out.clone()
            #noisy_out = apply_noise_add_and_mult(mixed_out, noise_minibatchsize,
            #                                        corruptions, self.normalized, self.dataset, noise_patch_lower_scale=
            #                                                noise_patch_lower_scale, noise_sparsity=noise_sparsity)
            #noisy_out = noise_up(mixed_out, robust_samples=robust_samples, add_noise_level=0.4, mult_noise_level=0.2,
            #                            sparse_level=noise_sparsity, l0_level=0.2)
            #noisy_out = torch.cat((noisy_out[:original_batchsize, :, :, :],
            #                       apply_noise_add_and_mult(noisy_out[original_batchsize:, :, :, :], noise_minibatchsize,
            #                                        corruptions, self.normalized, self.dataset, noise_patch_lower_scale=
            #                                                noise_patch_lower_scale, noise_sparsity=noise_sparsity)),
            #                       dim=0)
            out = mixed_out
            #plot_images(noisy_out, mixed, 3)

        out = self.blocks[0](out)

        for i, ResidualBlock in enumerate(self.blocks[1:]):
            out = ResidualBlock(out)
            if k == (i + 1):  # Do manifold mixup if k is greater 0
                out, targets = mixup_process(out, targets, robust_samples, self.num_classes, mixup_alpha, mixup_p,
                                         cutmix_alpha, cutmix_p, manifold=True, inplace=True)
                out = noise_up(out, robust_samples=robust_samples, add_noise_level=1.0, mult_noise_level=0.5,
                                        sparse_level=noise_sparsity, l0_level=0.2)
        return out, targets

