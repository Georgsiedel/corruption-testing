import re
import torch
import torchvision
import math
from torchvision import datasets
import torchvision.transforms as transforms
from skimage.util import random_noise
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.distributions as dist
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from experiments.data import normalization_values

class Noise_Sampler(torch.nn.Module):
    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Please provide a valid positive value for the num_classes. Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch, target):

        def apply_lp_corruption(batch, minibatchsize, combine_train_corruptions, train_corruptions,
                                concurrent_combinations, max, noise, epsilon):

            minibatches = batch.view(-1, minibatchsize, batch.size()[1], batch.size()[2], batch.size()[3])
            epsilon = float(epsilon)

            for id, minibatch in enumerate(minibatches):
                if combine_train_corruptions == True:
                    corruptions_list = random.sample(list(train_corruptions), k=concurrent_combinations)
                    for x, (noise_type, train_epsilon, max) in enumerate(corruptions_list):
                        train_epsilon = float(train_epsilon)
                        minibatch = sample_lp_corr_batch(noise_type, train_epsilon, minibatch, max)
                    minibatches[id] = minibatch
                else:
                    minibatch = sample_lp_corr_batch(noise, epsilon, minibatch, max)
                    minibatches[id] = minibatch
            batch = minibatches.view(-1, batch.size()[1], batch.size()[2], batch.size()[3])

            return batch

def get_image_mask(batch, noise_patch_lower_scale=1.0, ratio=[0.3, 3.3]):
        """Get image mask for Patched Noise (see e.g. Patch Gaussian paper).
        Args:
            batch (Tensor): batch of images to be masked.
            scale (sequence): range of proportion of masked area against input image.
            ratio (sequence): range of aspect ratio of masked area.
        """
        if noise_patch_lower_scale == 1.0:
            return torch.ones(batch.size(), dtype=torch.bool, device=device)

        img_c, img_h, img_w = batch.shape[-3], batch.shape[-2], batch.shape[-1]
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))

        patched_area = area * torch.empty(1).uniform_(noise_patch_lower_scale, 1.0).item()
        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        h = int(round(math.sqrt(patched_area * aspect_ratio)))
        w = int(round(math.sqrt(patched_area / aspect_ratio)))
        if h > img_h:
            h = img_h
            w = int(round(img_w * patched_area / area)) #reset patched area ratio when patch needs to be cropped due to aspect ratio
        if w > img_w:
            w = img_w
            h = int(round(img_h * patched_area / area)) #reset patched area ratio when patch needs to be cropped due to aspect ratio
        i = torch.randint(0, img_h - h + 1, size=(1,)).item()
        j = torch.randint(0, img_w - w + 1, size=(1,)).item()
        mask = torch.zeros(batch.size(), dtype=torch.bool, device=device)
        mask[:,:,i:i + h, j:j + w] = True

        return mask

def apply_noise(batch, minibatchsize, corruptions, concurrent_combinations, normalized, dataset,
                        manifold=False, manifold_factor=1, noise_sparsity=0.0, noise_patch_lower_scale=1.0):

    if corruptions is None:
        return batch

    #Calculate the mean values for each channel across all images
    mean, std = normalization_values(batch, dataset, normalized, manifold, manifold_factor)

    if manifold:
    # Throw out noise outside Gaussian, (L0) and Linf for manifold noise (since epsilon is dependent on dimensionality)
        if not isinstance(corruptions, dict):
            corruptions = [c for c in corruptions if c.get('noise_type') in {'gaussian', 'uniform-linf', 'standard', 'uniform-l0-impulse'}]
            corruptions = np.array(corruptions)
            if corruptions.size == 0:
                print('Warning: noise_type of p-norm outside L0 and Linf may not be applicable for manifold noise')
        else:
            if corruptions.get('noise_type') != ('gaussian' or 'uniform-linf' or 'standard' or 'uniform-l0-impulse'):
                print('Warning: noise_type of p-norm outside L0 and Linf may not be applicable for manifold noise')

    minibatches = batch.view(-1, minibatchsize, batch.size()[1], batch.size()[2], batch.size()[3])

    for id, minibatch in enumerate(minibatches):
        #no dict means corruption combination, so we choose randomly, dict means one single corruption
        if not isinstance(corruptions, dict): #in case of a combination of corruptions (combined_corruption = True)
            corruptions_list = random.sample(list(corruptions), k=concurrent_combinations)
        else:
            corruptions_list = [corruptions]

        #noisy_minibatch = torch.clone(minibatch)
        for _, (corruption) in enumerate(corruptions_list):
            if corruption['distribution'] == 'uniform':
                d = dist.Uniform(0, 1).sample()
            elif corruption['distribution'] == 'beta2-5':
                d = np.random.beta(2, 5)
            elif corruption['distribution'] == 'max':
                d = 1
            else:
                print('Unknown distribution for epsilon value of p-norm corruption. Max value chosen instead.')
                d = 1
            if d == 0: #dist sampling include lower bound but exclude upper, but we do not want eps = 0
                d = 1
            epsilon = float(d) * float(corruption['epsilon'])
            sparsity = random.random() * noise_sparsity
            noisy_minibatch = sample_lp_corr_batch(corruption['noise_type'], epsilon, minibatch, corruption['sphere'], mean, std, manifold, sparsity)

        mask = get_image_mask(minibatch, noise_patch_lower_scale = noise_patch_lower_scale, ratio = [0.3, 3.3])
        final_minibatch = torch.where(mask, noisy_minibatch, minibatch)
        minibatches[id] = final_minibatch

    batch = minibatches.view(-1, batch.size()[1], batch.size()[2], batch.size()[3])
    return batch

def sample_lp_corr_batch(noise_type, epsilon, batch, density_distribution_max, mean, std, manifold, sparsity=0.0):
    with torch.cuda.device(0):
        corruption = torch.zeros(batch.size(), dtype=torch.float16)

        if noise_type == 'uniform-linf':
            if density_distribution_max == True:  # sample on the hull of the norm ball
                rand = np.random.random(batch.shape)
                sign = np.where(rand < 0.5, -1, 1)
                corruption = torch.from_numpy(sign * epsilon)
            else: #sample uniformly inside the norm ball
                corruption = torch.cuda.FloatTensor(batch.shape).uniform_(-epsilon, epsilon)
        elif noise_type == 'gaussian': #note that this has no option for density_distribution=max
            corruption = torch.cuda.FloatTensor(batch.shape).normal_(0, epsilon)
        elif noise_type == 'uniform-l0-impulse':
            num_dimensions = torch.numel(batch[0])
            num_pixels = int(num_dimensions * epsilon)
            lower_bounds = torch.arange(0, batch.size(0) * batch[0].numel(), batch[0].numel(), device=device)
            upper_bounds = torch.arange(batch[0].numel(), (batch.size(0) + 1) * batch[0].numel(), batch[0].numel(), device=device)
            indices = torch.cat([torch.randint(l, u, (num_pixels,), device=device) for l, u in zip(lower_bounds, upper_bounds)])
            mask = torch.full(batch.size(), False, dtype=torch.bool, device=device)
            mask.view(-1)[indices] = True
            #apply sparsity: a share of the random impulse noise pixels is left out
            sparse_matrix = torch.cuda.FloatTensor(batch.shape).uniform_()
            mask[sparse_matrix < sparsity] = False

            if density_distribution_max == True:
                random_numbers = torch.randint(2, size=batch.size(), dtype=torch.float16, device=device)
            else:
                random_numbers = torch.cuda.FloatTensor(batch.shape).uniform_(0, 1)
            if manifold:
                random_numbers = random_numbers - 0.5

            #normalizing the impulse noise values
            random_numbers = (random_numbers - mean) / std
            batch_corr = torch.where(mask, random_numbers, batch)

            return batch_corr

        elif 'uniform-l' in noise_type:  #Calafiore1998: Uniform Sample Generation in lp Balls for Probabilistic Robustness Analysis
            img_corr = torch.zeros(batch[0].size(), dtype=torch.float16, device=device)
            #number of dimensions
            d = img_corr.numel()
            # extract Lp-number from args.noise variable
            lp = [float(x) for x in re.findall(r'-?\d+\.?\d*', noise_type)][0]
            u = dist.Gamma(1 / lp, 1).sample(img_corr.shape).to(device)
            u = u ** (1 / lp)
            sign = torch.where(torch.cuda.FloatTensor(img_corr.shape).uniform_(0, 1) < 0.5, -1, 1)
            norm = torch.sum(abs(u) ** lp) ** (1 / lp)  # scalar, norm samples to lp-norm-sphere
            if density_distribution_max == True:
                r = 1
            else:  # uniform density distribution
                r = dist.Uniform(0, 1).sample() ** (1.0 / d)
            img_corr = epsilon * r * u * sign / norm #image-sized corruption, epsilon * random radius * random array / normed
            corruption = img_corr.expand(batch.size()).to(device)

        elif noise_type == 'standard':
            return batch
        else:
            print('Unknown type of noise')

        corruption = corruption.to(device)#.clone()
        #sparsity is applied
        sparse_matrix = torch.cuda.FloatTensor(batch.shape).uniform_()
        #corruption[sparse_matrix < sparsity] = 0
        sparse_corruption = torch.where(sparse_matrix < sparsity, 0, corruption)

        corrupted_batch = batch + (sparse_corruption / std)
        if not manifold:
            corrupted_batch = torch.clamp(corrupted_batch, (0-mean)/std, (1-mean)/std)  #clip below lower and above upper bound
        return corrupted_batch

def sample_lp_corr_img(noise_type, epsilon, img, density_distribution_max):
    d = len(img.ravel())
    if epsilon == 0:
        img_corr = img
    else:
        if noise_type == 'uniform-linf':
            if density_distribution_max == True:  # sample on the hull of the norm ball
                rand = np.random.random(np.array(img).shape)
                sign = np.where(rand < 0.5, -1, 1)
                img_corr = img + (sign * epsilon)
            else: #sample uniformly inside the norm ball
                img_corr = dist.Uniform(img - epsilon, img + epsilon).sample()
            img_corr = np.clip(img_corr, 0, 1) # clip values below 0 and over 1
        elif noise_type == 'uniform-linf-brightness': #only max-distribution, every pixel gets same manipulation
            img_corr = img
            img_corr = random.choice([img_corr - epsilon, img_corr + epsilon])
            img_corr = np.clip(img_corr, 0, 1) # clip values below 0 and over 1
        elif noise_type == 'gaussian': #note that this has no option for density_distribution=max
            var = epsilon * epsilon
            img_corr = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=var, clip=True))
        elif noise_type == 'uniform-l0-impulse':
            num_pixels = int(img.numel() * epsilon)
            indices = torch.randperm(img.numel(), device=device)[:num_pixels]
            mask = torch.full(img.size(), False, dtype=torch.bool, device=device)
            mask.view(-1)[indices] = True
            if density_distribution_max == True:
                random_numbers = torch.randint(2, size=img.size(), dtype=torch.float16).to(device)

            else:
                random_numbers = torch.cuda.FloatTensor(img.shape).uniform_(0, 1)
            img_corr = torch.where(mask, random_numbers, img)
            return img_corr
        elif 'uniform-l' in noise_type:  #Calafiore1998: Uniform Sample Generation in lp Balls for Probabilistic Robustness Analysis
            lp = [float(x) for x in re.findall(r'-?\d+\.?\d*', noise_type)]  # extract Lp-number from args.noise variable
            lp = lp[0]
            u = np.random.gamma(1/lp, 1, size=(np.array(img).shape))  # image-sized array of Laplace-distributed random variables (distribution beta factor equalling Lp-norm)
            u = u ** (1/lp)
            rand = np.random.random(np.array(img).shape)
            sign = np.where(rand < 0.5, -1, 1)
            norm = np.sum(abs(u) ** lp) ** (1 / lp)  # scalar, norm samples to lp-norm-sphere
            if density_distribution_max == True:
                r = 1 # 1 to leave the sampled points on the hull of the norm ball, to sample uniformly within use this: np.random.random() ** (1.0 / d)
            else: #uniform density distribution
                r = np.random.random() ** (1.0 / d)
            corr = epsilon * r * u * sign / norm  #image-sized corruption, epsilon * random radius * random array / normed
            img_corr = img + corr  # construct corrupted image by adding sampled noise
            img_corr = np.clip(img_corr, 0, 1) #clip values below 0 and over 1
        else:
            img_corr = img
            print('Unknown type of noise')
    return img_corr

#Sample 3 images in original form and with a chosen maximum corruption of a chose Lp norm.
#Use this e.g. to estimate maximum Lp-corruptions which should not change the class, or which are quasi-imperceptible.
def sample_lp_corr_visualization(n_images = 3, seed = -1, noise_type = 'uniform-linf', epsilon = 8/255, density_distribution = "max", dataset = 'CIFAR10'):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == 'ImageNet' or dataset == 'TinyImageNet':
        trainset = torchvision.datasets.ImageFolder(root=f'./data/{dataset}/train', transform=transform)
        truncated_set = torch.utils.data.Subset(trainset, range(0, 100000, 5000))
        loader = torch.utils.data.DataLoader(truncated_set, batch_size=1, shuffle=False)
    else:
        load_helper = getattr(datasets, dataset)
        trainset = load_helper(root='./data', train=True, download=True, transform=transform)
        truncated_set = torch.utils.data.Subset(trainset, range(0, 100000, 5000))
        loader = torch.utils.data.DataLoader(truncated_set, batch_size=1, shuffle=False)

    fig, axs = plt.subplots(n_images, 2)
    j = seed
    for i in range(n_images):
        if seed == -1:
            j = random.randint(0, len(loader)) # selecting random images from the train dataset
        for id, (input, target) in enumerate(loader):
            if j == id:
                image = input
                corrupted_image = input
                break
        image = torch.squeeze(image)
        corrupted_image = torch.squeeze(corrupted_image)
        img_corr = sample_lp_corr_img(noise_type, epsilon, corrupted_image, density_distribution)
        corrupted_image = corrupted_image + img_corr  # construct corrupted image by adding sampled noise
        corrupted_image = np.clip(corrupted_image, 0, 1)  # clip values below 0 and over 1
        image = image.permute(1, 2, 0)
        corrupted_image = corrupted_image.permute(1, 2, 0)
        axs[i, 0].imshow(image)
        axs[i, 1].imshow(corrupted_image)
        j = j+1
    return fig

if __name__ == '__main__':
    fig = sample_lp_corr_img(3, -1, 'uniform-linf', 0.01, "max", "TinyImageNet")
    plt.show()