from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from PIL import Image
import torch.cuda.amp
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import Subset, Dataset, ConcatDataset, RandomSampler, BatchSampler, Sampler, DataLoader
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SwaLoader():
    def __init__(self, trainloader, batchsize, robust_samples):
        self.trainloader = trainloader
        self.batchsize = batchsize
        self.robust_samples = robust_samples

    def concatenate_collate_fn(self, batch):
        concatenated_batch = []
        for images, label in batch:
            concatenated_batch.extend(images)
        return torch.stack(concatenated_batch)

    def get_swa_dataloader(self):
        # Create a new DataLoader with the custom collate function
        swa_dataloader = DataLoader(
            dataset=self.trainloader.dataset,
            batch_size=self.batchsize,
            num_workers=self.trainloader.num_workers,
            collate_fn=self.concatenate_collate_fn
        )
        return swa_dataloader

class CustomDataset(Dataset):
    def __init__(self, np_images, original_dataset, resize):
        # Load images
        self.images = torch.from_numpy(np_images).permute(0, 3, 1, 2) / 255
        if resize == True:
            self.images = transforms.Resize(224, antialias=True)(self.images)

        # Extract labels from the original PyTorch dataset
        self.labels = [label for _, label in original_dataset]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Get image and label for the given index
        image = self.images[index]
        label = self.labels[index]

        return image, label

class CombinedDataset(Dataset):
    def __init__(self, original_dataset, generated_dataset, target_size, generated_ratio=0.0, transform=None):

        self.generated_ratio = generated_ratio
        self.transform = transform  # Save the transform passed to the constructor
        self.original_length = len(original_dataset)
        self.target_size = target_size

        if generated_dataset == None or generated_ratio == 0.0:
            self.images, self.labels = zip(*original_dataset)
            if isinstance(self.images[0], torch.Tensor):
                self.images = TF.to_pil_image(self.images)
            self.sources = [True] * self.original_length
        else:
            self.original_generated_length = len(generated_dataset['image'])

            self.num_generated = int(self.target_size * self.generated_ratio)
            self.num_original = self.target_size - self.num_generated
            print('new permutation')
            # Create a single permutation for the whole epoch
            original_perm = torch.randperm(self.original_length)
            generated_perm = torch.randperm(self.original_generated_length)

            original_indices = original_perm[0:self.num_original]
            generated_indices = generated_perm[0:self.num_generated]
            generated_images = list(map(Image.fromarray, generated_dataset['image'][generated_indices]))
            generated_labels = generated_dataset['label'][generated_indices]

            original_subset = Subset(original_dataset, original_indices)
            original_images, original_labels = zip(*original_subset)
            if isinstance(original_images[0], torch.Tensor):
                original_images = TF.to_pil_image(original_images)

            # Prepare lists for combined data
            self.images = [None] * self.target_size
            self.labels = [None] * self.target_size
            self.sources = [None] * self.target_size

            # Transform and append original data
            self.images[:self.num_original] = original_images
            self.labels[:self.num_original] = original_labels
            self.sources[:self.num_original] = [True] * self.num_original

            # Append NPZ data
            self.images[self.num_original:self.target_size] = generated_images
            self.labels[self.num_original:self.target_size] = generated_labels
            self.sources[self.num_original:self.target_size] = [False] * self.num_generated

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        source = self.sources[idx]

        # Apply the transformation if it exists
        if self.transform:
            image = self.transform(image)

        return image, label, source

class BalancedRatioSampler(Sampler):
    def __init__(self, dataset, generated_ratio, batch_size):
        super(BalancedRatioSampler, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.generated_ratio = generated_ratio
        self.size = len(dataset)

        self.num_generated = int(self.size * self.generated_ratio)
        self.num_original = self.size - self.num_generated
        self.num_generated_batch = int(self.batch_size * self.generated_ratio)
        self.num_original_batch = self.batch_size - self.num_generated_batch

    def __iter__(self):
        # Create a single permutation for the whole epoch.
        # generated permutation requires generated images appended to the back of the dataset!
        original_perm = torch.randperm(self.num_original)
        generated_perm = torch.randperm(self.num_generated) + self.num_original

        generated_residual_batch = self.num_generated_batch
        original_residual_batch = self.num_original_batch

        batch_starts = range(0, self.size, self.batch_size)  # Start points for each batch
        for i, start in enumerate(batch_starts):
            # Calculate end to avoid going out of bounds
            togo = self.size - start
            if togo < self.batch_size:
                generated_residual_batch = int(togo * self.generated_ratio)
                original_residual_batch = togo - generated_residual_batch
            # Slicing the permutation to get batch indices
            original_indices = original_perm[i * self.num_original_batch:i * self.num_original_batch + original_residual_batch]
            generated_indices = generated_perm[i * self.num_generated_batch:i * self.num_generated_batch + generated_residual_batch]

            # Combine
            batch_indices = torch.cat((original_indices, generated_indices))
            shuffled_batch_indices = batch_indices[torch.randperm(batch_indices.size(0))]

            yield shuffled_batch_indices.tolist()

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size

class AugmentedDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform augmentations and allow robust loss functions."""

  def __init__(self, dataset, transforms_preprocess, transforms_augmentation, transforms_generated=None,
               robust_samples=0):
    self.dataset = dataset
    self.preprocess = transforms_preprocess
    self.transforms_augmentation = transforms_augmentation
    self.transforms_generated = transforms_generated if transforms_generated else transforms_augmentation
    self.robust_samples = robust_samples
    self.original_length = getattr(dataset, 'original_length', None)
    self.generated_length = getattr(dataset, 'generated_length', None)

  def __getitem__(self, i):
    x, y, original = self.dataset[i]
    augment = self.transforms_augmentation if original == True else self.transforms_generated
    if self.robust_samples == 0:
      return augment(x), y
    elif self.robust_samples == 1:
      im_tuple = (self.preprocess(x), augment(x))
      return im_tuple, y
    elif self.robust_samples == 2:
      im_tuple = (self.preprocess(x), augment(x), augment(x))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)

def load_data(dataset, validontest, transforms_preprocess, transforms_augmentation = None, transforms_basic = None,
              transforms_generated = None, run=0, robust_samples=0, generated_ratio=0.0):

    # Trainset and Validset
    if transforms_augmentation is not None:
        if dataset == 'ImageNet' or dataset == 'TinyImageNet':
            base_trainset = torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}/train')
        else:
            load_helper = getattr(torchvision.datasets, dataset)
            base_trainset = load_helper(root='./experiments/data', train=True, download=True)

        if validontest == False:
            validsplit = 0.2
            train_indices, val_indices, _, _ = train_test_split(
                range(len(base_trainset)),
                base_trainset.targets,
                stratify=base_trainset.targets,
                test_size=validsplit,
                random_state=run)  # same validation split when calling train multiple times, but a random new validation on multiple runs
            base_trainset = Subset(base_trainset, train_indices)
            validset = Subset(base_trainset, val_indices)
            validset = list(map(transforms_preprocess, validset))
        else:
            if dataset == 'ImageNet' or dataset == 'TinyImageNet':
                validset = torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}/val',
                                                            transform=transforms_preprocess)
            elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
                load_helper = getattr(torchvision.datasets, dataset)
                validset = load_helper(root='./experiments/data', train=False, download=True,
                                       transform=transforms_preprocess)
            else:
                print('Dataset not loadable')

        generated_dataset = np.load(f'./experiments/data/{dataset}-add-1m-dm.npz') if generated_ratio > 0.0 else None
        trainset = CombinedDataset(base_trainset, generated_dataset, target_size=len(base_trainset),
                                   generated_ratio=generated_ratio, transform=transforms_basic)
        trainset = AugmentedDataset(trainset, transforms_preprocess, transforms_augmentation, transforms_generated,
                                    robust_samples)

    else:
        trainset = None
        validset = None

    #Testset
    if dataset == 'ImageNet' or dataset == 'TinyImageNet':
        testset = torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}/val',
                                                    transform=transforms_preprocess)
    elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
        load_helper = getattr(torchvision.datasets, dataset)
        testset = load_helper(root='./experiments/data', train=False, download=True, transform=transforms_preprocess)
    else:
        print('Dataset not loadable')
    num_classes = len(testset.classes)

    return trainset, validset, testset, num_classes

def load_data_c(dataset, testset, resize, test_transforms, subset, subsetsize):

    c_datasets = []
    #c-corruption benchmark: https://github.com/hendrycks/robustness
    corruptions_c = np.asarray(np.loadtxt('./experiments/data/c-labels.txt', dtype=list))

    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        #c-bar-corruption benchmark: https://github.com/facebookresearch/augmentation-corruption
        corruptions_bar = np.asarray(np.loadtxt('./experiments/data/c-bar-labels-cifar.txt', dtype=list))
        corruptions = [(string, 'c') for string in corruptions_c] + [(string, 'c-bar') for string in corruptions_bar]

        for corruption, set in corruptions:
            subtestset = testset
            np_data_c = np.load(f'./experiments/data/{dataset}-{set}/{corruption}.npy')
            np_data_c = np.array(np.array_split(np_data_c, 5))

            if subset == True:
                np.random.seed(0)
                selected_indices = np.random.choice(10000, subsetsize, replace=False)
                subtestset = Subset(testset, selected_indices)
                np_data_c = [intensity_dataset[selected_indices] for intensity_dataset in np_data_c]

            concat_intensities = ConcatDataset([CustomDataset(intensity_data_c, subtestset, resize) for intensity_data_c in np_data_c])
            c_datasets.append(concat_intensities)

    elif dataset == 'ImageNet' or dataset == 'TinyImageNet':
        #c-bar-corruption benchmark: https://github.com/facebookresearch/augmentation-corruption
        corruptions_bar = np.asarray(np.loadtxt('./experiments/data/c-bar-labels-IN.txt', dtype=list))
        corruptions = [(string, 'c') for string in corruptions_c] + [(string, 'c-bar') for string in corruptions_bar]
        for corruption, set in corruptions:
            intensity_datasets = [torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}-{set}/' + corruption + '/' + str(intensity),
                                                                   transform=test_transforms) for intensity in range(1, 6)]
            if subset == True:
                selected_indices = np.random.choice(len(intensity_datasets[0]), subsetsize, replace=False)
                intensity_datasets = [Subset(intensity_dataset, selected_indices) for intensity_dataset in intensity_datasets]
            concat_intensities = ConcatDataset(intensity_datasets)
            c_datasets.append(concat_intensities)
    else:
        print('No corrupted benchmark available other than CIFAR10-c, CIFAR100-c, TinyImageNet-c and ImageNet-c.')
        return

    if subset == True:
        c_datasets = ConcatDataset(c_datasets)
        c_datasets_dict = {'combined': c_datasets}
    else:
        c_datasets_dict = {label: dataset for label, dataset in zip([corr for corr, _ in corruptions], c_datasets)}

    return c_datasets_dict

def load_loader(trainset, validset, batchsize, number_workers, generated_ratio=0.0):
    if generated_ratio > 0.0:
        CustomSampler = BalancedRatioSampler(trainset, generated_ratio=generated_ratio,
                                             batch_size=batchsize)
    else:
        CustomSampler = BatchSampler(RandomSampler(trainset), batch_size=batchsize, drop_last=False)
    trainloader = DataLoader(trainset, batch_sampler=CustomSampler, pin_memory=False, num_workers=number_workers)
    validationloader = DataLoader(validset, batch_size=batchsize, pin_memory=False, num_workers=number_workers)

    return trainloader, validationloader

def create_transforms(dataset, aug_strat_check, train_aug_strat, resize = False, RandomEraseProbability = 0.0):
    # list of all data transformations used
    t = transforms.ToTensor()
    c32 = transforms.RandomCrop(32, padding=4)
    c64 = transforms.RandomCrop(64, padding=8)
    flip = transforms.RandomHorizontalFlip()
    r224 = transforms.Resize(224, antialias=True)
    r256 = transforms.Resize(256, antialias=True)
    c224 = transforms.CenterCrop(224)
    rrc224 = transforms.RandomResizedCrop(224, antialias=True)
    re = transforms.RandomErasing(p=RandomEraseProbability)#, value='random')

    # transformations of validation/test set and necessary transformations for training
    # always done (even for clean images while training, when using robust loss)
    if dataset == 'ImageNet':
        transforms_preprocess = transforms.Compose([t, r256, c224])
    elif resize == True:
        transforms_preprocess = transforms.Compose([t, r224])
    else:
        transforms_preprocess = transforms.Compose([t])

    # augmentations of training set before tensor transformation
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        transforms_aug_1 = transforms.Compose([flip, c32])
    elif dataset == 'TinyImageNet':
        transforms_aug_1 = transforms.Compose([flip, c64])
    else:
        transforms_aug_1 = transforms.Compose([flip])
    if aug_strat_check == True:
        tf = getattr(transforms, train_aug_strat)
        transforms_augmentation = transforms.Compose([tf(), transforms_preprocess])
    else:
        transforms_augmentation = transforms.Compose([transforms_preprocess])

    # augmentations of training set after tensor transformation
    transforms_aug_2 = transforms.Compose([re])

    transforms_augmentation = transforms.Compose([transforms_augmentation, transforms_aug_2])
    transforms_basic = transforms.Compose([transforms_aug_1])

    return transforms_preprocess, transforms_augmentation, transforms_basic

def normalization_values(batch, dataset, normalized, manifold=False, manifold_factor=1):

    if manifold:
        mean = torch.mean(batch, dim=(0, 2, 3), keepdim=True).to(device)
        std = torch.std(batch, dim=(0, 2, 3), keepdim=True).to(device)
        mean = mean.view(1, batch.size(1), 1, 1)
        std = ((1 / std) / manifold_factor).view(1, batch.size(1), 1, 1)
    elif normalized:
        if dataset == 'CIFAR10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1).to(device)
        elif dataset == 'CIFAR100':
            mean = torch.tensor([0.50707516, 0.48654887, 0.44091784]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.26733429, 0.25643846, 0.27615047]).view(1, 3, 1, 1).to(device)
        elif (dataset == 'ImageNet' or dataset == 'TinyImageNet'):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        else:
            print('no normalization values set for this dataset')
    else:
        mean = 0
        std = 1

    return mean, std

def apply_augstrat(batch, train_aug_strat):

    for id, img in enumerate(batch):
        img = img * 255.0
        img = img.type(torch.uint8)
        tf = getattr(transforms, train_aug_strat)
        img = tf()(img)
        img = img.type(torch.float32) / 255.0
        batch[id] = img

    return batch
