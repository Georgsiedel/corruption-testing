from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

import torch.cuda.amp
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import Subset, Dataset, ConcatDataset
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


class AugmentedDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform augmentations and allow robust loss functions."""

  def __init__(self, dataset, transforms_preprocess, transforms_augmentation, robust_samples=0):
    self.dataset = dataset
    self.preprocess = transforms_preprocess
    self.augment = transforms_augmentation
    self.robust_samples = robust_samples

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.robust_samples == 0:
      return self.augment(x), y
    elif self.robust_samples == 1:
      im_tuple = (self.preprocess(x), self.augment(x))
      return im_tuple, y
    elif self.robust_samples == 2:
      im_tuple = (self.preprocess(x), self.augment(x), self.augment(x))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)

def load_data(transforms_preprocess, dataset, validontest, transforms_augmentation = None, run=0, robust_samples=0):

    # Trainset
    if transforms_augmentation is not None:
        if dataset == 'ImageNet' or dataset == 'TinyImageNet':
            trainset = AugmentedDataset(torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}/train'),
                                        transforms_preprocess, transforms_augmentation, robust_samples=robust_samples)
        else:
            load_helper = getattr(torchvision.datasets, dataset)
            trainset = AugmentedDataset(load_helper(root='./experiments/data', train=True, download=True),
                                        transforms_preprocess, transforms_augmentation, robust_samples=robust_samples)
    else:
        trainset = None

    # Validationset
    if validontest == True:
        if dataset == 'ImageNet' or dataset == 'TinyImageNet':
            validset = torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}/val',
                                                        transform=transforms_preprocess)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            load_helper = getattr(torchvision.datasets, dataset)
            validset = load_helper(root='./experiments/data', train=False, download=True, transform=transforms_preprocess)
        else:
            print('Dataset not loadable')
    else:
        if dataset == 'ImageNet' or dataset == 'TinyImageNet':
            trainset_clean = torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}/train',
                                                              transform=transforms_preprocess)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            load_helper = getattr(torchvision.datasets, dataset)
            trainset_clean = load_helper(root='./experiments/data', train=True, download=True, transform=transforms_preprocess)
        else:
            print('Dataset not loadable')

        validsplit = 0.2
        train_indices, val_indices, _, _ = train_test_split(
            range(len(trainset)),
            trainset.targets,
            stratify=trainset.targets,
            test_size=validsplit,
            random_state=run)  # same validation split when calling train multiple times, but a random new validation on multiple runs
        trainset = Subset(trainset, train_indices)
        validset = Subset(trainset_clean, val_indices)

    #Testset
    if dataset == 'ImageNet' or dataset == 'TinyImageNet':
        testset = torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}/val',
                                                    transform=transforms_preprocess)
    elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
        load_helper = getattr(torchvision.datasets, dataset)
        testset = load_helper(root='./experiments/data', train=False, download=True, transform=transforms_preprocess)
    else:
        print('Dataset not loadable')


    num_classes = len(validset.classes)

    return trainset, validset, testset, num_classes

def load_data_c(dataset, testset, resize, test_transforms, subset, subsetsize):

    corruptions = np.loadtxt('./experiments/data/c-labels.txt', dtype=list)
    np.asarray(corruptions)
    c_datasets = []
    for corruption in corruptions:
        if dataset == 'CIFAR10' or dataset == 'CIFAR100':
            subtestset = testset
            np_data_c = np.load(f'./experiments/data/{dataset}-c/{corruption}.npy')
            np_data_c = np.array(np.array_split(np_data_c, 5))
            if subset == True:
                np.random.seed(0)
                selected_indices = np.random.choice(10000, subsetsize, replace=False)
                subtestset = Subset(testset, selected_indices)
                np_data_c = [intensity_dataset[selected_indices] for intensity_dataset in np_data_c]
            concat_intensities = ConcatDataset([CustomDataset(intensity_data_c, subtestset, resize) for intensity_data_c in np_data_c])

        elif dataset == 'ImageNet' or dataset == 'TinyImageNet':
            intensity_datasets = [torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}-c/' + corruption + '/' + str(intensity), transform=test_transforms) for intensity in range(1, 6)]
            if subset == True:
                selected_indices = np.random.choice(len(intensity_datasets[0]), subsetsize, replace=False)
                intensity_datasets = [Subset(intensity_dataset, selected_indices) for intensity_dataset in intensity_datasets]
            concat_intensities = ConcatDataset(intensity_datasets)
        else:
            print('No corrupted benchmark available other than CIFAR10-c, CIFAR100-c, TinyImageNet-c and ImageNet-c.')

        c_datasets.append(concat_intensities)

    if subset == True:
        c_datasets = ConcatDataset(c_datasets)
        c_datasets_dict = {'combined': c_datasets}
    else:
        c_datasets_dict = {label: dataset for label, dataset in zip(corruptions, c_datasets)}

    return c_datasets_dict

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
    elif dataset == 'ImageNet':
        transforms_aug_1 = transforms.Compose([flip])
    if aug_strat_check == True:
        tf = getattr(transforms, train_aug_strat)
        transforms_aug_1 = transforms.Compose([transforms_aug_1, tf()])

    # augmentations of training set after tensor transformation
    transforms_aug_2 = transforms.Compose([re])

    transforms_augmentation = transforms.Compose([transforms_aug_1, transforms_preprocess, transforms_aug_2])

    return transforms_preprocess, transforms_augmentation

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
