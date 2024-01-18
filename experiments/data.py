from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.cuda.amp
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import Subset

def load_data(transform_test, dataset, validontest, transform_train = None, run=0):

    load_helper = getattr(torchvision.datasets, dataset)

    # Trainset
    if transform_train is not None:
        if dataset == 'ImageNet' or dataset == 'TinyImageNet':
            trainset = torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}/train',
                                                    transform=transform_train)
        else:
            trainset = load_helper(root='./experiments/data', train=True, download=True, transform=transform_train)
    else:
        trainset = None

    # Validation/Testset
    if validontest == True:
        if dataset == 'ImageNet' or dataset == 'TinyImageNet':
            validset = torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}/val',
                                                        transform=transform_test)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            validset = load_helper(root='./experiments/data', train=False, download=True, transform=transform_test)
        else:
            print('Dataset not loadable')
    else:
        if dataset == 'ImageNet' or dataset == 'TinyImageNet':
            trainset_clean = torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}/train',
                                                              transform=transform_test)
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            trainset_clean = load_helper(root='./experiments/data', train=True, download=True, transform=transform_test)
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

    num_classes = len(validset.classes)

    return trainset, validset, num_classes

def create_transforms(dataset, train_aug_strat, resize = False, RandomEraseProbability = 0.0):
    # list of all data transformations used
    t = transforms.ToTensor()
    c32 = transforms.RandomCrop(32, padding=4)
    c64 = transforms.RandomCrop(64, padding=8)
    flip = transforms.RandomHorizontalFlip()
    r256 = transforms.Resize(256, antialias=True)
    c224 = transforms.CenterCrop(224)
    rrc224 = transforms.RandomResizedCrop(224, antialias=True)
    re = transforms.RandomErasing(p=RandomEraseProbability, value='random')
    tf = getattr(transforms, train_aug_strat)

    # transformations of validation/test set
    transforms_test = transforms.Compose([t])
    if dataset == 'ImageNet':
        transforms_test = transforms.Compose([transforms_test, r256, c224])
    elif resize == True:
        transforms_test = transforms.Compose([transforms_test, transforms.Resize(224, antialias=True)])
    else:
        transforms_test = transforms.Compose([transforms_test])

    # transformations of training set
    transforms_train = transforms.Compose([flip, t, re])
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        transforms_train = transforms.Compose([transforms_train, c32])
    elif dataset == 'TinyImageNet':
        transforms_train = transforms.Compose([transforms_train, c64])
    elif dataset == 'ImageNet':
        transforms_train = transforms.Compose([transforms_train, rrc224])

    return transforms_train, transforms_test

def normalization_values(dataset):
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

def apply_augstrat(batch, train_aug_strat, mini):

    for id, img in enumerate(batch):
        img = img * 255.0
        img = img.type(torch.uint8)
        tf = getattr(transforms, train_aug_strat)
        img = tf()(img)
        img = img.type(torch.float32) / 255.0
        batch[id] = img

    return batch

class AugmentedDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform augmentation and allow loss functions."""

  def __init__(self, dataset, preprocess, robust_samples=0):
    self.dataset = dataset
    self.preprocess = preprocess
    self.robust_samples = robust_samples

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.robust_samples == 0:
      return aug(x, self.preprocess), y
    elif self.robust_samples == 1:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess))
      return im_tuple, y
    elif self.robust_samples == 2:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                  aug(x, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)