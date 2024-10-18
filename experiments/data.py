import random
import time

import torch
from PIL import Image
import torch.cuda.amp
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import Subset, Dataset, ConcatDataset, RandomSampler, BatchSampler, Sampler, DataLoader
import numpy as np
import style_transfer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
            num_workers=0,
            collate_fn=self.concatenate_collate_fn,
            worker_init_fn=self.trainloader.worker_init_fn,
            generator=self.trainloader.generator
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
            #batch_indices = batch_indices[torch.randperm(batch_indices.size(0))]

            yield batch_indices.tolist()

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size

class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform augmentations and allow robust loss functions."""

    def __init__(self, images, labels, sources, transforms_preprocess, transforms_basic, transforms_augmentation,
                 transforms_generated=None, robust_samples=0):
        self.images = images
        self.labels = labels
        self.sources = sources
        self.preprocess = transforms_preprocess
        self.transforms_basic = transforms_basic
        self.transforms_augmentation = transforms_augmentation
        self.transforms_generated = transforms_generated if transforms_generated else transforms_augmentation
        self.robust_samples = robust_samples

    def __getitem__(self, i):
        x = self.images[i]
        aug_strat = self.transforms_augmentation if self.sources[i] == True else self.transforms_generated
        augment = transforms.Compose([self.transforms_basic, aug_strat])
        if self.robust_samples == 0:
          return augment(x), self.labels[i]
        elif self.robust_samples == 1:
          im_tuple = (self.preprocess(x), augment(x))
          return im_tuple, self.labels[i]
        elif self.robust_samples == 2:
          im_tuple = (self.preprocess(x), augment(x), augment(x))
          return im_tuple, self.labels[i]

    def __len__(self):
        return len(self.labels)

class RandomChoiceTransforms:
    def __init__(self, transforms, p):
        assert len(transforms) == len(p), "The number of transforms and probabilities must match."

        self.transforms = transforms
        self.p = p

    def __call__(self, x):
        choice = random.choices(self.transforms, self.p)[0]
        return choice(x)

class CustomTA_color(transforms_v2.TrivialAugmentWide):
    _AUGMENTATION_SPACE = {
    #"Identity": (lambda num_bins, height, width: None, False),
    #"ShearX": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    #"ShearY": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    #"TranslateX": (lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins), True),
    #"TranslateY": (lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins), True),
    #"Rotate": (lambda num_bins, height, width: torch.linspace(0.0, 135.0, num_bins), True),
    "Brightness": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "Color": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "Contrast": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "Sharpness": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "Posterize": (lambda num_bins, height, width: (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6))).round().int(), False),
    "Solarize": (lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins), False),
    "AutoContrast": (lambda num_bins, height, width: None, False),
    "Equalize": (lambda num_bins, height, width: None, False)
    }

class CustomTA_geometric(transforms_v2.TrivialAugmentWide):
    _AUGMENTATION_SPACE = {
    "Identity": (lambda num_bins, height, width: None, False),
    "ShearX": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "ShearY": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "TranslateX": (lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins), True),
    "TranslateY": (lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins), True),
    "Rotate": (lambda num_bins, height, width: torch.linspace(0.0, 135.0, num_bins), True),
    #"Brightness": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    #"Color": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    #"Contrast": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    #"Sharpness": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    #"Posterize": (lambda num_bins, height, width: (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6))).round().int(), False),
    #"Solarize": (lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins), False),
    #"AutoContrast": (lambda num_bins, height, width: None, False),
    #"Equalize": (lambda num_bins, height, width: None, False)
    }

class DataLoading():
    def __init__(self, dataset, epochs=200, generated_ratio=0.0, resize = False, run=0):
        self.dataset = dataset
        self.generated_ratio = generated_ratio
        self.resize = resize
        self.run = run
        self.epochs = epochs

    def create_transforms(self, aug_strat_check, train_aug_strat_orig, train_aug_strat_gen, RandomEraseProbability=0.0):
        # list of all data transformations used
        t = transforms.ToTensor()
        c32 = transforms.RandomCrop(32, padding=4)
        c64 = transforms.RandomCrop(64, padding=8)
        flip = transforms.RandomHorizontalFlip()
        r224 = transforms.Resize(224, antialias=True)
        r256 = transforms.Resize(256, antialias=True)
        c224 = transforms.CenterCrop(224)
        rrc224 = transforms.RandomResizedCrop(224, antialias=True)
        re = transforms.RandomErasing(p=RandomEraseProbability, scale=(0.02, 0.4), value='random')

        # transformations of validation/test set and necessary transformations for training
        # always done (even for clean images while training, when using robust loss)
        if self.dataset == 'ImageNet':
            self.transforms_preprocess = transforms.Compose([t, r256, c224])
        elif self.resize == True:
            self.transforms_preprocess = transforms.Compose([t, r224])
        else:
            self.transforms_preprocess = transforms.Compose([t])

        # standard augmentations of training set, without tensor transformation
        if self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
            self.transforms_basic = transforms.Compose([flip, c32])
        elif self.dataset == 'TinyImageNet':
            self.transforms_basic = transforms.Compose([flip, c64])
        else:
            self.transforms_basic = transforms.Compose([flip])

        # additional transforms with tensor transformation, Random Erasing after tensor transformation
        if aug_strat_check == True:

            TAc = CustomTA_color()
            TAg = CustomTA_geometric()

            def stylization():
                vgg, decoder = style_transfer.load_models()
                style_feats = style_transfer.load_feat_files()

                Stylize = style_transfer.NSTTransform(style_feats, vgg, decoder, alpha_min=0.2, alpha_max=1.0,
                                                      probability=0.9)
                return Stylize

            def transform_not_found(train_aug_strat, dataset):
                print('Training augmentation strategy', train_aug_strat, 'could not be found. Proceeding without '
                                                                         'augmentation strategy for.', dataset, '.')
                return transforms.Compose([self.transforms_preprocess, re])

            transform_map = {
                "TAorRE": RandomChoiceTransforms([transforms.Compose([TAc, self.transforms_preprocess]),
                                                  transforms.Compose([TAg, self.transforms_preprocess]),
                                                  transforms.Compose([self.transforms_preprocess, transforms.RandomErasing(p=1.0, scale=(0.02, 0.4), value='random')]),
                                                  transforms.Compose([self.transforms_preprocess, transforms.RandomErasing(p=1.0, scale=(0.02, 0.4), value=0)])],
                                            [0.4,0.3,0.15,0.15]),
                "TAc+TAg+RE": transforms.Compose([CustomTA_color(), CustomTA_geometric(), self.transforms_preprocess, re]),
                "TAc+TAgorRE": transforms.Compose([CustomTA_color(),
                                         RandomChoiceTransforms([transforms.Compose([CustomTA_geometric(), self.transforms_preprocess]),
                                                                 transforms.Compose(
                                             [self.transforms_preprocess,
                                              transforms.RandomErasing(p=1.0, scale=(0.02, 0.4), value='random')]),
                                                                transforms.Compose([
                                              self.transforms_preprocess,
                                              transforms.RandomErasing(p=1.0, scale=(0.02, 0.4), value=0)])],
                                        [6, 1, 1])]),
                "TAc+REorTAg": RandomChoiceTransforms([transforms.Compose([TAg, self.transforms_preprocess]),
                                                       transforms.Compose([TAc, self.transforms_preprocess,
                                                             transforms.RandomErasing(p=0.525, scale=(0.02, 0.4),
                                                                                      value='random')])],
                                        [6, 8]),
                "StyleTransfer": transforms.Compose([stylization(), self.transforms_preprocess, re]),
                "TAorStyle0.75": transforms.Compose([RandomChoiceTransforms([transforms_v2.TrivialAugmentWide(), stylization()], [1, 3]),
                                                    self.transforms_preprocess, re]),
                "TAorStyle0.5": transforms.Compose([RandomChoiceTransforms([transforms_v2.TrivialAugmentWide(), stylization()], [1, 1]),
                                                    self.transforms_preprocess, re]),
                "TAorStyle0.25": transforms.Compose([RandomChoiceTransforms([transforms_v2.TrivialAugmentWide(), stylization()], [3, 1]),
                                                    self.transforms_preprocess, re]),
                "TAorStyle0.1": transforms.Compose([RandomChoiceTransforms([transforms_v2.TrivialAugmentWide(), stylization()], [9, 1]),
                                                     self.transforms_preprocess, re]),
                "StyleAndTA": transforms.Compose([stylization(), transforms_v2.TrivialAugmentWide(), self.transforms_preprocess, re]),
                "TrivialAugmentWide": transforms.Compose([transforms_v2.TrivialAugmentWide(),
                                                          self.transforms_preprocess, re]),
                "RandAugment": transforms.Compose([transforms_v2.RandAugment(),
                                                          self.transforms_preprocess, re]),
                "AutoAugment": transforms.Compose([transforms_v2.AutoAugment(),
                                                          self.transforms_preprocess, re]),
                "AugMix": transforms.Compose([transforms_v2.AugMix(),
                                                          self.transforms_preprocess, re]),
                'None': transforms.Compose([self.transforms_preprocess, re])
            }
            self.transforms_augmentation = (transform_map[train_aug_strat_orig]
                if train_aug_strat_orig in transform_map
                else transform_not_found(train_aug_strat_orig, 'transforms_original'))

            self.transforms_generated = (transform_map[train_aug_strat_gen]
                                            if train_aug_strat_gen in transform_map
                                            else transform_not_found(train_aug_strat_gen, 'transforms_generated'))
        else:
            self.transforms_augmentation = transforms.Compose([self.transforms_preprocess, re])

    def load_base_data(self, validontest, run=0):

        # Trainset and Validset
        if self.transforms_augmentation is not None:
            if self.dataset == 'ImageNet' or self.dataset == 'TinyImageNet':
                self.base_trainset = torchvision.datasets.ImageFolder(root=f'./experiments/data/{self.dataset}/train')
            else:
                load_helper = getattr(torchvision.datasets, self.dataset)
                self.base_trainset = load_helper(root='./experiments/data', train=True, download=True)

            if validontest == False:
                validsplit = 0.2
                train_indices, val_indices, _, _ = train_test_split(
                    range(len(self.base_trainset)),
                    self.base_trainset.targets,
                    stratify=self.base_trainset.targets,
                    test_size=validsplit,
                    random_state=run)  # same validation split for same runs, but new validation on multiple runs
                self.base_trainset = Subset(self.base_trainset, train_indices)
                self.validset = Subset(self.base_trainset, val_indices)
                self.validset = list(map(self.transforms_preprocess, self.validset))
            else:
                if self.dataset == 'ImageNet' or self.dataset == 'TinyImageNet':
                    self.validset = torchvision.datasets.ImageFolder(root=f'./experiments/data/{self.dataset}/val',
                                                                transform=self.transforms_preprocess)
                elif self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
                    load_helper = getattr(torchvision.datasets, self.dataset)
                    self.validset = load_helper(root='./experiments/data', train=False, download=True,
                                           transform=self.transforms_preprocess)
                else:
                    print('Dataset not loadable')
        else:
            self.trainset = None
            self.validset = None

        #Testset
        if self.dataset == 'ImageNet' or self.dataset == 'TinyImageNet':
            self.testset = torchvision.datasets.ImageFolder(root=f'./experiments/data/{self.dataset}/val',
                                                        transform=self.transforms_preprocess)
        elif self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
            load_helper = getattr(torchvision.datasets, self.dataset)
            self.testset = load_helper(root='./experiments/data', train=False, download=True, transform=self.transforms_preprocess)
        else:
            print('Dataset not loadable')
        self.num_classes = len(self.testset.classes)

    def load_augmented_traindata(self, target_size, epoch=0, robust_samples=0):

        self.robust_samples = robust_samples
        self.target_size = target_size
        self.generated_dataset = np.load(f'./experiments/data/{self.dataset}-add-1m-dm.npz',
                                    mmap_mode='r') if self.generated_ratio > 0.0 else None
        self.epoch = epoch

        torch.manual_seed(self.epoch + self.epochs * self.run)
        np.random.seed(self.epoch + self.epochs * self.run)
        random.seed(self.epoch + self.epochs * self.run)

        # Prepare lists for combined data
        images = [None] * self.target_size
        labels = [None] * self.target_size
        sources = [None] * self.target_size

        if self.generated_dataset == None or self.generated_ratio == 0.0:
            images, labels = zip(*self.base_trainset)
            if isinstance(images[0], torch.Tensor):
                images = TF.to_pil_image(images)
            sources = [True] * len(self.base_trainset)
        else:
            self.num_generated = int(self.target_size * self.generated_ratio)
            self.num_original = self.target_size - self.num_generated
            # Create a single permutation for the whole epoch
            original_perm = torch.randperm(len(self.base_trainset))
            generated_perm = torch.randperm(len(self.generated_dataset['image']))

            original_indices = original_perm[0:self.num_original]
            generated_indices = generated_perm[0:self.num_generated]
            generated_images = list(map(Image.fromarray, self.generated_dataset['image'][generated_indices]))
            generated_labels = self.generated_dataset['label'][generated_indices]

            original_subset = Subset(self.base_trainset, original_indices)
            original_images, original_labels = zip(*original_subset)
            if isinstance(original_images[0], torch.Tensor):
                original_images = TF.to_pil_image(original_images)

            # Transform and append original data
            images[:self.num_original] = original_images
            labels[:self.num_original] = original_labels
            sources[:self.num_original] = [True] * self.num_original

            # Append NPZ data
            images[self.num_original:self.target_size] = generated_images
            labels[self.num_original:self.target_size] = generated_labels
            sources[self.num_original:self.target_size] = [False] * self.num_generated

        self.trainset = AugmentedDataset(images, labels, sources, self.transforms_preprocess,
                                         self.transforms_basic, self.transforms_augmentation, self.transforms_generated,
                                         robust_samples)

    def load_data_c(self, subset, subsetsize):

        c_datasets = []
        #c-corruption benchmark: https://github.com/hendrycks/robustness
        corruptions_c = np.asarray(np.loadtxt('./experiments/data/c-labels.txt', dtype=list))

        if self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
            #c-bar-corruption benchmark: https://github.com/facebookresearch/augmentation-corruption
            corruptions_bar = np.asarray(np.loadtxt('./experiments/data/c-bar-labels-cifar.txt', dtype=list))
            corruptions = [(string, 'c') for string in corruptions_c] + [(string, 'c-bar') for string in corruptions_bar]

            for corruption, set in corruptions:
                subtestset = self.testset
                np_data_c = np.load(f'./experiments/data/{self.dataset}-{set}/{corruption}.npy')
                np_data_c = np.array(np.array_split(np_data_c, 5))

                if subset == True:
                    np.random.seed(0)
                    selected_indices = np.random.choice(10000, subsetsize, replace=False)
                    subtestset = Subset(self.testset, selected_indices)
                    np_data_c = [intensity_dataset[selected_indices] for intensity_dataset in np_data_c]

                concat_intensities = ConcatDataset([CustomDataset(intensity_data_c, subtestset, self.resize) for intensity_data_c in np_data_c])
                c_datasets.append(concat_intensities)

        elif self.dataset == 'ImageNet' or self.dataset == 'TinyImageNet':
            #c-bar-corruption benchmark: https://github.com/facebookresearch/augmentation-corruption
            corruptions_bar = np.asarray(np.loadtxt('./experiments/data/c-bar-labels-IN.txt', dtype=list))
            corruptions = [(string, 'c') for string in corruptions_c] + [(string, 'c-bar') for string in corruptions_bar]
            for corruption, set in corruptions:
                intensity_datasets = [torchvision.datasets.ImageFolder(root=f'./experiments/data/{self.dataset}-{set}/' + corruption + '/' + str(intensity),
                                                                       transform=self.transforms_preprocess) for intensity in range(1, 6)]
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
            self.c_datasets_dict = {'combined': c_datasets}
        else:
            self.c_datasets_dict = {label: dataset for label, dataset in zip([corr for corr, _ in corruptions], c_datasets)}

        return self.c_datasets_dict

    def get_loader(self, batchsize, number_workers):
        self.number_workers = number_workers

        g = torch.Generator()
        g.manual_seed(self.epoch + self.epochs * self.run)

        if self.generated_ratio > 0.0:
            self.CustomSampler = BalancedRatioSampler(self.trainset, generated_ratio=self.generated_ratio,
                                                 batch_size=batchsize)
        else:
            self.CustomSampler = BatchSampler(RandomSampler(self.trainset), batch_size=batchsize, drop_last=False)
        self.trainloader = DataLoader(self.trainset, batch_sampler=self.CustomSampler, pin_memory=True,
                                      num_workers=number_workers, worker_init_fn=seed_worker, generator=g)
        self.validationloader = DataLoader(self.validset, batch_size=batchsize, pin_memory=False, num_workers=0)

        return self.trainloader, self.validationloader

    def update_trainset(self, epoch, start_epoch):

        if self.generated_ratio != 0.0 and epoch != 0 and epoch != start_epoch:
            self.load_augmented_traindata(self.target_size, epoch=epoch, robust_samples=self.robust_samples)

        g = torch.Generator()
        g.manual_seed(self.epoch + self.epochs * self.run)
        self.trainloader = DataLoader(self.trainset, batch_sampler=self.CustomSampler, pin_memory=True,
                                      num_workers=self.number_workers, worker_init_fn=seed_worker, generator=g)
        return self.trainloader

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
