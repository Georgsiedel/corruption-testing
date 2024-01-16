import argparse
import ast
import importlib
import numpy as np
from tqdm import tqdm
import shutil
import torch.nn as nn
import torch.cuda.amp
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as torchmodels
import torchvision.transforms as transforms
import copy

from experiments.jsd_loss import JsdCrossEntropy
import experiments.data as data
import experiments.checkpoints as checkpoints
import experiments.utils as utils
import experiments.models as low_dim_models

import torch.backends.cudnn as cudnn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#torch.backends.cudnn.enabled = False #this may resolve some cuDNN errors, but increases training time by ~200%
torch.cuda.set_device(0)
cudnn.benchmark = False #this slightly speeds up 32bit precision training (5%)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Error: Boolean value expected for argument {v}.')

class str2dictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Parse the dictionary string into a dictionary object
        # if values == '':

        try:
            dictionary = ast.literal_eval(values)
            if not isinstance(dictionary, dict):
                raise ValueError("Invalid dictionary format")
        except (ValueError, SyntaxError) as e:
            raise argparse.ArgumentTypeError(f"Invalid dictionary format: {values}") from e

        setattr(namespace, self.dest, dictionary)

parser = argparse.ArgumentParser(description='PyTorch Training with perturbations')
parser.add_argument('--resume', type=str2bool, nargs='?', const=False, default=False,
                    help='resuming from saved checkpoint in fixed-path repo defined below')
parser.add_argument('--traincorruptions', default={'noise_type': 'standard', 'epsilon': 0.0, 'sphere': False, 'distribution': 'max'},
                    type=str, action=str2dictAction, metavar='KEY=VALUE', help='dictionary for type of noise, epsilon value, '
                    'whether it is always the maximum noise value and a distribution from which various epsilon are sampled')
parser.add_argument('--run', default=0, type=int, help='run number')
parser.add_argument('--experiment', default=0, type=int,
                    help='experiment number - each experiment is defined in module config{experiment}')
parser.add_argument('--batchsize', default=128, type=int,
                    help='Images per batch - more means quicker training, but higher memory demand')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to choose')
parser.add_argument('--validontest', type=str2bool, nargs='?', const=True, default=True, help='For datasets wihtout '
                    'standout validation (e.g. CIFAR). True: Use full training data, False: Split 20% for valiationd')
parser.add_argument('--epochs', default=100, type=int, help="number of epochs")
parser.add_argument('--learningrate', default=0.1, type=float, help='learning rate')
parser.add_argument('--lrschedule', default='MultiStepLR', type=str, help='Learning rate scheduler from pytorch.')
parser.add_argument('--lrparams', default={'milestones': [85, 95], 'gamma': 0.2}, type=str, action=str2dictAction,
                    metavar='KEY=VALUE', help='parameters for the learning rate scheduler')
parser.add_argument('--earlystop', type=str2bool, nargs='?', const=False, default=False, help='Use earlystopping after '
                    'some epochs (patience) of no increase in performance')
parser.add_argument('--earlystopPatience', default=15, type=int,
                    help='Number of epochs to wait for a better performance if earlystop is True')
parser.add_argument('--optimizer', default='SGD', type=str, help='Optimizer from torch.optim')
parser.add_argument('--optimizerparams', default={'momentum': 0.9, 'weight_decay': 5e-4}, type=str,
                    action=str2dictAction, metavar='KEY=VALUE', help='parameters for the optimizer')
parser.add_argument('--modeltype', default='wideresnet', type=str,
                    help='Modeltype to train, use either default WRN28 or model from pytorch models')
parser.add_argument('--modelparams', default={}, type=str, action=str2dictAction, metavar='KEY=VALUE',
                    help='parameters for the chosen model')
parser.add_argument('--resize', type=str2bool, nargs='?', const=False, default=False,
                    help='Resize a model to 224x224 pixels, standard for models like transformers.')
parser.add_argument('--aug_strat_check', type=str2bool, nargs='?', const=True, default=False,
                    help='Whether to use an auto-augmentation scheme')
parser.add_argument('--train_aug_strat', default='TrivialAugmentWide', type=str, help='auto-augmentation scheme')
parser.add_argument('--jsd_loss', type=str2bool, nargs='?', const=False, default=False,
                    help='Whether to use Jensen-Shannon-Divergence loss function (enforcing smoother models)')
parser.add_argument('--mixup_alpha', default=0.0, type=float, help='Mixup Alpha parameter, Pytorch suggests 0.2. If '
                    'both mixup and cutmix are >0, mixup or cutmix are selected by 0.5 chance')
parser.add_argument('--cutmix_alpha', default=0.0, type=float, help='Cutmix Alpha parameter, Pytorch suggests 1.0. If '
                    'both mixup and cutmix are >0, mixup or cutmix are selected by 0.5 chance')
parser.add_argument('--mixup_manifold', type=str2bool, nargs='?', const=False, default=False,
                    help='Whether to apply mixup in the embedding layers of the network')
parser.add_argument('--combine_train_corruptions', type=str2bool, nargs='?', const=True, default=True,
                    help='Whether to combine all training noise values by drawing from the randomly')
parser.add_argument('--concurrent_combinations', default=1, type=int, help='How many of the training noise values should '
                    'be applied at once on one image. USe only if you defined multiple training noise values.')
parser.add_argument('--number_workers', default=4, type=int, help='How many workers are launched to parallelize data '
                    'loading. Experimental. 4 for ImageNet, 1 for Cifar. More demand GPU memory, but maximize GPU usage.')
parser.add_argument('--lossparams', default={'num_splits': 3, 'alpha': 12, 'smoothing': 0}, type=str, action=str2dictAction, metavar='KEY=VALUE',
                    help='parameters for the JSD loss function')
parser.add_argument('--RandomEraseProbability', default=0.0, type=float,
                    help='probability of applying random erasing to an image')
parser.add_argument('--warmupepochs', default=5, type=int,
                    help='Number of Warmupepochs for stable training early on. Start with factor 10 lower learning rate')
parser.add_argument('--normalize', type=str2bool, nargs='?', const=False, default=False,
                    help='Whether to normalize input data to mean=0 and std=1')
parser.add_argument('--pixel_factor', default=1, type=int, help='default is 1 for 32px (CIFAR10), '
                    'e.g. 2 for 64px images. Scales convolutions automatically in the same model architecture')
parser.add_argument('--minibatchsize', default=8, type=int, help='batchsize, for which a new corruption type is sampled. '
                    'batchsize must be a multiple of minibatchsize. in case of p-norm corruptions with 0<p<inf, the same '
                    'corruption is applied for all images in the minibatch')

args = parser.parse_args()
configname = (f'experiments.configs.config{args.experiment}')
config = importlib.import_module(configname)
if args.combine_train_corruptions == True:
    train_corruptions = config.train_corruptions
else:
    train_corruptions = args.train_corruptions
crossentropy = nn.CrossEntropyLoss(label_smoothing=args.lossparams["smoothing"])
jsdcrossentropy = JsdCrossEntropy(**args.lossparams)

def train_epoch(pbar):
    model.train()
    correct, total, train_loss, avg_train_loss = 0, 0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        if args.jsd_loss == True:
            inputs_orig, inputs_pert = copy.deepcopy(inputs), copy.deepcopy(inputs)
        if args.aug_strat_check == True:
            inputs = data.apply_augstrat(inputs, args.train_aug_strat)
            if args.jsd_loss == True:
                inputs_pert = data.apply_augstrat(inputs_pert, args.train_aug_strat)
        if args.jsd_loss == True:
            inputs = torch.cat((inputs_orig, inputs, inputs_pert), 0)

        inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device)
        with torch.cuda.amp.autocast():
            outputs, mixed_targets = model(inputs, targets)
            if args.jsd_loss == True:
                loss = jsdcrossentropy(outputs, mixed_targets)
            else:
                loss = crossentropy(outputs, mixed_targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        if np.ndim(mixed_targets) == 2:
            _, mixed_targets = mixed_targets.max(1)
        if args.jsd_loss == True:
            mixed_targets = torch.cat((mixed_targets, mixed_targets, mixed_targets), 0)
        total += mixed_targets.size(0)
        correct += predicted.eq(mixed_targets).sum().item()
        avg_train_loss = train_loss / (batch_idx + 1)
        pbar.set_description('[Train] Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(
            avg_train_loss, 100. * correct / total, correct, total))
        pbar.update(1)

    train_acc = 100. * correct / total
    return train_acc, avg_train_loss

def valid_epoch(pbar):
    model.eval()
    with torch.no_grad():
        test_loss, correct, total, avg_test_loss = 0, 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(validationloader):

            inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device)

            with torch.cuda.amp.autocast():
                outputs, targets = model(inputs, targets)
                loss = crossentropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            avg_test_loss = test_loss / (batch_idx + 1)
            pbar.set_description(
                '[Valid] Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(avg_test_loss, 100. * correct / total,
                                                                    correct, total))
            pbar.update(1)

    acc = 100. * correct / total
    return acc, avg_test_loss

if __name__ == '__main__':
    # Load and transform data
    print('Preparing data..')
    transform_train, transform_valid = data.create_transforms(args.dataset, args.resize, args.RandomEraseProbability)
    trainset, validset, num_classes = data.load_data(transform_valid, args.dataset, args.validontest, transform_train, run=args.run)
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, pin_memory=True, collate_fn=None, num_workers=args.number_workers)
    validationloader = DataLoader(validset, batch_size=args.batchsize, shuffle=True, pin_memory=True, num_workers=args.number_workers)

    # Construct model
    print(f'\nBuilding {args.modeltype} model with {args.modelparams} | Augmentation strategy: {args.aug_strat_check}'
          f' | JSD loss: {args.jsd_loss}')
    if args.dataset == 'CIFAR10' or 'CIFAR100' or 'TinyImageNet':
        model_class = getattr(low_dim_models, args.modeltype)
        model = model_class(dataset=args.dataset, normalized =args.normalize, corruptions = train_corruptions, num_classes=num_classes,
                            factor=args.pixel_factor, mixup_alpha=args.mixup_alpha, mixup_manifold=args.mixup_manifold,
                            cutmix_alpha=args.cutmix_alpha, noise_minibatchsize=args.minibatchsize,
                            concurrent_combinations = args.concurrent_combinations, **args.modelparams)
    else:
        model_class = getattr(torchmodels, args.modeltype)
        model = model_class(num_classes = num_classes, **args.modelparams)
    model = torch.nn.DataParallel(model).to(device)

    # Define Optimizer, Learningrate Scheduler, Scaler, and Early Stopping
    opti = getattr(optim, args.optimizer)
    optimizer = opti(model.parameters(), lr=args.learningrate, **args.optimizerparams)
    schedule = getattr(optim.lr_scheduler, args.lrschedule)
    scheduler = schedule(optimizer, **args.lrparams)
    if args.warmupepochs > 0:
        warmupscheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmupepochs)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmupscheduler, scheduler], milestones=[args.warmupepochs])
    scaler = torch.cuda.amp.GradScaler()
    Checkpointer = checkpoints.Checkpoint(earlystopping=args.earlystop, patience=args.earlystopPatience, verbose=False,
                                          model_path='experiments/trained_models/checkpoint.pt',
                                          best_model_path = 'experiments/trained_models/best_checkpoint.pt')

    # Some necessary parameters
    total_steps = utils.calculate_steps(args.dataset, args.batchsize, args.epochs, args.warmupepochs, args.validontest)
    train_accs, train_losses, valid_accs, valid_losses = [], [], [], []
    training_folder = 'combined' if args.combine_train_corruptions == True else 'separate'
    filename_spec = str(f"_{args.noise}_eps_{args.epsilon}_{args.max}_" if
                        args.combine_train_corruptions == False else f"_")
    start_epoch, end_epoch = 0, args.epochs

    # Resume from checkpoint
    if args.resume == True:
        start_epoch, model, optimizer, scheduler = Checkpointer._load_model(model, optimizer, scheduler, best=False)
        print('\nResuming from checkpoint at epoch', start_epoch)
        # load prior learning curve values
        train_accs, train_losses, valid_accs, valid_losses = utils.load_learning_curves(args.dataset,
                        args.modeltype, args.lrschedule, args.experiment, args.run, training_folder, filename_spec)

    # Training loop
    with tqdm(total=total_steps) as pbar:
        with torch.autograd.set_detect_anomaly(False, check_nan=False): #this may resolve some Cuda/cuDNN errors.
            # check_nan=True increases 32bit precision train time by ~20% and causes errors due to nan values for mixed precision training.
            for epoch in range(start_epoch, end_epoch):
                train_acc, train_loss = train_epoch(pbar)
                valid_acc, valid_loss = valid_epoch(pbar)
                train_accs.append(train_acc)
                valid_accs.append(valid_acc)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)

                if args.lrschedule == 'ReduceLROnPlateau':
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()
                # Check for best model, save model(s) and learning curve and check for earlystopping conditions
                Checkpointer._earlystopping(valid_acc, model)
                Checkpointer._save_checkpoint(model, optimizer, scheduler, epoch)
                utils.save_learning_curves(args.dataset, args.modeltype, args.lrschedule, args.experiment, args.run,
                                           train_accs, valid_accs, train_losses, valid_losses, training_folder, filename_spec)
                if Checkpointer.early_stop:
                    end_epoch = epoch
                    break

    # Save final model
    end_epoch, model, optimizer, scheduler = Checkpointer._load_model(model, optimizer, scheduler, best=False)
    Checkpointer._save_final_model(model, optimizer, scheduler, end_epoch, path = f'./experiments/trained_models/{args.dataset}'
                                                    f'/{args.modeltype}/config{args.experiment}_{args.lrschedule}_'
                                                    f'{training_folder}{filename_spec}run_{args.run}.pth')
    # print results
    print("Maximum validation accuracy of", max(valid_accs), "achieved after", np.argmax(valid_accs) + 1, "epochs; "
         "Minimum validation loss of", min(valid_losses), "achieved after", np.argmin(valid_losses) + 1, "epochs; ")
    # save config file
    shutil.copyfile(f'./experiments/configs/config{args.experiment}.py',
                    f'./results/{args.dataset}/{args.modeltype}/config{args.experiment}_{args.lrschedule}_{training_folder}.py')