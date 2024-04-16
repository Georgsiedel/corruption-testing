import argparse
import importlib
import numpy as np
from tqdm import tqdm
import torch.cuda.amp
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
import torchvision.models as torchmodels

import experiments.data as data
import experiments.utils as utils
import experiments.losses as losses
import experiments.models as low_dim_models
from experiments.eval_corruptions import compute_c_corruptions
from experiments.eval_adversarial import fast_gradient_validation

import torch.backends.cudnn as cudnn
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#torch.backends.cudnn.enabled = False #this may resolve some cuDNN errors, but increases training time by ~200%
torch.cuda.set_device(0)
cudnn.benchmark = False #this slightly speeds up 32bit precision training (5%)

parser = argparse.ArgumentParser(description='PyTorch Training with perturbations')
parser.add_argument('--resume', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='resuming from saved checkpoint in fixed-path repo defined below')
parser.add_argument('--train_corruptions', default={'noise_type': 'standard', 'epsilon': 0.0, 'sphere': False, 'distribution': 'max'},
                    type=str, action=utils.str2dictAction, metavar='KEY=VALUE', help='dictionary for type of noise, epsilon value, '
                    'whether it is always the maximum noise value and a distribution from which various epsilon are sampled')
parser.add_argument('--run', default=1, type=int, help='run number')
parser.add_argument('--experiment', default=0, type=int,
                    help='experiment number - each experiment is defined in module config{experiment}')
parser.add_argument('--batchsize', default=128, type=int,
                    help='Images per batch - more means quicker training, but higher memory demand')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to choose')
parser.add_argument('--validontest', type=utils.str2bool, nargs='?', const=True, default=True, help='For datasets wihtout '
                    'standout validation (e.g. CIFAR). True: Use full training data, False: Split 20% for valiationd')
parser.add_argument('--epochs', default=100, type=int, help="number of epochs")
parser.add_argument('--learningrate', default=0.1, type=float, help='learning rate')
parser.add_argument('--lrschedule', default='MultiStepLR', type=str, help='Learning rate scheduler from pytorch.')
parser.add_argument('--lrparams', default={'milestones': [85, 95], 'gamma': 0.2}, type=str, action=utils.str2dictAction,
                    metavar='KEY=VALUE', help='parameters for the learning rate scheduler')
parser.add_argument('--earlystop', type=utils.str2bool, nargs='?', const=False, default=False, help='Use earlystopping after '
                    'some epochs (patience) of no increase in performance')
parser.add_argument('--earlystopPatience', default=15, type=int,
                    help='Number of epochs to wait for a better performance if earlystop is True')
parser.add_argument('--optimizer', default='SGD', type=str, help='Optimizer from torch.optim')
parser.add_argument('--optimizerparams', default={'momentum': 0.9, 'weight_decay': 5e-4}, type=str,
                    action=utils.str2dictAction, metavar='KEY=VALUE', help='parameters for the optimizer')
parser.add_argument('--modeltype', default='wideresnet', type=str,
                    help='Modeltype to train, use either default WRN28 or model from pytorch models')
parser.add_argument('--modelparams', default={}, type=str, action=utils.str2dictAction, metavar='KEY=VALUE',
                    help='parameters for the chosen model')
parser.add_argument('--resize', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Resize a model to 224x224 pixels, standard for models like transformers.')
parser.add_argument('--aug_strat_check', type=utils.str2bool, nargs='?', const=True, default=False,
                    help='Whether to use an auto-augmentation scheme')
parser.add_argument('--train_aug_strat', default='TrivialAugmentWide', type=str, help='auto-augmentation scheme')
parser.add_argument('--loss', default='CrossEntropyLoss', type=str, help='loss function to use, chosen from torch.nn loss functions')
parser.add_argument('--lossparams', default={}, type=str, action=utils.str2dictAction, metavar='KEY=VALUE',
                    help='parameters for the standard loss function')
parser.add_argument('--trades_loss', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='whether or not to use trades loss for training')
parser.add_argument('--trades_lossparams',
                    default={'step_size': 0.003, 'epsilon': 0.031, 'perturb_steps': 10, 'beta': 1.0, 'distance': 'l_inf'},
                    type=str, action=utils.str2dictAction, metavar='KEY=VALUE', help='parameters for the trades loss function')
parser.add_argument('--robust_loss', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='whether or not to use robust (JSD/stability) loss for training')
parser.add_argument('--robust_lossparams', default={'num_splits': 3, 'alpha': 12}, type=str, action=utils.str2dictAction,
                    metavar='KEY=VALUE', help='parameters for the robust loss function. If 3, JSD will be used.')
parser.add_argument('--mixup', default={'alpha': 0.2, 'p': 0.0}, type=str, action=utils.str2dictAction, metavar='KEY=VALUE',
                    help='Mixup parameters, Pytorch suggests 0.2 for alpha. Mixup, Cutmix and RandomErasing are randomly '
                    'chosen without overlapping based on their probability, even if the sum of the probabilities is >1')
parser.add_argument('--cutmix', default={'alpha': 1.0, 'p': 0.0}, type=str, action=utils.str2dictAction, metavar='KEY=VALUE',
                    help='Cutmix parameters, Pytorch suggests 1.0 for alpha. Mixup, Cutmix and RandomErasing are randomly '
                    'chosen without overlapping based on their probability, even if the sum of the probabilities is >1')
parser.add_argument('--manifold', default={'apply': False, 'noise_factor': 4}, type=str, action=utils.str2dictAction, metavar='KEY=VALUE',
                    help='Choose whether to apply noisy mixup in manifold layers')
parser.add_argument('--combine_train_corruptions', type=utils.str2bool, nargs='?', const=True, default=True,
                    help='Whether to combine all training noise values by drawing from the randomly')
parser.add_argument('--concurrent_combinations', default=1, type=int, help='How many of the training noise values should '
                    'be applied at once on one image. USe only if you defined multiple training noise values.')
parser.add_argument('--number_workers', default=2, type=int, help='How many workers are launched to parallelize data '
                    'loading. Experimental. 4 for ImageNet, 1 for Cifar. More demand GPU memory, but maximize GPU usage.')
parser.add_argument('--RandomEraseProbability', default=0.0, type=float,
                    help='probability of applying random erasing to an image')
parser.add_argument('--warmupepochs', default=5, type=int,
                    help='Number of Warmupepochs for stable training early on. Start with factor 10 lower learning rate')
parser.add_argument('--normalize', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to normalize input data to mean=0 and std=1')
parser.add_argument('--pixel_factor', default=1, type=int, help='default is 1 for 32px (CIFAR10), '
                    'e.g. 2 for 64px images. Scales convolutions automatically in the same model architecture')
parser.add_argument('--minibatchsize', default=8, type=int, help='batchsize, for which a new corruption type is sampled. '
                    'batchsize must be a multiple of minibatchsize. in case of p-norm corruptions with 0<p<inf, the same '
                    'corruption is applied for all images in the minibatch')
parser.add_argument('--validonc', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to do a validation on a subset of c-data every epoch')
parser.add_argument('--validonadv', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to do a validation with an FGSM adversarial attack every epoch')
parser.add_argument('--swa', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to use stochastic weight averaging over the last epochs')
parser.add_argument('--noise_sparsity', default=0.0, type=float,
                    help='probability of not applying a calculated noise value to a dimension of an image')
parser.add_argument('--noise_patch_lower_scale', default=1.0, type=float, help='lower bound of the scale to choose the '
                    'area ratio of the image from, which gets perturbed by random noise')

args = parser.parse_args()
configname = (f'experiments.configs.config{args.experiment}')
config = importlib.import_module(configname)
if args.combine_train_corruptions == True:
    train_corruptions = config.train_corruptions
else:
    train_corruptions = args.train_corruptions

def train_epoch(pbar):
    model.train()
    correct, total, train_loss, avg_train_loss = 0, 0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()

        if criterion.robust_samples >= 1:
            inputs = torch.cat(inputs, 0)
        inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device)
        with torch.cuda.amp.autocast():
            outputs, mixed_targets = model(inputs, targets, criterion.robust_samples, train_corruptions, args.mixup['alpha'],
                                           args.mixup['p'], args.manifold['apply'], args.manifold['noise_factor'],
                                           args.cutmix['alpha'], args.cutmix['p'], args.minibatchsize,
                                           args.concurrent_combinations, args.noise_sparsity, args.noise_patch_lower_scale)
            criterion.update(model, optimizer)
            loss = criterion(outputs, mixed_targets, inputs, targets)
        loss.retain_grad()

        Scaler.scale(loss).backward()

        Scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
        #for name, param in model.named_parameters():
        #        print(name, param.grad)
        Scaler.step(optimizer)
        Scaler.update()
        torch.cuda.synchronize()
        train_loss += loss.item()

        _, predicted = outputs.max(1)
        if np.ndim(mixed_targets) == 2:
            _, mixed_targets = mixed_targets.max(1)
        if criterion.robust_samples >= 1:
            mixed_targets = torch.cat([mixed_targets] * (criterion.robust_samples+1), 0)

        total += mixed_targets.size(0)
        correct += predicted.eq(mixed_targets).sum().item()
        avg_train_loss = train_loss / (batch_idx + 1)
        pbar.set_description('[Train] Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(
            avg_train_loss, 100. * correct / total, correct, total))
        pbar.update(1)

    train_acc = 100. * correct / total
    return train_acc, avg_train_loss

def valid_epoch(pbar, net):
    net.eval()
    with torch.no_grad():
        test_loss, correct, total, avg_test_loss, adv_acc, acc_c, adv_correct = 0, 0, 0, 0, 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(validationloader):

            inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device)

            with torch.cuda.amp.autocast():

                if args.validonadv == True:
                    adv_inputs, outputs = fast_gradient_validation(model_fn=model, eps=8/255, x=inputs, y=None, norm=np.inf, criterion=criterion)
                    _, adv_predicted = model(adv_inputs).max(1)
                    adv_correct += adv_predicted.eq(targets).sum().item()
                else:
                    outputs = net(inputs)

                loss = criterion.test(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            avg_test_loss = test_loss / (batch_idx + 1)
            pbar.set_description(
                '[Valid] Loss: {:.3f} | Acc: {:.3f} ({}/{}) | Adversarial Acc: {:.3f}'.format(avg_test_loss, 100. * correct / total,
                                                                    correct, total, 100. * adv_correct / total))
            pbar.update(1)

        if args.validonc == True:
            pbar.set_description(
                '[Valid] Robust Accuracy Calculation. Last Robust Accuracy: {:.3f}'.format(Traintracker.valid_accs_robust[-1] if Traintracker.valid_accs_robust else 0))
            acc_c = compute_c_corruptions(args.dataset, testsets_c, net, batchsize=200,
                                          num_classes=num_classes, eval_run = True)[0]
        pbar.update(1)

    acc = 100. * correct / total
    adv_acc = 100. * adv_correct / total
    return acc, avg_test_loss, acc_c, adv_acc

if __name__ == '__main__':
    # Load and transform data
    print('Preparing data..')
    transforms_preprocess, transforms_augmentation = data.create_transforms(args.dataset, args.aug_strat_check, args.train_aug_strat, args.resize, args.RandomEraseProbability)
    lossparams = args.trades_lossparams | args.robust_lossparams | args.lossparams
    criterion = losses.Criterion(args.loss, trades_loss=args.trades_loss, robust_loss=args.robust_loss, **lossparams)
    trainset, validset, testset, num_classes = data.load_data(transforms_preprocess, args.dataset, args.validontest, transforms_augmentation, run=args.run, robust_samples=criterion.robust_samples)
    testsets_c = data.load_data_c(args.dataset, testset, args.resize, transforms_preprocess, args.validonc, subsetsize=200)
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, pin_memory=True, collate_fn=None, num_workers=args.number_workers)
    validationloader = DataLoader(validset, batch_size=args.batchsize, shuffle=True, pin_memory=True, num_workers=args.number_workers)

    # Construct model
    print(f'\nBuilding {args.modeltype} model with {args.modelparams} | Augmentation strategy: {args.aug_strat_check}'
          f' | Loss Function: {args.loss}')
    if args.dataset == 'CIFAR10' or 'CIFAR100' or 'TinyImageNet':
        model_class = getattr(low_dim_models, args.modeltype)
        model = model_class(dataset=args.dataset, normalized =args.normalize, num_classes=num_classes,
                            factor=args.pixel_factor, **args.modelparams)
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
    if args.swa == True:
        swa_model = AveragedModel(model)
        swa_start = args.epochs * 0.9
        swa_scheduler = SWALR(optimizer, anneal_strategy="linear", anneal_epochs=5, swa_lr=args.learningrate / 10)
    Scaler = torch.cuda.amp.GradScaler()
    Checkpointer = utils.Checkpoint(args.combine_train_corruptions, args.dataset, args.modeltype, args.experiment,
                                    train_corruptions, args.run, earlystopping=args.earlystop, patience=args.earlystopPatience,
                                    verbose=False,  model_path='experiments/trained_models/checkpoint.pt',
                                                    swa_model_path='experiments/trained_models/swa_checkpoint.pt',
                                                    best_model_path = 'experiments/trained_models/best_checkpoint.pt')
    Traintracker = utils.TrainTracking(args.dataset, args.modeltype, args.lrschedule, args.experiment, args.run,
                            args.combine_train_corruptions, args.validonc, args.validonadv, args.swa, train_corruptions)

    # Calculate steps and epochs
    total_steps = utils.calculate_steps(args.dataset, args.batchsize, args.epochs, args.warmupepochs, args.validontest)
    start_epoch, end_epoch = 0, args.epochs

    # Resume from checkpoint
    if args.resume == True:
        start_epoch, model, optimizer, scheduler = Checkpointer.load_model(model, optimizer, scheduler, 'checkpoint')
        Traintracker.load_learning_curves()
        if args.swa == True:
            start_epoch, swa_model, optimizer, swa_scheduler = Checkpointer.load_model(swa_model, optimizer, scheduler, 'swa_checkpoint')
        print('\nResuming from checkpoint at epoch', start_epoch)

    # Training loop
    with tqdm(total=total_steps) as pbar:
        with torch.autograd.set_detect_anomaly(False, check_nan=False): #this may resolve some Cuda/cuDNN errors.
            # check_nan=True increases 32bit precision train time by ~20% and causes errors due to nan values for mixed precision training.
            for epoch in range(start_epoch, end_epoch):
                train_acc, train_loss = train_epoch(pbar)
                valid_acc, valid_loss, valid_acc_robust, valid_acc_adv = valid_epoch(pbar, model)

                if args.lrschedule == 'ReduceLROnPlateau':
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()

                if args.swa == True and epoch > swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                    valid_acc_swa, valid_loss_swa, valid_acc_robust_swa, valid_acc_adv_swa = valid_epoch(pbar, swa_model)
                else:
                    valid_acc_swa, valid_acc_robust_swa, valid_acc_adv_swa = valid_acc, valid_acc_robust, valid_acc_adv

                # Check for best model, save model(s) and learning curve and check for earlystopping conditions
                Checkpointer.earlystopping(valid_acc)
                Checkpointer.save_checkpoint(model, optimizer, scheduler, epoch)
                if args.swa == True:
                    Checkpointer.save_swa_checkpoint(swa_model, optimizer, swa_scheduler, epoch)
                Traintracker.save_metrics(train_acc, valid_acc, valid_acc_robust, valid_acc_adv, valid_acc_swa,
                             valid_acc_robust_swa, valid_acc_adv_swa, train_loss, valid_loss)
                Traintracker.save_learning_curves()
                if Checkpointer.early_stop:
                    end_epoch = epoch
                    break

    # Save final model
    if args.swa == True:
        torch.optim.swa_utils.update_bn(trainloader, swa_model)
        model = swa_model
        valid_acc_swa, valid_loss_swa, acc_c_swa = valid_epoch(pbar, swa_model)
    Checkpointer.save_final_model(model, optimizer, scheduler, end_epoch)
    Traintracker.print_results()
    Traintracker.save_config()
