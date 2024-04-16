import torch
import torch.backends.cudnn as cudnn
import torchvision.models as torchmodels
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassCalibrationError
import argparse
import importlib
import numpy as np
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"

import experiments.models as low_dim_models
import experiments.eval_adversarial as eval_adversarial
import experiments.eval_corruptions as eval_corruptions
import experiments.data as data
import experiments.utils as utils

parser = argparse.ArgumentParser(description='PyTorch Robustness Testing')
parser.add_argument('--resume', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='resuming evaluation')
parser.add_argument('--runs', default=1, type=int, help='run number')
parser.add_argument('--experiment', default=0, type=int,
                    help='experiment number - each experiment is defined in module config{experiment}')
parser.add_argument('--batchsize', default=1000, type=int,
                    help='Images per batch - more means quicker training, but higher memory demand')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to choose')
parser.add_argument('--modeltype', default='wideresnet', type=str,
                    help='Modeltype to train, use either default WRN28 or model from pytorch models')
parser.add_argument('--modelparams', default={}, type=str, action=utils.str2dictAction, metavar='KEY=VALUE',
                    help='parameters for the chosen model')
parser.add_argument('--resize', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Resize a model to 224x224 pixels, standard for models like transformers.')
parser.add_argument('--combine_test_corruptions', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to combine all testing noise values by drawing from the randomly')
parser.add_argument('--combine_train_corruptions', type=utils.str2bool, nargs='?', const=True, default=True,
                    help='Whether to combine all training noise values by drawing from the randomly')
parser.add_argument('--number_workers', default=1, type=int, help='How many workers are launched to parallelize data '
                    'loading. Experimental. 4 for ImageNet, 1 for Cifar. More demand GPU memory, but maximize GPU usage.')
parser.add_argument('--normalize', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to normalize input data to mean=0 and std=1')
parser.add_argument('--pixel_factor', default=1, type=int, help='default is 1 for 32px (CIFAR10), '
                    'e.g. 2 for 64px images. Scales convolutions automatically in the same model architecture')
parser.add_argument('--test_on_c', type=utils.str2bool, nargs='?', const=True, default=True,
                    help='Whether to test on corrupted benchmark sets C and C-bar')
parser.add_argument('--calculate_adv_distance', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to calculate adversarial distance with adv_distance_params')
parser.add_argument('--adv_distance_params', default={'setsize': 500, 'nb_iters': 200, 'eps_iter': 0.0003, 'norm': 'np.inf',
                        "epsilon": 0.1, "clever": True, "clever_batches": 50, "clever_samples": 50},
                    type=str, action=utils.str2dictAction, metavar='KEY=VALUE', help='parameters for the trades loss function')
parser.add_argument('--calculate_autoattack_robustness', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to calculate adversarial accuracy with Autoattack with autoattack_params')
parser.add_argument('--autoattack_params', default={'setsize': 500, 'epsilon': 8/255, 'norm': 'Linf'},
                    type=str, action=utils.str2dictAction, metavar='KEY=VALUE', help='parameters for the trades loss function')
parser.add_argument('--test_count', default=75, type=int, help='how many metrics are calculated in total per model')
parser.add_argument('--model_count', default=1, type=int, help='how many models are evaluated for this experiment')

args = parser.parse_args()
configname = (f'experiments.configs.config{args.experiment}')
config = importlib.import_module(configname)
test_corruptions = config.test_corruptions
train_corruptions = config.train_corruptions

def compute_clean(testloader, model, num_classes):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        calibration_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='l2')
        all_targets = torch.empty(0)
        all_targets_pred = torch.empty((0, num_classes))
        all_targets, all_targets_pred = all_targets.to(device), all_targets_pred.to(device)

        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            with torch.cuda.amp.autocast():
                targets_pred = model(inputs)

            _, predicted = targets_pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_targets = torch.cat((all_targets, targets), 0)
            all_targets_pred = torch.cat((all_targets_pred, targets_pred), 0)

        acc = 100.*correct/total
        rmsce_clean = float(calibration_metric(all_targets_pred, all_targets).cpu())
        print("Clean Accuracy ", acc, "%, RMSCE Calibration Error: ", rmsce_clean)

        return acc, rmsce_clean

if __name__ == '__main__':
    Testtracker = utils.TestTracking(args.dataset, args.modeltype, args.experiment, args.runs,
                                args.combine_train_corruptions, args.combine_test_corruptions, args.test_on_c,
                                args.calculate_adv_distance, args.calculate_autoattack_robustness, train_corruptions,
                                test_corruptions)

    # Load data
    test_transforms,_ = data.create_transforms(args.dataset, aug_strat_check=False, train_aug_strat='None', resize=args.resize)
    _, _, testset, num_classes = data.load_data(test_transforms, args.dataset, validontest=True)
    testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=False, pin_memory=True,
                            num_workers=args.number_workers)

    for run in range(args.runs):
        for model in range(args.model_count):

            Testtracker.initialize(run, model)

            # Load model
            if args.dataset == 'CIFAR10' or 'CIFAR100' or 'TinyImageNet':
                model_class = getattr(low_dim_models, args.modeltype)
                model = model_class(dataset=args.dataset, normalized=args.normalize, num_classes=num_classes,
                                    factor=args.pixel_factor, **args.modelparams)
            else:
                model_class = getattr(torchmodels, args.modeltype)
                model = model_class(num_classes=num_classes, **args.modelparams)
            model = torch.nn.DataParallel(model).to(device)
            cudnn.benchmark = True
            model.load_state_dict(torch.load(Testtracker.filename)["model_state_dict"], strict=False)

            # Clean Test Accuracy
            acc, rmsce = compute_clean(testloader, model, num_classes)
            Testtracker.track_results([acc, rmsce])

            if args.test_on_c == True:  # C-dataset robust accuracy
                testsets_c = data.load_data_c(args.dataset, testset, args.resize, test_transforms, subset=False, subsetsize=None)
                accs_c = eval_corruptions.compute_c_corruptions(args.dataset, testsets_c, model, args.batchsize, num_classes,
                                                                eval_run=False)
                Testtracker.track_results([accs_c])

            if args.calculate_adv_distance == True:  # adversarial distance calculation
                adv_acc, mean_dist1, mean_dist2, mean_clever, clever_sorted, dist1_sorted = eval_adversarial.compute_adv_distance(
                    testset, args.number_workers, model, args.adv_distance_params)
                Testtracker.track_results([adv_acc, mean_dist1, mean_dist2, mean_clever])
                Testtracker.save_adv_distance(dist1_sorted, clever_sorted)

            if args.calculate_autoattack_robustness == True:  # adversarial accuracy calculation
                adv_acc_aa, mean_dist_aa = eval_adversarial.compute_adv_acc(args.autoattack_params, testset, model,
                                                                            args.number_workers, args.batchsize)
                Testtracker.track_results([adv_acc_aa, mean_dist_aa])

            # Robust Accuracy on p-norm noise - either combined or separate noise types
            accs = eval_corruptions.select_p_corruptions(testloader, model, test_corruptions, args.dataset, args.combine_test_corruptions)
            Testtracker.track_results([accs])

            print(Testtracker.accs)

    Testtracker.create_report()