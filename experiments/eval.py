import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.models as torchmodels
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassCalibrationError
device = "cuda" if torch.cuda.is_available() else "cpu"

import experiments.models as low_dim_models
import experiments.eval_adversarial as eval_adversarial
import experiments.eval_corruptions as eval_corruptions
import experiments.data as data

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

def eval_metric(modelfilename, test_corruptions, combine_test_corruptions, test_on_c, modeltype, modelparams, resize,
                dataset, batchsize, workers, normalized, calculate_adv_distance, adv_distance_params,
                calculate_autoattack_robustness, autoattack_params, pixel_factor):

    #Load data
    test_transforms, _ = data.create_transforms(dataset, aug_strat_check=False, train_aug_strat='None', resize=resize)
    _, _, testset, num_classes = data.load_data(test_transforms, dataset, validontest = True)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle =False, pin_memory=True, num_workers=workers)

    #Load model
    if dataset == 'CIFAR10' or 'CIFAR100' or 'TinyImageNet':
        model_class = getattr(low_dim_models, modeltype)
        model = model_class(dataset = dataset, normalized = normalized, num_classes=num_classes, factor=pixel_factor,
                            **modelparams)
    else:
        model_class = getattr(torchmodels, modeltype)
        model = model_class(num_classes = num_classes, **modelparams)
    model = torch.nn.DataParallel(model).to(device)
    cudnn.benchmark = True
    model.load_state_dict(torch.load(modelfilename)["model_state_dict"], strict=False)

    accs = []

    # Clean Test Accuracy
    acc, rmsce = compute_clean(testloader, model, num_classes)
    accs = accs + [acc, rmsce]

    if test_on_c == True: # C-dataset robust accuracy
        testsets_c = data.load_data_c(dataset, testset, resize, test_transforms, subset=False, subsetsize=None)
        accs_c = eval_corruptions.compute_c_corruptions(dataset, testsets_c, model, batchsize, num_classes, eval_run = False)
        accs = accs + accs_c
    if calculate_adv_distance == True: # adversarial distance calculation
        adv_acc_high_iter_pgd, mean_dist1, mean_dist2, mean_clever_score = eval_adversarial.compute_adv_distance(testset, workers, model, adv_distance_params)
        accs = accs + [adv_acc_high_iter_pgd, mean_dist1, mean_dist2, mean_clever_score]
    if calculate_autoattack_robustness == True: # adversarial accuracy calculation
        adv_acc_aa, mean_dist_aa = eval_adversarial.compute_adv_acc(autoattack_params, testset, model, workers, batchsize)
        accs = accs + [adv_acc_aa, mean_dist_aa]
    if combine_test_corruptions: # combined p-norm corruption robust accuracy
        acc = eval_corruptions.compute_p_corruptions(testloader, model, test_corruptions, normalized, dataset)
        print(acc, "% Accuracy on combined Lp-norm Test Noise")
        accs.append(acc)
    else: # separate p-norm corruption robust accuracy
        for id, (test_corruption) in enumerate(test_corruptions):
            acc = eval_corruptions.compute_p_corruptions(testloader, model, test_corruption, normalized, dataset)
            print(acc, "% Accuracy on random test corruptions of type:", test_corruption['noise_type'], test_corruption['epsilon'])
            accs.append(acc)

    return accs