import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchmetrics.classification import MulticlassCalibrationError
device = "cuda" if torch.cuda.is_available() else "cpu"

from experiments.noise import apply_lp_corruption

def compute_p_corruptions(testloader, model, test_corruptions, normalized, dataset):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            inputs_pert = apply_lp_corruption(inputs, 1, test_corruptions, 1, False, dataset)

            with torch.cuda.amp.autocast():
                targets_pred, targets = model(inputs_pert, targets)

            _, predicted = targets_pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        return acc

def compute_c_corruptions(dataset, testloader, model, batchsize, num_classes, test_transforms, resize):
    accs_c = []
    corruptions = np.loadtxt('./experiments/data/c-labels.txt', dtype=list)
    np.asarray(corruptions)
    rmsce_c_list = []
    print(f"Testing on {dataset}-c Benchmark Noise (Hendrycks 2019)")
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        for corruption in corruptions:
            test_loader_c = np.load(f'./experiments/data/{dataset}-c/{corruption}.npy')
            test_loader_c = torch.from_numpy(test_loader_c).permute(0, 3, 1, 2) / 255
            if resize == True:
                test_loader_c = transforms.Resize(224, antialias=True)(test_loader_c)
            acc, rmsce_c = compute_cifar_c(testloader, test_loader_c, model, batchsize, num_classes)
            accs_c.append(acc)
            rmsce_c_list.append(rmsce_c)
            print(acc, f"% mean (avg. over 5 intensities) Accuracy on {dataset}-c corrupted data of type",
                  corruption)

    elif dataset == 'ImageNet' or dataset == 'TinyImageNet':
        for corruption in corruptions:
            acc_intensities = []

            for intensity in range(1, 6):
                load_c = torchvision.datasets.ImageFolder(
                    root=f'./experiments/data/{dataset}-c/' + corruption + '/' + str(intensity),
                    transform=test_transforms)
                test_loader_c = DataLoader(load_c, batch_size=batchsize, shuffle=False)
                acc, rmsce_c = compute_imagenet_c(test_loader_c, model, num_classes)
                acc_intensities.append(acc)
                rmsce_c_list.append(rmsce_c)
            acc = sum(acc_intensities) / 5
            accs_c.append(acc)
            print(acc, f"% mean (avg. over 5 intensities) Accuracy on {dataset}-c corrupted data of type",
                  corruption)

    else:
        print('No corrupted benchmark available other than CIFAR10-c, CIFAR100-c, TinyImageNet-c and ImageNet-c.')

    rmsce_c = np.average(np.asarray(rmsce_c_list))
    print("Robust Accuracy all (19 corruptions): ", sum(accs_c[0:19]) / 19, "%, "
                                                                          "Robust Accuracy original (15 corruptions): ",
          sum(accs_c[0:15]) / 15, "%, "
                                "Robust Accuracy ex noise (15 corruptions): ",
          (sum(accs_c[3:15]) + sum(accs_c[16:19])) / 15, "%, "
                                                     "RMSCE-C: ", rmsce_c)
    accs_c.append(sum(accs_c[0:19]) / 19)
    accs_c.append(sum(accs_c[0:15]) / 15)
    accs_c.append((sum(accs_c[3:15]) + sum(accs_c[16:19])) / 15)
    accs_c.append(rmsce_c)

    return accs_c

def compute_imagenet_c(loader_c, model, num_classes):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        calibration_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='l2')
        all_targets = torch.empty(0)
        all_targets_pred = torch.empty((0, num_classes))
        all_targets, all_targets_pred = all_targets.to(device), all_targets_pred.to(device)

        for batch_idx, (inputs, targets) in enumerate(loader_c):
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            with torch.cuda.amp.autocast():
                targets_pred, targets = model(inputs, targets)

            _, predicted = targets_pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_targets = torch.cat((all_targets, targets), 0)
            all_targets_pred = torch.cat((all_targets_pred, targets_pred), 0)

        rmsce_c = float(calibration_metric(all_targets_pred, all_targets).cpu())
        acc = 100. * correct / total

        return acc, rmsce_c

def compute_cifar_c(loader, loader_c, model, batchsize, num_classes):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        calibration_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='l2')
        all_targets = torch.empty(0)
        all_targets_pred = torch.empty((0, num_classes))
        all_targets, all_targets_pred = all_targets.to(device), all_targets_pred.to(device)

        for intensity in range(5):
            for batch_idx, (inputs, targets) in enumerate(loader):
                for id, label in enumerate(targets):

                    input_c = loader_c[intensity * 10000 + batch_idx * batchsize + id]
                    inputs[id] = input_c

                inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
                with torch.cuda.amp.autocast():
                    targets_pred, targets = model(inputs, targets)

                _, predicted = targets_pred.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                all_targets = torch.cat((all_targets, targets), 0)
                all_targets_pred = torch.cat((all_targets_pred, targets_pred), 0)

        acc = 100.*correct/total
        rmsce_clean = float(calibration_metric(all_targets_pred, all_targets).cpu())

        return acc, rmsce_clean