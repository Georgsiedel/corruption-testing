import torch
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassCalibrationError
device = "cuda" if torch.cuda.is_available() else "cpu"

from experiments.noise import apply_noise

def compute_p_corruptions(testloader, model, test_corruptions, dataset):
    with torch.no_grad():
        model.eval()
        correct, total = 0, 0
        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            inputs_pert = apply_noise(inputs, 1, test_corruptions, 1, False, dataset)

            with torch.cuda.amp.autocast():
                targets_pred = model(inputs_pert)

            _, predicted = targets_pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        return acc

def compute_c_corruptions(dataset, testsets_c, model, batchsize, num_classes, eval_run = False):
    accs_c, rmsce_c_list = [], []
    if eval_run == False:
        print(f"Testing on {dataset}-c Benchmark Noise (Hendrycks 2019)")

    for corruption, corruption_testset in testsets_c.items():
        testloader_c = DataLoader(corruption_testset, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=1)
        acc, rmsce_c = compute_c(testloader_c, model, num_classes)
        accs_c.append(acc)
        rmsce_c_list.append(rmsce_c)
        if eval_run == False:
            print(acc, f"% mean (avg. over 5 intensities) Accuracy on {dataset}-c corrupted data of type", corruption)

    rmsce_c = np.average(np.asarray(rmsce_c_list))
    if eval_run == False:
        print("Robust Accuracy all (19 corruptions): ", sum(accs_c[0:19]) / 19, "%,"
            "Robust Accuracy original (15 corruptions): ", sum(accs_c[0:15]) / 15, "%, "
            "Robust Accuracy ex noise (15 corruptions): ", (sum(accs_c[3:15]) + sum(accs_c[16:19])) / 15, "%, "
            "RMSCE-C: ", rmsce_c)
        accs_c.append(sum(accs_c[0:19]) / 19)
        accs_c.append(sum(accs_c[0:15]) / 15)
        accs_c.append((sum(accs_c[3:15]) + sum(accs_c[16:19])) / 15)
        accs_c.append(rmsce_c)

    return accs_c

def compute_c(loader_c, model, num_classes):
    with torch.no_grad():
        model.eval()
        correct, total = 0, 0
        calibration_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='l2')
        all_targets = torch.empty(0)
        all_targets_pred = torch.empty((0, num_classes))
        all_targets, all_targets_pred = all_targets.to(device), all_targets_pred.to(device)

        for batch_idx, (inputs, targets) in enumerate(loader_c):
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            with torch.cuda.amp.autocast():
                targets_pred = model(inputs)

            _, predicted = targets_pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_targets = torch.cat((all_targets, targets), 0)
            all_targets_pred = torch.cat((all_targets_pred, targets_pred), 0)

        rmsce_c = float(calibration_metric(all_targets_pred, all_targets).cpu())
        acc = 100. * correct / total

        return acc, rmsce_c

