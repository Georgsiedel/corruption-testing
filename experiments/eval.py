from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import copy
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
import gc
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from experiments.network import WideResNet
from experiments.sample_corrupted_img import sample_lp_corr
import experiments.adversarial_eval as adv_eval
from torchmetrics.classification import MulticlassCalibrationError

def compute_metric(loader, net, noise_type, epsilon, max, combine, resize, dataset, normalize):
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs_pert = inputs
        if combine == True:
            corruptions = noise_type #this is just a helper
            for id, img in enumerate(inputs):
                (n, e, m) = random.choice(corruptions)
                e = float(e)
                if m == 'True':
                    inputs_pert[id] = sample_lp_corr(n, e, img, 'max')
                else:
                    inputs_pert[id] = sample_lp_corr(n, e, img, 'other')
        else:
            for id, img in enumerate(inputs):
                epsilon = float(epsilon)
                if max == 'True':
                    inputs_pert[id] = sample_lp_corr(noise_type, epsilon, img, 'max')
                else:
                    inputs_pert[id] = sample_lp_corr(noise_type, epsilon, img, 'other')
        if resize == True:
            inputs_pert = transforms.Resize(224)(inputs_pert)

        if dataset == 'CIFAR10' and normalize == True:
            inputs_pert = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))(inputs_pert)
        elif dataset == 'CIFAR100' and normalize == True:
            inputs_pert = transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))(inputs_pert)
        elif (dataset == 'ImageNet' or dataset == 'TinyImageNet') and normalize == True:
            inputs_pert = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(inputs_pert)

        inputs_pert, targets = inputs_pert.to(device, dtype=torch.float), targets.to(device)
        targets_pert = targets
        targets_pert_pred = net(inputs_pert)

        _, predicted = targets_pert_pred.max(1)
        total += targets_pert.size(0)
        correct += predicted.eq(targets_pert).sum().item()

    acc = 100.*correct/total
    return(acc)

def compute_clean(loader, net, resize, dataset, normalize, num_classes):
    with torch.no_grad():
        net.eval()
        correct = 0
        total = 0
        calibration_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=20, norm='l2')
        rmsce_batches = []
        n_images = []

        for batch_idx, (inputs, targets) in enumerate(loader):
            if resize == True:
                inputs = transforms.Resize(224)(inputs)
            if dataset == 'CIFAR10' and normalize == True:
                inputs = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))(inputs)
            elif dataset == 'CIFAR100' and normalize == True:
                inputs = transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))(inputs)
            elif (dataset == 'ImageNet' or dataset == 'TinyImageNet') and normalize == True:
                inputs = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(inputs)

            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            targets_pred = net(inputs)

            _, predicted = targets_pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            rmsce = calibration_metric(targets_pred, targets)
            rmsce_batches += [rmsce.item() ** 2] #in order to average values over all batches later
            n_images += [len(targets)] #in order to put the correct weighting on the last (smaller) batch

        acc = 100.*correct/total
        rmsce_batches = np.asarray(rmsce_batches)
        n_images = np.asarray(n_images)
        rmsce = np.sqrt(sum(rmsce_batches * n_images) / sum(n_images)) #weighted sum and average over batches and sqrt-operation we undid above
        return acc, rmsce

def compute_metric_imagenet_c(loader_c, net):
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader_c):

        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
        targets_pred = net(inputs)

        _, predicted = targets_pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return (acc)

def compute_metric_cifar_c(loader, loader_c, net, batchsize):
    net.eval()
    correct = 0
    total = 0
    for intensity in range(5):
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs_c = copy.deepcopy(inputs)
            for id, label in enumerate(targets):

                input_c = loader_c[intensity * 10000 + batch_idx * batchsize + id]
                inputs_c[id] = input_c

            inputs_c, targets = inputs_c.to(device, dtype=torch.float), targets.to(device)
            targets_pred = net(inputs_c)

            _, predicted = targets_pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            gc.collect()

    acc = 100. * correct / total
    return (acc)

def eval_metric(modelfilename, test_corruptions, combine_test_corruptions, test_on_c, modeltype, modelparams, resize,
                dataset, batchsize, workers, normalize, calculate_adv_distance, adv_distance_params, calculate_autoattack_robustness, autoattack_params):
    if dataset == 'ImageNet':
        test_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor()])
    elif dataset == 'CIFAR10' and resize == True:
        test_transforms = transforms.Compose([transforms.Resize(224),
                                              transforms.ToTensor()])
    else:
        test_transforms = transforms.Compose([transforms.ToTensor()])

    if dataset == 'ImageNet' or dataset == 'TinyImageNet':
        testset = torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}/val', transform=test_transforms)
        test_loader = DataLoader(testset, batch_size=batchsize, shuffle =False,
                                 pin_memory=True, num_workers=workers)
    else:
        load_helper = getattr(datasets, dataset)
        testset = load_helper("./experiments/data", train=False, download=True, transform=test_transforms)
        test_loader = DataLoader(testset, batch_size=batchsize, shuffle =False,
                                 pin_memory=True, num_workers=workers)
    num_classes = len(testset.classes)
    #Load model
    if modeltype == 'wrn28':
        model = WideResNet(depth = 28, widen_factor = 10, dropout_rate=modelparams['dropout_rate'], num_classes=num_classes)
    else:
        torchmodel = getattr(models, modeltype)
        model = torchmodel(num_classes = num_classes, **modelparams)
    model = model.to(device)

    if device == "cuda":
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    model.load_state_dict(torch.load(modelfilename)["net"])

    accs = []
    acc, rmsce = compute_clean(test_loader, model, resize, dataset, normalize, num_classes)
    accs = accs + [acc, rmsce]
    print("Clean Accuracy ",acc,"%, RMSCE Calibration Error: ", rmsce)

    if test_on_c == True:
        corruptions = np.loadtxt('./experiments/data/c-labels.txt', dtype=list)
        np.asarray(corruptions)
        corruptions = np.delete(corruptions, 0) #delete the 'standard' out of the list, this is only for labeling.
        print(f"Testing on {dataset}-c Benchmark Noise (Hendrycks 2019)")
        if dataset == 'CIFAR10' or dataset == 'CIFAR100':
            for corruption in corruptions:
                test_loader_c = np.load(f'./experiments/data/{dataset}-c/{corruption}.npy')
                test_loader_c = torch.from_numpy(test_loader_c)
                test_loader_c = test_loader_c.permute(0, 3, 1, 2)
                test_loader_c = test_loader_c / 255.0
                if resize == True:
                    test_loader_c = transforms.Resize(224)(test_loader_c)
                if normalize == True and dataset == 'CIFAR10':
                    test_loader_c = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))(test_loader_c)
                if normalize == True and dataset == 'CIFAR100':
                    test_loader_c = transforms.Normalize((0.50707516, 0.48654887, 0.44091784),
                                                    (0.26733429, 0.25643846, 0.27615047))(test_loader_c)
                acc = compute_metric_cifar_c(test_loader, test_loader_c, model, batchsize)
                accs.append(acc)
                print(acc, f"% mean (avg. over 5 intensities) Accuracy on {dataset}-c corrupted data of type", corruption)

        elif dataset == 'ImageNet' or dataset == 'TinyImageNet':
            for corruption in corruptions:
                acc_intensities = []

                for intensity in range(1, 6):
                    load_c = datasets.ImageFolder(root=f'./experiments/data/{dataset}-c/'+corruption+'/'+str(intensity),
                                    transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()]))
                    load_c = load_c / 255
                    if normalize == True:
                        load_c = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(load_c)
                    test_loader_c = DataLoader(load_c, batch_size=batchsize, shuffle=False)
                    acc = compute_metric_imagenet_c(test_loader_c, model)
                    acc_intensities.append(acc)
                acc = sum(acc_intensities) / 5
                accs.append(acc)
                print(acc, f"% mean (avg. over 5 intensities) Accuracy on {dataset}-c corrupted data of type", corruption)

        else:
            print('No corrupted benchmark available other than CIFAR10-c, CIFAR100-c, TinyImageNet-c and ImageNet-c.')

    if combine_test_corruptions:
        acc = compute_metric(test_loader, model, test_corruptions, test_corruptions, test_corruptions,
                             combine_test_corruptions, resize, dataset, normalize)
        print(acc, "% Accuracy on combined Lp-norm Test Noise")
        accs.append(acc)
    else:
        for id, (noise_type, test_epsilon, max) in enumerate(test_corruptions):
            acc = compute_metric(test_loader, model, noise_type, test_epsilon, max, combine_test_corruptions, resize, dataset, normalize)
            print(acc, "% Accuracy on random test corupptions of type:", noise_type, test_epsilon, "with maximal-perturbation =", max)
            accs.append(acc)
    if calculate_adv_distance == True:
        print(f"{adv_distance_params['norm']}-Adversarial Distance calculation using PGD attack with {adv_distance_params['nb_iters']} iterations of "
              f"stepsize {adv_distance_params['eps_iter']}")
        adv_acc_high_iter_pgd, dst0, idx0, dst1, idx1, dst2, idx2 = adv_eval.compute_adv_distance(testset, workers, model, adv_distance_params)
        accs.append(adv_acc_high_iter_pgd)
        mean_dist0, mean_dist1, mean_dist2 = [np.asarray(torch.tensor(d).cpu()).mean() for d in [dst0, dst1, dst2]]
        accs = accs + [mean_dist0, mean_dist1, mean_dist2]
    if calculate_autoattack_robustness == True:
        print(f"{autoattack_params['norm']} Adversarial Accuracy calculation using AutoAttack attack with epsilon={autoattack_params['epsilon']}")
        load_helper = getattr(datasets, dataset)
        testset = load_helper("./experiments/data", train=False, download=True, transform=transforms.Compose([test_transforms, transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]))
        adv_acc_aa, mean_dist_aa = adv_eval.compute_adv_acc(autoattack_params, testset, model, workers, batchsize)
        accs = accs + [adv_acc_aa, mean_dist_aa]
    return accs