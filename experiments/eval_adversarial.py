from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torchattacks

device = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
from torch.utils.data import DataLoader
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from autoattack import AutoAttack
from art.estimators.classification.pytorch import PyTorchClassifier
from art.metrics import clever_u, clever_t
import matplotlib.pyplot as plt
from cleverhans.torch.utils import optimize_linear

def fast_gradient_validation(
    model_fn,
    x,
    eps,
    norm,
    valid_loss,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    sanity_checks=False,
):
    """
    PyTorch implementation of the Fast Gradient Method. from Cleverhans package
    """

    if norm not in [np.inf, 1, 2]:
        raise ValueError(
            "Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm)
        )
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
    if targeted:
        valid_loss = -valid_loss

    # Define gradient of loss wrt input
    valid_loss.backward()
    optimal_perturbation = optimize_linear(x.grad, eps, norm)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
            )
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x

def adv_valid(inputs, labels, epsilon, model, criterion):
    inputs.requires_grad_()
    with torch.torch.enable_grad():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        adv_inputs = fast_gradient_validation(model_fn=model, eps=epsilon, x=inputs, y=labels, norm=np.inf, valid_loss=loss)
        adv_outputs = model(adv_inputs)
    return adv_outputs, outputs, loss

def pgd_with_early_stopping(model, inputs, labels, clean_predicted, eps, number_iterations, epsilon_iters, norm):

    for i in range(number_iterations):
        adv_inputs = projected_gradient_descent(model,
                                                inputs,
                                                eps=eps,
                                                eps_iter=epsilon_iters,
                                                nb_iter=1,
                                                norm=norm,
                                                y = labels,
                                                rand_init=False,
                                                sanity_checks=False)

        adv_outputs = model(adv_inputs)
        _, adv_predicted = torch.max(adv_outputs.data, 1)

        label_flipped = bool(adv_predicted!=clean_predicted)
        if label_flipped:
            break
        inputs = adv_inputs.clone()
    return adv_inputs, adv_predicted

def adv_distance(testloader, model, number_iterations, epsilon, eps_iter, norm, setsize):
    distance_list_1, image_idx_1 = [], []
    distance_list_2, image_idx_2 = [], []
    model.eval()
    correct, total = 0, 0
    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        adv_inputs, adv_predicted = pgd_with_early_stopping(model, inputs, labels, predicted, epsilon, number_iterations, eps_iter, norm)
        distance = torch.norm((inputs - adv_inputs), p=norm)
        if (predicted == labels):
            distance_list_1.append(distance)
            image_idx_1.append(i)
            distance_list_2.append(distance) #only originally correctly classified distances are counted
            image_idx_2.append(i) #only originally correctly classified points
        else:
            distance_list_1.append(torch.tensor([0.0])) #originally misclassified distances are counted as 0
            image_idx_1.append(i) #all points, also originally misclassified ones

        correct += (adv_predicted == labels).sum().item()
        total += labels.size(0)
        if (i+1) % 20 == 0:
            print(f"Completed: {i+1} of {setsize}, mean_distances: {sum(distance_list_1)/len(distance_list_1)}, {sum(distance_list_2)/len(distance_list_2)}, correct: {correct}, total: {total}, accuracy: {correct / total * 100}%")
    print(distance_list_1)
    adv_acc = correct / total
    return distance_list_1, image_idx_1, distance_list_2, image_idx_2, adv_acc

def clever_score(testloader, model, clever_batches, clever_samples, epsilon, norm, num_classes):

    torch.cuda.empty_cache()
    clever_scores = []
    image_ids = []
    images, _ = next(iter(testloader))
    model = PyTorchClassifier(model=model,
                            loss=torch.nn.CrossEntropyLoss(),
                            optimizer=torch.optim.SGD(model.parameters(), momentum= 0.9, weight_decay= 1e-4, lr=0.01),
                            input_shape=images[0].size(),
                            nb_classes=num_classes)
    # Iterate through each image for CLEVER score calculation
    for batch_idx, (inputs, targets) in enumerate(testloader):
        for id, input in enumerate(inputs):
            clever_score = clever_u(model,
                                    input.numpy(),
                                    nb_batches=clever_batches,
                                    batch_size=clever_samples,
                                    radius=epsilon,
                                    norm=norm,
                                    pool_factor=10)

            # Append the calculated CLEVER score to the list
            clever_scores.append(clever_score)
            # Append the image ID to the list
            image_ids.append(id)

    return clever_scores, image_ids

def compute_adv_distance(testset, workers, model, adv_distance_params):

    epsilon = adv_distance_params["epsilon"]
    eps_iter = adv_distance_params["eps_iter"]
    nb_iters = adv_distance_params["nb_iters"]
    norm = adv_distance_params["norm"]
    clever_batches = adv_distance_params["clever_batches"]
    clever_samples = adv_distance_params["clever_samples"]

    print(f"{norm}-Adversarial Distance upper bound calculation using PGD attack with "
          f"{nb_iters} iterations of stepsize {eps_iter}")
    num_classes = len(testset.classes)
    truncated_testset, _ = torch.utils.data.random_split(testset,
                                                         [adv_distance_params["setsize"], len(testset)-adv_distance_params["setsize"]],
                                                         generator=torch.Generator().manual_seed(42))
    truncated_testloader = DataLoader(truncated_testset, batch_size=1, shuffle=False,
                                       pin_memory=True, num_workers=workers)

    dst1, idx1, dst2, idx2, adv_acc = adv_distance(testloader=truncated_testloader, model=model,
        number_iterations=nb_iters, epsilon=epsilon, eps_iter=eps_iter, norm=norm, setsize=adv_distance_params["setsize"])
    mean_dist1, mean_dist2 = [np.asarray(torch.tensor(d).cpu()).mean() for d in [dst1, dst2]]
    adv_dist_list = np.asarray([t.item() for t in dst1])
    sorted_indices = np.argsort(adv_dist_list)
    adv_distance_sorted = adv_dist_list[sorted_indices]

    if adv_distance_params['clever'] == True:
        print(f"{norm}-Adversarial Distance (statistical) lower bound calculation using Clever Score with "
              f"{clever_batches} batches with {clever_samples} samples each.")
        clever_scores, clever_id = clever_score(testloader=truncated_testloader, model=model, clever_batches=clever_batches,
                             clever_samples=clever_samples, epsilon=epsilon, norm=norm, num_classes=num_classes)
        clever_scores_sorted = np.asarray(clever_scores)[sorted_indices]
        mean_clever_score = np.asarray(torch.tensor(clever_scores).cpu()).mean()
    else:
        mean_clever_score = 0.0
    plt.figure(figsize=(15, 5))
    plt.scatter(range(len(adv_distance_sorted)), adv_distance_sorted, s=3, label="PGD Adversarial Distance")
    if adv_distance_params['clever'] == True:
        plt.scatter(range(len(clever_scores_sorted)), clever_scores_sorted, s=3, label="Clever Score")
    plt.xlabel("Sorted Image ID")
    plt.ylabel("Distance")
    plt.legend()
    #plt.show()
    #plt.savefig(f'results/{dataset}/{modeltype}/config{experiment}_{lrschedule}_{training_folder}_learning_curve'
    #            f'{filename_spec}run_{run}.svg')
    #plt.close()

    return adv_acc*100, mean_dist1, mean_dist2, mean_clever_score

def compute_adv_acc(autoattack_params, testset, model, workers, batchsize=50):
    print(f"{autoattack_params['norm']} Adversarial Accuracy calculation using AutoAttack attack "
          f"with epsilon={autoattack_params['epsilon']}")
    truncated_testset, _ = torch.utils.data.random_split(testset, [autoattack_params["setsize"],
                                len(testset)-autoattack_params["setsize"]], generator=torch.Generator().manual_seed(42))
    truncated_testloader = DataLoader(truncated_testset, batch_size=autoattack_params["setsize"], shuffle=False,
                                       pin_memory=True, num_workers=workers)
    adversary = AutoAttack(model, norm=autoattack_params['norm'], eps=autoattack_params['epsilon'], version='standard')
    correct, total = 0, 0
    distance_list = []
    if autoattack_params["norm"] == 'Linf':
        autoattack_params["norm"] = np.inf
    else:
        autoattack_params["norm"] = autoattack_params["norm"][1:]
    for batch_id, (inputs, targets) in enumerate(truncated_testloader):
        adv_inputs, adv_predicted = adversary.run_standard_evaluation(inputs, targets, bs=batchsize, return_labels=True)

        for i, (input) in enumerate(inputs):
            distance = torch.linalg.vector_norm((input - adv_inputs[i]), ord=autoattack_params["norm"])
            distance_list.append(distance)

    mean_aa_dist = np.asarray(torch.tensor(distance_list).cpu()).mean()
    correct += (adv_predicted == targets).sum().item()
    total += targets.size(0)
    adv_acc = correct / total
    return adv_acc, mean_aa_dist