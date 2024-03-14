import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from experiments.losses import JsdCrossEntropy
from experiments.losses import Trades

def plot_images(images, corrupted_images, number):
    fig, axs = plt.subplots(number, 2)
    images, corrupted_images = images.cpu(), corrupted_images.cpu()
    for i in range(number):
        image = images[i]
        image = torch.squeeze(image)
        image = image.permute(1, 2, 0)
        axs[i, 0].imshow(image)
        corrupted_image = corrupted_images[i]
        corrupted_image = torch.squeeze(corrupted_image)
        corrupted_image = corrupted_image.permute(1, 2, 0)
        axs[i, 1].imshow(corrupted_image)
    #return fig
    plt.show()

def calculate_steps(dataset, batchsize, epochs, warmupepochs, validontest):
    #+0.5 is a way of rounding up to account for the last partial batch in every epoch
    if dataset == 'ImageNet':
        steps = round(1281167/batchsize + 0.5) * (epochs + warmupepochs)
        if validontest == True:
            steps += (round(50000/batchsize + 0.5) * (epochs + warmupepochs))
    if dataset == 'TinyImageNet':
        steps = round(100000/batchsize + 0.5) * (epochs + warmupepochs)
        if validontest == True:
            steps += (round(10000/batchsize + 0.5) * (epochs + warmupepochs))
    elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
        steps = round(50000 / batchsize + 0.5) * (epochs + warmupepochs)
        if validontest == True:
            steps += (round(10000/batchsize + 0.5) * (epochs + warmupepochs))
    total_steps = int(steps)
    return total_steps

def get_criterion(loss_function, lossparams):
    if loss_function == 'jsd':
        criterion, robust_samples = JsdCrossEntropy(**lossparams), lossparams["num_splits"] - 1
    elif loss_function == 'trades':
        criterion, robust_samples = Trades(**lossparams), 0
    else:
        criterion, robust_samples = torch.nn.CrossEntropyLoss(label_smoothing=lossparams["smoothing"]), 0
    test_criterion = torch.nn.CrossEntropyLoss()
    return criterion, test_criterion, robust_samples

def create_report(avg_test_metrics, max_test_metrics, std_test_metrics, train_corruptions, test_corruptions,
                combine_train_corruptions, combine_test_corruptions, dataset, modeltype, lrschedule, experiment,
                test_on_c, calculate_adv_distance, calculate_autoattack_robustness, runs):

    training_folder = 'combined' if combine_train_corruptions == True else 'separate'

    if combine_train_corruptions == True:
        train_corruptions_string = ['config_model']
    else:
        train_corruptions_string = np.array([','.join(map(str, row.values())) for row in train_corruptions])

    test_corruptions_string = np.array(['Standard_Acc', 'RMSCE'])
    if test_on_c == True:
        test_corruptions_label = np.loadtxt('./experiments/data/c-labels.txt', dtype=list)
        test_corruptions_string = np.append(test_corruptions_string, test_corruptions_label, axis=0)
        test_corruptions_string = np.append(test_corruptions_string, ['mCE-19', 'mCE-15', 'mCE-19_exNoise', 'RMSCE_C'],
                                            axis=0)

    if calculate_adv_distance == True:
        test_corruptions_string = np.append(test_corruptions_string, ['Acc_from_PGD_adv_distance_calculation',
                                                                      'Mean_PGD_adv_distance_with_misclassified_images_0)',
                                                                      'Mean_PGD_adv_distance_misclassified-images_not_included)',
                                                                      'Mean_CLEVER_score'],
                                            axis=0)
    if calculate_autoattack_robustness == True:
        test_corruptions_string = np.append(test_corruptions_string,
                                            ['Adversarial_accuracy_autoattack', 'Mean_adv_distance_autoattack)'],
                                            axis=0)
    if combine_test_corruptions == True:
        test_corruptions_string = np.append(test_corruptions_string, ['Combined Noise'])
    else:
        test_corruptions_labels = np.array([','.join(map(str, row.values())) for row in test_corruptions])
        test_corruptions_string = np.append(test_corruptions_string, test_corruptions_labels)

    avg_report_frame = pd.DataFrame(avg_test_metrics, index=test_corruptions_string, columns=train_corruptions_string)
    avg_report_frame.to_csv(f'./results/{dataset}/{modeltype}/config{experiment}_{lrschedule}_{training_folder}_'
                            f'metrics_test_avg.csv', index=True, header=True,
                            sep=';', float_format='%1.4f', decimal=',')
    if runs >= 2:
        max_report_frame = pd.DataFrame(max_test_metrics, index=test_corruptions_string, columns=train_corruptions_string)
        std_report_frame = pd.DataFrame(std_test_metrics, index=test_corruptions_string, columns=train_corruptions_string)
        max_report_frame.to_csv(f'./results/{dataset}/{modeltype}/config{experiment}_{lrschedule}_{training_folder}_'
                                f'metrics_test_max.csv', index=True, header=True,
                                sep=';', float_format='%1.4f', decimal=',')
        std_report_frame.to_csv(f'./results/{dataset}/{modeltype}/config{experiment}_{lrschedule}_{training_folder}_'
                                f'metrics_test_std.csv', index=True, header=True,
                                sep=';', float_format='%1.4f', decimal=',')

def save_learning_curves(dataset, modeltype, lrschedule, experiment, run, train_accs, valid_accs, valid_accs_robust,
                         valid_accs_adv, valid_accs_swa, valid_accs_robust_swa, swa, validonc, validonadv, train_losses,
                         valid_losses, training_folder, filename_spec):

    learning_curve_frame = pd.DataFrame({"train_accuracy": train_accs, "train_loss": train_losses,
                                             "valid_accuracy": valid_accs, "valid_loss": valid_losses})
    if validonc == True:
        learning_curve_frame.insert(4, "valid_accuracy_robust", valid_accs_robust)
    if validonadv == True:
        learning_curve_frame.insert(5, "valid_accuracy_adversarial", valid_accs_adv)
    if swa == True:
        learning_curve_frame.insert(6, "valid_accuracy_swa", valid_accs_swa)
        learning_curve_frame.insert(7, "valid_accuracy_robust_swa", valid_accs_robust_swa)

    x = list(range(1, len(train_accs) + 1))
    plt.figure()
    plt.plot(x, train_accs, label='Train Accuracy')
    plt.plot(x, valid_accs, label='Validation Accuracy')
    if validonc == True:
        plt.plot(x, valid_accs_robust, label='Robust Validation Accuracy')
    if validonadv == True:
        plt.plot(x, valid_accs_adv, label='Adversarial Validation Accuracy')
    if swa == True:
        plt.plot(x, valid_accs_swa, label='SWA Validation Accuracy')
        plt.plot(x, valid_accs_robust_swa, label='SWA Robust Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(np.linspace(1, len(train_accs), num=10, dtype=int))
    plt.legend(loc='best')

    learning_curve_frame.to_csv(f'./results/{dataset}/{modeltype}/config{experiment}_{lrschedule}_{training_folder}'
                                f'_learning_curve{filename_spec}run_{run}.csv',
                                index=False, header=True, sep=';', float_format='%1.4f', decimal=',')
    plt.savefig(f'results/{dataset}/{modeltype}/config{experiment}_{lrschedule}_{training_folder}_learning_curve'
                f'{filename_spec}run_{run}.svg')
    plt.close()

def load_learning_curves(dataset, modeltype, lrschedule, experiment, run, training_folder, filename_spec, validonc, validonadv, swa):
    learning_curve_frame = pd.read_csv(f'./results/{dataset}/{modeltype}/config{experiment}_{lrschedule}_{training_folder}'
                                f'_learning_curve{filename_spec}run_{run}.csv', sep=';', decimal=',')
    train_accuracy = learning_curve_frame.iloc[:, 0].values.tolist()
    train_loss = learning_curve_frame.iloc[:, 1].values.tolist()
    valid_accuracy = learning_curve_frame.iloc[:, 2].values.tolist()
    valid_loss = learning_curve_frame.iloc[:, 3].values.tolist()
    if validonc == True:
        valid_accuracy_robust = learning_curve_frame.iloc[:, 4].values.tolist()
    else:
        valid_accuracy_robust = []

    if validonadv == True:
        valid_accuracy_adv = learning_curve_frame.iloc[:, 5].values.tolist()
    else:
        valid_accuracy_adv = []

    if swa == True:
        valid_accuracy_swa = learning_curve_frame.iloc[:, 6].values.tolist()
        valid_accuracy_robust_swa = learning_curve_frame.iloc[:, 7].values.tolist()
    else:
        valid_accuracy_swa = []
        valid_accuracy_robust_swa = []

    return train_accuracy, train_loss, valid_accuracy, valid_loss, valid_accuracy_robust, valid_accuracy_adv, valid_accuracy_swa, valid_accuracy_robust_swa
