import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import shutil

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

class Checkpoint:
    """Early stops the training if validation loss doesn't improve after a given patience.
    credit to https://github.com/Bjarten/early-stopping-pytorch/tree/master for early stopping functionality"""

    def __init__(self, final_model_path, earlystopping=False, patience=7, verbose=False, delta=0, trace_func=print,
                 model_path='experiments/trained_models/checkpoint.pt',
                 swa_model_path='experiments/trained_models/swa_checkpoint.pt',
                 best_model_path='experiments/trained_models/best_checkpoint.pt'
                 ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = False
        self.val_loss_min = 1000  # placeholder initial value
        self.delta = delta
        self.trace_func = trace_func
        self.earlystopping = earlystopping
        self.model_path = model_path
        self.swa_model_path = swa_model_path
        self.best_model_path = best_model_path
        self.final_model_path = final_model_path

    def earlystopping(self, val_acc):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.best_model = False
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and self.earlystopping == True:
                self.early_stop = True
                print("Early stopping")
        else:
            self.best_score = score
            self.counter = 0
            self.best_model = True

    def load_model(self, model, optimizer, scheduler, path='checkpoint'):
        if path == 'checkpoint':
            checkpoint = torch.load(self.model_path)
        elif path == 'best_checkpoint':
            checkpoint = torch.load(self.best_model_path)
        elif path == 'swa_checkpoint':
            checkpoint = torch.load(self.swa_model_path)
        else:
            print('only swa_checkpoint, best_checkpoint or checkpoint can be loaded')

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch, model, optimizer, scheduler

    def save_checkpoint(self, model, optimizer, scheduler, epoch):

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, self.model_path)

        if self.best_model == True:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, self.best_model_path)

    def save_swa_checkpoint(self, swa_model, optimizer, swa_scheduler, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': swa_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': swa_scheduler.state_dict(),
        }, self.swa_model_path)

    def save_final_model(self, model, optimizer, scheduler, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, self.final_model_path)

class TrainTracking:
    def __init__(self, dataset, modeltype, lrschedule, experiment, run, combine_train_corruptions, validonc, validonadv,
                 swa, train_corruptions):
        self.dataset = dataset
        self.modeltype = modeltype
        self.lrschedule = lrschedule
        self.experiment = experiment
        self.run = run
        self.validonc = validonc
        self.validonadv = validonadv
        self.swa = swa
        self.train_accs, self.train_losses, self.valid_accs, self.valid_losses, self.valid_accs_robust = [],[],[],[],[]
        self.valid_accs_adv, self.valid_accs_swa, self.valid_accs_robust_swa, self.valid_accs_adv_swa = [],[],[],[]

    def load_learning_curves(self):

        learning_curve_frame = pd.read_csv(f'./results/{self.dataset}/{self.modeltype}/config{self.experiment}_'
                                           f'learning_curve_run_{self.run}.csv', sep=';', decimal=',')
        train_accs = learning_curve_frame.iloc[:, 0].values.tolist()
        train_losses = learning_curve_frame.iloc[:, 1].values.tolist()
        valid_accs = learning_curve_frame.iloc[:, 2].values.tolist()
        valid_losses = learning_curve_frame.iloc[:, 3].values.tolist()

        valid_accs_robust, valid_accs_adv, valid_accs_swa, valid_accs_robust_swa, valid_accs_adv_swa = [],[],[],[],[]
        if self.validonc == True:
            valid_accs_robust = learning_curve_frame.iloc[:, 4].values.tolist()
        if self.validonadv == True:
            valid_accs_adv = learning_curve_frame.iloc[:, 5].values.tolist()
        if self.swa == True:
            valid_accs_swa = learning_curve_frame.iloc[:, 6].values.tolist()
            valid_accs_robust_swa = learning_curve_frame.iloc[:, 7].values.tolist()
            valid_accs_adv_swa = learning_curve_frame.iloc[:, 8].values.tolist()

        self.train_accs = train_accs
        self.train_losses = train_losses
        self.valid_accs = valid_accs
        self.valid_losses = valid_losses
        self.valid_accs_robust = valid_accs_robust
        self.valid_accs_adv = valid_accs_adv
        self.valid_accs_swa = valid_accs_swa
        self.valid_accs_robust_swa = valid_accs_robust_swa
        self.valid_accs_adv_swa = valid_accs_adv_swa

    def save_metrics(self, train_acc, valid_acc, valid_acc_robust, valid_acc_adv, valid_acc_swa,
                             valid_acc_robust_swa, valid_acc_adv_swa, train_loss, valid_loss):

        self.train_accs.append(train_acc)
        self.train_losses.append(train_loss)
        self.valid_accs.append(valid_acc)
        self.valid_losses.append(valid_loss)
        self.valid_accs_robust.append(valid_acc_robust)
        self.valid_accs_adv.append(valid_acc_adv)
        self.valid_accs_swa.append(valid_acc_swa)
        self.valid_accs_robust_swa.append(valid_acc_robust_swa)
        self.valid_accs_adv_swa.append(valid_acc_adv_swa)

    def save_learning_curves(self):

        learning_curve_frame = pd.DataFrame({"train_accuracy": self.train_accs, "train_loss": self.train_losses,
                                                 "valid_accuracy": self.valid_accs, "valid_loss": self.valid_losses})
        if self.validonc == True:
            learning_curve_frame.insert(4, "valid_accuracy_robust", self.valid_accs_robust)
        if self.validonadv == True:
            learning_curve_frame.insert(5, "valid_accuracy_adversarial", self.valid_accs_adv)
        if self.swa == True:
            learning_curve_frame.insert(6, "valid_accuracy_swa", self.valid_accs_swa)
            learning_curve_frame.insert(7, "valid_accuracy_robust_swa", self.valid_accs_robust_swa)
            learning_curve_frame.insert(8, "valid_accuracy_adversarial_swa", self.valid_accs_adv_swa)
        learning_curve_frame.to_csv(f'./results/{self.dataset}/{self.modeltype}/config{self.experiment}_'
                                    f'learning_curve_run_{self.run}.csv',
                                    index=False, header=True, sep=';', float_format='%1.4f', decimal=',')

        x = list(range(1, len(self.train_accs) + 1))
        plt.figure()
        plt.plot(x, self.train_accs, label='Train Accuracy')
        plt.plot(x, self.valid_accs, label='Validation Accuracy')
        if self.validonc == True:
            plt.plot(x, self.valid_accs_robust, label='Robust Validation Accuracy')
        if self.validonadv == True:
            plt.plot(x, self.valid_accs_adv, label='Adversarial Validation Accuracy')
        if self.swa == True:
            plt.plot(x, self.valid_accs_swa, label='SWA Validation Accuracy')
            plt.plot(x, self.valid_accs_robust_swa, label='SWA Robust Validation Accuracy')
            plt.plot(x, self.valid_accs_adv_swa, label='SWA Adversarial Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.xticks(np.linspace(1, len(self.train_accs), num=10, dtype=int))
        plt.legend(loc='best')
        plt.savefig(f'results/{self.dataset}/{self.modeltype}/config{self.experiment}_learning_curve_run_{self.run}.svg')
        plt.close()

    def save_config(self):
        shutil.copyfile(f'./experiments/configs/config{self.experiment}.py',
                        f'./results/{self.dataset}/{self.modeltype}/config{self.experiment}.py')

    def print_results(self):
        print("Maximum validation accuracy of", max(self.valid_accs), "achieved after",
              np.argmax(self.valid_accs) + 1, "epochs; ")
        if self.validonc:
            print("Maximum robust validation accuracy of", max(self.valid_accs_robust), "achieved after",
                  np.argmax(self.valid_accs_robust) + 1, "epochs; ")
        if self.validonadv:
            print("Maximum robust validation accuracy of", max(self.valid_accs_adv), "achieved after",
                  np.argmax(self.valid_accs_adv) + 1, "epochs; ")

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
        test_corruptions_string = np.append(test_corruptions_string,
                                            ['mCE-19', 'mCE-15', 'mCE-19_exNoise', 'RMSCE_C'],
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

    avg_report_frame = pd.DataFrame(avg_test_metrics, index=test_corruptions_string,
                                    columns=train_corruptions_string)
    avg_report_frame.to_csv(f'./results/{dataset}/{modeltype}/config{experiment}_{lrschedule}_{training_folder}_'
                            f'metrics_test_avg.csv', index=True, header=True,
                            sep=';', float_format='%1.4f', decimal=',')
    if runs >= 2:
        max_report_frame = pd.DataFrame(max_test_metrics, index=test_corruptions_string,
                                        columns=train_corruptions_string)
        std_report_frame = pd.DataFrame(std_test_metrics, index=test_corruptions_string,
                                        columns=train_corruptions_string)
        max_report_frame.to_csv(
            f'./results/{dataset}/{modeltype}/config{experiment}_{lrschedule}_{training_folder}_'
            f'metrics_test_max.csv', index=True, header=True,
            sep=';', float_format='%1.4f', decimal=',')
        std_report_frame.to_csv(
            f'./results/{dataset}/{modeltype}/config{experiment}_{lrschedule}_{training_folder}_'
            f'metrics_test_std.csv', index=True, header=True,
            sep=';', float_format='%1.4f', decimal=',')


class TestTracking:
    def __init__(self, dataset, modeltype, lrschedule, experiment, runs, combine_train_corruptions, combine_test_corruptions,
                      test_on_c, calculate_adv_distance, calculate_autoattack_robustness, train_corruptions, test_corruptions):
        self.dataset = dataset
        self.modeltype = modeltype
        self.lrschedule = lrschedule
        self.experiment = experiment
        self.runs = runs
        self.combine_train_corruptions = combine_train_corruptions
        self.combine_test_corruptions = combine_test_corruptions
        self.test_on_c = test_on_c
        self.calculate_adv_distance = calculate_adv_distance
        self.calculate_autoattack_robustness = calculate_autoattack_robustness
        self.train_corruptions = train_corruptions
        self.test_corruptions = test_corruptions

        self.results_folder = f'./results/{self.dataset}/{self.modeltype}/config{self.experiment}'

    def create_report(self, avg_test_metrics, max_test_metrics, std_test_metrics):

        if self.combine_train_corruptions == True:
            train_corruptions_string = ['config_model']
        else:
            train_corruptions_string = np.array([','.join(map(str, row.values())) for row in self.train_corruptions])

        test_corruptions_string = np.array(['Standard_Acc', 'RMSCE'])
        if self.test_on_c == True:
            test_corruptions_label = np.loadtxt('./experiments/data/c-labels.txt', dtype=list)
            test_corruptions_string = np.append(test_corruptions_string, test_corruptions_label, axis=0)
            test_corruptions_string = np.append(test_corruptions_string,
                                                ['mCE-19', 'mCE-15', 'mCE-19_exNoise', 'RMSCE_C'],
                                                axis=0)

        if self.calculate_adv_distance == True:
            test_corruptions_string = np.append(test_corruptions_string, ['Acc_from_PGD_adv_distance_calculation',
                                                                          'Mean_PGD_adv_distance_with_misclassified_images_0)',
                                                                          'Mean_PGD_adv_distance_misclassified-images_not_included)',
                                                                          'Mean_CLEVER_score'],
                                                axis=0)
        if self.calculate_autoattack_robustness == True:
            test_corruptions_string = np.append(test_corruptions_string,
                                                ['Adversarial_accuracy_autoattack', 'Mean_adv_distance_autoattack)'],
                                                axis=0)
        if self.combine_test_corruptions == True:
            test_corruptions_string = np.append(test_corruptions_string, ['Combined Noise'])
        else:
            test_corruptions_labels = np.array([','.join(map(str, row.values())) for row in self.test_corruptions])
            test_corruptions_string = np.append(test_corruptions_string, test_corruptions_labels)

        avg_report_frame = pd.DataFrame(avg_test_metrics, index=test_corruptions_string,
                                        columns=train_corruptions_string)
        avg_report_frame.to_csv(f'{self.results_folder}_metrics_test_avg.csv', index=True, header=True,
                                sep=';', float_format='%1.4f', decimal=',')
        if self.runs >= 2:
            max_report_frame = pd.DataFrame(max_test_metrics, index=test_corruptions_string,
                                            columns=train_corruptions_string)
            std_report_frame = pd.DataFrame(std_test_metrics, index=test_corruptions_string,
                                            columns=train_corruptions_string)
            max_report_frame.to_csv(
                f'{self.results_folder}_metrics_test_max.csv', index=True, header=True,
                sep=';', float_format='%1.4f', decimal=',')
            std_report_frame.to_csv(
                f'{self.results_folder}_metrics_test_std.csv', index=True, header=True,
                sep=';', float_format='%1.4f', decimal=',')
