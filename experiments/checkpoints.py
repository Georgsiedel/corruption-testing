#credit to https://github.com/Bjarten/early-stopping-pytorch/tree/master
import numpy as np
import torch

class Checkpoint:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, earlystopping = False, patience=7, verbose=False, delta=0, trace_func=print,
                 model_path= 'experiments/trained_models/checkpoint.pt',
                 best_model_path = 'experiments/trained_models/best_checkpoint.pt'):
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
        self.val_loss_min = 1000 #placeholder initial value
        self.delta = delta
        self.trace_func = trace_func
        self.earlystopping = earlystopping
        self.model_path = model_path
        self.best_model_path = best_model_path

    def _earlystopping(self, val_acc, model):

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

    def _load_model(self, model, optimizer, scheduler, best=False):
        checkpoint = torch.load(self.best_model_path) if best == True else torch.load(self.model_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch, model, optimizer, scheduler

    def _save_checkpoint(self, model, optimizer, scheduler, epoch):

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

    def _save_final_model(self, model, optimizer, scheduler, epoch, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, path)





