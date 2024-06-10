import torch
import numpy as np
import torch.nn as nn


class Setting(object):
    """ Base class to define and organize settings """

    def update(self, **kwargs):
        """ Update parameters' value """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def print(self):
        print(self.__dict__)


class OptimizerSetting(Setting):
    """ Gather all optimizer parameters"""
    def __init__(self, num_epochs=200, lr=.1, optimizer_name='sgd', device=torch.device('cuda'), lr_scheduler=None,
                 verbose=True, weight_decay=0, momentum=0, loss_function=None, early_stopping=True,
                 early_stopping_criterion='accuracy', delta=0, patience=10, batch_size=1, **kwargs):
        super().__init__()
        self.num_epochs = num_epochs
        self.optimizer_name = optimizer_name.lower()
        self.device = device
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.verbose = verbose
        self.loss_function = nn.CrossEntropyLoss() if loss_function is None else loss_function
        self.batch_size = batch_size

        # Early stopping parameters
        self.early_stopping = early_stopping
        self.early_stopping_criterion = early_stopping_criterion
        self.delta = delta
        self.patience = patience

    def get_optimizer(self, model):
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(model.parameters(), lr=self.lr)
        elif self.optimizer_name == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            return torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                   momentum=self.momentum)

    def get_lr_scheduler(self, optimizer):
        if (self.lr_scheduler is not None) or (type(self.lr_scheduler) == bool):
            scheduler = self.lr_scheduler(optimizer)
            self.lr_scheduler = True
            return scheduler
        else:
            self.lr_scheduler = False

    def get_loss_function(self):
        return self.loss_function.to(device=self.device)

    def get_early_stopping(self, validation_data=None):
        if validation_data is None:
            self.early_stopping = False
        else:
            if self.early_stopping:
                return EarlyStopping(patience=self.patience, delta=self.delta, verbose=self.verbose,
                                     criterion=self.early_stopping_criterion)


class OptimizationMeter(object):
    def __init__(self, *args):
        for val in args:
            setattr(self, val, [])

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                tmp = getattr(self, key)
                tmp.append(value)
                setattr(self, key, tmp)
            else:
                setattr(self, key, [value])


class EarlyStopping:
    """
    Early stops the training if validation loss/accuracy doesn't improve after a given patience.
    See https://github.com/Bjarten/early-stopping-pytorch]
    """

    def __init__(self, patience=7, verbose=False, delta=1e-5, path='checkpoint', trace_func=print, criterion='loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
            criterion (string): criterion used for early stopping ('loss' or 'accuracy')
                            Default: loss
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.criterion = criterion.lower()
        self.val_best = np.Inf if self.criterion == 'loss' else 0
        self.delta = delta
        self.path = path + str(np.random.randint(1000)) + '.pt'
        self.trace_func = trace_func

    def __call__(self, val_loss, val_acc, model):

        score = -val_loss if self.criterion == 'loss' else val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, val, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            if self.criterion == 'loss':
                val = -val
                self.trace_func(
                    f'Validation loss decreased ({self.val_best:.6f} --> {val:.6f}).  Saving model ...')
            else:
                self.trace_func(
                    f'Validation accuracy increased ({self.val_best:.6f} --> {val:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_best = val

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))
        return model
