import numpy as np
import torch

from attacks.utils import *
import torchattacks
from attacks.utils import bounded_cross_entropy
import wandb


torch.set_default_tensor_type(torch.DoubleTensor)

class UAPPGD(Attack):
    """ UAP: Universal Adversarial Perturbation from  [Shafahi et al., 2020]

     beta: clipping parameter of the cross-entropy loss (Default: 9). Note that it strongly affects the performance
     eps: radius of the ball on the adversarial noise
     batch_size (Default: 128)

    source: https://arxiv.org/pdf/1811.11304.pdf

     """

    def __init__(self, model, data_train=None, steps=10, batch_size=128, beta=9, step_size=1, norm='l2', eps=.1,
                 optimizer='adam', criterion="max-ce", time_limit=np.Inf, wandb_log=False):
        super().__init__("UAPPGD", model)
        self.attack = None
        self.beta = beta
        self.steps = steps
        self.step_size = step_size
        self.batch_size = batch_size
        self.norm = norm
        self.eps = eps
        self.optimizer = optimizer
        self.coeff = -1. if self._targeted is False else 1.
        self.time_limit = time_limit
        self.wandb = wandb_log

        if criterion == 'max-ce':
            """ Clamped cross entropy """
            ce = torch.nn.CrossEntropyLoss(reduction='none')  # Was mean before
            self.criterion = lambda var1, var2: torch.sum(torch.clamp_max(ce(var1, var2), max=self.beta), dim=0)
        elif criterion == 'ce':
            """ cross-entropy """
            ce = torch.nn.CrossEntropyLoss(reduction='sum')
            self.criterion = lambda var1, var2: ce(var1, var2)
        elif criterion == 'bce':
            self.criterion = lambda var1, var2: bounded_cross_entropy(var1, var2)

        if data_train is not None:
            self.update(dataset=data_train, model=self.model.eval())

    def project(self, attack):
        if self.norm.lower() == 'l2':
            attack_norm = torch.norm(attack, p='fro', keepdim=False)
            if float(attack_norm) > self.eps:
                return self.eps * attack / attack_norm
            else:
                return attack
        elif self.norm.lower() == 'linf':
            return torch.clamp(attack, min=-self.eps, max=self.eps)

    def update(self, dataset, model):

        # data loader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # variable
        x, _ = dataset[0]
        attack = torch.nn.parameter.Parameter(torch.zeros(x.shape, device=self.device), requires_grad=True)
        start_time = time.time()

        # optimizer
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD([attack], lr=self.step_size)
        else:
            optimizer = torch.optim.Adam([attack], lr=self.step_size)

        flag = False
        for _ in range(int(self.steps)):
            if flag:
                break
            else:
                for x, y in data_loader:
                    loss_full = 0
                    if not flag:
                        x, y = x.to(device=self.device), y.to(device=self.device)

                        optimizer.zero_grad()
                        loss = self.criterion(model(x + attack), y)
                        loss.backward()
                        optimizer.step()

                        with torch.no_grad():
                            loss_full += loss.data

                            # Projection
                            attack = self.project(attack)

                            # Time constraint
                            current_time = time.time()
                            # print(str(start_time), '\t', str(current_time), '\t', str(current_time - start_time))
                            if np.abs(current_time - start_time) > self.time_limit:
                                flag = True
                                break
                        attack.requires_grad = True
                    if self.wandb is True:
                        wandb.log({"loss": loss_full})

        self.attack = attack.detach()

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)

        # Check if the UAP attack has been learned
        if self.attack is None:
            print('The UAP attack has not been learned. It is now being learned on the given dataset.')
            # dataset = QuickAttackDataset(images=images, labels=labels)
            # self.update(dataset=dataset, model=self.model)

        return images + self.attack  # not used in the neurips submission, instead we were projecting back
