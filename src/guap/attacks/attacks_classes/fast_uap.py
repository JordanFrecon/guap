from torchattacks.attack import Attack
from copy import deepcopy
import torch
import torchattacks
import numpy as np
import torch.nn as nn
torch.set_default_tensor_type(torch.DoubleTensor)
import time

# This file is intended to replace usual_attacks.py


class QuickAttackDataset(torch.utils.data.Dataset):
    """ A Dataset class to quickly build dataset from images and labels """

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.images[item], self.labels[item]


class FastUAP(Attack):
    """ Fast UAP: Fast Universal Adversarial Perturbation from  [Jiazhu Dai and Le Shu, 2021]

     Pytorch implementation aiming to reproduce the TensorFlow version developed in
     https://github.com/FallLeaf0914/fast-universal

     model: torch model to fool.
     data_train: dataset on which the attack is learned
     steps: number of maximum steps. (default: 100)
     fooling_rate: fooling rate to reach on data_train. (default: 1)
     norm: norm of the ball ('l2' or 'linf') constraining the adversarial noise. (default: 'linf')
     eps: radius of the ball on the adversarial noise. (default: inf)
     overshoot: overshoot parameter used in DeepFool. (default: 0.02)
     steps_deepfool: number of steps used in DeepFool. (default: 50)

     source: https://doi.org/10.1016/j.neucom.2020.09.052

     """

    def __init__(self, model, steps=10, fooling_rate=.8, eps=np.inf, norm='linf', data_train=None, overshoot=0.02,
                 steps_deepfool=10, time_limit=np.Inf):
        super().__init__('FastUAP', model)
        self.steps = steps
        self.fooling_rate = fooling_rate
        self.eps = eps
        self.norm = norm
        self.overshoot = overshoot
        self.steps_deepfool = steps_deepfool
        self.time_limit = time_limit

        if data_train is not None:
            self.update(dataset=data_train, model=model)

    def compute_fooling_rate(self, dataset, attack, batch_size=4):
        with torch.no_grad():
            n_img = len(dataset)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            fr = 0
            for x, y in data_loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                #y_pred = self.model.eval()(x).argmax(dim=1)
                y_attk = self.model.eval()(x + attack).argmax(dim=1)
                fr += float(y_attk.eq(y).sum())
                #fr += torch.sum(y_pred != y_attk)

            return fr / n_img

    def project(self, attack):
        if self.norm.lower() == 'l2':
            attack_norm = torch.norm(attack, p='fro', keepdim=False)
            if attack_norm > self.eps:
                return self.eps * attack / attack_norm
            else:
                return attack
        elif self.norm.lower() == 'linf':
            return torch.clamp(attack, min=-self.eps, max=self.eps)

    def update(self, dataset, model):

        # data loader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        # solvers
        deepfool = torchattacks.DeepFool(model=self.model, overshoot=self.overshoot, steps=self.steps_deepfool)
        deepfoolcos = DeepFoolCosinus(model=self.model, overshoot=self.overshoot, steps=self.steps_deepfool)

        # variable
        x, _ = dataset[0]
        attack = torch.zeros_like(x, device=self.device)
        fooling_rate_old = 0
        start_time = time.time()
        flag = False

        for iteration in range(int(self.steps)):
            #fooling_rate = self.compute_fooling_rate(dataset=dataset, attack=attack)
            #if fooling_rate >= self.fooling_rate or fooling_rate < fooling_rate_old:
            #    break
            #else:
            if not flag:
                for x, y in data_loader:

                    x, y = x.to(device=self.device), y.to(device=self.device)

                    # If the attack of 'x' does not fool the model ...
                    if model(x + attack).argmax(dim=1) == model(x).argmax(dim=1):

                        if torch.all(attack.eq(torch.zeros_like(attack))):
                            # if the attack is zero, compute the perturbation  that has the smallest magnitude and
                            # fools the model. The authors suggest to resort to DeepFool
                            delta_attack = deepfool(x, y) - x
                        else:
                            # Otherwise, compute the perturbation that has similar orientation to the current
                            # perturbation and that fools the model.
                            delta_attack = deepfoolcos(x, y, attack) - x
                        attack = self.project(attack + delta_attack)

                    current_time = time.time()
                    print(str(start_time), '\t', str(current_time), '\t', str(current_time - start_time))
                    if np.abs(current_time - start_time) > self.time_limit:
                        flag = True
                        break
        self.attack = attack

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)

        # Check if the Fast-UAP attack has been learned
        if self.attack is None:
            print('The Fast-UAP attack has not been learned. It is now being learned on the given dataset.')
            dataset = QuickAttackDataset(images=images, labels=labels)
            self.update(dataset=dataset, model=self.model)

        return torch.clamp(images + self.attack, min=0, max=1)


class DeepFoolCosinus(Attack):
    r"""
    'DeepFoolCosinus: variant of Deep Fool where, given some perturbation eps_old, find

            arg max_eps cosinus(eps, eps_old) s.t. image + eps + eps_old fools the classifier
    """

    def __init__(self, model, steps=1, overshoot=0.02):
        super().__init__("DeepFoolCosinus", model)
        self.steps = steps
        self.overshoot = overshoot
        self._supported_mode = ['default']

    def forward(self, images, labels, attack_init, return_target_labels=False):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        batch_size = len(images)
        correct = torch.tensor([True] * batch_size)
        target_labels = labels.clone().detach().to(self.device)
        curr_steps = 0

        adv_images = []
        for idx in range(batch_size):
            image = images[idx:idx + 1].clone().detach() + attack_init.clone().detach()
            adv_images.append(image)

        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                if not correct[idx]: continue
                early_stop, pre, adv_image = self._forward_indiv(adv_images[idx], labels[idx], attack_init)
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            curr_steps += 1

        adv_images = torch.cat(adv_images).detach()

        if return_target_labels:
            return adv_images, target_labels

        return adv_images

    def _forward_indiv(self, image, label, attack_init):
        image.requires_grad = True
        fs = self.model(image)[0]
        _, pre = torch.max(fs, dim=0)
        if pre != label:
            return True, pre, image

        ws = self._construct_jacobian(fs, image)
        image = image.detach()

        f_0 = fs[label]
        w_0 = ws[label]

        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0

        cosinus_best = - np.inf
        delta_best = 0
        target_label = 0

        for kk in range(len(wrong_classes)):

            delta = (torch.abs(f_prime[kk]) * w_prime[kk, :, :, :, :] \
                     / (torch.norm(w_prime[kk, :, :, :, :], p=2) ** 2))

            cosinus = torch.tensordot(nn.Flatten()(delta), nn.Flatten()(attack_init)) / \
                      (torch.norm(nn.Flatten()(delta), p=2) * torch.norm(nn.Flatten()(attack_init), p=2))

            if cosinus > cosinus_best:
                cosinus_best = cosinus
                delta_best = delta
                target_label = kk

        adv_image = image + (1 + self.overshoot) * delta_best
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return False, target_label, adv_image,

    def _construct_jacobian(self, y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx + 1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)
