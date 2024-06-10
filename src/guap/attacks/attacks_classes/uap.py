import torch
import torch.nn as nn
import torchattacks
import numpy as np

from torchattacks.attack import Attack
torch.set_default_tensor_type(torch.DoubleTensor)


class UAP(Attack):
    r"""
    'Universal adversarial perturbations'
    [https://arxiv.org/abs/1610.08401]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        desired_accuracy (float): Percentage of minimum accuracy to reach (Default: 1)
        max_iter: Maximum number of iterations (Default: 100)
        steps (int): number of steps. (Default: 50) : DeepFool parameter
        overshoot (float): parameter for enhancing the noise. (Default: 0.02) : DeepFool parameter

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = UAP(model, steps=50, overshoot=0.02)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, data_train=None, desired_foolingrate=1., max_iter=100, steps=50, overshoot=0.02):
        super().__init__("UAP", model)
        self.desired_foolingrate = desired_foolingrate
        self.reached_foolingrate = None
        self.max_iter = max_iter
        self.steps = steps
        self.overshoot = overshoot
        self._supported_mode = ['default']
        self.universal_noise = None
        if data_train is not None:
            self.update(dataset=data_train)

    def update(self, dataset):
        r"""
        Overridden.
        """
        # Initialization
        images = dataset
        n_img = len(images)
        idx_list = np.arange(0, n_img)
        fooling_rate = 0.0
        iteration = 0
        deepfool_attack = torchattacks.DeepFool(self.model.eval(), steps=self.steps, overshoot=self.overshoot)

        # Initializing the perturbation to 0s
        adv_noise = torch.zeros_like(images[0][0], device=self.device)

        # Begin of the main loop on Universal Adversarial Perturbations algorithm
        while fooling_rate < self.desired_foolingrate and iteration < self.max_iter:
            np.random.shuffle(idx_list)
            target_labels = []
            for idx in idx_list:
                x, _ = images[idx]
                x = x.to(device=self.device)

                # Get the original prediction from the model
                original_pred = (self.model.eval()(x.reshape(1, 3, 32, 32)).max(1)[1])

                # Generating the adversarial example using the current adversarial noise
                crt_adv = x + adv_noise
                adv_pred = (self.model.eval()(crt_adv.reshape(1, 3, 32, 32)).max(1)[1])

                # If the prediction did not change then the adversarial noise needs to be updated
                if original_pred == adv_pred:

                    # Compute the adversarial example using deepfool
                    adv_deepfool = deepfool_attack(x.reshape(1, 3, 32, 32), original_pred)

                    # Adding the new adversarial noise found and projecting it in the L2-ball of the original example
                    unproj_v = adv_noise + adv_deepfool
                    adv_noise = unproj_v * torch.clamp(x / torch.norm(unproj_v, p=2), max=1, min=0)
                else:
                    fooling_rate = fooling_rate + (1 / n_img)
                    target_labels.append(adv_pred)
            iteration = iteration + 1

        self.desired_foolingrate = fooling_rate
        self.universal_noise = adv_noise.detach()

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        # labels = labels.clone().detach().to(self.device)

        return torch.clamp(images + self.universal_noise, max=1, min=0)
