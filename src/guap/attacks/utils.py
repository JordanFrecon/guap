import torch
from utils.optimization import Setting
import time


def bounded_cross_entropy(var1, var2, lmax=4, reduction='mean', rescale=True):
    """ Bounded cross entropy. If rescale=true, yields value in [0,1] """
    softmax = torch.nn.Softmax(dim=1)
    probability = softmax(var1)
    mapped_probability = np.exp(-lmax) + (1 - 2 * np.exp(-lmax)) * probability
    log_mapped_proba = torch.log(mapped_probability)
    nlloss = torch.nn.NLLLoss(reduction=reduction)
    return nlloss(log_mapped_proba, var2) / lmax if rescale else nlloss(log_mapped_proba, var2)


class AttackSetting(Setting):
    """ Class of Attack parameters """

    def __init__(self, norm='l2', eps=None, targeted=False, **kwargs):
        super().__init__()
        self.norm = norm.lower()
        if eps is None:
            self.eps = 8 / 255 if (self.norm == 'linf') else 0.5
        else:
            self.eps = eps
        self.targeted = targeted
        self.update(**kwargs)


def project_onto_l1_ball(x, eps):
    """
    Compute Euclidean projection onto the L1 ball
    Pytorch version of Adrien Gaidon's work.
    Reference
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)


def project_onto_lp_perturbations_ball(d, constr_set='l2ball'):
    # Projection ||d||_2 = 1 (l2sphere) or ||d||_2 <= 1 (l2ball) or ||d||_1 <= (l1ball)
    n_atom = d.shape[-1]
    for ind in range(n_atom):
        d_norm = torch.norm(d[..., ind], p='fro', keepdim=True)
        if constr_set == 'l2sphere':
            # Project onto the sphere
            d[..., ind] = torch.div(d[..., ind], d_norm)
        elif constr_set == 'l2ball':
            # Project onto the ball
            d[..., ind] = torch.div(d[..., ind], torch.maximum(d_norm, torch.ones_like(d_norm)))
        else:
            d[..., ind] = project_onto_l1_ball(d[..., ind], eps=1)
    return d


def get_slices(n, step):
    """ Return the slices (of size step) of N values """
    n_range = range(n)
    return [list(n_range[ii:min(ii + step, n)]) for ii in range(0, n, step)]


class Attack(object):
    r"""
    Base class for all attacks.

    .. note::
        It automatically set device to the device where given model is.
        It basically changes training mode to eval during attack process.
        To change this, please see `set_training_mode`.
    """

    def __init__(self, name, model):
        r"""
        Initializes internal attack state.

        Arguments:
            name (str): name of attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]
        self.device = next(model.parameters()).device

        self._attack_mode = 'default'
        self._targeted = False
        self._return_type = 'float'
        self._supported_mode = ['default']

        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False

    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def get_mode(self):
        r"""
        Get attack mode.

        """
        return self._attack_mode

    def set_mode_default(self):
        r"""
        Set attack mode as default mode.

        """
        self._attack_mode = 'default'
        self._targeted = False
        print("Attack mode is changed to 'default.'")

    def set_mode_targeted_by_function(self, target_map_function=None):
        r"""
        Set attack mode as targeted.

        Arguments:
            target_map_function (function): Label mapping function.
                e.g. lambda images, labels:(labels+1)%10.
                None for using input labels as targeted labels. (Default)

        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = 'targeted'
        self._targeted = True
        self._target_map_function = target_map_function
        print("Attack mode is changed to 'targeted.'")

    def set_mode_targeted_least_likely(self, kth_min=1):
        r"""
        Set attack mode as targeted with least likely labels.
        Arguments:
            kth_min (str): label with the k-th smallest probability used as target labels. (Default: 1)

        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = "targeted(least-likely)"
        self._targeted = True
        self._kth_min = kth_min
        self._target_map_function = self._get_least_likely_label
        print("Attack mode is changed to 'targeted(least-likely).'")

    def set_mode_targeted_random(self, n_classses=None):
        r"""
        Set attack mode as targeted with random labels.
        Arguments:
            num_classses (str): number of classes.

        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = "targeted(random)"
        self._targeted = True
        self._n_classses = n_classses
        self._target_map_function = self._get_random_target_label
        print("Attack mode is changed to 'targeted(random).'")

    def set_return_type(self, type):
        r"""
        Set the return type of adversarial images: `int` or `float`.

        Arguments:
            type (str): 'float' or 'int'. (Default: 'float')

        .. note::
            If 'int' is used for the return type, the file size of
            adversarial images can be reduced (about 1/4 for CIFAR10).
            However, if the attack originally outputs float adversarial images
            (e.g. using small step-size than 1/255), it might reduce the attack
            success rate of the attack.

        """
        if type == 'float':
            self._return_type = 'float'
        elif type == 'int':
            self._return_type = 'int'
        else:
            raise ValueError(type + " is not a valid type. [Options: float, int]")

    def set_training_mode(self, model_training=False, batchnorm_training=False, dropout_training=False):
        r"""
        Set training mode during attack process.

        Arguments:
            model_training (bool): True for using training mode for the entire model during attack process.
            batchnorm_training (bool): True for using training mode for batchnorms during attack process.
            dropout_training (bool): True for using training mode for dropouts during attack process.

        .. note::
            For RNN-based models, we cannot calculate gradients with eval mode.
            Thus, it should be changed to the training mode during the attack.
        """
        self._model_training = model_training
        self._batchnorm_training = batchnorm_training
        self._dropout_training = dropout_training

    def save(self, data_loader, save_path=None, verbose=True, return_verbose=False):
        r"""
        Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.

        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (Default: True)
            return_verbose (bool): True for returning detailed information. (Default: False)

        """
        if (verbose == False) and (return_verbose == True):
            raise ValueError("Verobse should be True if return_verbose==True.")

        if save_path is not None:
            image_list = []
            label_list = []

        correct = 0
        total = 0
        l2_distance = []

        total_batch = len(data_loader)

        given_training = self.model.training

        for step, (images, labels) in enumerate(data_loader):
            start = time.time()
            adv_images = self.__call__(images, labels)

            batch_size = len(images)

            if save_path is not None:
                image_list.append(adv_images.cpu())
                label_list.append(labels.cpu())

            if self._return_type == 'int':
                adv_images = adv_images.float() / 255

            if verbose:
                with torch.no_grad():
                    if given_training:
                        self.model.eval()
                    outputs = self.model(adv_images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = (predicted == labels.to(self.device))
                    correct += right_idx.sum()
                    end = time.time()
                    delta = (adv_images - images.to(self.device)).view(batch_size, -1)
                    l2_distance.append(torch.norm(delta[~right_idx], p=2, dim=1))

                    rob_acc = 100 * float(correct) / total
                    l2 = torch.cat(l2_distance).mean().item()
                    progress = (step + 1) / total_batch * 100
                    elapsed_time = end - start
                    self._save_print(progress, rob_acc, l2, elapsed_time, end='\r')

        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end='\n')

        if save_path is not None:
            x = torch.cat(image_list, 0)
            y = torch.cat(label_list, 0)
            torch.save((x, y), save_path)
            print('- Save complete!')

        if given_training:
            self.model.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time

    def _save_print(self, progress, rob_acc, l2, elapsed_time, end):
        print('- Save progress: %2.2f %% / Robust accuracy: %2.2f %% / L2: %1.5f (%2.3f it/s) \t' \
              % (progress, rob_acc, l2, elapsed_time), end=end)

    def _get_target_label(self, images, labels=None):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        if self._target_map_function:
            return self._target_map_function(images, labels)
        raise ValueError('Please define target_map_function.')

    def _get_least_likely_label(self, images, labels=None):
        r"""
        Function for changing the attack mode.
        Return least likely labels.
        """
        outputs = self.model(images)
        if self._kth_min < 0:
            pos = outputs.shape[1] + self._kth_min + 1
        else:
            pos = self._kth_min
        _, target_labels = torch.kthvalue(outputs.data, pos)
        target_labels = target_labels.detach()
        return target_labels.long().to(self.device)

    def _get_random_target_label(self, images, labels=None):
        if self._n_classses is None:
            outputs = self.model(images)
            if labels is None:
                _, labels = torch.max(outputs, dim=1)
            n_classses = outputs.shape[-1]
        else:
            n_classses = self._n_classses

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            t = self.random_int(0, len(l))
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    def random_int(self, low=0, high=1, shape=[1]):
        t = low + (high - low) * torch.rand(shape).to(self.device)
        return t.long()

    def _to_uint(self, images):
        r"""
        Function for changing the return type.
        Return images as int.
        """
        return (images * 255).type(torch.uint8)

    def __str__(self):
        info = self.__dict__.copy()

        del_keys = ['model', 'attack']

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info['attack_mode'] = self._attack_mode
        info['return_type'] = self._return_type

        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __call__(self, *input, **kwargs):
        given_training = self.model.training

        if self._model_training:
            self.model.train()
            for _, m in self.model.named_modules():
                if not self._batchnorm_training:
                    if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if 'Dropout' in m.__class__.__name__:
                        m = m.eval()

        else:
            self.model.eval()

        images = self.forward(*input, **kwargs)

        if given_training:
            self.model.train()

        if self._return_type == 'int':
            images = self._to_uint(images)

        return images
        
        
def get_target(img, label, classifier, targeted):
    with torch.no_grad():
        if targeted:
            f_x = classifier(img)
            _, index = f_x.sort()
            return index[:, -2]
        else:
            f_x = classifier(img)
            _, index = f_x.sort()
            return label
            
def attack_status(loss, perturbation, norm):
    with torch.no_grad():
        n_batch = perturbation.shape[0]
        device = perturbation.device

        if norm.lower() == 'l2':
            bound = torch.mean(torch.sqrt(torch.sum(perturbation ** 2, dim=[1, 2, 3]))).data
        else:
            bound = torch.zeros(1, device=device)
            for ind in range(n_batch):
                bound += torch.max(torch.abs(perturbation[ind, :, :, :]))/n_batch

        return loss.item(), bound.item()
        
def proj_constraint_adversary(img, norm=None, radius=None, valid=True):
    if norm is None or radius is None:
        if valid:
            return lambda var: torch.clamp(var, min=0, max=1)
        else:
            return lambda var: var
    elif norm.lower() == 'l1':
        if valid:
            return lambda var: torch.clamp(img + project_onto_l1_ball(var-img, radius), min=0, max=1)
        else:
            return lambda var: img + project_onto_l1_ball(var - img, radius)
    elif norm.lower() == 'l2':
        if valid:
            return lambda var: torch.clamp(img + project_onto_l2_ball(var-img, radius), min=0, max=1)
        else:
            return lambda var: img + project_onto_l2_ball(var - img, radius)
    elif norm.lower() == 'linf':
        if valid:
            return lambda var: torch.clamp(img + torch.clamp(var-img, min=-radius, max=radius), min=0, max=1)
        else:
            return lambda var: img + torch.clamp(var - img, min=-radius, max=radius)
    else:
        raise ValueError
        
def project_onto_l2_ball(var_x, eps):
    """ Compute Euclidean projection onto the L2 ball """

    batch_size = var_x.shape[0]
    x_norms = torch.norm(var_x.view(batch_size, -1), p=2, dim=1)
    factor = eps / x_norms
    factor = torch.min(factor, torch.ones_like(x_norms))
    var_x = var_x * factor.view(-1, 1, 1, 1)

    return var_x

