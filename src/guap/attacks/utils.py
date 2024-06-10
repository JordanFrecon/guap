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

