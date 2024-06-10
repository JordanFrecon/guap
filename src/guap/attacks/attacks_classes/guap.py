from attacks.utils import *
from utils.optimization import *
from tqdm import trange
from attacks.utils import bounded_cross_entropy
from torchattacks import Attack

class GUAPAttackSetting(AttackSetting):
    """ Special class of GUAP attacks """

    def __init__(self, num_perturbations=1, perturbations=None, smooth_param=np.inf, device=torch.device('cuda'),
                 initialization='orthogonal',
                 unsupervised_target='prediction', **kwargs):
        super().__init__(**kwargs)
        self.num_perturbations = num_perturbations
        self.perturbations = perturbations
        self.device = device
        self.unsupervised_target = unsupervised_target
        self.initialization = initialization
        # if = inf then no max smoothing, else max is replaced by logsumexp with coeff smooth_param
        self.smooth_param = smooth_param


class GUAP(Attack):
    """
    GUAP: Generalized Universal Adversarial Perturbations

    Arguments:
        shape (list[int]): shape of the input data to attack (Default: None).
                        - Example: [3,32,32] for RGB images of 32x32 pixels
        norm (str): Lp-norm of the attack. ['Linf', 'L2'] (Default: 'L2')
        eps (float): maximum perturbation. (Default: None)
        data_train (Dataset): dataset used to learn the perturbations.
        num_epochs (int): number of epochs to learn the perturbations. (Default: 100).
        n_atoms (int): number of adversarial perturbations atoms. (Default: 10).
        lr (int): learning rate (Default: 10 for norm='L2' and eps/steps for norm='Linf')
        trials (int): number of trials to find the best unsupervised attacks to unseen examples. (Default: 10)
        batch_size (int): batch size used to compute the gradients. Larger values speed up the computation at the cost
                          of a higher memory storage. It has no impact on the results. (Default: len(data_train)).
        optimizer (str): choice of nonconvex solver. (default: 'vmilan')
                        - vmilan
                        - saga
                        - saga-me


    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, 'H = height`
                    and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 <= y_i <= ` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    """

    def __init__(self, model, optimizer='vmilan', perturbations=None,
                 time_limit=np.Inf, choice_loss='bce', **kwargs):
        super().__init__("GUAP", model.eval())

        # Attack and algorithmic parameters
        self.bar = None
        self.criterion = None
        self.data_loader = None
        self.atk_setting = GUAPAttackSetting(device=self.device, **kwargs)
        self.opt_setting = OptimizerSetting(device=self.device, **kwargs)
        self.perturbations = perturbations

        # Choice of optimizer
        self.optim_meter = None
        self.optimizer = optimizer.lower()
        self.time_limit = time_limit
        self.choice_loss = choice_loss

    def update(self, dataset, model):
        """
        Learn/update the adversarial perturbations

        Arguments:
            dataset (Dataset): dataset used to learn the perturbations.
            model (nn.Module): model to attack.
        """

        self.initialize_optimizer(dataset)

        if self.optimizer == 'vmilan':
            self.opt_vMILAN(model=model)
        elif self.optimizer == 'saga':
            self.opt_SAGA(model=model)
        elif self.optimizer == 'saga-me':
            self.opt_SAGA_memoryEfficient(model=model)
        else:
            self.opt_SAGA_memoryEfficientv2(model=model)

    def initialize_perturbations(self, dataset):
        x, _ = next(iter(dataset))
        dict_shape = x.shape.__add__((self.atk_setting.num_perturbations,))

        if self.atk_setting.initialization == 'orthogonal':
            shape_tot = x.shape.numel()
            d = torch.rand(shape_tot, shape_tot, device=self.device)
            q, r = torch.linalg.qr(d)
            d = q[:, 0:self.atk_setting.num_perturbations]
            d = d / torch.max(torch.abs(d))
            self.perturbations = self.projection(d.reshape(dict_shape))
        elif self.atk_setting.initialization == 'zero':
            self.perturbations = torch.zeros(dict_shape, device=self.device)
        else:
            coeff = 1e-4 if self.atk_setting.norm == 'l2' else 1.
            self.perturbations = self.projection(coeff * (-1 + 2 * torch.rand(dict_shape, device=self.device)))

    def projection(self, var):
        if self.atk_setting.norm == 'l2':
            """ In order to respect l2 bound, D has to lie inside a l2 unit ball """
            return project_onto_lp_perturbations_ball(var, constr_set='l2ball')

        elif self.atk_setting.norm == 'linf':
            """ In order to respect linf bound, D has to lie inside a linf unit ball """
            return torch.clamp(var, min=-1, max=1)

    def get_loss(self, x, y, guap, reduction='sum'):

        # Hard max (or equivalently min-)
        coeff = self.atk_setting.smooth_param
        if coeff == np.inf:
            """ Max """
            comp = np.Inf * torch.ones(x.shape[0], device=self.device)
            for _ in range(self.atk_setting.num_perturbations):
                val = -self.criterion(self.model(x + guap[..., _]), y)
                comp = torch.min(comp, val)
        else:
            """ Softmax """
            comp_stack = torch.zeros(self.atk_setting.num_perturbations, x.shape[0], device=self.device)
            for _ in range(self.atk_setting.num_perturbations):
                comp_stack[_, :] = self.criterion(self.model(x + guap[..., _]), y)
            soft_max = torch.nn.functional.softmax(comp_stack / coeff, dim=0)

            # Weighted sum of input values based on softmax probabilities
            comp = -torch.sum(soft_max * comp_stack, dim=0)

        if reduction == 'sum':
            loss = torch.sum(comp)
        else:
            loss = torch.mean(comp)

        return loss

    def forward(self, images, labels):

        # Put data on device
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Parameters
        n_samples = images.shape[0]

        # Pick the best ones
        criterion = nn.CrossEntropyLoss(reduction='none')
        val = torch.zeros([n_samples, self.atk_setting.num_perturbations], device=self.device)
        for _ in range(self.atk_setting.num_perturbations):
            val[:, _] = -criterion(self.model(images + self.perturbations[..., _]), labels)
        ind = torch.argmin(val, dim=1)

        # Form the adversarial images
        adv = torch.zeros_like(images)
        for _ in range(n_samples):
            adv[_] = images[_] + self.perturbations[_, ..., ind[_]]

        return adv

    def initialize_optimizer(self, dataset):

        opt = self.opt_setting
        batch_size = len(dataset) if opt.batch_size is None else opt.batch_size
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Choice of loss function H
        """ The reduction is done at the self.get_loss level (default: sum) """
        if self.choice_loss == 'bce':
            self.criterion = lambda var1, var2: bounded_cross_entropy(var1, var2, reduction='none')
        elif self.choice_loss == 'max-ce':
            """ Clamped cross entropy """
            ce = torch.nn.CrossEntropyLoss(reduction='none')
            self.criterion = lambda var1, var2: torch.clamp_max(ce(var1, var2), max=9)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')

        # Initialization of the perturbations
        if self.perturbations is None:
            self.initialize_perturbations(dataset)

        # Algorithm
        self.optim_meter = OptimizationMeter()
        self.bar = trange(int(opt.num_epochs)) if opt.verbose else range(int(opt.num_epochs))

    """ ----------------------- Below: various optimizers ----------------------- """

    def opt_vMILAN(self, model):
        """ Gradient Based Optimizer"""
        d = self.perturbations
        d_old = d.clone().detach()

        # Linesearch parameters
        gamma = 1
        delta = .9
        beta = .5
        flag_stop = False
        index_i = 0

        self.model = model
        for iteration in self.bar:
            if flag_stop is False:

                # Go through all samples
                loss_full = 0
                fr = 0

                # Prepare computation graph
                d.grad = torch.zeros_like(d)
                d.detach()
                d.requires_grad = True

                for x, y in self.data_loader:
                    # Load data
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)

                    # Compute loss and accumulate gradients
                    loss = self.get_loss(x, y, d)
                    loss.backward()

                    with torch.no_grad():
                        loss_full += float(loss)
                        # fr += float(model(x).argmax(dim=1).ne(model(x + dv).argmax(dim=1)).sum()) / (n_img )

                # Forward-Backward step
                with torch.no_grad():

                    # Memory
                    loss_old = loss_full
                    d_old = d.copy_(d)
                    grad = d.grad

                    # Update
                    d = self.projection(d - self.opt_setting.lr * grad)

                    # added distance
                    dist_d = d - d_old

                    # First order approximation of the difference in loss
                    h = torch.sum(dist_d * grad) + .5 * (gamma / self.opt_setting.lr) * torch.norm(dist_d, 'fro') ** 2

                    flag = False
                    index_i = np.maximum(index_i - 2, 0)
                    while not flag:
                        d_new = d_old + (delta ** index_i) * dist_d

                        # Compute new loss
                        loss_new = 0
                        for x, y in self.data_loader:
                            x = x.to(device=self.device)
                            y = y.to(device=self.device)
                            loss_new += self.get_loss(x, y, d_new)

                        # Check the sufficient decrease condition
                        if loss_new <= loss_old + beta * (delta ** index_i) * h:
                            # Then it's fine !
                            d = d_new
                            flag = True
                        else:
                            # Then we need to change index_i
                            index_i = index_i + 1
                            if index_i > 100:
                                # We have reached a stationary point
                                flag_stop = True
                                flag = True

                # Keep track of loss and fooling rate
                self.optim_meter.update(loss=loss_full, fooling_rate=fr)

                # Display the atoms
            self.bar.set_description('EPOCH: %d - fooling rate: %.2f' % (iteration + 1, 100 * fr))

        # Assign the perturbations and close display
        self.perturbations = d.detach()

    def opt_SAGA(self, model):
        """ ProxSAGA based stochastic solver """
        d = self.perturbations
        self.model = model

        # Parameters
        n_samples = len(self.data_loader.dataset)
        index = get_slices(n_samples, self.data_loader.batch_size)
        nb_slices = len(index)

        # Initialization of variables
        dshape_aug = torch.Size([nb_slices]).__add__(d.shape)
        g = torch.zeros(dshape_aug, device=self.device)
        gtilde = torch.zeros(dshape_aug, device=self.device)
        gbar = torch.zeros_like(d)
        loss_all = np.NaN * torch.ones(n_samples)

        iteration = 0
        while iteration < self.bar.__len__():
            for ii, (x, y) in enumerate(self.data_loader):

                # Load data
                ind = index[ii]
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                # Prepare computation graph
                d.detach()
                d.requires_grad = True
                d.grad = torch.zeros_like(d)

                # Compute loss and accumulate gradients
                loss = self.get_loss(x, y, d)
                loss.backward()

                with torch.no_grad():
                    loss_all[ii] = float(loss)
                    loss_full = torch.mean(loss_all)  # should be nanmean
                    fr = 0

                    # instant gradient
                    g[ii] = d.grad

                    # projected gradient step
                    alpha = (g[ii] - gtilde[ii]) / len(ind) + gbar
                    d = self.projection(d - self.opt_setting.lr * alpha)

                    # Update
                    gbar = (1 / n_samples) * (g[ii] - gtilde[ii]) + gbar
                    gtilde[ii] = g[ii]

                    # Display the atoms

                with torch.no_grad():
                    self.optim_meter.update(loss=loss_full, fooling_rate=fr)
                    self.bar.set_description('EPOCH: %d - fooling rate: %.2f' % (iteration, 100 * fr))
                    self.bar.update()
                    iteration += 1
                    if iteration > self.bar.__len__():
                        break

        # Assign the perturbations and close display
        self.perturbations = d.detach()

    def opt_SAGA_memoryEfficient(self, model):
        """ ProxSAGA based stochastic solver """
        d = self.perturbations
        self.model = model

        # Parameters
        n_samples = len(self.data_loader.dataset)
        index = get_slices(n_samples, self.data_loader.batch_size)
        nb_slices = len(index)

        # Initialization of variables
        g_old = torch.zeros_like(d)
        gbar = torch.zeros_like(d)
        loss_all = np.NaN * torch.ones(n_samples)

        iteration = 0
        while iteration < self.bar.__len__():
            for ii, (x, y) in enumerate(self.data_loader):

                # Load data
                ind = index[ii]
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                # Prepare computation graph
                d.detach()
                d.requires_grad = True
                d.grad = torch.zeros_like(d)

                # Compute loss and accumulate gradients
                loss = self.get_loss(x, y, d)
                loss.backward()

                with torch.no_grad():
                    # loss_all[ii] = float(loss)
                    loss_full = float(loss)  # torch.mean(loss_all) #should be nanmean
                    fr = 0

                    # instant gradient
                    g_inst = d.grad

                    # projected gradient step
                    alpha = (g_inst - g_old) / len(ind) + gbar
                    d = self.projection(d - self.opt_setting.lr * alpha)

                    # Update
                    gbar = (1 / n_samples) * (g_inst - g_old) + gbar
                    g_old = g_inst

                with torch.no_grad():
                    self.optim_meter.update(loss=loss_full, fooling_rate=fr)
                    self.bar.set_description('EPOCH: %d - fooling rate: %.2f' % (iteration, 100 * fr))
                    self.bar.update()
                    iteration += 1
                    if iteration > self.bar.__len__():
                        break

        # Assign the perturbations and close display
        self.perturbations = d.detach()

    def opt_SAGA_memoryEfficientv2(self, model):
        """ ProxSAGA based stochastic solver """
        d = self.perturbations
        self.model = model

        # Parameters
        n_samples = len(self.data_loader.dataset)
        index = get_slices(n_samples, self.data_loader.batch_size)
        nb_slices = len(index)

        # Initialization of variables
        gbar = torch.zeros_like(d)
        loss_all = np.NaN * torch.ones(n_samples)

        # Time constraint
        start_time = time.time()
        flag = False

        iteration = 0
        while iteration < self.bar.__len__() and not flag:
            for ii, (x, y) in enumerate(self.data_loader):

                # Load data
                ind = index[ii]
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                # Prepare computation graph
                d.detach()
                d.requires_grad = True
                d.grad = torch.zeros_like(d)

                # Compute loss and accumulate gradients
                loss = self.get_loss(x, y, d)
                loss.backward()

                with torch.no_grad():
                    loss_all[ii] = float(loss)
                    loss_full = torch.mean(loss_all)  # should be nanmean
                    fr = 0

                    # instant gradient
                    g_inst = d.grad

                    # projected gradient step
                    alpha = g_inst / len(ind) + gbar
                    d = self.projection(d - self.opt_setting.lr * alpha)

                    # Update
                    gbar = (1 / n_samples) * g_inst + gbar

                with torch.no_grad():
                    self.optim_meter.update(loss=loss_full, fooling_rate=fr)
                    self.bar.set_description('EPOCH: %d - fooling rate: %.2f' % (iteration, 100 * fr))
                    self.bar.update()
                    iteration += 1
                    current_time = time.time()
                    if iteration > self.bar.__len__() or current_time-start_time > self.time_limit:
                        flag = True
                        break

        # Assign the perturbations and close display
        self.perturbations = d.detach()
