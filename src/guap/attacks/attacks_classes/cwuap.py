from attacks.utils import *
import torchattacks
import time

torch.set_default_tensor_type(torch.DoubleTensor)


class CWUAP(Attack):
    """ CWUAP: Class-wise Universal Adversarial Perturbation from [Benz et al., 2021]

     eps: radius of the ball on the adversarial noise
     batch_size:

    source: https://arxiv.org/pdf/2104.03000.pdf

     """

    def __init__(self, model, data_train=None, steps=10, batch_size=128, beta=9, step_size=1, norm='l2', eps=.1,
                 optimizer='adam', time_limit=np.Inf):
        super().__init__("CWUAP", model)
        self.attack = None
        self.beta = beta
        self.steps = steps
        self.step_size = step_size
        self.batch_size = batch_size
        self.norm = norm
        self.eps = eps
        self.optimizer = optimizer
        self.num_labels = None
        self.coeff = -1. if self._targeted is False else 1.
        self.time_limit = time_limit

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
        # Get the number of labels
        y = [a.item() for (_, a) in dataset]
        labels = np.unique(y)
        self.num_labels = len(labels)

        # Get the shape of the attack
        x, _ = dataset[0]
        attack_shape = torch.Size([self.num_labels]).__add__(x.shape)
        self.attack = torch.zeros(attack_shape, device=self.device)

        # Split the dataset
        dataset_split = {k: [] for k in range(self.num_labels)}
        for x, y in dataset:
            dataset_split[y.item()].append([x, y])

        # Learn the universal perturbations
        for _ in range(self.num_labels):
            self.attack[_] = self.update_indiv(dataset=dataset_split[_], model=model)

    def update_indiv(self, dataset, model):

        # data loader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        # variable
        x, _ = dataset[0]
        attack = torch.zeros(x.shape, device=self.device, requires_grad=True)
        start_time = time.time()

        # optimizer
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD([attack], lr=self.step_size)
        else:
            optimizer = torch.optim.Adam([attack], lr=self.step_size)

        flag = False
        for _ in range(int(self.steps)):
            while not flag:
                for x, y in data_loader:
                    x, y = x.to(device=self.device), y.to(device=self.device)

                    optimizer.zero_grad()
                    #loss = self.coeff * torch.clamp_max(criterion(model(x + attack), y), max=self.beta)
                    loss = self.coeff * torch.clamp(criterion(model(x + attack), y), max=self.beta, min=-self.beta)
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        attack = self.project(attack)
                    attack.requires_grad = True

                    current_time = time.time()
                    print(str(start_time), '\t', str(current_time), '\t', str(current_time - start_time))
                    if np.abs(current_time - start_time) > self.time_limit/self.num_labels:
                        flag = True
                        break

        return attack.detach()

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Check if the UAP attack has been learned
        if self.attack is None:
            print('The UAP attack has not been learned. It is now being learned on the given dataset.')
            # dataset = QuickAttackDataset(images=images, labels=labels)
            # self.update(dataset=dataset, model=self.model)

        # return torch.clamp(images + self.attack, min=0, max=1)
        return images + self.attack[labels]
