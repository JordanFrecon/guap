import numpy as np
import utils.data as data
import utils.display as dsp
import torch
import torchvision
import torchvision.transforms as transforms
from models.PyTorch_CIFAR10.cifar10_models.resnet import resnet18
import random
import matplotlib.pyplot as plt
from attacks import GUAP

seed = 1
torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# ------ INSTRUCTIONS ------- #
# To make it work, you first need to download the weights of the neural networks
# Go to models/Pytorch_Cifar10 and run 'python train.py --download_weights 1'


if __name__ == '__main__':

    # CUDA for PyTorch & precision
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    pin_memory = True if use_cuda else False

    ###########################################
    #           Dataset: CIFAR10              #
    ###########################################
    print("Small batch of 250 images")
    dataset = torchvision.datasets.CIFAR10
    n_val = 250
    n_tst = 500
    batch_size = 250
    n_class = 10

    # Pre-processing
    preprocessing = transforms.Compose([transforms.ToTensor()])

    # Split into validation and test set
    raw_test_data = dataset(root='data', train=False, download=True, transform=preprocessing)
    val_loader, tst_loader, val1_set, val2_loader = data.prepare_data(raw_data=raw_test_data, batch_size=batch_size,
                                                                      n_class=10, n_val=n_val, n_tst=n_tst,
                                                                      n_val1=n_val // 2)

    ###########################################
    #                 Models                  #
    ###########################################

    # Attack setting
    norm = 'linf'
    budget = 8/255

    # Normalize layer
    class Normalize(torch.nn.Module):
        def __init__(self, mean, std):
            super(Normalize, self).__init__()
            self.register_buffer('mean', torch.Tensor(mean))
            self.register_buffer('std', torch.Tensor(std))

        def forward(self, input_var):
            mean = self.mean.reshape(1, 3, 1, 1)
            std = self.std.reshape(1, 3, 1, 1)
            return (input_var - mean) / std


    # Model used to learn the attacks
    normalize = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    model = torch.nn.Sequential(
        normalize,
        resnet18(pretrained=True).eval().to(device=device)
    )
    model = model.eval()

    # Compute validation accuracy
    val_acc = 0
    for x, y in val_loader:
        x, y = x.to(device=device), y.to(device=device)
        val_acc += float(model(x).argmax(dim=1).eq(y).sum())
    val_acc = val_acc / n_val
    print('Validation accuracy: ', str(val_acc))

    ###########################################
    #               Adversary                 #
    ###########################################

    # Learn adversarial perturbations
    guap = GUAP(model=model, norm=norm, eps=budget, num_perturbations=5, num_epochs=100,
                batch_size=batch_size, lr=1, verbose=True)

    guap.update(dataset=val1_set, model=model)

    # Display loss
    plt.figure(1)
    plt.plot(guap.optim_meter.loss)
    plt.xlabel(r'Steps')
    plt.ylabel(r'Training loss')

    # Display fooling rate
    plt.figure(2)
    plt.plot(guap.optim_meter.fooling_rate)
    plt.xlabel(r'Steps')
    plt.ylabel(r'Training fooling rate')

    # Display atoms
    dsp.plot_perturbations(guap.perturbations)
