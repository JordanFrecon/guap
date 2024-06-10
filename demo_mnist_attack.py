import numpy as np
import utils.data as data
import utils.display as dsp
import torch
import torchvision
import torchvision.transforms as transforms
import os
import random
from attacks import GUAP
from models.MLP.MLP import CiresanMLP, affine_transformation


if __name__ == '__main__':

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # ------------------------------ DATASET ------------------------------ #

    # Seed
    seed = 1
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Dataset: MNIST
    dataset = torchvision.datasets.MNIST

    # Data transform
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(affine_transformation)
                                    ])

    # Data splits
    val1_set = dataset(root='data', train=True, download=True, transform=transform)
    test_set = dataset(root='data', train=False, download=True, transform=transform)

    # Data loaders
    batch_size = 500
    n_val = 5000
    n_tst = 5000
    n_class = 10
    val_loader, _, val1_set, _ = data.prepare_data(raw_data=val1_set, batch_size=batch_size,
                                                                      n_class=10, n_val=n_val, n_tst=n_tst,
                                                                      n_val1=n_val)
    tst_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, pin_memory=True)

    # -------------------------- MODEL TRAINING -------------------------- #

    model = CiresanMLP(network_id=5)
    filename = os.path.join('models', 'MLP', 'model_ciresan_mnist_id5.pt')
    model.load_state_dict(torch.load(filename, map_location=device))
    model = model.to(device=device)

    # Compute validation accuracy
    val_acc = 0
    for x, y in val_loader:
        x, y = x.to(device=device), y.to(device=device)
        val_acc += float(model(x).argmax(dim=1).eq(y).sum())
    val_acc = val_acc / val_loader.dataset.__len__()
    print('Validation accuracy: ', str(val_acc))

    # -------------------------- ADVERSARIAL ATTACK -------------------------- #

    # Attack setting (typical: l2 [.25, .5] and linf [4/255, 8/255])
    norm = 'linf'
    budget = 8/255

    # Learn adversarial perturbations
    guap = GUAP(model=model.eval(), norm=norm, eps=budget, num_perturbations=5,
                num_epochs=5e1, batch_size=10, lr=1e3, verbose=True)

    guap.update(dataset=val1_set, model=model.eval())

    # Display perturbations
    perturbations = guap.perturbations.reshape(1, 28, 28, guap.atk_setting.num_perturbations)
    dsp.plot_perturbations(perturbations)

    # Display performance
    attacker = guap
    fr = 0
    for x, y in tst_loader:
        x, y = x.to(device=device), y.to(device=device)
        a = torch.clamp(attacker(x,y), min=-1, max=1)
        fr += float(model(x).argmax(dim=1).eq(model(a).argmax(dim=1)).sum())
    fr = fr / tst_loader.dataset.__len__()
    fr = 1-fr
    print('Test fooling rate: ' + str(fr))