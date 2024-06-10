import numpy as np
import matplotlib.pyplot as plt
import torch
torch.set_default_tensor_type(torch.DoubleTensor)


def extract_part(dico):
    # Extract positive and negative parts
    dico = dico.cpu().numpy()
    dico_positive = np.where(dico > 0, dico, 0)
    dico_negative = np.abs(np.where(dico < 0, dico, 0))
    vceil = np.max([dico.max(), np.abs(dico.min())])
    return dico_positive * (1 / vceil), dico_negative * (1 / vceil)
    #return dico_positive, dico_negative


def plot_perturbations(D, cmap=None, num=None, width=None):
    n_dict = D.shape[3]

    if width is None:
        fig = plt.subplots(nrows=2, ncols=n_dict)
    else:
        fig, ax = plt.subplots(2, n_dict, figsize=set_size(width, subplots=(2, n_dict)))

    for ind_dic in range(n_dict):
        if cmap is not None:
            plt.set_cmap(cmap)
        [d1_p, d1_n] = extract_part(D[:, :, :, ind_dic])
        ax = plt.subplot(2, n_dict, ind_dic + 1)
        plt.imshow(np.transpose(d1_p, (1, 2, 0)))
        if num is None:
            val = ind_dic+1
        else:
            val = num[ind_dic]
        ax.title.set_text(r'$\varepsilon^+_{' + str(val) + '}$')
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        ax = plt.subplot(2, n_dict, n_dict + ind_dic + 1)
        plt.imshow(np.transpose(d1_n, (1, 2, 0)))
        ax.title.set_text(r'$\varepsilon^-_{' + str(val) + '}$')
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
    plt.show()
    return fig


def plot_perturbations_augmented(D, cov, figure=None, cmap=None):
    n_dict = D.shape[3]
    if figure is not None:
        fig = plt.figure(figure)
    fig = plt.subplots(nrows=3, ncols=n_dict)
    for ind_dic in range(n_dict):
        if cmap is not None:
            plt.set_cmap(cmap)
        [d1_p, d1_n] = extract_part(D[:, :, :, ind_dic])
        ax = plt.subplot(3, n_dict, ind_dic + 1)
        plt.set_cmap("jet")
        plt.imshow(np.transpose(d1_p, (1, 2, 0)))
        ax.title.set_text(r'$\varepsilon^+_{' + str(ind_dic+1) + '}$')
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        ax = plt.subplot(3, n_dict, n_dict + ind_dic + 1)
        plt.set_cmap("jet")
        plt.imshow(np.transpose(d1_n, (1, 2, 0)))
        ax.title.set_text(r'$\varepsilon^-_{' + str(ind_dic + 1) + '}$')
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        ax = plt.subplot(3, n_dict, 2*n_dict + ind_dic + 1)
        plt.set_cmap('Greys')
        plt.imshow(cov[ind_dic])
        ax.title.set_text(r'fr')
    plt.show()
    return fig



def plot_attack(image, label, attack, model, labels):
    device = attack.device
    model = model.to(device=device)
    image_cast = image[None, :, :, :].to(device=device)
    label_cast = label.to(device=device)
    adversary = attack(image_cast, label_cast)

    adv_noise = adversary - image_cast
    adv_noise_positive, adv_noise_negative = extract_part(adv_noise.squeeze().to('cpu'))

    original_label = labels[model(image_cast).argmax(dim=1).item()]
    predicted_label = labels[model(adversary).argmax(dim=1).item()]

    fig = plt.subplots(nrows=1, ncols=4)
    ax = plt.subplot(1, 4, 1)
    plt.imshow(np.transpose(image, (1, 2, 0)))
    ax.set_title(original_label)
    ax = plt.subplot(1, 4, 4)
    plt.imshow(np.transpose(adversary.squeeze().to('cpu'), (1, 2, 0)))
    ax.set_title(predicted_label)
    ax = plt.subplot(1, 4, 2)
    plt.imshow(np.transpose(adv_noise_positive, (1, 2, 0)))
    ax.set_title('noise added')
    ax = plt.subplot(1, 4, 3)
    plt.imshow(np.transpose(adv_noise_negative, (1, 2, 0)))
    ax.set_title('noise subtracted')

    return fig


def set_size(width, fraction=1, subplots=(1, 1), ratio=None):
    """Set figure dimensions to avoid scaling in LaTeX.
     Parameters ----------
        width: float Document textwidth or columnwidth in pts fraction: float,
        optional Fraction of the width which you wish the figure to occupy Returns
         -------
    fig_dim: tuple Dimensions of figure in inches """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height     # https://disq.us/p/2940ij3
    if ratio is None:
        ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

