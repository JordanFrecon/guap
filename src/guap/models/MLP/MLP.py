import torch
import torch.nn as nn


def affine_transformation(var_in, cmin=-1, cmax=1, eps=1e-5):
    return (cmin+eps) + (cmax - cmin - 2 * eps) * var_in.flatten()


class ScaledTanh(torch.nn.Module):
    def __init__(self):
        super(ScaledTanh, self).__init__()
        self.A = 1.7159
        self.B = 2 / 3

    def forward(self, x):
        return self.A * torch.tanh(self.B * x)


class CiresanMLP(nn.Module):
    """
    Reproduction of the simple MLPs from (Ciresan et al., 2010)
    The 'network_id' refers the network identifier of Table 1 in [https://arxiv.org/pdf/1003.0358.pdf]
    """
    def __init__(self, network_id=1):
        super().__init__()

        if network_id == 1:
            # Input layer
            self.input = torch.nn.Sequential(
                torch.nn.Linear(784, 1000),
                ScaledTanh(),
            )

            # Hidden layer
            self.hidden = torch.nn.Sequential(
                torch.nn.Linear(1000, 500),
                ScaledTanh(),
            )

            # Output layer
            self.output = nn.Linear(500, 10)

        elif network_id == 5:

            # Input layer
            self.input = torch.nn.Sequential(
                torch.nn.Linear(784, 1000),
                ScaledTanh(),
            )

            # Hidden layers
            self.hidden = nn.ModuleList()
            for _ in range(8):
                self.hidden.append(nn.Sequential(
                    torch.nn.Linear(1000, 1000),
                    ScaledTanh(),
                ))

            # Output layer
            self.output = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.input(x)
        for i in range(len(self.hidden)):
            x = self.hidden[i](x)
        x = self.output(x)
        return x
