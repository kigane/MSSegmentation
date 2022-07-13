import torch
import torch.nn as nn

def get_activation(act):
    if act == 'elu':
        activation = nn.ELU()
    elif act == 'relu':
        activation = nn.ReLU()
    elif act == 'lrelu':
        activation = nn.LeakyReLU(0.1)
    else:
        raise NotImplementedError(f'{act} is not supported yet')
    return activation