'''
Implementations of useful loss functions used in registration
'''

import torch
import numpy as np

def MSE_loss(x_tr, template):
    '''
    mean[(x_tr-template_unsqueezed)**2]
    '''
    return torch.linalg.vector_norm(torch.flatten(x_tr - template), ord=2)**2/np.prod(x_tr.shape)

def get_loss(args):
    if args.loss=="MSE":
        return MSE_loss
    else:
        raise NotImplementedError(f"Loss {args.loss} not implemented")