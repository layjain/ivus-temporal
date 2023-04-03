'''
Implementations of useful loss functions used in registration
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

def MSE_loss(parameters, x_tr, template):
    '''
    mean[(x_tr-template_unsqueezed)**2]
    '''
    return torch.linalg.vector_norm(torch.flatten(x_tr - template), ord=2)**2/np.prod(x_tr.shape)

def regularization_penalty(parameter, threshold, order):
    '''
    penalty = mean(relu(|parameter|-threshold)**order)
    '''
    B, P = parameter.shape
    if threshold < 0:
        raise ValueError("Invalid threshold")

    activation = nn.ReLU()
    thresholded_parameters = activation(torch.flatten(parameter) - threshold) + activation(-torch.flatten(parameter) - threshold)
    penalty = torch.pow(torch.linalg.vector_norm(thresholded_parameters, ord=order), order) / (B * P)
    return penalty

def Regularized_MSE(parameters, x_tr, template, lambda_tr=None, threshold_tr=None, lambda_rot=0, threshold_rot=0):
    '''
    MSE + 
    lambda_tr * (relu(translation-threshold_tr, 0))**2 + 
    lambda_rot * (relu(rotation-rotation_tr, 0))**2
    '''
    mse_loss = MSE_loss(parameters, x_tr, template)
    penalty_tr = regularization_penalty(parameters.translation, threshold_tr, 2)
    penalty_rot = regularization_penalty(parameters.rotation, threshold_rot, 2)

    loss = mse_loss + lambda_rot * penalty_rot + lambda_tr * penalty_tr

    return loss

def CC_loss(parameters, x_tr, template, win=9, return_var=False):
    '''
    Adapted from
    https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/losses.py
    '''
    B, T, W, H = x_tr.shape
    Ii = torch.flatten(x_tr, start_dim=0, end_dim=1).unsqueeze(1) # B*T x 1 x H x W
    expanded_template = template.expand(-1, T, -1, -1) # B x T x W x H
    Ji = torch.flatten(expanded_template, start_dim=0, end_dim=1).unsqueeze(1) # B*T x 1 x H x W

    if Ji.shape != Ii.shape:
        raise ValueError(f"Shape mismatch between template and transformed clip")

    # get dimension of volume
    # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats] <-- wrong
    # assumes Ii, Ji are sized [batch_size, 1, *vol_shape]
    ndims = len(list(Ii.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    win = [win] * ndims

    # compute filters
    sum_filt = torch.ones([1, 1, *win]).to(Ii.device)

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    # get convolution function
    conv_fn = getattr(F, 'conv%dd' % ndims)

    # compute CC squares
    I2 = Ii * Ii
    J2 = Ji * Ji
    IJ = Ii * Ji

    I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
    J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
    I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = (IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size)/win_size
    I_var = (I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size)/win_size
    J_var = (J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size)/win_size

    cc = cross * cross / (I_var * J_var + 1e-15) # 1e-5 was way to close to (1/255)**2

    if return_var:
        return -torch.mean(cc), cross[0,0], I_var[0,0], J_var[0,0], cc[0,0]

    return -torch.mean(cc)

def self_mse(images, args):
    x = images.squeeze()
    B, T, H, W = x.shape
    assert H==W
    assert B==args.batch_size
    assert T==args.clip_len

    mean_x = torch.mean(x, dim=1, keepdim=True)
    return  MSE_loss(None, x, mean_x)

def get_loss(args):
    if args.loss=="MSE":
        return MSE_loss
    elif args.loss=="RegularizedMSE":
        lambda_tr, threhsold_tr, lambda_rot, threshold_rot = args.loss_hyperparams
        return lambda **kwargs: Regularized_MSE(**kwargs, lambda_tr=lambda_tr, threshold_tr=threhsold_tr, lambda_rot=lambda_rot, threshold_rot=threshold_rot)
    elif args.loss=="CC":
        win, = args.loss_hyperparams
        return lambda **kwargs: CC_loss(**kwargs, win=round(win))
    else:
        raise NotImplementedError(f"Loss {args.loss} not implemented")