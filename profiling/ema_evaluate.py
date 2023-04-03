import os
import torch
import copy
import numpy as np

from models.encoder_classifier import EncoderClassifier2D

def get_model(args, epoch):
    checkpoint_path = os.path.join(args.output_dir, 'saved_models', f'model_{epoch}.pth')
    ckpt = torch.load(checkpoint_path)
    # Overwrite args
    args = ckpt['args']
    model = EncoderClassifier2D(args)
    model_dict = model.state_dict()
    assert set(model_dict.keys()) == set(ckpt['model'].keys())
    model_dict.update(ckpt['model'])
    model.load_state_dict(model_dict)
    model.to(args.device)
    return model

def ema_evalute(args, halflife=None, alpha=None):
    if (halflife is not None) and (alpha is not None):
            raise ValueError("Cant specify both alpha and halflife")
    if alpha is None:
        alpha = 1 - np.exp(np.log(0.5) / halflife)
    # EMA model
    model = get_model(args, 0)
    model_ema = copy.deepcopy(model)
    for param, ema_param in zip(model.parameters(), model_ema.parameters()):
        ema_param.data.copy_(param.data)
    
    # Update EMA model weights
    for param, ema_param in zip(model.parameters(), model_ema.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    for bn, bn_ema in zip(model.modules(), model_ema.modules()):
        if isinstance(bn, torch.nn.BatchNorm2d):
            bn_ema.running_mean.mul_(alpha).add_(1 - alpha, bn.running_mean)
            bn_ema.running_var.mul_(alpha).add_(1 - alpha, bn.running_var)
            