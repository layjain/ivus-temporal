import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from . import encoder_localizer
from . import utils
from . import stn

class RegistrationModel(nn.Module):
    def __init__(self, args):
        super(RegistrationModel, self).__init__()
        self.args = args
        self.encoder_localizer = encoder_localizer.EncoderLocalizer(args)
        self.stn = stn.get_stn(args)
        self.unet = utils.get_unet(in_channels=args.clip_len)

    def forward(self, x):
        '''
        clip --encoder_transformer--> parameters --STN--> stable_clip
        clip --unet--> tweak --add_average--> template
        '''
        B, T, ONE, W, H = x.shape # B x (C'=T) x (T'=1) x W x H
        assert ONE==1, f"input shape: {x.shape}"
        assert H==W, f"input shape: {x.shape}"
        transform_parameters = self.encoder_localizer(x) # B x (T*p)
