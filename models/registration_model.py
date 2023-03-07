import torch
import torch.nn as nn
import torch.nn.functional as F
# import time

from . import encoder_localizer
from . import utils
from . import stns

class RegistrationModel(nn.Module):
    def __init__(self, args):
        super(RegistrationModel, self).__init__()
        self.args = args
        self.encoder_localizer = encoder_localizer.EncoderLocalizer(args)
        self.stn = stns.get_stn(args)
        self.unet = utils.get_unet(in_channels=args.clip_len)

    def forward(self, x):
        '''
        clip --encoder_transformer--> parameters --STN--> stable_clip
        clip --unet--> tweak --add_average--> template
        '''
        B, T, ONE, W, H = x.shape # B x (C'=T) x (T'=1) x W x H
        assert ONE==1, f"input shape: {x.shape}"
        assert H==W, f"input shape: {x.shape}"
        # t0 = time.time()
        transform_parameters = self.encoder_localizer(x) # B x (T*p)
        # t1 = time.time() - t0
        x_transformed = self.stn.transform(x, transform_parameters) # B x T x 1 x W x H
        # t2 = time.time() - t1 - t0
        template = self.unet(x.squeeze()) + torch.mean(x, dim=1, keepdim=False) # B x 1 x W x H
        # t3 = time.time() - t2 - t1 - t0

        # print(f"Model: {t1} {t2} {t3}")

        return x_transformed.squeeze(2), template # B x T x W x H, B x 1 x W x H