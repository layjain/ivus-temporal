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
        self.unet = utils.get_unet(in_channels=args.clip_len, args=args)
        if args.zero_init:
            self._zero_init()

    def _zero_init(self):
        # encoder-localizer
        if self.encoder_localizer.translation_head is not None:
            nn.init.constant_(self.encoder_localizer.translation_head.transformer_FC.weight, 0)
            nn.init.constant_(self.encoder_localizer.translation_head.transformer_FC.bias, 0)
        if self.encoder_localizer.rotation_head is not None:
            nn.init.constant_(self.encoder_localizer.rotation_head.transformer_FC.weight, 0)
            nn.init.constant_(self.encoder_localizer.rotation_head.transformer_FC.bias, 0)
        # unet
        if self.unet is not None:
            raise NotImplementedError("TODO: Zero-Init UNET")

    def _get_template_base(self, x, x_transformed):
        '''
        base to add on to the template
        '''
        if self.args.new_template:
            base = torch.mean(x_transformed, dim=1, keepdim=False) # B x 1 x W x H
        else:
            base = torch.mean(x, dim=1, keepdim=False) # B x 1 x W x H
        return base

    def _unet_pred(self, x):
        if self.args.no_unet:
            return torch.zeros_like(x)[:,:1,:,:]
        else:
            return self.unet(x)

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
        template = self._unet_pred(x.squeeze()) + self._get_template_base(x=x, x_transformed=x_transformed)
        # t3 = time.time() - t2 - t1 - t0

        # print(f"Model: {t1} {t2} {t3}")

        return transform_parameters, x_transformed.squeeze(2), template # B x T x W x H, B x 1 x W x H