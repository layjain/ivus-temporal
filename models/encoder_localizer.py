import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from . import utils
from . import stns

class EncoderLocalizer(nn.Module):
    '''
    EncoderClassifier2D uses the color channels (C' = T, T' = 1) 
    to obtain the encoding and subsequent classification
    (Hint: Take a look at From3D under utils)

    Use CRW pattern for consistency with past experiments
    The localizer uses different heads for predicting different transformations.
    '''
    def __init__(self, args):
        super(EncoderLocalizer, self).__init__()
        self.args=args
        self.encoder=utils.make_encoder(args)
        self.infer_dims(args)
        self.mlp_encoding_size=args.mlp_encoding_size # default 128
        self.make_heads()

    def make_heads(self):
        if 'Translation' in self.args.transforms:
            self.translation_head = self.make_head(depth=getattr(self.args, 'head_depth', 0), dim_out=2*self.args.clip_len)
        else:
            self.translation_head = lambda x: torch.zeros(x.shape[0], 2)

        if 'Rotation' in self.args.transforms:
            self.rotation_head = self.make_head(depth=getattr(self.args, 'head_depth', 0), dim_out=self.args.clip_len)
        else:
            self.rotation_head = lambda x: torch.zeros(x.shape[0], 1)

    def infer_dims(self, args):
        in_sz = args.img_size
        dummy = torch.zeros(1, args.model_in_channels, 1, in_sz, in_sz).to(next(self.encoder.parameters()).device)
        dummy_out = self.encoder(dummy) # Recall: B*N, C, T, H, W, dummy_out: B*N, 512, T, H', W'
        self.enc_hid_dim = dummy_out.shape[1]

    def make_head(self, depth=1, tanh=True, dim_out=0):
        head = []

        if depth > 0:
            dims = [self.enc_hid_dim] + [self.enc_hid_dim] * depth + [self.args.mlp_encoding_size]
            for d1, d2 in zip(dims, dims[1:]):
                h = nn.Linear(d1, d2)
                head += [h, nn.BatchNorm1d(d2), nn.ReLU()]
        else:
            raise ValueError(f"Invalid head-depth {depth}<=0")

        # Add a final layer with num_classes=2 outputs
        mlp = nn.Sequential(*head)
        mlp.add_module('transformer_FC', nn.Linear(self.args.mlp_encoding_size, dim_out))
        if tanh:
            mlp.add_module("tanh",nn.Tanh())
        return mlp

    def forward(self, x):
        '''
        clip --encoder--> encoding ---average--> feats --FC--> BxP output 
        '''
        B, T, ONE, W, H = x.shape # B x (C'=T) x (T'=1) x W x H
        assert ONE==1, f"input shape: {x.shape}"
        assert H==W, f"input shape: {x.shape}"
        maps = self.encoder(x) # B x 256 x 1 x W' x H'
        H_prime, W_prime = maps.shape[-2:]
        feats = maps.sum(-1).sum(-1) / (H_prime*W_prime) # B x 256 x 1
        feats = feats.transpose(-1, -2).reshape(-1, self.enc_hid_dim) # B x 1 x 512 --> B x 512
        translations=self.translation_head(feats) # B x (2*T)
        rotations=self.rotation_head(feats) # B x T
        parameters=stns.parameters.Parameters(translation=translations, rotation=rotations)
        return parameters
