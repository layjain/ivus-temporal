import torch.nn as nn
import torch

from .encoders import resnet


class From3D(nn.Module):
    ''' Use a 2D convnet as a 3D convnet '''
    def __init__(self, resnet):
        super(From3D, self).__init__()
        self.model = resnet
    
    def forward(self, x):
        N, C, T, h, w = x.shape
        xx = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, h, w) # N*T x C x h x w
        m = self.model(xx)

        return m.view(N, T, *m.shape[-3:]).permute(0, 2, 1, 3, 4)

def make_encoder(args):
    if args.encoder_type=="resnet18":
        net = resnet.resnet18(num_channels=args.model_in_channels) # HACK
    elif args.encoder_type=="resnet50":
        net = resnet.resnet50(num_channels=args.model_in_channels)
    else:
        raise NotImplementedError(f"Encoder Type {args.encoder_type} not implemented.")
    
    if hasattr(net, 'modify'):
        net.modify(remove_layers=args.remove_layers)

    if 'Conv2d' in str(net):
        net = From3D(net)

    return net

################
# Registration #
################s

def get_unet(in_channels, args):
    if args.no_unet:
        # B x T x W x H --> B x 1 x W x H of zeros
        model = lambda x: torch.zeros_like(x)[:,:1,:,:]
    else:
        import monai
        model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=1,
        dropout = 0.1)

    return model