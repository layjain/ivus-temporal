import torch.nn as nn

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
        net = resnet.resnet18(num_channels=args.clip_len) # HACK
    elif args.encoder_type=="resnet50":
        net = resnet.resnet50(num_channels=args.clip_len)
    else:
        raise NotImplementedError(f"Encoder Type {args.encoder_type} not implemented.")
    
    if hasattr(net, 'modify'):
        net.modify(remove_layers=args.remove_layers)

    if 'Conv2d' in str(net):
        net = From3D(net)

    return net