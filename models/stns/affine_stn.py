from . import base

class AffineSTN(base.BaseSTN):
    '''
    P = T * 6
    x: B x T x 1 x H x W
    '''
    def _transform(self, x, p):
        raise NotImplementedError