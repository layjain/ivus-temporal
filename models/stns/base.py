import torch
from . import parameters

class BaseSTN():
    '''
    frames x params -> transformed_frames
    '''
    def __init__(self):
        pass

    def transform(self, x, p:parameters.Parameters):
        B_t, P_t = p.translation.shape
        B_r, P_r = p.rotation.shape

        B_, T, ONE, H, W = x.shape
        assert ONE == 1
        
        if B_!=B_t:
            raise ValueError(f"Batch sizes inconsistent: {B_t}!={B}")
        if B_!=B_r:
            raise ValueError(f"Batch sizes inconsistent: {B_r}!={B}")

        if (P_t != 2*T):
            raise ValueError(f"Can't transform: {P_t}!=2x{T}")

        if (P_r != T):
            raise ValueError(f"Can't transform: {P_r}!={T}")

        return self._transform(x, p)
    def _transform(self, x, p):
        raise NotImplementedError