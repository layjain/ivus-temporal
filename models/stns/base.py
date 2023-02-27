class BaseSTN():
    '''
    frames x params -> transformed_frames
    '''
    def __init__(self):
        pass
    def transform(self, x, p):
        B, P = p.shape
        B_, T, ONE, H, W = x.shape
        assert ONE == 1
        if B_!=B:
            raise ValueError(f"Batch sizes inconsistent: {B_}!={B}")
        if (P%T != 0):
            raise ValueError(f"Can't transform: {P}%{T}!=0")
        return self._transform(x, p)
    def _transform(self, x, p):
        raise NotImplementedError