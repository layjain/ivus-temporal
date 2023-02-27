import torch
import torch.nn.functional as F

from . import base

class RigidSTN(base.BaseSTN):
    """
    P = T * 3 (v_x, v_y, omega)
    x: B x T x 1 x H x W
    """

    def _transform(self, x, p):
        B, T, _, H, W = x.shape
        _, P = p.shape
        assert P == 3 * T
        p = p.view(B, T, 3)  # B x T x 3
        p = p.view(B*T, 3) # (B*T)x 3
        translations, omegas = p[...,:2], p[..., 2]
        # https://discuss.pytorch.org/t/differentiable-and-learnable-rotations-with-grid-sample/148796
        _R = torch.stack(
            [
                torch.stack(
                    [
                        torch.stack(
                            [
                                torch.cos(t).unsqueeze(dim=0),
                                -torch.sin(t).unsqueeze(dim=0),
                                torch.zeros(1),
                            ]
                        ),
                        torch.stack(
                            [
                                torch.sin(t).unsqueeze(dim=0),
                                torch.cos(t).unsqueeze(dim=0),
                                torch.zeros(1),
                            ]
                        ),
                        torch.stack(
                            [
                                torch.zeros(1),
                                torch.zeros(1),
                                torch.ones(1),
                            ]
                        ),
                    ]
                )
                for t in omegas # B*T x 2 x 3
            ]
        ).squeeze()
        _T = torch.stack(
            [
                torch.stack(
                    [
                        torch.stack(
                            [
                                torch.ones(1),
                                torch.zeros(1),
                                v_x.unsqueeze(dim=0)
                            ]
                        ),
                        torch.stack(
                            [
                                torch.zeros(1),
                                torch.ones(1),
                                v_y.unsqueeze(dim=0)
                            ]
                        ),
                        torch.stack(
                            [
                                torch.zeros(1),
                                torch.zeros(1),
                                torch.ones(1),
                            ]
                        ),
                    ]
                )
                for (v_x, v_y) in translations
            ]
        ).squeeze() # B*T x 3 x 3
        A = torch.matmul(_R, _T) # B*T x 3 x 3
        A = A[:,:2,:]
        grid = F.affine_grid(A, torch.Size((B*T, 1, H, W)))
        x = x.view(B*T, 1, H, W)
        x = F.grid_sample(x, grid) # B*T x 1 x H x W
        x = x.view(B, T, 1, H, W)
        return x


