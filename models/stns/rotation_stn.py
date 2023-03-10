import torch
import torch.nn.functional as F
import numpy as np

from . import rigid_stn

class RotationSTN(rigid_stn.RigidSTN):
    """
    P = T * 1 (omega)
    x: B x T x 1 x H x W
    """

    def _transform(self, x, p):
        # using the same code structure as RigidSTN for consistency
        B, T, _, H, W = x.shape
        _, P = p.shape
        assert P == T
        p = p.view(B*T, 1) # (B*T) x 3
        omegas = p[...,0] * np.pi # [-1, 1] --> [-pi, pi]
        _R = torch.stack(
            [
                torch.stack(
                    [
                        torch.stack(
                            [
                                torch.cos(t).unsqueeze(dim=0),
                                -torch.sin(t).unsqueeze(dim=0),
                                torch.zeros(1).to(p.device),
                            ]
                        ),
                        torch.stack(
                            [
                                torch.sin(t).unsqueeze(dim=0),
                                torch.cos(t).unsqueeze(dim=0),
                                torch.zeros(1).to(p.device),
                            ]
                        ),
                        torch.stack(
                            [
                                torch.zeros(1).to(p.device),
                                torch.zeros(1).to(p.device),
                                torch.ones(1).to(p.device),
                            ]
                        ),
                    ]
                )
                for t in omegas # B*T x 2 x 3
            ]
        ).squeeze()
        A = _R[:,:2,:]
        grid = F.affine_grid(A, torch.Size((B*T, 1, H, W)), align_corners=False)
        x = x.view(B*T, 1, H, W)
        x = F.grid_sample(x, grid, align_corners=False) # B*T x 1 x H x W
        x = x.view(B, T, 1, H, W)
        return x