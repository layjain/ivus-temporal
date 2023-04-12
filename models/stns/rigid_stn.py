import torch
import torch.nn.functional as F
import numpy as np
# import time

from . import base
from . import parameters

class RigidSTN(base.BaseSTN):
    """
    P = T * 3 (v_x, v_y, omega)
    x: B x T x 1 x H x W
    """

    def _transform(self, x, p:parameters.Parameters):
        # t0 = time.time()
        B, T, _, H, W = x.shape
        translations, omegas = p.translation, p.rotation # Bx(2*T), BxT
        translations = translations.view(-1, 2) # (B*T) x 2
        omegas = omegas.view(-1,) # (B*T)
        omegas = np.pi * omegas # [-1, 1]  -> [-pi, pi]
        # https://discuss.pytorch.org/t/differentiable-and-learnable-rotations-with-grid-sample/148796
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
        _T = torch.stack(
            [
                torch.stack(
                    [
                        torch.stack(
                            [
                                torch.ones(1).to(p.device),
                                torch.zeros(1).to(p.device),
                                v_x.unsqueeze(dim=0)
                            ]
                        ),
                        torch.stack(
                            [
                                torch.zeros(1).to(p.device),
                                torch.ones(1).to(p.device),
                                v_y.unsqueeze(dim=0)
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
                for (v_x, v_y) in translations
            ]
        ).squeeze() # B*T x 3 x 3
        A = torch.matmul(_R, _T) # B*T x 3 x 3
        if len(A.shape) == 2: # if B = T = 1
            assert A.shape == (3,3)
            A = A.unsqueeze(0)
        A = A[:,:2,:]
        grid = F.affine_grid(A, torch.Size((B*T, 1, H, W)), align_corners=False, padding_mode="border")
        x = x.view(B*T, 1, H, W)
        x = F.grid_sample(x, grid, align_corners=False) # B*T x 1 x H x W
        # t2 = time.time() - t1 - t0
        x = x.view(B, T, 1, H, W)
        # t3 = time.time() - t2 - t1 - t0
        # print(f"STN: {t1} {t2} {t3}")
        return x


