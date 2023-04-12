import torch
import torchvision
import torchvision.transforms.functional as F

import numpy as np

class RegistrationTransform(object):
    def __init__(self, args):
        '''
        Adapted from utils/augs.py
        All transformations are deterministic
        '''
        self.aug_list = args.aug_list
        self.args = args
    
    def __call__(self, vid):
        T, H, W, C = vid.shape
        assert C == 1
        assert H == W
        assert T == self.args.clip_len
        
        to_apply = []
        for aug_string in self.aug_list:
            if aug_string == "rot":
                rotaffine_params = torchvision.transforms.RandomAffine.get_params(
                    degrees=(-180, 180),
                    shears=None,
                    scale_ranges=None,
                    translate=None,
                    img_size=[W, W],
                )
                rot = lambda img: F.affine(img, *rotaffine_params)
                to_apply.append(rot)
            elif aug_string == "affine":
                affine_params = torchvision.transforms.RandomAffine.get_params(
                    degrees=(-180, 180),
                    shears=(-20, 20),
                    scale_ranges=(0.8, 1.2),
                    translate=(0.1, 0.1),
                    img_size=[W, W],
                )
                affine = lambda img: F.affine(img, *affine_params)
                to_apply.append(affine)
            
            elif aug_string == "flip":
                if torch.rand(1) < 0.5:
                    to_apply.append(F.hflip)
                if torch.rand(1) < 0.5:
                    to_apply.append(F.vflip)

            elif aug_string == "cj":
                cj_params = torchvision.transforms.ColorJitter.get_params(
                        brightness=(0.8, 1.2), contrast=(0.7, 1.3), saturation=(0.8, 1.2), hue=None
                    )
                fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = cj_params
                def cj(img):
                    for fn_id in fn_idx:
                        if fn_id == 0 and brightness_factor is not None:
                            img = F.adjust_brightness(img, brightness_factor)
                        elif fn_id == 1 and contrast_factor is not None:
                            img = F.adjust_contrast(img, contrast_factor)
                        elif fn_id == 2 and saturation_factor is not None:
                            img = F.adjust_saturation(img, saturation_factor)
                        elif fn_id == 3 and hue_factor is not None:
                            img = F.adjust_hue(img, hue_factor)

                    return img
                to_apply.append(cj)

            elif aug_string == "blur":
                sigma = torchvision.transforms.GaussianBlur.get_params(0.1, 3)
                blur = lambda img: F.gaussian_blur(img, (7,7), [sigma, sigma])
                to_apply.append(blur)

            else:
                raise ValueError(f"Invalid Augmentation {aug_string}")
                
        def apply_sequential(frame, to_apply):
            """
            frame: torch,  [...,H,W]
            """
            for f in to_apply:
                frame = f(frame)
            return frame

        stacked = np.stack(
            [
                apply_sequential(
                    torch.from_numpy(np.float32(v)).transpose(0, 2), to_apply
                )
                for v in vid
            ]
        )  # T x C x W x H

        return torch.from_numpy(stacked).transpose(1, 3) # T x H x W x C