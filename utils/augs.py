import torch
import torchvision
import torchvision.transforms.functional as F

import numpy as np

# from utils.ilab_utils import cart2polar


class CustomNormalize(object):
    def __init__(self):
        pass

    def __call__(self, img):  # img: 1, W, H tensor --OK
        ONE, W, H = img.shape
        assert ONE == 1
        assert W == H

        mean, std = img.mean([1, 2]), img.std([1, 2])
        return F.normalize(img, mean, std)  # Make image 0 mean and unit std

    def __repr__(self):
        return "CustomNormalize"


class CustomRandomize(object):
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            return self.transform(img)
        return img

    def __repr__(self):
        return f"CustomRandomize({str(self.transform)}, {self.p})"


class CustomFloatify(object):
    def __init__(self, factor=255.0, check=True):
        self.factor = factor
        self.check = check

    def __call__(self, img):
        if self.check:
            if self.factor != 1:
                if torch.max(img).item() <= 1:
                    raise ValueError(
                        f"Potentially floatifying good image, set check=False to suppress this error"
                    )
        return img / self.factor

    def __repr__(self):
        return f"CustomFloatify({self.factor})"


class CustomRTConvert(object):
    """
    Preserves the size
    """

    def __init__(self):
        pass

    def __call__(self, img):  # img: 1, W, H tensor
        img = img[0].numpy()
        img = torch.from_numpy(cart2polar(img)).unsqueeze(0)
        return img

    def __repr__(self):
        return "CustomRTConvert"


class TrainTransform(object):
    def __init__(self, aug_list, img_size, deterministic_intensity=False, concat_struts=False):
        self.aug_list = aug_list
        self.img_size = img_size
        self.det_in = deterministic_intensity
        self.struts = concat_struts
        if self.struts:
            if not self.det_in:
                raise ValueError("Must set det_in=True for Struts")
            self.segmentor = torch.load("/data/vision/polina/users/layjain/ivus-temporal/struts/model_10600.pt").to('cpu')
            self.segmentor.eval()
        

    def __call__(self, vid):
        T, H, W, C = vid.shape  # To assert correct shape
        assert C == 1
        assert H == W
        to_apply, seg_apply = [], [self.get_segmentation]

        # No resizing is done if the sizes match: 
        # https://pytorch.org/vision/main/_modules/torchvision/transforms/functional.html#resize
        resize = torchvision.transforms.Resize((self.img_size, self.img_size))
        to_apply.append(resize) # segmentations already resized

        floatify = CustomFloatify()
        to_apply.append(floatify)

        for aug_string in self.aug_list:
            if aug_string == "rrc":  # randomized, to prevent edge effects (?)
                random_crop = torchvision.transforms.RandomResizedCrop(
                    self.img_size, scale=(0.8, 0.95), ratio=(0.7, 1.3), interpolation=2
                )
                to_apply.append(random_crop)
                if not self.det_in:
                    raise NotImplementedError("TODO: Deterministic RRC")
            elif aug_string == "affine":  # deterministic
                affine_params = torchvision.transforms.RandomAffine.get_params(
                    degrees=(-180, 180),
                    shears=(-20, 20),
                    scale_ranges=(0.8, 1.2),
                    translate=None,
                    img_size=[W, W],
                )
                affine = lambda img: F.affine(img, *affine_params)
                to_apply.append(affine); seg_apply.append(affine)
            elif aug_string == "flip":  # deterministic
                if torch.rand(1) < 0.5:
                    to_apply.append(F.hflip); seg_apply.append(F.hflip)
                if torch.rand(1) < 0.5:
                    to_apply.append(F.vflip); seg_apply.append(F.vflip)
            elif aug_string == "cj":  # randomized
                if not self.det_in:
                    cj = torchvision.transforms.ColorJitter(
                        brightness=0.2, contrast=0.3, saturation=0.2
                    )
                else: # deterministic version
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

                to_apply.append(cj); seg_apply.append(cj)
            elif aug_string == "blur":  # randomized
                if not self.det_in:
                    blur = torchvision.transforms.GaussianBlur(
                        kernel_size=7, sigma=(0.1, 3)
                    )
                    blur = CustomRandomize(blur, p=0.5)
                else: # deterministic version
                    sigma = torchvision.transforms.GaussianBlur.get_params(0.1, 3)
                    blur = lambda img: F.gaussian_blur(img, (7,7), [sigma, sigma])

                to_apply.append(blur); seg_apply.append(blur)
            elif aug_string == "sharp":  # randomized
                if not self.det_in:
                    sharpness = torchvision.transforms.RandomAdjustSharpness(2.)
                else:
                    if torch.rand(1) < 0.5:
                        sharpness = lambda img: F.adjust_sharpness(img, 2.)
                    else:
                        sharpness = lambda img: img
                to_apply.append(sharpness); seg_apply.append(sharpness)
            elif aug_string == "rt":  # deterministic
                rt = CustomRTConvert()
                to_apply.append(rt); seg_apply.append(rt)
            elif aug_string == "norm":  # deterministic
                normalize = CustomNormalize()
                to_apply.append(normalize) # No need to norm the segmentations
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

        if self.struts:
            seg_stacked = np.stack(
                [
                    apply_sequential(
                        v, seg_apply
                    )
                    for v in vid
                ]
            )  # T x C x W x H
            stacked = np.concatenate((stacked, seg_stacked)) # 2T x C x W x H

        # plain = torchvision.transforms.Compose(
        #     [
        #         torchvision.transforms.Resize((self.img_size, self.img_size)),
        #     ]
        # )

        return torch.from_numpy(stacked) #, plain(torch.from_numpy(np.float32(vid[0])).transpose(0, 2))
    
    def get_segmentation(self, frame):
        """
        frame: H, W, 1 (torch)
        """
        H, W, C = frame.shape
        assert C==1
        assert H==W

        if np.max(frame) <= 1:
            raise ValueError("TODO: Skip floatifying good image for struts")

        image = np.expand_dims((frame/255).squeeze(), axis=(0,1)).astype(np.float32) # 1 x 1 x H x W
        image = torch.from_numpy(image)
        image = F.resize(image, (self.img_size, self.img_size))

        logits = self.segmentor(image)
        segmentation = torch.sigmoid(logits).detach().to('cpu')
        segmentation = torch.squeeze(segmentation)
        segmentation = torch.unsqueeze(segmentation, dim=0)
        
        return segmentation # 1, H, W


class ValTransform(object):
    def __init__(self, img_size, aug_list, concat_struts=False):
        self.img_size = img_size
        self.aug_list = aug_list
        self.struts = concat_struts
        if self.struts:
            self.segmentor = torch.load("/data/vision/polina/users/layjain/ivus-temporal/struts/model_10600.pt").to('cpu')
            self.segmentor.eval()

    def get_segmentation(self, frame):
        """
        frame: H, W, 1 (torch)
        (Same function as above)
        """
        H, W, C = frame.shape
        assert C==1
        assert H==W

        if np.max(frame) <= 1:
            raise ValueError("TODO: Skip floatifying good image for struts")

        image = np.expand_dims((frame/255).squeeze(), axis=(0,1)).astype(np.float32) # 1 x 1 x H x W
        image = torch.from_numpy(image)
        image = F.resize(image, (self.img_size, self.img_size))

        logits = self.segmentor(image)
        segmentation = torch.sigmoid(logits).detach().to('cpu')
        segmentation = torch.squeeze(segmentation)
        segmentation = torch.unsqueeze(segmentation, dim=0)
        
        return segmentation # 1, H, W

    def __call__(self, vid):
        T, H, W, C = vid.shape  # To assert correct shape
        assert C == 1
        assert H == W
        to_compose, to_seg = [], [self.get_segmentation]

        resize = torchvision.transforms.Resize((self.img_size, self.img_size))
        to_compose.append(resize)

        floatify = CustomFloatify()
        to_compose.append(floatify)

        for aug_string in self.aug_list:
            if aug_string == "rt":
                rt = CustomRTConvert()
                to_compose.append(rt); to_seg.append(rt)

            elif aug_string == "norm":
                normalize = CustomNormalize()
                to_compose.append(normalize)

        composed = torchvision.transforms.Compose(to_compose)
        stacked = np.stack(
            [composed(torch.from_numpy(np.float32(v)).transpose(0, 2)) for v in vid]
        )  # T x N x H' x W'
        def apply_sequential(frame, to_apply):
            """
            frame: torch,  [...,H,W]
            """
            for f in to_apply:
                frame = f(frame)
            return frame

        if self.struts:
            seg_stacked = np.stack(
                [
                    apply_sequential(
                        v, to_seg
                    )
                    for v in vid
                ]
            )  # T x C x W x H
            stacked = np.concatenate((stacked, seg_stacked)) # 2T x C x W x H


        # plain = torchvision.transforms.Compose(
        #     [
        #         torchvision.transforms.Resize((self.img_size, self.img_size)),
        #     ]
        # )

        return torch.from_numpy(stacked) #, plain(torch.from_numpy(np.float32(vid[0])).transpose(0, 2))
