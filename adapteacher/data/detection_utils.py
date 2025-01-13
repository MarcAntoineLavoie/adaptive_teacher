# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torchvision.transforms as transforms
from adapteacher.data.transforms.augmentation_impl import (
    GaussianBlur,
)
from torch import rand
import torchvision.transforms.functional as F
import cv2
import numpy as np

def build_strong_augmentation(cfg, is_train):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    if is_train:
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

        if cfg.INPUT.USE_RANDOM_NOISE:
            randcrop_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomErasing(
                        p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
                    ),
                    transforms.RandomErasing(
                        p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
                    ),
                    transforms.RandomErasing(
                        p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
                    ),
                    transforms.ToPILImage(),
                ]
            )
            augmentation.append(randcrop_transform)

        logger.info("Augmentations used in training: " + str(augmentation))
    return transforms.Compose(augmentation)

def build_strong_augmentation_detect(cfg, is_train):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    if is_train:
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

        if cfg.INPUT.USE_RANDOM_NOISE:
            randcrop_transform = Compose_detect(
                [
                    transforms.ToTensor(),
                    RandomErasing_detect(
                        p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
                    ),
                    RandomErasing_detect(
                        p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
                    ),
                    RandomErasing_detect(
                        p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
                    ),
                    transforms.ToPILImage(),
                ]
            )
            augmentation.append(randcrop_transform)

        logger.info("Augmentations used in training: " + str(augmentation))
    return Compose_detect(augmentation)

class Compose_detect(transforms.Compose):
    def __call__(self, img):
        regions = []
        for t in self.transforms:
            if isinstance(t, RandomErasing_detect):
                img, region = t(img)
                regions.append(region)
            elif isinstance(t, Compose_detect):
                img, regions = t(img)
            else:
                img = t(img)
        return img, regions
    
class RandomErasing_detect(transforms.RandomErasing):
    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if rand(1) < self.p:

            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [float(self.value)]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, (list, tuple)):
                value = [float(v) for v in self.value]
            else:
                value = self.value

            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    f"{img.shape[-3]} (number of input channels)"
                )

            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
            # return F.erase(img, x, y, h, w, v, self.inplace), [x,y,h,w]
            return F.erase(img, x, y, h, w, v, self.inplace), [y,x,w,h]
        return img, []

######
#Taken from https://github.com/K2OKOH/MAD
######

def ImageDCT(img):
    h,w,c = img.shape
    img_dct = np.zeros((h,w,c))
    for i in range(c):
        single_channel = img[:, :, i].astype(float)
        # img_ = np.float32(img_)
        img_dct[:,:,i] = cv2.dct(single_channel)
    return img_dct

def ImageIDCT(img_dct):
    h,w,c = img_dct.shape
    img = np.zeros((h,w,c))
    for i in range(c):
        single_channel = img_dct[:, :, i].astype(float)
        # img_ = np.float32(img_)
        img[:,:,i] = cv2.idct(single_channel).clip(0,1)
    return img

def FrequencyBandpassFilter(img,r1=0.005,r2=0.7,pass_val=0.99,low_floor=0.5,high_floor=0.2,noise_scale=0.1):
    '''
    img: hxwxc np.array image input
    r1: float for low freq cutoff
    r2: float for high freq cutoff
    '''

    h,w,c = img.shape
    img_dct = ImageDCT(img)
    mask = np.ones_like(img_dct)*pass_val
    band_low = int(min(h,w) * r1)
    band_high = int(min(h,w) * r2)
    out_of_square = min(h,w)
    
    for x in range(h):
        for y in range(w):
            if (max(x, y) <= band_low):
                mask[x,y,:] = (1-low_floor)*(band_low - max(x,y))/band_low + low_floor
            elif (band_high <= max(x,y) <= out_of_square):
                mask[x,y,:] = (max(x,y) - band_high)/(out_of_square-band_high)*(1-high_floor) + high_floor
            elif max(x,y) > out_of_square:
                mask[x,y,:] = 1
            else:
                mask[x,y,:] = high_floor
    n_mask = 1 - mask
    inv_img_dct = img_dct * mask
    var_img_dct = img_dct * n_mask * (np.random.normal(1,scale=noise_scale,size=(1,1,c))).clip(0,2)
    new_img = ImageIDCT(inv_img_dct + var_img_dct)

    return new_img

