import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import numpy as np

from typing import List, Dict


def get_transforms(options: List[Dict]):
    tfms = []
    for option in options:
        ty = option["ty"]
        conf = option["config"]
        if ty == "Pad":
            tfms.append(A.PadIfNeeded(conf))
        elif ty == "RandomCrop":
            tfms.append(A.RandomCrop(conf))
        elif ty == "HorizontalFlip":
            tfms.append(A.HorizontalFlip(conf))
        elif ty == "CoarseDropout":
            tfms.append(A.CoarseDropout(conf))
        elif ty == "Normalize":
            tfms.append(A.Normalize(conf))

    tfms.append(ToTensorV2())
    tfms = A.Compose(tfms)

    return lambda img: tfms(image=np.array(img))["image"]
