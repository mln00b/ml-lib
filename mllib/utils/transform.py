import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import cv2

from typing import List, Dict

def get_transforms(options: Dict):
    tfms = []

    if "Normalize" in options:
        config = options["Normalize"]
        tfms.append(
            A.Normalize(
                mean=config["mean"],
                std=config["std"],
            )
        )

    if "RandomPadAndCrop" in options:
        config = options["RandomPadAndCrop"]
        prob = config.get("prob") or 1.0
        crop_size = config["cropSize"]
        pad_size = config["padSize"]
        tfms.append(
            A.Sequential([
                A.PadIfNeeded(
                    min_height=pad_size["height"],
                    min_width=pad_size["width"],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(0,0,0)
                ),
                A.RandomCrop(
                    height=crop_size["height"],
                    width=crop_size["width"]
                )
            ], p=prob)
        )

    if "Rotate" in options:
        config = options["Rotate"]
        limit = config["limit"]
        prob = config["prob"]
        tfms.append(
            A.Rotate(
                limit=limit,
                p=prob
            )
        )
    
    if "HorizontalFlip" in options:
        config = options["HorizontalFlip"]
        tfms.append(
            A.HorizontalFlip(
                prob=config["prob"],
            )
        )
    

    tfms.append(ToTensorV2())

    return A.Compose(tfms)
