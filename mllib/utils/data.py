from PIL import Image
import numpy as np

from typing import List

def get_mean_std(images: np.array = None, image_paths: List[str] = None, do_norm: bool = True):
    assert images or image_paths, "Both images and image_paths can't be None"
    if not images and image_paths:
        images = np.array([np.array(Image.open(f).convert("RGB")) for f in image_paths])

    mean = np.mean(images, axis=(0,1,2), keepdims=True).squeeze()
    std = np.std(images, axis=(0,1,2), keepdims=True).squeeze()

    if do_norm:
        mean /= 255.0
        std /= 255.0

    return mean, std