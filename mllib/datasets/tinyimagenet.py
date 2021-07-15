from torch.utils.data import Dataset, DataLoader
from PIL import Image

import os

from typing import List

DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
ZIP_NAME = "tiny-imagenet-200.zip"

DATASET_MEAN = [0.48043839, 0.44820218, 0.39760034]  # train+test ds
DATASET_STD = [0.27698959, 0.26908774, 0.28216029]  # train+test ds


class TinyImageNetDataset(Dataset):
    def __init__(self, image_paths: List[str], image_labels: List[str], transform=None):
        super(TinyImageNetDataset, self).__init__

        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transform = transform

        labels_ls = list(set(image_labels))
        labels_ls.sort()
        self.label_map = {label: i for i, label in enumerate(labels_ls)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        lbl = self.label_map[self.image_labels[idx]]

        if self.transform:
            img = self.transform(img)

        return img, lbl


def get_train_test_images_and_labels(base_dir: str = ".", split: int = 0.7):
    data_dir = os.path.join(base_dir, "tiny-imagenet-200")
    if not os.path.exists(data_dir):
        os.system(f"wget {DATASET_URL}")
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        os.system(f"unzip -qq {ZIP_NAME} -d {base_dir}")

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for (root, _, files) in os.walk(f"{data_dir}/train"):
        images = [f for f in files if ".JPEG" in f]
        for name in images:
            img_path = os.path.join(root, name)
            label = root.split("/")[-2]
            train_images.append(img_path)
            train_labels.append(label)

    with open(f"{data_dir}/val/val_annotations.txt") as f:
        lines = f.readlines()
        for line in lines:
            splits = line.split("\t")
            img_path = os.path.join(f"{data_dir}/val/images/{splits[0]}")
            label = splits[1]
            test_images.append(img_path)
            test_labels.append(label)

    return train_images, train_labels, test_images, test_labels
