from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import os

from typing import List

DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
ZIP_NAME = "tiny-imagenet-200.zip"

DATASET_MEAN = [0.48043839, 0.44820218, 0.39760034]  # train+test ds
DATASET_STD = [0.27698959, 0.26908774, 0.28216029]  # train+test ds


class TinyImageNetDataset(Dataset):
    def __init__(self, base_dir: str = ".", data_dir: str = "tiny-imagenet-200", mode: str = None, transform: transforms.Compose = None):
        super(TinyImageNetDataset, self).__init__

        dataset_dir = os.path.join(base_dir, data_dir)
        if not os.path.exists(dataset_dir):
            os.system(f"wget {DATASET_URL}")
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            os.system(f"unzip -qq {ZIP_NAME} -d {base_dir}")

        image_paths = []
        image_labels = []
        if mode == "train":
            for (root, _, files) in os.walk(f"{dataset_dir}/train"):
                images = [f for f in files if ".JPEG" in f]
                for name in images:
                    img_path = os.path.join(root, name)
                    label = root.split("/")[-2]
                    image_paths.append(img_path)
                    image_labels.append(label)
        elif mode == "test":
            with open(f"{dataset_dir}/val/val_annotations.txt") as f:
                lines = f.readlines()
                for line in lines:
                    splits = line.split("\t")
                    img_path = os.path.join(
                        f"{dataset_dir}/val/images/{splits[0]}")
                    label = splits[1]
                    image_paths.append(img_path)
                    image_labels.append(label)

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

        if self.transform is not None:
            img = self.transform(img)

        return img, lbl


train_transform_options = [
    {
        "ty": "Pad",
        "config": {
            "min_height": 70, "min_width": 70, "always_apply": True
        }
    },
    {
        "ty": "RandomCrop",
        "config": {
            "height": 64, "width": 64, "p": 1
        }
    },
    {
        "ty": "HorizontalFlip",
        "config": {
            "p": 0.5
        }
    },
    {
        "ty": "CoarseDropout",
        "config": {
            "max_holes": 1, "min_holes": 1, "max_height": 32, "max_width": 32, "p": 0.8, "fill_value": tuple([x * 255.0 for x in DATASET_MEAN]),
            "min_height": 32, "min_width": 32
        }
    },
    {
        "ty": "Normalize",
        "config": {
            "mean": DATASET_MEAN, "std": DATASET_STD, "always_apply": True
        }
    },
]

test_transform_options = [
    {
        "ty": "Normalize",
        "config": {
            "mean": DATASET_MEAN, "std": DATASET_STD, "always_apply": True
        }
    },
]


def get_train_test_transform_options():
    return train_transform_options, test_transform_options


def get_train_test_dataset(base_dir: str = ".", data_dir: str = "tiny-imagenet-200", train_transforms: transforms.Compose = None, test_transforms: transforms.Compose = None):
    train_ds = TinyImageNetDataset(
        base_dir=base_dir, data_dir=data_dir, transform=train_transforms, mode="train")
    test_ds = TinyImageNetDataset(
        base_dir=base_dir, data_dir=data_dir, transform=test_transforms, mode="test")
    return train_ds, test_ds


def get_train_test_dataloaders(base_dir: str = ".", data_dir: str = "tiny-imagenet-200", train_transforms: transforms.Compose = None, test_transforms: transforms.Compose = None, train_bs: int = 64, test_bs: int = 64, num_workers: int = 4, use_cuda: bool = False):
    train_ds, test_ds = get_train_test_dataset(
        base_dir, data_dir, train_transforms, test_transforms)

    train_dataloader_args = dict(shuffle=True, batch_size=train_bs, num_workers=num_workers,
                                 pin_memory=True) if use_cuda else dict(shuffle=True, batch_size=train_bs)
    test_dataloader_args = dict(shuffle=True, batch_size=test_bs, num_workers=num_workers,
                                pin_memory=True) if use_cuda else dict(shuffle=True, batch_size=test_bs)

    train_loader = DataLoader(train_ds, **train_dataloader_args)
    test_loader = DataLoader(test_ds, **test_dataloader_args)

    return train_loader, test_loader
