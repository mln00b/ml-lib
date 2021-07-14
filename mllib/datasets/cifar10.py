from torchvision import datasets

def get_train_test_dataset(train_transforms, test_transforms):
    train_dataset = datasets.CIFAR10(train=True, transform=train_transforms)
    test_dataset = datasets.CIFAR10(train=False, transform=test_transforms)
    
    return train_dataset, test_dataset