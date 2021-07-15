import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from typing import Tuple, List


def plot_misclassified(data: List, title: str, r: int = 5, c: int = 4):
    fig, axs = plt.subplots(r, c, figsize=(15, 10))
    fig.tight_layout()

    for i in range(r):
        for j in range(c):
            axs[i][j].axis('off')
            axs[i][j].set_title(
                f"Target: {str(data[(i*c)+j]['target'])}\nPred: {str(data[(i*c)+j]['pred'])}")
            axs[i][j].imshow(data[(i*c)+j]['data'])


def inverse_normalize(tensor: torch.Tensor, mean: Tuple = (0.1307,), std: Tuple = (0.3081,)):
    # Not mul by 255 here
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def get_misclassified(model: torch.nn.Module, test_loader: DataLoader, title: str, device: torch.device, mean: Tuple, std: Tuple, n: int = 20, r: int = 5, c: int = 4):
    model.eval()
    wrong = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).item()
            if not correct:
                wrong.append({
                    "data": inverse_normalize(data, mean, std).squeeze().cpu(),
                    "target": target.item(),
                    "pred": pred.item()
                })

    plot_misclassified(wrong[:n], title, r, c)


def plot_single_run(title: str, train_losses: List[float], train_acc: List[float], test_losses: List[float], test_acc: List[float]):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title)
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc[4000:])
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


def plot_multi_runs(ls_title: List[str], ls_train_losses: List[List[float]], ls_train_acc: List[List[float]], ls_test_losses: List[List[float]], ls_test_acc: List[List[float]]):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(" | ".join(ls_title))

    for train_losses in ls_train_losses:
        axs[0, 0].plot(train_losses)
    axs[0, 0].legend(ls_title)
    axs[0, 0].set_title("Training Loss")

    for train_acc in ls_train_acc:
        axs[1, 0].plot(train_acc[4000:])
    axs[1, 0].legend(ls_title)
    axs[1, 0].set_title("Training Accuracy")

    for test_losses in ls_test_losses:
        axs[0, 1].plot(test_losses)
    axs[0, 1].legend(ls_title)
    axs[0, 1].set_title("Test Loss")

    for test_acc in ls_test_acc:
        axs[1, 1].plot(test_acc)
    axs[1, 1].legend(ls_title)
    axs[1, 1].set_title("Test Accuracy")
