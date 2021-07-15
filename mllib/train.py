import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .test import test_classification

from typing import Callable


def train_one_epoch_classification(model: torch.nn.Module, device: torch.device, train_loader: DataLoader, optimizer: torch.optim, loss_fn: Callable):
    train_losses = []
    train_acc = []

    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        y_pred = model(data)

        loss = loss_fn(y_pred, target)
        train_losses.append(loss)

        loss.backward()
        optimizer.step()

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

    return train_losses, train_acc


def train_and_test_classification(model: torch.nn.Module, device: torch.device, train_loader: DataLoader, test_loader: DataLoader, optimizer: torch.optim, loss_fn: Callable, epochs: int):
    train_losses = []
    test_losses = []
    train_acc = []
    test_accs = []
    for epoch in range(epochs):
        print("EPOCH:", epoch)
        train_epoch_losses, train_epoch_acc = train_one_epoch_classification(
            model, device, train_loader, optimizer, loss_fn)
        test_loss, test_acc = test_classification(
            model, device, test_loader, loss_fn)
        train_losses.extend(train_epoch_losses)
        train_acc.extend(train_epoch_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    return train_losses, test_losses, train_acc, test_accs
