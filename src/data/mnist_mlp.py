import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from src.utils.config import MNIST_MLP, SEED


def data_mnist_mlp():
    transform = transforms.ToTensor()

    # download train and test datasets
    train_dataset = datasets.MNIST(
        root=MNIST_MLP["data_dir"],
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=MNIST_MLP["data_dir"],
        train=False,
        download=True,
        transform=transform,
    )

    # train/val split
    n_val = int(len(train_dataset) * MNIST_MLP["val_ratio"])
    n_train = len(train_dataset) - n_val

    train_set, val_set = random_split(
        train_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=MNIST_MLP["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=MNIST_MLP["batch_size"],
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=MNIST_MLP["batch_size"],
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
