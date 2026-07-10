import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from src.utils.config import MNIST_CNN, SEED


def data_mnist_cnn():
    transform = transforms.ToTensor()

    # download train and test datasets
    train_dataset = datasets.MNIST(
        root=MNIST_CNN["data_dir"],
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=MNIST_CNN["data_dir"],
        train=False,
        download=True,
        transform=transform,
    )

    # train/val split
    n_val = int(len(train_dataset) * MNIST_CNN["val_ratio"])
    n_train = len(train_dataset) - n_val

    train_set, val_set = random_split(
        train_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=MNIST_CNN["batch_size"],
        shuffle=True,
        # num_workers=4,    # useless if only CPU. Parallelization for data loading
        # pin_memory=True   # accelerates transfer CPU -> GPU, but in my case it's useless
    )

    val_loader = DataLoader(
        val_set,
        batch_size=MNIST_CNN["batch_size"],
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=MNIST_CNN["batch_size"],
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
