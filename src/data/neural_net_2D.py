import torch

from src.utils.config import NN_2D


def data_neural_net_2D():
    sigma = NN_2D["sigma"]
    n = NN_2D["n_per_class"]

    points_0 = torch.normal(mean=0.0, std=sigma, size=(n, 2))
    
    r = torch.empty(n).uniform_(3.2, 4.8)
    theta = torch.empty(n).uniform_(0, 2 * torch.pi)
    points_1 = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    labels_0 = torch.zeros(n)

    labels_1 = torch.ones(n)

    X = torch.cat([points_0, points_1], dim=0)
    Y = torch.cat([labels_0, labels_1], dim=0)

    # train / validation split
    nb_elements = 2 * n
    indices = torch.randperm(nb_elements)
    n_train = int(nb_elements * 0.8)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    return X_train, Y_train, X_val, Y_val