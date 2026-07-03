import torch
from src.utils.config import SIGMA, N

# data for logistic regression

def data_logistic_regression():
    c0 = torch.tensor([-2, -2])
    c1 = torch.tensor([2, 2])
    sigma = SIGMA # standard deviation
    n = N # number of points per class
    
    points_0 = c0 + torch.normal(mean=0.0, std=sigma, size=(n, 2))     # instead of np.random.normal
    points_1 = c1 + torch.normal(mean=0.0, std=sigma, size=(n, 2))
    labels_0 = torch.zeros(n)
    labels_1 = torch.ones(n)

    X = torch.cat([points_0, points_1], dim=0)   # instead of np.concatenate([...], axis=0)
    y = torch.cat([labels_0, labels_1], dim=0)

    return X, y