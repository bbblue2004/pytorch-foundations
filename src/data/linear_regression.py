import torch

# data for linear regression

def data_linear_regression():
    X = torch.arange(-3, 3, 0.1).view(-1, 1)
    fX = 3 * X + 2
    Y = fX + 0.6 * torch.randn(X.size())
    return X, fX, Y