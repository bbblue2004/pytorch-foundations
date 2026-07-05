import torch
from src.utils.config import LIN_REG

# data for linear regression

def data_linear_regression():
    X = torch.arange(-3, 3, 0.1).view(-1, 1)
    fX = LIN_REG["w_true"] * X + LIN_REG["b_true"]
    Y = fX + LIN_REG["noise"] * torch.randn(X.size())
    return X, fX, Y