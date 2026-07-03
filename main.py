import torch
from src.train.linear_regression import train_linear_regression




if __name__ == "__main__":
    torch.manual_seed(42)
    train_linear_regression()