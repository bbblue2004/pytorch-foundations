import torch
from src.train.linear_regression import train_linear_regression
from src.train.logistic_regression import train_logistic_regression
from src.utils.config import SEED



if __name__ == "__main__":
    torch.manual_seed(SEED)
    # train_linear_regression()
    train_logistic_regression()