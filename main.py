import torch
from torch import nn

from src.data import lin_reg_data
from src.models.linear_regression import LinearRegression
from src.utils.config import EPOCHS, LR
from src.utils.visualization import *


def linear_regression():
    dirname = "01_linear_regression"

    X, fX, Y = lin_reg_data()
    plot_data(X, fX, Y, dirname)

    lr = LinearRegression(X.shape[1], Y.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(lr.parameters(), lr=LR)
    losses = []

    for epoch in range(1, EPOCHS + 1):
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS}")
        optimizer.zero_grad()
        Yhat = lr.forward(X)
        loss = loss_fn(Yhat, Y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    w, b = lr.linear.weight.item(), lr.linear.bias.item()
    plot_line(X, Y, w, b, dirname)

    plot_loss(losses, dirname)

    x_test = torch.tensor([[11.0]])
    print(lr.forward(x_test).item())


if __name__ == '__main__':
    linear_regression()