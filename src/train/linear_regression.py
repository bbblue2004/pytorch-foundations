import torch
from torch import nn

from src.data.linear_regression import data_linear_regression
from src.models.linear_regression import LinearRegression
from src.utils.config import EPOCHS, LR
from src.utils.visualization import LinRegPlot, plot_lin_reg, plot_loss
from src.utils.metrics import regression_metrics


def train_linear_regression():
    dirname = "01_linear_regression"

    X, fX, Y = data_linear_regression()
    plot_lin_reg(LinRegPlot(X, Y, fX, dirname), "data.png")

    model = LinearRegression(X.shape[1], Y.shape[1])
    loss_fn = nn.MSELoss()                  # MSE for continuous variables
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)           # Another possibility is Adam
    losses = []

    for epoch in range(1, EPOCHS + 1):      # no need for dataloader and batches, as the dataset is small
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS}")
        optimizer.zero_grad()
        Yhat = model(X)
        loss = loss_fn(Yhat, Y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    w, b = model.linear.weight.item(), model.linear.bias.item()
    print(f"After training: w={w:.3f}, b={b:.3f} | actual: w=3.0, b=2.0")

    plot_lin_reg(LinRegPlot(X, Y, fX, dirname, w, b), "line.png")

    plot_loss(losses, dirname)

    model.eval()           # mode evaluation from now on
    with torch.no_grad():                # to ensure that pytorch does not compute the graph for inference
        Yhat = model(X)
        metrics = regression_metrics(Y, Yhat)
        print(f"Metrics: MSE = {metrics['mse']}, RMSE = {metrics['rmse']}, MAE = {metrics['mae']}, R2 = {metrics['r2']}")