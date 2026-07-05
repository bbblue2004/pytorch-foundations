import torch
from torch import nn

from src.data.linear_regression import data_linear_regression
from src.models.linear_regression import LinearRegression
from src.utils.config import LIN_REG
from src.utils.visualization import LinRegPlot, plot_linear_regression, plot_loss
from src.utils.metrics import linear_regression_metrics


def train_linear_regression():
    dirname = "01_linear_regression"

    X, fX, Y = data_linear_regression()
    plot_linear_regression(LinRegPlot(X, Y, fX, dirname, LIN_REG["w_true"], LIN_REG["b_true"]), "data.png")

    model = LinearRegression(X.shape[1], Y.shape[1])
    loss_fn = nn.MSELoss()                  # MSE for continuous variables
    optimizer = torch.optim.SGD(model.parameters(), lr=LIN_REG["lr"])           # Another possibility is Adam
    losses = []

    model.train()
    for epoch in range(1, LIN_REG["epochs"] + 1):      # no need for dataloader and batches, as the dataset is small
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{LIN_REG['epochs']}")
        optimizer.zero_grad()
        Yhat = model(X)
        loss = loss_fn(Yhat, Y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    w = model.linear.weight.item()
    b = model.linear.bias.item()
    print(f"After training: w={w:.3f}, b={b:.3f} | actual: w={LIN_REG['w_true']}, b={LIN_REG['b_true']}")

    plot_linear_regression(LinRegPlot(X, Y, fX, dirname, LIN_REG['w_true'], LIN_REG['b_true'], w, b), "fit.png")

    plot_loss(losses, dirname)

    model.eval()               # mode evaluation from now on
    with torch.no_grad():                # to ensure that pytorch does not compute the graph for inference
        Yhat = model(X)
        metrics = linear_regression_metrics(Y, Yhat)
        print(f"Metrics: MSE = {metrics['mse']:.4f}, RMSE = {metrics['rmse']:.4f}, MAE = {metrics['mae']:.4f}, R2 = {metrics['r2']:.4f}")