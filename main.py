import torch
from torch import nn

from src.data import lin_reg_data
from src.models.linear_regression import LinearRegression
from src.utils.config import EPOCHS, LR
from src.utils.visualization import LinRegPlot, plot_lin_reg, plot_loss


def linear_regression():
    dirname = "01_linear_regression"

    X, fX, Y = lin_reg_data()
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

    with torch.no_grad():                # to ensure that pytorch does not compute the graph for inference
        x_test = torch.tensor([[11.0]])
        print(model(x_test).item())


if __name__ == '__main__':
    torch.manual_seed(42)
    linear_regression()