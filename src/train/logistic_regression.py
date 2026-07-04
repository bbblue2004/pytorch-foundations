import torch
from torch import nn

from src.data.logistic_regression import data_logistic_regression
from src.models.logistic_regression import LogisticRegression
from src.utils.config import EPOCHS, LR
from src.utils.visualization import plot_loss
from src.utils.metrics import logistic_regression_metrics


def train_logistic_regression():
    dirname = "02_logistic_regression"

    X, y = data_logistic_regression()
    # plot_lin_reg(LinRegPlot(X, y, fX, dirname), "data.png")

    model = LogisticRegression(X.shape[1], 1)
    loss_fn = nn.BCEWithLogitsLoss()               # BCEWithLogitsLoss is binary cross-entropy that expects logits as inputs instead of probabilities
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)           # Another possibility is Adam
    losses = []

    for epoch in range(1, EPOCHS + 1):
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS}")
        optimizer.zero_grad()
        logits = model(X)                 # logits : [400, 1]
        loss = loss_fn(logits, y.float().view(-1, 1))    # y : [400] -> [400, 1]
        losses.append(loss.item())
        loss.backward()
        optimizer.step()


    # plot_lin_reg(LinRegPlot(X, y, fX, dirname, w, b), "line.png")

    plot_loss(losses, dirname)

    model.eval()           # mode evaluation from now on
    with torch.no_grad():                # to ensure that pytorch does not compute the graph for inference
        logits = model(X)
        metrics = logistic_regression_metrics(y, logits)
        print("========== Metrics ==========")
        print("Confusion matrix")
        print(f"              Pred 0    Pred 1")
        print(f"Actual 0      {metrics['tn']:6d}    {metrics['fp']:6d}")
        print(f"Actual 1      {metrics['fn']:6d}    {metrics['tp']:6d}")
        print()
        print(
            f"Accuracy = {metrics['accuracy']:.4f}, "
            f"Precision = {metrics['precision']:.4f}, "
            f"Recall = {metrics['recall']:.4f}, "
            f"F1-score = {metrics['f1']:.4f}"
        )