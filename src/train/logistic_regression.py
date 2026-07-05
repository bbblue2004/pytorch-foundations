import torch
from torch import nn

from src.data.logistic_regression import data_logistic_regression
from src.models.logistic_regression import LogisticRegression
from src.utils.config import LOG_REG
from src.utils.visualization import plot_loss, LogRegPlot, plot_logistic_regression
from src.utils.metrics import binary_classification_metrics


def train_logistic_regression():
    dirname = "02_logistic_regression"

    X, Y = data_logistic_regression()
    plot_logistic_regression(LogRegPlot(X, Y, dirname), "data.png")

    model = LogisticRegression(X.shape[1], 1)
    loss_fn = nn.BCEWithLogitsLoss()               # BCEWithLogitsLoss is binary cross-entropy that expects logits as inputs instead of probabilities
    optimizer = torch.optim.SGD(model.parameters(), lr=LOG_REG["lr"])           # Another possibility is Adam
    losses = []

    model.train()
    for epoch in range(1, LOG_REG["epochs"] + 1):
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{LOG_REG['epochs']}")
        optimizer.zero_grad()
        logits = model(X)                 # logits : [400, 1]
        loss = loss_fn(logits, Y.float().view(-1, 1))    # Y : [400] -> [400, 1]
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    w = model.linear.weight.squeeze().detach()   # shape [2] : w1, w2
    b = model.linear.bias.item()
    plot_logistic_regression(LogRegPlot(X, Y, dirname, w, b), "fit.png")

    plot_loss(losses, dirname)

    model.eval()           # mode evaluation from now on
    with torch.no_grad():                # to ensure that pytorch does not compute the graph for inference
        logits = model(X)
        metrics = binary_classification_metrics(Y, logits)
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