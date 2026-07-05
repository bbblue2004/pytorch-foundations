import torch
from torch import nn

from src.data.neural_net_2D import data_neural_net_2D
from src.models.neural_net_2D import NeuralNet2D
from src.utils.config import NN_2D
from src.utils.visualization import plot_loss, NN2DPlot, plot_neural_net_2D
from src.utils.metrics import binary_classification_metrics, show_confusion_matrix


def train_neural_net_2D():
    dirname = "03_neural_net_2D"

    X_train, Y_train, X_val, Y_val = data_neural_net_2D()

    # for the figures, we use all the training + validation points
    X_all = torch.cat([X_train, X_val], dim=0)
    Y_all = torch.cat([Y_train, Y_val], dim=0)
    plot_neural_net_2D(NN2DPlot(X_all, Y_all, dirname), "data.png")

    model = NeuralNet2D(
        NN_2D["input_size"], 
        NN_2D["hidden_size"], 
        NN_2D["output_size"]
    )
    loss_fn = nn.BCEWithLogitsLoss()               # same as logistic regression (binary classification)
    optimizer = torch.optim.SGD(model.parameters(), lr=NN_2D["lr"])           # Another possibility is Adam
    train_losses = []
    val_losses = []

    
    for epoch in range(1, NN_2D["epochs"] + 1):
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{NN_2D['epochs']}")
        
        # on training data
        model.train()
        optimizer.zero_grad()
        train_logits = model(X_train)
        train_loss = loss_fn(train_logits, Y_train.float().view(-1, 1))
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

        # on validation data
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, Y_val.float().view(-1, 1))
            val_losses.append(val_loss.item())
        

    plot_neural_net_2D(NN2DPlot(X_all, Y_all, dirname, model), "fit.png")

    plot_loss(train_losses, dirname, val_losses=val_losses)

    model.eval()
    with torch.no_grad():
        # here we compare the train metrics to the validation metrics
        train_metrics = binary_classification_metrics(Y_train, model(X_train))
        val_metrics = binary_classification_metrics(Y_val, model(X_val))
        print("========== Metrics ==========\n")

        print("1. Train_metrics")
        show_confusion_matrix(train_metrics)

        print("\n2. Validation metrics")
        show_confusion_matrix(val_metrics)
        