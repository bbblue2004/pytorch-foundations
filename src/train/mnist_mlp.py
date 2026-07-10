import torch
from torch import nn
from tqdm import tqdm
from src.data.mnist_mlp import data_mnist_mlp
from src.models.mnist_mlp import MNISTMLP
from src.utils.config import MNIST_MLP
from src.utils.visualization import plot_loss
from src.utils.metrics import test_multiclass_accuracy, update_conf_mat, metrics_from_conf_mat


def train_mnist_mlp():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     # works for both cpu and gpu
    print(f"Using device: {device}")

    dirname = "04_mnist_mlp"
    train_loader, val_loader, test_loader = data_mnist_mlp()
    model = MNISTMLP(
        MNIST_MLP["input_size"], 
        MNIST_MLP["hidden_size"], 
        MNIST_MLP["output_size"]
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()               # it's not a binary classification anymore
    optimizer = torch.optim.SGD(model.parameters(), lr=MNIST_MLP["lr"])           # Another possibility is Adam
    train_losses = []
    val_losses = []

    for epoch in range(1, MNIST_MLP["epochs"] + 1):

        # training steps

        model.train()
        epoch_train_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{MNIST_MLP['epochs']}"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            train_logits = model(images)
            train_loss = loss_fn(train_logits, labels)    # this time, the shape must be [batch_size, output_size]
            epoch_train_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # validation metrics

        model.eval()
        epoch_val_loss = 0
        conf_mat = torch.zeros(10, 10, dtype=torch.int64)

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{MNIST_MLP['epochs']}"):
                images = images.to(device)
                labels = labels.to(device)

                val_logits = model(images)
                val_loss = loss_fn(val_logits, labels)
                epoch_val_loss += val_loss.item()
                preds = val_logits.argmax(dim=1)
                update_conf_mat(conf_mat, labels, preds)

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_metrics = metrics_from_conf_mat(conf_mat)
        val_accuracy = val_metrics["accuracy"]

        print(f"train loss: {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f} | val accuracy: {val_accuracy:.4f}")
        print()
        
    plot_loss(train_losses, dirname, val_losses=val_losses)

    model.eval()
    with torch.no_grad():
        # here we test the model on the test dataset
        test_metrics = test_multiclass_accuracy(model, test_loader, device)

        print("========== metrics on test dataset ==========\n")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Average F1-score: {test_metrics['macro-f1']:.4f}")