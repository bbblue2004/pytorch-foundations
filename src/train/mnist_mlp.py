import torch
from torch import nn
from tqdm import tqdm
from src.data.mnist_mlp import data_mnist_mlp
from src.models.mnist_mlp import MNISTMLP
from src.utils.config import MNIST_MLP
from src.utils.visualization import plot_loss
from src.utils.metrics import binary_classification_metrics, show_confusion_matrix


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

        model.train()
        epoch_train_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{MNIST_MLP['epochs']}"):
            # on training data
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

        model.eval()
        epoch_val_loss = 0

        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{MNIST_MLP['epochs']}"):
            images = images.to(device)
            labels = labels.to(device)
            val_logits = model(images)
            val_loss = loss_fn(val_logits, labels)
            epoch_val_loss += val_loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print()
        

    plot_loss(train_losses, dirname, val_losses=val_losses)

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        # here we test the model on the test dataset
        for images, labels in tqdm(test_loader, desc="Test"):
            images = images.to(device)
            labels = labels.to(device)
            test_logits = model(images)
            preds = test_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)                      # the last batch might contain less than batch_size elemnts

        print("========== Metrics ==========\n")
        accuracy = correct / total
        print(f"Accuracy = {accuracy}")

