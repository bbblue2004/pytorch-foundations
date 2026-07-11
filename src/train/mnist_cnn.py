import torch
from torch import nn
from torchvision.utils import save_image
from tqdm import tqdm
from src.data.mnist_cnn import data_mnist_cnn
from src.models.mnist_cnn import MNISTCNN
from src.utils.config import MNIST_CNN
from src.utils.visualization import plot_loss, _figure_path, plot_mnist_cnn
from src.utils.metrics import test_multiclass_accuracy, update_conf_mat, metrics_from_conf_mat


def train_mnist_cnn():
    torch.set_num_threads(4)    # limits the cpu threads used for computations

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    dirname = "05_mnist_cnn"
    train_loader, val_loader, test_loader = data_mnist_cnn()

    ## save an example
    images, labels = next(iter(train_loader))    # first batch of the iterator train_loader
    # cpu is used here and not device (which might be gpu)
    save_image(images[:8].cpu(), _figure_path(dirname, "sample_images.png"), nrow=4)   # or directly through matplotlib


    model = MNISTCNN().to(device)
    # model = torch.compile(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=MNIST_CNN["lr"])    # actually mini-batch SGD, not 1 by 1 but batch
    train_losses = []
    val_losses = []

    for epoch in range(1, MNIST_CNN["epochs"] + 1):

        model.train()
        epoch_train_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{MNIST_CNN['epochs']}"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            train_logits = model(images)
            train_loss = loss_fn(train_logits, labels)
            epoch_train_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0
        conf_mat = torch.zeros(10, 10, dtype=torch.int64)

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{MNIST_CNN['epochs']}"):
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

        print(
            f"train loss: {avg_train_loss:.4f} | "
            f"val loss: {avg_val_loss:.4f} | "
            f"val accuracy: {val_accuracy:.4f}"
        )
        print()

    ## plots
    plot_loss(train_losses, dirname, val_losses=val_losses)

    images, labels = next(iter(train_loader))
    image = images[0:1].to(device)
    plot_mnist_cnn(model, image, dirname, device)    # plots the filters from layer conv1 and feature maps from after

    model.eval()
    with torch.no_grad():
        test_metrics = test_multiclass_accuracy(model, test_loader, device)

        print("========== metrics on test dataset ==========\n")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Average F1-score: {test_metrics['macro-f1']:.4f}")
