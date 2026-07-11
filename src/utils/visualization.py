from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from pathlib import Path


FIGURES_DIR = Path("figures")

# helper function that ensures that the folder figures/ exists
def _figure_path(dirname: str, filename: str) -> Path:
    path = FIGURES_DIR / dirname / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path



def plot_loss(losses, dirname, filename = "loss.png", val_losses=None):
    plt.plot(range(1, len(losses) + 1), losses, label = "train loss")
    
    if val_losses is not None:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="val loss")
    
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Evolution of loss value")

    if val_losses is not None:
        plt.legend()

    plt.savefig(_figure_path(dirname, filename))
    plt.close()



### 1. Linear Regression

# creating a dataclass is a convenient solution here

@dataclass
class LinRegPlot:
    X: torch.Tensor
    Y: torch.Tensor
    fX: torch.Tensor
    dirname: str
    w_true: float     
    b_true: float
    w_pred: float | None = None      # optional parameters
    b_pred: float | None = None


def plot_linear_regression(plot: LinRegPlot, filename):
    x = plot.X.numpy().flatten()        # flatten means transforming a tensor into a 1-dimensional vector
    y = plot.Y.numpy().flatten()
    fx = plot.fX.numpy().flatten()

    fig, ax = plt.subplots(figsize=(8, 5))           

    ax.scatter(x, y, alpha=0.65, s=24, label="Random points", color="blue", zorder=3)     # or tab:blue    # zorder means in the foreground or background
    ax.plot(x, fx, "--", color="green", linewidth=2, label=f"y = {plot.w_true}x + {plot.b_true}", zorder=2)

    if plot.w_pred is not None and plot.b_pred is not None:
        ax.plot(x, plot.w_pred * x + plot.b_pred, "-", color="tab:red", linewidth=2, label=f"Fitted params w = {plot.w_pred:.2f}, b = {plot.b_pred:.2f}", zorder=2)
        ax.set_title("Linear regression")
    else:
        ax.set_title("Training data")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.savefig(_figure_path(plot.dirname, filename), dpi=120, bbox_inches="tight")
    plt.close(fig)




### 2. Logistic Regression


@dataclass
class LogRegPlot:
    X: torch.Tensor      # [n, 2]
    Y: torch.Tensor      # [n]
    dirname: str
    w: torch.Tensor | None = None
    b: float | None = None


def plot_logistic_regression(plot: LogRegPlot, filename):
    X = plot.X.numpy()
    Y = plot.Y.numpy().flatten()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], alpha=0.65, s=24,
               label="Class 0", color="tab:blue", edgecolors="k", zorder=3)
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], alpha=0.65, s=24,
               label="Class 1", color="tab:orange", edgecolors="k", zorder=3)

    if plot.w is not None and plot.b is not None:
        # frontier : w1 * x1 + w2 * x2 + b = 0  ->  x2 = -(w1 * x1 + b) / w2
        w1, w2 = plot.w[0].item(), plot.w[1].item()
        b = plot.b
        x1_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
        x2_line = -(w1 * x1_line + b) / w2
        ax.plot(x1_line, x2_line, "-", color="tab:red", linewidth=2, label="Decision boundary", zorder=2)
        ax.set_title("Logistic regression")
    else:
        ax.set_title("Training data")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.savefig(_figure_path(plot.dirname, filename), dpi=120, bbox_inches="tight")
    plt.close(fig)



### 3. Neural Network on 2D data

@dataclass
class NN2DPlot:
    X: torch.Tensor
    Y: torch.Tensor
    dirname: str
    model: nn.Module | None = None


def plot_neural_net_2D(plot: NN2DPlot, filename):
    X = plot.X.numpy()
    Y = plot.Y.numpy().flatten()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], alpha=0.65, s=24,
               label="Class 0", color="tab:blue", edgecolors="k", zorder=3)
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], alpha=0.65, s=24,
               label="Class 1", color="tab:orange", edgecolors="k", zorder=3)

    if plot.model is not None:
        plot.model.eval()
        with torch.no_grad():
            # Grid on the plane
            margin = 0.5
            x1_min, x1_max = X[:, 0].min() - margin, X[:, 0].max() + margin
            x2_min, x2_max = X[:, 1].min() - margin, X[:, 1].max() + margin

            x1 = torch.linspace(x1_min, x1_max, 200)
            x2 = torch.linspace(x2_min, x2_max, 200)
            grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing="xy")

            grid = torch.stack([grid_x1.ravel(), grid_x2.ravel()], dim=1)
            probs = torch.sigmoid(plot.model(grid)).reshape(grid_x1.shape)

        # Line where probability = 0.5
        ax.contour(
            grid_x1.numpy(), grid_x2.numpy(), probs.numpy(),
            levels=[0.5], colors="tab:red", linewidths=2, zorder=2,
        )
        ax.plot([], [], color="tab:red", linewidth=2, label="Decision boundary")
        ax.set_title("Neural network (2D)")
    else:
        ax.set_title("Training data")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.savefig(_figure_path(plot.dirname, filename), dpi=120, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_mnist_cnn(model, image, dirname, device):

    model.eval()
    image = image.to(device)

    ## plot convolution filters
    conv1 = model.net[1]
    filters = conv1.weight.detach().cpu()

    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    for i, ax in enumerate(axes.flat):     # axes.flat: array of axes -> list
        ax.imshow(filters[i, 0].numpy(), cmap="gray")
        ax.set_title(f"Filter {i}")
        ax.axis("off")
    fig.suptitle("Conv1 filters (from learned weights)")   # global title
    plt.savefig(_figure_path(dirname, "filters.png"))
    plt.close()

    ## plot feature maps on an image
    x = image
    x = model.net[0](x)
    x = model.net[1](x)
    x = model.net[2](x)

    features = x.cpu()    # because numpy/matplotlib need cpu tensors
    fig = plt.figure(figsize=(12, 8))

    ax0 = fig.add_subplot(3, 3, 1)
    ax0.imshow(image.cpu().squeeze().numpy(), cmap="gray", vmin=0, vmax=1)   # vmin and vmax to fix color scale
    ax0.set_title("Input")
    ax0.axis("off")

    for i in range(6):
        ax = fig.add_subplot(3, 3, i + 2)
        ax.imshow(features[0, i].numpy(), cmap="viridis")
        ax.set_title(f"Activation {i}")
        ax.axis("off")
    fig.suptitle("Conv1 feature maps on a sample image")
    fig.savefig(_figure_path(dirname, "feature_maps.png"), dpi=120, bbox_inches="tight")   # dpi: dots per inch (resolution) and bbox_tight to avoid white margins
    plt.close(fig)