from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path

from src.utils.config import LIN_REG


FIGURES_DIR = Path("figures")

# helper function that ensures that the folder figures/ exists
def _figure_path(dirname: str, filename: str) -> Path:
    path = FIGURES_DIR / dirname / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path



def plot_loss(losses, dirname, filename = "loss.png"):
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Evolution of loss value")
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
    w: float | None = None     # optional parameters
    b: float | None = None


def plot_linear_regression(plot: LinRegPlot, filename: str) -> None:
    x = plot.X.numpy().flatten()        # flatten means transforming a tensor into a 1-dimensional vector
    y = plot.Y.numpy().flatten()
    fx = plot.fX.numpy().flatten()

    fig, ax = plt.subplots(figsize=(8, 5))           

    ax.scatter(x, y, alpha=0.65, s=24, label="Random points", color="blue", zorder=3)     # or tab:blue    # zorder means in the foreground or background
    ax.plot(x, fx, "--", color="green", linewidth=2, label=f"y = {LIN_REG['w_true']}x + {LIN_REG['b_true']}", zorder=2)

    if plot.w is not None and plot.b is not None:
        ax.plot(x, plot.w * x + plot.b, "-", color="tab:red", linewidth=2, label=f"Fitted params w = {plot.w:.2f}, b = {plot.b:.2f}", zorder=2)
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


def plot_logistic_regression(plot: LogRegPlot, filename: str) -> None:
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