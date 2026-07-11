import torch
import argparse
from src.train.linear_regression import train_linear_regression
from src.train.logistic_regression import train_logistic_regression
from src.train.neural_net_2D import train_neural_net_2D
from src.train.mnist_mlp import train_mnist_mlp
from src.train.mnist_cnn import train_mnist_cnn
from src.utils.config import SEED


EXPERIMENTS = {
    "linreg": train_linear_regression,
    "logreg": train_logistic_regression,
    "nn2d": train_neural_net_2D,
    "mnist_mlp": train_mnist_mlp,
    "mnist_cnn": train_mnist_cnn,
}

EXPERIMENT_LABELS = {
    "linreg": "Linear regression",
    "logreg": "Logistic regression",
    "nn2d": "Neural network on 2D data",
    "mnist_mlp": "MLP on MNIST",
    "mnist_cnn": "CNN on MNIST",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a pytorch-foundations experiment"
    )
    parser.add_argument(
        "--exp",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default=None,
        help="Which experiment to run (--exp all to run every experiment)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.exp is None:
        print("No experiment selected. Use --exp <name> or --exp all")
        print("Examples:")
        print("  python main.py --exp mnist_mlp")
        print("  python main.py --exp all")

    else:
        torch.manual_seed(SEED)

        if args.exp == "all":
            total = len(EXPERIMENTS)
            for i, (name, train) in enumerate(EXPERIMENTS.items(), start=1):
                label = EXPERIMENT_LABELS[name]
                print(f"\n========== Starting experiment {i}/{total}: {label} ==========\n")
                train()
        else:
            print(f"\n========== Starting: {EXPERIMENT_LABELS[args.exp]} ==========\n")
            EXPERIMENTS[args.exp]()
