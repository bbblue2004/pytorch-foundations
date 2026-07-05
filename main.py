import torch
import argparse
from src.train.linear_regression import train_linear_regression
from src.train.logistic_regression import train_logistic_regression
from src.train.neural_net_2D import train_neural_net_2D
from src.utils.config import SEED


EXPERIMENTS = {
    "linreg": train_linear_regression,
    "logreg": train_logistic_regression,
    "nn2d": train_neural_net_2D,
    # "mnist_mlp": train_mnist_mlp,
    # "mnist_cnn": train_mnist_cnn,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a pytorch-foundations experiment"
    )
    parser.add_argument(
        "--exp",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default="all",
        help="Which experiment to run (default: all)",
    )
    return parser.parse_args()



if __name__ == "__main__":
    torch.manual_seed(SEED)

    args = parse_args()

    if args.exp == "all":
        for train in EXPERIMENTS.values():
            train()
    else:
        EXPERIMENTS[args.exp]()