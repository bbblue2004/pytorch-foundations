# pytorch-foundations

This project contains simple ML projects. The goal is to learn pytorch.
It contains:
- a linear regression
- a logistic regression
- a neural network on 2D data
- a MLP on MNIST
- a CNN on MNIST


## Project structure

```
pytorch-foundations/
  main.py                       # runs all experiments
  requirements.txt
  README.md
  src/
    data/                       # dataset generators    
    models/                     # nn.Module architectures
    train/                      # training pipelines
    utils/                      # config, metrics, plots
  figures/
    01_linear_regression/       # experiment 1 outputs
    02_logistic_regression      # experiment 2 outputs
```



## Quick start

**Requirements:** Python 3.10+

```bash
git clone https://github.com/bbblue2004/pytorch-foundations.git
cd pytorch-foundations
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python main.py
```

`main.py` can run any of these 5 experiments (the __main__ has to be configured):

1. Linear regression
2. Logistic regression
3. Neural network on 2D data
4. MLP on MNIST
5. CNN on MNIST

Metrics are printed in the terminal; plots are saved in each experiment subfolder under figures/.



## 1. Linear regression

Implementation of a simple linear regression in pytorch.

### Data

X = [-3, 3] with a step of 0.1  (60 points)
f(X) = 3 * X + 2  (true parameters (3, 2))
Y = f(X) + 0.6 * N(0, 1)  (random gaussian noise)

### model

Use of nn.Module, so small implementation

### metrics

MSE (mean square error)
RMSE (root mean square error)
MAE (mean absolute error)
R2 = ESS / TSS = 1 - RSS/TSS. We remind that TSS = ESS + RSS. A good R2 is (very) close to 1

### figures

The main directory is 01_linear_regression.
data.png displays f(X) the theoretical repartition and Y the random dataset
line.png compares f(X) to f_pred(X), where f_pred(X) = w_pred * X + b_pred
loss.png shows the evolution of the loss