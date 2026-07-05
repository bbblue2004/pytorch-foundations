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
    03_neural_net_2D/           # experiment 3 outputs
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

`main.py` can run any of these 5 experiments using --exp :

1. Linear regression  (--exp linreg)
2. Logistic regression  (--exp logreg)
3. Neural network on 2D data  (--exp nn2d)
4. MLP on MNIST
5. CNN on MNIST

Metrics are printed in the terminal; plots are saved in each experiment subfolder under figures/.



## 1. Linear regression

Implementation of a simple linear regression in pytorch.

### Data

X = [-3, 3] with a step of 0.1  (60 points)
f(X) = 3 * X + 2  (true parameters (3, 2))
Y = f(X) + 0.6 * N(0, 1)  (random gaussian noise)

### Model

Use of nn.Module, so small implementation

### Metrics

MSE (mean square error)
RMSE (root mean square error)
MAE (mean absolute error)
R2 = ESS / TSS = 1 - RSS/TSS. We remind that TSS = ESS + RSS. A good R2 is (very) close to 1.

### Figures

The main directory is 01_linear_regression.
data.png displays f(X) the theoretical repartition and Y the random dataset
fit.png compares f(X) to the fitted linear function Yhat = w_pred * X + b_pred
loss.png shows the evolution of the loss


## 2. Logistic Regression

Implementation of a simple logistic regression in pytorch.

### Data

I reused the linear dataset from the repo mini-neural-network-from-scratch.

### Model

Use of nn.Module, so small implementation

### Metrics

Accuracy: (TP + TN) / (TP + TN + FP + FN)
Precision: TP / (TP + FP)
Recall: TP / (TP + FN)
F1-score: harmonic mean of precision and recall.

### Figures

The main directory is 02_logistic_regression.
data.png displays the points from the two (slightly overlapping) classes
fit.png shows the predicted linear frontier on the training data
loss.png shows the evolution of the loss


## 3. Neural network on 2D Data

### Data

I reused the non-linear dataset from the repo mini-neural-network-from-scratch, with slight modifications of SIGMA and N_PER_CLASS.
This time, split between training and validation sets (80/20) to ensure that overfitting is avoided.

### Model

Use of nn.Module with nn.Sequential()

### Metrics

Same metrics as logistic regression (binary classification in both cases).

### Figures

The main directory is 03_neural_net_2D.
data.png displays the points from the two (slightly overlapping) non linearly-separable classes
fit.png shows the predicted linear frontier on the full (training + validation) data.
loss.png shows the evolution of the training + validation losses