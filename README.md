# pytorch-foundations

A progressive PyTorch tutorial: 5 experiments from linear regression to CNN on MNIST.

Five small experiments in increasing complexity, from linear regression to a CNN on MNIST.
Plots are saved in `figures/` and metrics are printed in the terminal.


| #   | Experiment            | Typical result (CPU, default config)              |
| --- | --------------------- | ------------------------------------------------- |
| 1   | Linear regression     | R² ≈ 0.97, fitted `w≈2.85`, `b≈1.46` (true: 3, 2) |
| 2   | Logistic regression   | Accuracy ≈ 93%, F1 ≈ 0.93                         |
| 3   | Neural net on 2D data | Val accuracy ≈ 95%, F1 ≈ 0.95                     |
| 4   | MLP on MNIST          | Test accuracy ≈ 91%, macro-F1 ≈ 0.91              |
| 5   | CNN on MNIST          | Test accuracy ≈ 98%, macro-F1 ≈ 0.98              |




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
    02_logistic_regression/     # experiment 2 outputs
    03_neural_net_2D/           # experiment 3 outputs
    04_mnist_mlp/               # experiment 4 outputs
    05_mnist_cnn/               # experiment 5 outputs
```



## Quick start

**Requirements:** Python 3.10+

```bash
git clone https://github.com/bbblue2004/pytorch-foundations.git
cd pytorch-foundations
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python main.py --exp mnist_cnn   # or: linreg, logreg, nn2d, mnist_mlp, all
```

`main.py` runs one of these 5 experiments via `--exp` (or all with `--exp all`) :

1. Linear regression  (--exp linreg)
2. Logistic regression  (--exp logreg)
3. Neural network on 2D data  (--exp nn2d)
4. MLP on MNIST  (--exp mnist_mlp)
5. CNN on MNIST   (--exp mnist_cnn)
6. All of the above  (--exp all)

Metrics are printed in the terminal; plots are saved in each experiment subfolder under figures/.

## 1. Linear regression

**Run:** `python main.py --exp linreg`

Synthetic dataset: fit `Y ≈ w·X + b` with Gaussian noise. Introduces `nn.Module`, `MSELoss`, and SGD on the full dataset (no DataLoader).

### Data

- X ∈ [-3, 3], step 0.1 (60 points)
- True function: `f(X) = 3·X + 2`
- `Y = f(X) + 0.6 · N(0, 1)`



### Model

Single `nn.Linear(1, 1)` — weights `w` and bias `b` learned by gradient descent.

### Results

After 50 epochs: `w≈2.85`, `b≈1.46` (target: 3, 2) — **R² ≈ 0.97**, MSE ≈ 0.87.

### Metrics

MSE, RMSE, MAE, R² = 1 − RSS/TSS (close to 1 is good).

### Figures

`figures/01_linear_regression/` — `data.png`, `fit.png`, `loss.png`.

Linear regression fit

## 2. Logistic Regression

**Run:** `python main.py --exp logreg`

Binary classification on 2D Gaussian blobs. Introduces `BCEWithLogitsLoss` (logits, no sigmoid in the forward pass).

### Data

Two overlapping classes (200 points each), inspired by [mini-neural-network-from-scratch](https://github.com/ppries/mini-neural-network-from-scratch).

### Model

`nn.Linear(2, 1)` — linear decision boundary in the plane.

### Results

After 50 epochs: **accuracy ≈ 93%**, precision ≈ 0.93, recall ≈ 0.93, **F1 ≈ 0.93**.

### Metrics

Accuracy, precision, recall, F1-score (confusion matrix printed in the terminal).

### Figures

`figures/02_logistic_regression/` — `data.png`, `fit.png`, `loss.png`.

Logistic regression decision boundary

## 3. Neural network on 2D Data

**Run:** `python main.py --exp nn2d`

Same family of 2D data as experiment 2, but **non-linearly separable**. Introduces a hidden layer, train/val split (80/20), and validation metrics.

### Data

Two moons-style classes (250 points each). 80% train / 20% validation.

### Model

`nn.Sequential`: `Linear(2,16) → ReLU → Linear(16,1)` — non-linear decision boundary.

### Results

After 1000 epochs: train accuracy ≈ 95%, **val accuracy ≈ 95%**, val F1 ≈ 0.95.

### Metrics

Same as experiment 2, reported separately on train and validation sets.

### Figures

`figures/03_neural_net_2D/` — `data.png`, `fit.png`, `loss.png`.

Neural net decision boundary

## 4. MNIST MLP

**Run:** `python main.py --exp mnist_mlp`

First image dataset: `torchvision` MNIST + `DataLoader`. Multiclass classification (10 digits).

### Data

MNIST (60k train + 10k test). 90/10 train/val split on the training set. Images as tensors `[N, 1, 28, 28]`.

### Model

`Flatten → Linear(784,128) → ReLU → Linear(128,10)` — logits for 10 classes.

### Results

After 5 epochs (CPU): **test accuracy ≈ 91%**, macro-F1 ≈ 0.91.

### Metrics

Accuracy and macro-F1 on the test set; 10×10 confusion matrix.

### Figures

`figures/04_mnist_mlp/` — `loss.png`.

MLP training loss

## 5. MNIST CNN

**Run:** `python main.py --exp mnist_cnn`

### Data

Same pipeline as experiment 4 (torchvision MNIST, 90/10 train/val split, separate test set).
Images are tensors of shape `[N, 1, 28, 28]` with values in `[0, 1]` (`ToTensor` only).

**DataLoader:** `num_workers=0` is often best on CPU/Windows for MNIST; `num_workers>0` mainly helps when a GPU waits on data loading. `pin_memory=True` only speeds up CPU→GPU transfer.

**Batch size:** mini-batch SGD updates weights once per batch. Larger batches are faster per epoch but change optimization (fewer, smoother updates); smaller batches can improve accuracy at the cost of speed.

### Model

LeNet-5–style CNN adapted from [LeCun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf). MNIST is 28×28; we pad to 32×32 so conv/pool dimensions match the original architecture.


| Layer                             | Output shape      | Notes                     |
| --------------------------------- | ----------------- | ------------------------- |
| `ZeroPad2d(2)`                    | `[N, 1, 32, 32]`  | pad 28→32                 |
| Conv 5×5, 6 filters + ReLU        | `[N, 6, 28, 28]`  | paper used **tanh**       |
| MaxPool 2×2                       | `[N, 6, 14, 14]`  | paper used **avg pool**   |
| Conv 5×5, 16 filters + ReLU       | `[N, 16, 10, 10]` |                           |
| MaxPool 2×2                       | `[N, 16, 5, 5]`   |                           |
| Conv 5×5, 120 filters (C5) + ReLU | `[N, 120, 1, 1]`  | 5×5 kernel on 5×5 spatial |
| Flatten                           | `[N, 120]`        |                           |
| Linear 120→84 + ReLU              |                   | F6 in the paper           |
| Linear 84→10                      | logits            | 10 digit classes          |


Unlike the MLP, there is no `Flatten` at the input: the network keeps the 2D structure and learns local filters before the fully connected head.

### Metrics

Same as experiment 4 (accuracy + macro-F1 on the test set).

### Results

After 5 epochs (CPU): **test accuracy ≈ 98%**, macro-F1 ≈ 0.98.

### Figures

The main directory is `05_mnist_cnn/`.


| File                | Description                          |
| ------------------- | ------------------------------------ |
| `loss.png`          | Train / val loss per epoch           |
| `sample_images.png` | 8 MNIST digits from a training batch |
| `filters.png`       | 6 learned 5×5 kernels from Conv1     |
| `feature_maps.png`  | Conv1 activations on one input digit |


MNIST samples

