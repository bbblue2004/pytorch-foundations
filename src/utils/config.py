SEED = 42

LIN_REG = {
    "epochs": 50,
    "lr": 0.01,
    "w_true": 3.0,
    "b_true": 2.0,
    "noise": 0.6,
}

LOG_REG = {
    "epochs": 50,
    "lr": 0.1,
    "n_per_class": 200,
    "sigma": 2.0,
}

NN_2D = {
    "epochs": 1000,
    "lr": 0.1,
    "n_per_class": 250,
    "sigma": 1.4,
    "input_size": 2,
    "hidden_size": 16,
    "output_size": 1
}

MNIST_MLP = {
    "epochs": 5,
    "lr": 0.01,
    "input_size": 28*28,
    "hidden_size": 128,
    "output_size": 10,
    "val_ratio": 0.1,
    "data_dir": "data",
    "batch_size": 64
}

MNIST_CNN = {
    "epochs": 5,
    "lr": 0.01,
    "val_ratio": 0.1,
    "data_dir": "data",
    "batch_size": 32
}