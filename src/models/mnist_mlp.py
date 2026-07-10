from torch import nn


class MNISTMLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                     # shape [N, 1, 28, 28] -> [N, 784]
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)                # returns logits
