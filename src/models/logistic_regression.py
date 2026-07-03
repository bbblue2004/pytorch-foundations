from torch import nn


class LogisticRegression(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)             # no torch.sigmoid here, for stability reasons