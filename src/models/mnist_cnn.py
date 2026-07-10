from torch import nn


class MNISTCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # input shape is [N, depth=1, height=28, width=28]
            nn.ZeroPad2d(2),      # [N, 1, 28, 28] -> [N, 1, 32, 32]
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),  # out: shape [N, 6, 28, 28] because 6 feature maps
            nn.Tanh(),    # or nn.ReLU(), but that's how the paper did it
            nn.MaxPool2d(kernel_size=2, stride=2),    # out: shape [N, 6, 14, 14] (each square 2x2 -> 1x1)
            # no activation here because the operation max(...) is already non-linear
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),  # out: shape [N, 16, 10, 10]
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # out: shape [N, 16, 5, 5]
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0) ,  # out: shape [N, 120, 1, 1]
            nn.Tanh(),

            nn.Flatten(),    # out: shape [N, ]
            nn.Linear(120, 84),   # out: shape [N, 84]
            nn.Tanh(),
            nn.Linear(84, 10)    # out: shape [N, 10]
        )

    def forward(self, x):
        return self.net(x)                # returns logits
