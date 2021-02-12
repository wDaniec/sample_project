import torch.nn as nn
import torch.nn.functional as F


class LinearRegression(nn.Module):
    def __init__(self, out_size, in_size=32*32*3):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(in_size, out_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.fc(x)
        