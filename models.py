import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.conv1 = nn.Linear(784, 10)

    def forward(self, x):
        s = self.conv1(x)
        return s


class OneConv_kernel3(nn.Module):
    def __init__(self):
        super(OneConv_kernel3, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)

    def forward(self, x):
        s = self.conv1(x)
        return s


class TwoConv_kernel3(nn.Module):
    def __init__(self):
        super(OneConv_kernel3, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, padding=1)
        self.conv2 = nn.Conv2d(5, 10, 3, padding=1)

    def forward(self, x):
        z = F.relu(self.conv1(x))
        s = self.conv2(z)
        return s


class ThreeConv_kernel3(nn.Module):
    def __init__(self):
        super(OneConv_kernel3, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 7, 3, padding=1)
        self.conv3 = nn.Conv2d(7, 10, 3, padding=1)

    def forward(self, x):
        w = F.relu(self.conv1(x))
        z = F.relu(self.conv2(w))
        s = self.conv3(z)
        return s
