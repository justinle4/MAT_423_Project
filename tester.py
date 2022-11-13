import torch
import statistics as st
import torch.nn.functional as F

y_value = 5
y = torch.tensor(y_value)
x = torch.rand((1, 784))
print(x.size(dim=1))
