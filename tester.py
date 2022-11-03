import torch
import statistics as st

y = torch.zeros(10, 1)
y[3] = 1
print(y)
print(torch.argmax(y).item())
print(y.shape)

y = [1, 2, 3, 4]
y = st.mean(y)
print(y)