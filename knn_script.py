import pandas
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import statistics
import numpy as np
from scipy import stats
import statistics as st
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import *

# hyper parameters:
cfg = {
    'sample_size_known': 500,
    'sample_size_test': 100,
    'num_epochs': 40,
    'path_data': "C:\\Users\\drpot\\PycharmProjects\\Fall2022Project\\Cropped Training Data\\",
    'shuffle_dataset_known': True,
    'shuffle_dataset_test': False,
    'transform': transforms.ToTensor(),
    'k': 75,
    'norm': 'fro'
}

# dataset as tensors
myDataSet_known = torchvision.datasets.MNIST("C:\\Users\\drpot\\", train=True, transform=cfg['transform'])
myDataSet_test = torchvision.datasets.MNIST("C:\\Users\\drpot\\", train=False, transform=cfg['transform'])


# using the DataLoader class to take a random sample from the datasets
dataloader_known = DataLoader(myDataSet_known, shuffle=cfg['shuffle_dataset_known'],
                              batch_size=cfg['sample_size_known'])
dataloader_test = DataLoader(myDataSet_test, shuffle=cfg['shuffle_dataset_test'], batch_size=cfg['sample_size_test'])

# grab only ONE sample from the known dataset. This sample will be randomized each time to test the effectiveness
# of our k-nearest-neighbors algorithm
x, y = next(iter(dataloader_known))
x = torch.squeeze(x, dim=1)

# grab only ONE sample from the test dataset. We will use these tensors and try to predict what images they are.
x_test, y_test = next(iter(dataloader_test))
x_test = torch.squeeze(x_test, dim=1)


df = pd.DataFrame(columns=('K-Value', 'Accuracy'))
# perform k-nearest-neighbors for varying values of k.
for k in range(5, 80, 5):   # k goes from 5 to 75 in increments of 5 (5, 10, 15, ... , 70, 75)
    success = 0
    for i in range(x_test.size(dim=0)):
        norm_list = []
        for j in range(x.size(dim=0)):
            diff = x_test[i] - x[j]
            norm = torch.linalg.norm(diff, ord=float('inf')).item()
            norm_list.append(norm)
        partitioned = np.argpartition(norm_list, k)
        k_smallest = partitioned[:k]
        neighbor_list = []
        for h in k_smallest:
            neighbor_list.append(y[h].item())
        guess = st.mode(neighbor_list)
        if int(guess) == y[i].item():
            success += 1
    accuracy = success / cfg['sample_size_test']
    df.loc[int(k/5 - 1)] = (str(k), accuracy)

print(df)
df.to_csv(f"{cfg['norm']}_norm_results.tsv", sep="\t")
