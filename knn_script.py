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
    'shuffle_dataset_test': True,
    'transform': transforms.ToTensor(),
    'k': 5,
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


df = pd.DataFrame(columns=('Trial', 'Accuracy'))

# perform k-nearest-neighbors for the SAME value of k (k=5) but shuffle dataset for multiple trials
k = cfg['k']
for trial in range(1, 20):   # k goes from 5 to 75 in increments of 5 (5, 10, 15, ... , 70, 75)
    x, y = next(iter(dataloader_known))
    x = torch.squeeze(x, dim=1)
    x_test, y_test = next(iter(dataloader_test))
    x_test = torch.squeeze(x_test, dim=1)
    success = 0
    for i in range(x_test.size(dim=0)):
        norm_list = []
        for j in range(x.size(dim=0)):
            diff = x_test[i] - x[j]
            norm = torch.linalg.norm(diff, ord=cfg['norm']).item()
            norm_list.append(norm)
        partitioned = np.argpartition(norm_list, k)
        k_smallest = partitioned[:k]
        neighbor_list = []
        for h in k_smallest:
            neighbor_list.append(y[h].item())
        guess = st.mode(neighbor_list)
        if int(guess) == y_test[i].item():
            success += 1
    accuracy = success / cfg['sample_size_test']
    df.loc[trial] = (str(trial), accuracy)
    print(f'Trial {trial} complete')

print(df)
df.to_csv('trial_runs.tsv', sep='\t')
