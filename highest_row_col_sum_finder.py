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
    'shuffle_dataset_known': False,
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



df = pd.DataFrame(columns=('Identity', 'Brightest Row', 'Brightest Column'))
for i in range(x.size(dim=0)):
    col_sums = torch.sum(x[i], dim=0)
    row_sums = torch.sum(x[i], dim=1)
    whitest_col = torch.argmax(col_sums).item()
    whitest_row = torch.argmax(row_sums).item()
    df.loc[i] = (y[i].item(), whitest_row, whitest_col)

print(df)
df.to_csv(f"brightest_rows_cols.tsv", sep="\t")



