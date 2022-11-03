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
    'sample_size_test': 50,
    'num_epochs': 40,
    'path_data': "C:\\Users\\drpot\\PycharmProjects\\Fall2022Project\\Cropped Training Data\\",
    'shuffle_dataset_known': False,
    'shuffle_dataset_test': False,
    'transform': transforms.ToTensor(),
    'k': 75,
    'norm': 'inf'
}

# dataset as images
myDataSet_image_known = torchvision.datasets.MNIST("C:\\Users\\drpot\\", train=True)
myDataSet_image_test = torchvision.datasets.MNIST("C:\\Users\\drpot\\", train=False)

# to show off a picture of the image from the dataset:
# x_img, y = myDataSet_image_known.__getitem__(0)
# plt.figure(0); plt.clf()
# plt.imshow(x_img, cmap='gray')
# plt.title("An image with label "+str(y))
# plt.show()


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
# x.shape: 500 x 28 x 28
# y.shape: 500 x 1

# Note: x is a 500 x 28 x 28 tensor. The first dimension encodes which image in the sample we are looking at.
# the 28 x 28 dimensions tell us the grayscale intensity of each pixel (from 0 to 1, where 0 is black and 1 is white).
# y is a 500 x 1 vector. This vector encodes the identity of each image. For example, the 15th entry of this vector
# will tell us the identity of the 15th image in our sample (a digit from 0 to 9).


# grab only ONE sample from the test dataset. We will use these tensors and try to predict what images they are.
x_test, y_test = next(iter(dataloader_test))
x_test = torch.squeeze(x_test, dim=1)
# x_test.shape: 15 x 28 x 28
# y_test.shape: 15 x 1


# we have a sample of known images and images we want to test. How can we use these to implement k-nearest-neighbors?
# note: if we do x[k] (for some integer k), this will give us the grayscale tensor of the kth image of the sample.
# similarly, y[k] will give us the identity of the kth image of the sample.

success = 0
for i in range(x_test.size(dim=0)):
    norm_list = []
    for j in range(x.size(dim=0)):
        diff = x_test[i] - x[j]
        norm = torch.linalg.norm(diff, ord=float('inf')).item()
        norm_list.append(norm)
    partitioned = np.argpartition(norm_list, cfg['k'])
    k_smallest = partitioned[:cfg['k']]
    neighbor_list = []
    for h in k_smallest:
        neighbor_list.append(y[h].item())
    guess = st.mode(neighbor_list)
    if int(guess) == y[i].item():
        success += 1
        print(f"Correct Guess! Number was {int(guess)}")


print(f"Number of successes: {success}")
print(f"Total number of pictures guessed: {x_test.size(dim=0)}")


# some things we can do:
# plot the accuracy for different values of k
# make a scatter plot of matrix norm vs what the number actually is.
# can start on the presentation. First slides will be motivation -> refresher k-nearest neighbors -> our implementation
# make a group GitHub repository and store our code into it.
# I do not believe the code is bugged; these images are very similar due to the fact that most of it is black background
# (meaning zeros on the matrix). We can devote a part of the presentation to addressing the limitations of
# k-nearest neighbors in a context like this to better motivate the need to use machine learning architecture.
# shuffle our samples and run the program more times to see a pattern.
# if anyone is into stats, maybe come up with some confidence interval for the mean accuracy.












