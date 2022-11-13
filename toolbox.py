# torch
import torch
# classic libraries
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('agg')  # need it on the server (no GUI)
import matplotlib.pyplot as plt
import random, os, sys, json


def plot_stat_training(df):
    ''' statistics over epochs '''
    # init
    nbEpochs = len(df) - 1
    # plot
    plt.figure(1);
    plt.clf()
    plt.ioff()
    plt.plot(df['epoch'], df['loss_train'], '-o', label='loss train')
    plt.plot(df['epoch'], df['loss_test'], '-o', label='loss test')
    plt.plot(df['epoch'], df['accuracy_train'], '-o', label='accuracy train')
    plt.plot(df['epoch'], df['accuracy_test'], '-o', label='accuracy test')
    plt.grid(b=True, which='major')
    plt.xlabel(r'epoch')
    plt.ylabel(r'loss')
    plt.legend(loc=0)
    plt.axis([-.5, nbEpochs + .5, 0, 1.01])
    plt.draw()
    plt.savefig('stat_epochs.pdf')
    plt.close()


def plot_save_proba(s, nameFile):
    plt.figure(2);
    plt.clf()
    plt.ioff()
    plt.imshow(s)
    plt.colorbar()
    plt.draw()
    plt.savefig(nameFile)
    plt.close()


def plot_save_mask(s, nameFile):
    plt.figure(2);
    plt.clf()
    plt.ioff()
    plt.imshow(s > .5, cmap="gray")
    plt.draw()
    plt.savefig(nameFile)
    plt.close()


def compute_jaccard_batch(mask1, mask2):
    ''' Jaccard index between two mini-batch of images
      . mask1,mask2: boolean matrix of the same size NxHxW
      . jaccard index: intersection/union (always between 0 and 1)
    '''
    num = np.minimum(mask1, mask2)
    num_sum = np.sum(np.sum(num, axis=2), axis=1)
    den = np.maximum(mask1, mask2)
    den_sum = np.sum(np.sum(den, axis=2), axis=1)
    return num_sum / den_sum


def save_df_network(myModel, cfg, df):
    plot_stat_training(df)
