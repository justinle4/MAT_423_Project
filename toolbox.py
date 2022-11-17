# torch
import torch
# classic libraries
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('agg')  # need it on the server (no GUI)
import matplotlib.pyplot as plt
import random, os, sys, json


def plot_stat_training(df, folder_name):
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
    plt.savefig(folder_name + '/stat_epochs.pdf')
    plt.close()


def save_df_network(myModel, cfg, df):
    plot_stat_training(df, cfg['folder_result'] + '/Report_' + cfg['str_time'])
    df.to_csv(cfg['folder_result'] + '/Report_' + cfg['str_time'] + '/stat_epochs.tsv', sep='\t')