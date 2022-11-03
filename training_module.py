# torch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import *
# classic
import sys
import time
import datetime
import os
from PIL import Image
import pandas as pd
import statistics
import numpy as np
from scipy import stats
import statistics as st
import matplotlib
import matplotlib.pyplot as plt
# personal
from models import *
from toolbox import *

def trainModel(myModel, cfg):
    myDataSet_train = torchvision.datasets.MNIST("C:\\Users\\drpot\\", train=True, transform=cfg['transform'])
    myDataSet_test = torchvision.datasets.MNIST("C:\\Users\\drpot\\", train=False, transform=cfg['transform'])

    if myDataSet_train.__len__() == 0:
        print("--- Problem initialization: dataSet empty\n")
        sys.exit(2)

    if myDataSet_test.__len__() == 0:
        print("--- Problem initialization: dataSet empty\n")
        sys.exit(2)

    dataLoader_train = DataLoader(myDataSet_train, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                                  sampler=myDataSet_train.train_sampler)
    dataLoader_valid = DataLoader(myDataSet_test, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                                  sampler=myDataSet_test.test_sampler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    myModel = torch.nn.DataParallel(myModel)
    myModel.to(device)

    weight = torch.tensor(cfg['weight'])
    criterion = nn.MSELoss(weight=weight.to(device))
    optimizer = torch.optim.SGD(myModel.parameters(), lr=cfg['learning_rate'])

    # training
    t0 = time.time()
    df = pd.DataFrame(columns=('epoch', 'loss_train', 'loss_test', 'accuracy_train,' 'accuracy_test'))

    print("--- Training Begins ---")
    for epoch in range(cfg['num_epochs']):
        myModel.train()
        accuracy_train = 0
        accuracy_test = 0
        loss_train = []
        loss_test = []
        for X, y_value in dataLoader_train:
            # forward pass
            yhat = myModel(X.to(device))

            # loss
            y = torch.zeros((10, 1))
            y[y_value] = 1
            myLoss_train = criterion(yhat, y.to(device))
            loss_train.append(myLoss_train)

            # backpropagation
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # accuracy
            if torch.argmax(X).item() == y_value:
                accuracy_train += 1

        accuracy_train = st.mean(accuracy_train)
        # evaluate model
        myModel.eval()
        with torch.no_grad():
            for X, y_value in dataLoader_valid:
                yhat = myModel(X.to(device))
                y = torch.zeros(10, 1)
                y[y_value] = 1
                myLoss_test = criterion(yhat, y)
                loss_test.append(myLoss_test)
                if torch.argmax(X).item() == y_value:
                    accuracy_test += 1


        # insert results into dataframe
        df.loc[epoch] = [epoch, np.mean(loss_train), np.mean(loss_test),
                         accuracy_train/len(myDataSet_train), accuracy_test/len(myDataSet_test)]
        print(f"Epoch {epoch + 1} complete")

    seconds_elapsed = time.time() - t0
    cfg['time_training'] = '{:.0f} minutes, {:.0f} seconds'.format(seconds_elapsed // 60, seconds_elapsed % 60)
    print(f"--- Training Complete in {cfg['time_training']}  ---")
    cfg['str_time'] = datetime.datetime.now().replace(microsecond=0).isoformat(sep='_').replace(':', 'h', 1).replace(
        ':', 'm', 1)

