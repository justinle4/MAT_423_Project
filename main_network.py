from models import *
from training_module import *
import matplotlib.pyplot as plt

# declaring hyper parameters
cfg = {
    'num_epochs': 20,
    'learning_rate': 0.01,
    'nbr_classes': 10,
    'weight': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'batch_size': 32,
    'pct_train_set': .8,
    'shuffle_dataset': True,
    'name_classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'num_workers': 0,  # num_workers of 4 for training on Agave, 0 on a Windows machine
    'folder_result': 'C:\\Users\\drpot\\PycharmProjects\\MAT 423 Project\\Results',
    # agave result folder: '../results_ML'
    # personal machine result folder: "C:\\Users\\drpot\\PycharmProjects\\Fall2022Project\\Results
    'path_data': "C:\\Users\\drpot\\PycharmProjects\\Fall2022Project\\Cropped Training Data\\",
    # agave path data: "../Slime_Mold_Project/Slime_Mold_Dataset/"
    # personal machine path data: "C:\\Users\\drpot\\PycharmProjects\\Fall2022Project\\Cropped Training Data\\"
    'model': 'Linear',
    'transform': transforms.ToTensor()
}


# training
myModel = eval(cfg['model'])()
df_training = trainModel(myModel, cfg)