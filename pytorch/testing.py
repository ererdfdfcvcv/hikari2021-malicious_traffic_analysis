import pickle
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
import torch
from network import Net
from dataset import CustomDataLoader
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import re

model_path = 'pytorch\models\20220518185019_5.model'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("pytorch/training_data.pickle", 'rb') as handle:
    training_data = pickle.load(handle)
    x_test = training_data[1]
    y_test = training_data[3]



net = torch.load(model_path)


net.predict