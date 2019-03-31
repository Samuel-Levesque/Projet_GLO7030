import warnings
warnings.filterwarnings('ignore') # to suppress some matplotlib deprecation warnings

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import ast
import cv2

import matplotlib.pyplot as plt
import matplotlib.style as style

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import os
import glob
import time
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import torchvision
from torchvision import transforms, utils

from data_set_file import DoodlesDataset



path = 'D:/User/William/Documents/Devoir/Projet Deep/data/test/'
csv_file="test_simplified.csv"

data_set=DoodlesDataset(csv_file,path,mode="test")

loader=DataLoader(data_set,batch_size=1)

i=1
for image in loader:
    print(i)
    i+=1
    print(image)