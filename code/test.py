#!/usr/bin/env python
# coding: utf-8

# In[52]:


# Imports
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import skimage
torch.manual_seed(5)
import time
import csv
import math
#Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion() 


# In[5]:

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# torshion multiplied by 100


# define the CNN Models
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.baseConv1 = nn.Conv2d(3, 16, 3, 2)
        self.baseConv2 = nn.Conv2d(16, 32, 3, 2)
        self.baseConv3 = nn.Conv2d(32, 64, 3,2)

        self.pool = nn.MaxPool2d(2, 2)

        self.classFc1 = nn.Linear(4032, 300)
        self.classFc2 = nn.Linear(300, 1)

        self.regFc1 = nn.Linear(4032, 1000)
        self.regFc2 = nn.Linear(1000, 100)
        self.regFc3 = nn.Linear(100, 7)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.pool(F.relu(self.baseConv1(x)))
        x = self.pool(F.relu(self.baseConv2(x)))
        x = self.pool(F.relu(self.baseConv3(x)))
        # flatten the Base
        x = x.view(x.size(0), -1)

        x1 = F.relu(self.classFc1(x))
        x1 = self.dropout(x1)
        x1 = self.classFc2(x1)
        x1 = F.sigmoid(x1)
        #x = self.fc2(x)
        #x = F.relu(self.fc2(x))
        x2 = F.relu(self.regFc1(x))
        x2 = self.dropout(x2)
        x2 = F.relu(self.regFc2(x2))
        x2 = self.dropout(x2)
        x2 = self.regFc3(x2)

        return x1, x2 #F.relu(x)

net = Net()


# # >>> EVALUATE TESTING DATASET


# TESTing Dataset

def model_test(modelname):


    ModelSavePath = '/raid/daulet/rgb_resubmit/saved_models/' + modelname +'.pth'

    t0 = time.clock()

    pathTestFiles = '/raid/daulet/rgb_resubmit/data/labels/testing/'
    nameTestFiles = [ 'main/test_dataS2.csv', 
                      'bandhigh/bandhigh_labels0.csv', 'bandhigh/bandhigh_labels1.csv', 'bandhigh/bandhigh_labels2.csv', 
                      'bandhigh/bandhigh_labels3.csv', 'bandhigh/bandhigh_labels4.csv', 'bandhigh/bandhigh_labels5.csv',
                      'bandlow/bandlow_labels0.csv', 'bandlow/bandlow_labels1.csv', 'bandlow/bandlow_labels2.csv', 
                      'both/both_labels0.csv', 'both/both_labels1.csv', 'both/both_labels2.csv']

    pathPredFiles = '/raid/daulet/rgb_resubmit/dgxpredictions/' + modelname + '/'
    namePredFiles = ['pred_main_reshaped_testS2.csv', 
                      'pred_bandhigh_labels0.csv', 'pred_bandhigh_labels1.csv', 'pred_bandhigh_labels2.csv', 
                      'pred_bandhigh_labels3.csv', 'pred_bandhigh_labels4.csv', 'pred_bandhigh_labels5.csv',
                      'pred_bandlow_labels0.csv', 'pred_bandlow_labels1.csv', 'pred_bandlow_labels2.csv', 
                      'pred_both_labels0.csv', 'pred_both_labels1.csv', 'pred_both_labels2.csv']

    
    
    for n in range(4,4,4,4,13):
        
        testFilename = os.path.join(pathTestFiles, nameTestFiles[n])
        testFile = pd.read_csv(testFilename, header = None)
        
        predFile = os.path.join(pathPredFiles, namePredFiles[n])
        
        file = open(testFilename)
        numline = len(file.readlines())
        print(n,numline)
        
        for i in range(0,numline):
            
            img_name = testFile.iloc[i, 0]
            impath = '/raid/daulet/rgb_resubmit/data/images/'    
        
            # read the image
            test_img_name = os.path.join(impath, img_name)

            #print(test_img_name)
            test_image = io.imread(test_img_name)
            #io.imshow(test_image)

            test_image = test_image.transpose((2, 0, 1))
            test_image = np.reshape(test_image, (1, 3, 480, 640))

            # load the model
            #mdevice = torch.device('cpu')
            loaded_model = Net()
            loaded_model.load_state_dict(torch.load(ModelSavePath, map_location = device))

            # predict the labels
            test_image = torch.from_numpy(test_image).float()
            test_image = test_image.to(device)
            loaded_model = loaded_model.to(device)

            #loaded_model( skimage.img_as_float(test_image))
            label1, label2 = loaded_model(test_image)

            label1 = label1.cpu().detach().numpy()
            label2 = label2.cpu().detach().numpy()

            #print(labels_predicted)
            label1.shape = (len(label1))
            label2.shape = (7*len(label2))

            line = [None]*9
            testIm = test_img_name.split('/')
            testImFinal = testIm.pop()
            line[0] = testImFinal
            line[1] = label1[0]
            line[2:9] = label2[0:7]
            line[8] = line[8]/100.0
            
            if(os.path.exists(pathPredFiles) == False):
                os.makedirs(pathPredFiles)
            with open(predFile, 'a') as writeFile:
                writer = csv.writer(writeFile, delimiter=',')
                writer.writerows([line])

    t1 = time.clock()
    total = t1-t0

    print('Testing set results finished')
    print(total)


# In[ ]:


modelname = 'model_BOTH5_ep20'


model_test(modelname)

