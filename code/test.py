"""
Testing the model.
"""

# Imports
import os
import time
import csv
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import skimage
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from models import Net

def main():
    
    net = Net()
    modelname = 'model_BOTH5_ep20'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ModelSavePath = '/raid/daulet/rgb_resubmit/saved_models/' + modelname +'.pth'
    pathTestFiles = '/raid/daulet/rgb_resubmit/data/labels/testing/'
    nameTestFiles = [ 'main/test_dataS2.csv']
    pathPredFiles = '/raid/daulet/rgb_resubmit/dgxpredictions/' + modelname + '/'
    namePredFiles = ['pred_main_reshaped_testS2.csv'] 
    
    for n in range(4, 13):
        
        testFilename = os.path.join(pathTestFiles, nameTestFiles[n])
        testFile = pd.read_csv(testFilename, header=None)
        predFile = os.path.join(pathPredFiles, namePredFiles[n])
        
        file = open(testFilename)
        numline = len(file.readlines())
        
        for i in range(0, numline):
            
            img_name = testFile.iloc[i, 0]
            impath = '/raid/daulet/rgb_resubmit/data/images/'    
        
            # read the image
            test_img_name = os.path.join(impath, img_name)
            test_image = io.imread(test_img_name)
            test_image = test_image.transpose((2, 0, 1))
            test_image = np.reshape(test_image, (1, 3, 480, 640))

            # load the model
            loaded_model = Net()
            loaded_model.load_state_dict(torch.load(ModelSavePath, map_location=device))

            # predict the labels
            test_image = torch.from_numpy(test_image).float()
            test_image = test_image.to(device)
            loaded_model = loaded_model.to(device)

            label1, label2 = loaded_model(test_image)
            label1 = label1.cpu().detach().numpy()
            label2 = label2.cpu().detach().numpy()

            label1.shape = (len(label1))
            label2.shape = (7 * len(label2))

            line = [None] * 9
            testIm = test_img_name.split('/')
            testImFinal = testIm.pop()
            line[0] = testImFinal
            line[1] = label1[0]
            line[2:9] = label2[0:7]
            line[8] = line[8] / 100.0
            
            if(os.path.exists(pathPredFiles) == False):
                os.makedirs(pathPredFiles)
            with open(predFile, 'a') as writeFile:
                writer = csv.writer(writeFile, delimiter=',')
                writer.writerows([line])


if __name__ == '__main__':
    main()

