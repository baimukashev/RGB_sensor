#!/usr/bin/env python
# coding: utf-8

# RGB RAL_ICRA_2020 paper 
# Daulet Baimukashev
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
import math

torch.manual_seed(5)

#Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion() 

class rgbSensorDataset(Dataset):
    """RGB dataset """
    def __init__(self, csv_file, root_dir, transform = None):
        """
        init 
        """
        self.labels_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.labels_file)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir,self.labels_file.iloc[idx, 0])
        image = io.imread(img_name)
        
        # !!! choose the LABEL
        #angle = self.labels_file.iloc[idx, 9]
        labels = self.labels_file.iloc[idx, [1,2,3,4,5,6,7,8]]

        labels = np.array([labels])
        #labels[0,0] = 1000.0*labels[0,0]*math.cos(math.radians(angle))
        labels[0,3] = labels[0,3] + labels[0,6]
        labels[0,7] = 100.0*labels[0,7]
        labels = labels.astype('float').reshape(-1, 8)
        #print(img_name)
        #print(labels.dtype)
        #print('given labels::', labels)
        #labels = labels.astype('float').reshape(-1, 2)        

        sample = {'image':image, 'labels':labels, 'image_name':img_name}
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    

class Resize(object):
    """Resize the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, labels, img_name = sample['image'], sample['labels'], sample['image_name']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        image =  transform.resize(image, (new_h, new_w))

        return {'image': image, 'labels': labels, 'image_name':img_name}

class ImageCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, labels, img_name = sample['image'], sample['labels'], sample['image_name']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = 100
        left = 130

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'labels': labels, 'image_name':img_name}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels, img_name = sample['image'], sample['labels'], sample['image_name']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        #print('imshape :', image.shape)
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels),
                'image_name':img_name
               }

csv_file_train = '/raid/daulet/rgb_resubmit/data/labels/reshaped_trainABC.csv'
csv_file_test  = '/raid/daulet/rgb_resubmit/data/labels/reshaped_validABC.csv'
images_folder_train  = '/raid/daulet/rgb_resubmit/data/images/'
images_folder_test  = '/raid/daulet/rgb_resubmit/data/images/'

transformed_rgb_dataset = rgbSensorDataset(csv_file = csv_file_train, 
                                            root_dir = images_folder_train, 
                                           transform=transforms.Compose([ToTensor()]))
transformed_rgb_dataset_test = rgbSensorDataset(csv_file = csv_file_test, 
                                            root_dir = images_folder_test, 
                                transform=transforms.Compose([ToTensor()]))
batchSize = 64


data_train = DataLoader(transformed_rgb_dataset, batch_size = batchSize,shuffle=True, num_workers= 4,drop_last=True)
data_valid = DataLoader(transformed_rgb_dataset_test, batch_size= batchSize,shuffle=True, num_workers= 4,drop_last=True)

print('Lenght of train_data:', len(data_train))

iValid = int(round(len(data_train)/batchSize))/2+1

# Model preparation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# define the CNN Models
       

# define the CNN Models
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.baseConv1 = nn.Conv2d(3, 32, 5, 2)
        self.baseConv2 = nn.Conv2d(32, 64, 5, 2)
        self.baseConv3 = nn.Conv2d(64, 128, 3, 2)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.classFc1 = nn.Linear(8064, 300)
        self.classFc2 = nn.Linear(300, 1)
                
        self.regFc1 = nn.Linear(8064, 1000)
        self.regFc2 = nn.Linear(1000, 100)
        self.regFc3 = nn.Linear(100, 7) 
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.baseConv1(x)))
        x = self.pool(F.relu(self.baseConv2(x)))
        x = self.pool(F.relu(self.baseConv3(x)))
        # flatten the Base
        x = x.view(x.size(0), -1)

        x1 = F.relu(self.classFc1(x))
        #x1 = self.dropout(x1)
        x1 = self.classFc2(x1)
        x1 = F.sigmoid(x1)
        #x = self.fc2(x)
        #x = F.relu(self.fc2(x))
        x2 = F.relu(self.regFc1(x))
        #x2 = self.dropout(x2)
        x2 = F.relu(self.regFc2(x2))
        #x2 = self.dropout(x2)
        x2 = self.regFc3(x2)

        return x1, x2 #F.relu(x)

net = Net()
net = nn.DataParallel(net)
net.to(device)


criterion1 = nn.BCELoss()
criterion2 = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)

modelname  = 'model_Test9_ep5'
modelname2 = 'model_Test9_ep10'
modelname3 = 'model_Test9_ep20'

ModelSavePath = '/raid/daulet/rgb_resubmit/saved_models/' + modelname + '.pth'
ModelSavePath2 = '/raid/daulet/rgb_resubmit/saved_models/' + modelname2 + '.pth'
ModelSavePath3 = '/raid/daulet/rgb_resubmit/saved_models/' + modelname3 + '.pth'

#ModelSavePath1 = '/raid/daulet/rgb_resubmit/saved_models/normalFz_Z4_ep10.pth'

# train the network
for epoch in range(50):  # loop over the dataset multiple times
    print('epoch', epoch)
    running_loss = 0.0
    if epoch == 10:
        #print('save only ep10')
        torch.save(net.module.state_dict(), ModelSavePath)
    if epoch == 20:
        #print('save only ep10')
        torch.save(net.module.state_dict(), ModelSavePath2)
    for i, data in enumerate(data_train):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['image']
        inputs = inputs.float()
        inputs = inputs.to(device)

        labels = data['labels']
        labels = labels.type(torch.cuda.FloatTensor)
        labels = labels.reshape((batchSize,8)).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output1, output2 = net(inputs)
        
        output2a = output2[:,0:3]
        output2b = output2[:,3:]        
       

        loss1 = criterion1(output1, labels[:,0])

        loss2 = criterion2(output2a, labels[:,1:4])
        loss3 = criterion2(output2b, labels[:,4:])

        loss = loss1 + loss2 + loss3

        loss.mean().backward()
        optimizer.step()
       
        # print statistics
        if i % 2000  == 1999:
            print('i ==', i)
            correct = 0
            total = 0
            count_test = 0
            avg_loss = 0.0
            for ix, datax in enumerate(data_valid):
                count_test = count_test + 1
                # get the inputs; data is a list of [inputs, labels]
                inputs = datax['image']
                inputs = inputs.float()
                inputs = inputs.to(device)

                labels = datax['labels']
                labels = labels.type(torch.cuda.FloatTensor)
                labels = labels.reshape((batchSize,8)).to(device)

                output_test1, output_test2 = net(inputs)
                
                loss1 = criterion1(output_test1, labels[:,0])
                loss2 = criterion2(output_test2, labels[:,1:])

                loss = loss1 + loss2 + loss3

                avg_loss = avg_loss + loss.mean().item()

                if count_test > 20:
                    break
            
            avg_loss = avg_loss / count_test
            print("RMSE Loss: {}, average RMSE: {} ".format(loss.mean().item(), avg_loss))

# save the model

torch.save(net.module.state_dict(), ModelSavePath3)

