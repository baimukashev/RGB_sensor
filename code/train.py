"""
Script for training the DNN model.
"""

# Imports
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from models import Net

class rgbSensorDataset(Dataset):
    """RGB dataset """
    def __init__(self, csv_file, root_dir, transform=None):
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
        
        labels = self.labels_file.iloc[idx, [1, 2, 3, 4, 5, 6, 7, 8]]
        labels = np.array([labels])
        labels[0, 3] = labels[0, 3] + labels[0, 6]
        labels[0, 7] = 100.0 * labels[0, 7]
        labels = labels.astype('float').reshape(-1, 8)
        
        sample = {'image':image, 'labels':labels, 'image_name':img_name}
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class Resize(object):
    """Resize the image in a sample."""

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

        image = transform.resize(image, (new_h, new_w))

        return {'image':image, 'labels':labels, 'image_name':img_name}

class ImageCrop(object):
    """Crop randomly the image in a sample."""

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

        image = image[top: top + new_h, left: left + new_w]

        return {'image': image, 'labels': labels, 'image_name':img_name}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels, img_name = sample['image'], sample['labels'], sample['image_name']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels),
                'image_name':img_name
               }

def main():
    
    csv_file_train = '/raid/daulet/rgb_resubmit/data/labels/reshaped_trainABC.csv'
    csv_file_test  = '/raid/daulet/rgb_resubmit/data/labels/reshaped_validABC.csv'
    images_folder_train  = '/raid/daulet/rgb_resubmit/data/images/'
    images_folder_test  = '/raid/daulet/rgb_resubmit/data/images/'

    transformed_rgb_dataset = rgbSensorDataset(csv_file =csv_file_train, 
            root_dir=images_folder_train, transform=transforms.Compose([ToTensor()]))
    transformed_rgb_dataset_test = rgbSensorDataset(csv_file = csv_file_test,
            root_dir=images_folder_test, transform=transforms.Compose([ToTensor()]))
    
    batchSize = 64
    data_train = DataLoader(transformed_rgb_dataset, batch_size=batchSize, shuffle=True,
            num_workers=4 ,drop_last=True)
    data_valid = DataLoader(transformed_rgb_dataset_test, batch_size=batchSize, shuffle=True,
            num_workers=4, drop_last=True)

    print('Lenght of train_data:', len(data_train))
    iValid = int(round(len(data_train) / batchSize))/2 + 1

    # Model preparation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net()
    net = nn.DataParallel(net)
    net.to(device)

    criterion1 = nn.BCELoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)

    modelname  = 'model_Test9_ep5'
    ModelSavePath = '/raid/daulet/rgb_resubmit/saved_models/' + modelname + '.pth'

    # train the network
    for epoch in range(50):  # loop over the dataset multiple times
        print('epoch', epoch)
        running_loss = 0.0
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
            
            output2a = output2[:, 0:3]
            output2b = output2[:, 3:]        
            loss1 = criterion1(output1, labels[:, 0])
            loss2 = criterion2(output2a, labels[:, 1:4])
            loss3 = criterion2(output2b, labels[:, 4:])

            loss = loss1 + loss2 + loss3

            loss.mean().backward()
            optimizer.step()
           
            # print statistics
            if i % 2000  == 1999:
                print('i== ', i)
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
                    labels = labels.reshape((batchSize, 8)).to(device)

                    output_test1, output_test2 = net(inputs)
                    
                    loss1 = criterion1(output_test1, labels[:, 0])
                    loss2 = criterion2(output_test2, labels[:, 1:])
                    loss = loss1 + loss2 + loss3

                    avg_loss = avg_loss + loss.mean().item()

                    if count_test > 20:
                        break
                
                avg_loss = avg_loss / count_test
                print("RMSE Loss: {}, average RMSE: {} ".format(loss.mean().item(), avg_loss))

    # save the model
    torch.save(net.module.state_dict(), ModelSavePath3)


if __name__ == '__main__':
    main()


