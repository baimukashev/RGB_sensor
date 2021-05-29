These are guidelines to run the training and testing scripts for DL model. 

Prerequisites:
1. Pytorch 
2. Pandas
3. Numpy
4. Matplotlib

How to create conda environment and install the packages:
1. Download and install Anaconda for Python 3 from https://www.anaconda.com/products/individual 
2. Create conda environment: ```conda create -n rgb_env python=3.7```
3. Activate the conda environment: ```conda activate rgb_env```
4. Install packages:
  - Pytorch: ```conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch```
  - Other: ```conda install -c anaconda pandas, numpy, matplotlib``` 

How to run the scripts:
1. Copy the images dataset to path ```data/images/```
2. To train the model run the command ```python train.py```
3. To train the model run the command ```python test.py```
