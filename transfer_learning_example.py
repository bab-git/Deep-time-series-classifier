#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:08:41 2019

@author: bhossein
Transfer learning example on caltech101
"""

from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch
from torchvision import transforms, datasets, models

from sklearn.model_selection import train_test_split
import os
import numpy as np
from shutil import copyfile

import matplotlib.pyplot as plt

from torchsummary import summary

# Image manipulations
from PIL import Image
os.chdir('/home/bhossein/BMBF project/code_resources/Transfer learning')

import seaborn as sns

import pandas as pd

from torchvision import transforms

# %%============ Parameters
# Location of data
datadir = '101_ObjectCategories/'
traindir = datadir + 'train/'
validdir = datadir + 'valid/'
testdir = datadir + 'test/'

#main_dr = datadir + 'all/'
#list_class = os.listdir(main_dr)
#test_size = 0.5
#for class_dr in list_class:
#    print(class_dr)
#    class_dr = class_dr + '/'
#    class_pth = main_dr + class_dr+'/'
#    
#    list_iamges = os.listdir(class_pth)
#    
#    idx = np.arange(len(list_iamges))
#    trn_idx, tst_idx = train_test_split(idx, test_size = test_size)
#    val_idx, tst_idx= train_test_split(tst_idx, test_size = test_size)
#    
#        
#    os.mkdir(traindir+class_dr)
#    tr_class_path = traindir+class_dr
#    for idx in trn_idx:
#        copyfile(class_pth+list_iamges[idx], tr_class_path+list_iamges[idx])
#        
#    
#    os.mkdir(testdir+class_dr)
#    tst_class_path = testdir+class_dr
#    for idx in tst_idx:
#        copyfile(class_pth+list_iamges[idx], tst_class_path+list_iamges[idx])
#
#    os.mkdir(validdir+class_dr)
#    val_class_path = validdir+class_dr
#    for idx in val_idx:
#        copyfile(class_pth+list_iamges[idx], val_class_path+list_iamges[idx])
                
    
save_file_name = 'vgg16-transfer-4.pt'
checkpoint_path = 'vgg16-transfer-4.pth'


# Change to fit hardware
batch_size = 128


# Whether to train on a gpu
train_on_gpu = cuda.is_available()
print('Train on gpu: '+ str(train_on_gpu))

# Number of gpus
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# %%
# Empty lists
categories = []
img_categories = []
n_train = []
n_valid = []
n_test = []
hs = []
ws = []

# Iterate through each category
for d in os.listdir(traindir):
    categories.append(d)
    # Number of each image
    train_imgs = os.listdir(traindir + d)
    valid_imgs = os.listdir(validdir + d)
    test_imgs = os.listdir(testdir + d)
    n_train.append(len(train_imgs))
    n_valid.append(len(valid_imgs))
    n_test.append(len(test_imgs))
    
    # Find stats for train images
    for i in train_imgs:
        img_categories.append(d)
        img = Image.open(traindir + d + '/' + i)
        img_array = np.array(img)
        # Shape
        hs.append(img_array.shape[0])
        ws.append(img_array.shape[1])
    
# Dataframe of categories
cat_df = pd.DataFrame({'category': categories,
                       'n_train': n_train,
                       'n_valid': n_valid, 'n_test': n_test}).\
    sort_values('category')

# Dataframe of training images
image_df = pd.DataFrame({
    'category': img_categories,
    'height': hs,
    'width': ws
})
    
cat_df.sort_values('n_train', ascending=False, inplace=True)
cat_df.head()

cat_df.set_index('category')['n_train'].plot.bar(
    color='r', figsize=(20, 6))
plt.xticks(rotation=80)
plt.ylabel('Count')
plt.title('Training Images by Category')

# Only top 50 categories
cat_df.set_index('category').iloc[:50]['n_train'].plot.bar(
    color='r', figsize=(20, 6))
plt.xticks(rotation=80)
plt.ylabel('Count')
plt.title('Training Images by Category')

img_dsc = image_df.groupby('category').describe()
img_dsc.head()
sns.kdeplot(img_dsc['height']['mean'], label='Average Height')


def imshow(image):
    """Display image"""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# Example image
x = Image.open(traindir + 'ewer/image_0004.jpg')
np.array(x).shape
imshow(x)


# %%================ Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'val':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Test does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# %%
ex_img = Image.open(traindir+'elephant/image_0024.jpg')
imshow(ex_img)
    
t = image_transforms['train']    
plt.figure(figsize=(24, 24))

for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
#    imshow(t(ex_img).permute(1,2,0))
    img = t(ex_img)
    img = img.numpy().transpose((1, 2, 0))
    
    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    
    ax.imshow(img)
plt.tight_layout()    

#%% data loader
# Datasets from each folder
data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'val':
    datasets.ImageFolder(root=validdir, transform=image_transforms['val']),
    'test':
    datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
}

# Dataloader iterators
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
}


trainiter = iter(dataloaders['train'])
features, labels = next(trainiter)
features.shape, labels.shape

n_classes = len(cat_df)
print('There are %d different classes.' % (n_classes))

len(data['train'].classes)

# %% pretatined network

model = models.vgg16(pretrained=True)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False
    
n_inputs = model.classifier[6].in_features

model.classifier[6] = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, n_classes), nn.LogSoftmax(dim = 1),
        )

total_params = sum(p.numel() for p in model.parameters())
print(str(total_params)+ ': total parameters.')
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(str(total_trainable_params)+': training parameters.')

# %% to GPU
model = model.to(device)
model2 = model.to('cpu')
#from torchsummary import summary
summary(model2, input_size=(3, 224, 224), batch_size=batch_size, device = 'cpu')

#Mapping of Classes to Indexes¶

model.class_to_idx = data['train'].class_to_idx

model.idx_to_class = {
        idx: class_ for class_, idx in model.class_to_idx.items()
        }

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

for p in optimizer.param_groups[0]['params']:
    if p.requires_grad:
        print(p.shape)

# %%Training
        