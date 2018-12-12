"""Put cifar datasets in a less shitty format and save a pandas df of the locations.
PATH: where to store standard format cifar
NEW_PATH: Where to store train, val and dict of pands dfs. in proper format
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd

from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--PATH', type=str)
parser.add_argument('--NEW_PATH', type=str)
parser.add_argument('--DATASET', type=str)
args = parser.parse_args()


def make_data_normal(PATH, NEW_PATH):
    NEW_PATH = Path(NEW_PATH)
    files = {}

    datasets = {}
    dataloaders = {}

    if args.DATASET == 'CIFAR10':
        trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        datasets['train'] = torchvision.datasets.CIFAR10(root=str(PATH), train=True, transform=trans, download=True)
        dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=1, shuffle=True, num_workers=0)
        datasets['val'] = torchvision.datasets.CIFAR10(root=str(PATH), train=False, transform=trans, download=True)
        dataloaders['val'] = torch.utils.data.DataLoader(datasets['val'], batch_size=1, shuffle=False, num_workers=0)

    elif args.DATASET == 'MNIST':
        trans = transforms.Compose([transforms.ToTensor()])
        datasets['train'] = torchvision.datasets.MNIST(root=str(PATH), train=True, transform=trans, download=True)
        dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=1, shuffle=True, num_workers=0)
        datasets['val'] = torchvision.datasets.MNIST(root=str(PATH), train=False, transform=trans, download=True)
        dataloaders['val'] = torch.utils.data.DataLoader(datasets['val'], batch_size=1, shuffle=False, num_workers=0)

    # train set
    SAVE_PATH = NEW_PATH / 'train' 
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    files['train'] = pd.DataFrame()
    
    # validation set
    SAVE_PATH = NEW_PATH / 'val' 
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    files['val'] = pd.DataFrame()
    for i, batch in enumerate(dataloaders['val']):
        inputs, target = batch[0].numpy(), batch[1].numpy()
        image = inputs*255 # need this to save as png without problem
        if args.DATASET == 'CIFAR10':
            image = np.moveaxis(np.squeeze(image), 0, 2)
            im = Image.fromarray(image.astype('uint8')).convert('RGB')
        else:
            im = Image.fromarray(np.squeeze(image).astype('uint8'), 'L')
        loc = str(SAVE_PATH) + '/val_'+str(i)+'.png'
        im.save(loc)
        
        files['val'] = files['val'].append({'path': loc, 'class': int(np.squeeze(target))}, ignore_index=True)
    
    SAVE_PATH = NEW_PATH / 'train' 
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(dataloaders['train']):
        inputs, target = batch[0].numpy(), batch[1].numpy()
        image = inputs*255 # need this to save as png without problem
        if args.DATASET == 'CIFAR10':
            image = np.moveaxis(np.squeeze(image), 0, 2)
            im = Image.fromarray(image.astype('uint8')).convert('RGB')
        else:
            im = Image.fromarray(np.squeeze(image).astype('uint8'), 'L')
        loc = str(SAVE_PATH) + '/train_'+str(i)+'.png'
        im.save(loc)
        
        files['train'] = files['train'].append({'path': loc, 'class': int(np.squeeze(target))}, ignore_index=True)
        
    with open(str(NEW_PATH)+'/files_df.pkl', 'wb') as f:
        pickle.dump(files, f, pickle.HIGHEST_PROTOCOL)

    # Make a 10% sample:
    sample_df ={}
    sample_df['train'] = files['train'].sample(frac=.1)
    sample_df['val'] = files['val'].sample(frac=.1)

    with open(str(NEW_PATH)+'/sample_df.pkl', 'wb') as f:
        pickle.dump(sample_df, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    make_data_normal(args.PATH, args.NEW_PATH)