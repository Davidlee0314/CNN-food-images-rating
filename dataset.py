import os
import math
from PIL import Image

from torch.utils.data import Dataset
from natsort import natsort
import pandas as pd

from configs import configs

class MeanDataset(Dataset):
    def __init__(self, root, train, transform=None):
        self.root = root
        
        # train 0:tr 1:val 2:pb
        self.train = train

        if(self.train == 0):
            self.transform = transform['tr']
            self.total_imgs, self.y = self.get_data()
        elif(self.train == 1):
            self.transform = transform['val']
            self.total_imgs, self.y = self.get_data()
        elif(self.train == 2):
            self.transform = transform['val']
            self.total_imgs = self.get_data()
        self.transform_imgnet = transform['imagenet']

    def get_data(self):
        data_list = []
        data_list = os.listdir(self.root)
        data_list = natsort.natsorted(data_list)
        data_num = int(data_list[-1].split('.')[0]) + 1
        
        df = pd.read_csv(configs.train_label_dir, sep='\t', header=None)
        df.columns = ['mean', 'std', 'j1', 'j2', 'j3']
        label = df.loc[:data_num - 1, 'mean'].to_numpy().reshape(-1, 1)
        
        # use original data
        train_data_num = math.floor(len(data_list) * 0.8)
        if(self.train == 0):
            # return data_list[:train_data_num], label[:train_data_num]
            return data_list, label
        elif(self.train == 1):
            return data_list[train_data_num:], label[train_data_num:]
        elif(self.train == 2):
            return data_list

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.root, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        if((self.train == 0) or (self.train == 1)):
            return tensor_image, self.transform_imgnet(image), self.y[idx, :]
        else:
            return tensor_image, self.transform_imgnet(image)

class StdDataset(Dataset):
    def __init__(self, root, train, transform=None):
        self.root = root
        
        # train 0:tr 1:val 2:pb
        self.train = train

        if(self.train == 0):
            self.transform = transform['tr']
            self.total_imgs, self.y = self.get_data()
        elif(self.train == 1):
            self.transform = transform['val']
            self.total_imgs, self.y = self.get_data()
        elif(self.train == 2):
            self.transform = transform['val']
            self.total_imgs = self.get_data()

    def get_data(self):
        data_list = []
        data_list = os.listdir(self.root)
        data_list = natsort.natsorted(data_list)
        data_num = int(data_list[-1].split('.')[0]) + 1
        
        df = pd.read_csv(configs.train_label_dir, sep='\t', header=None)
        df.columns = ['mean', 'std', 'j1', 'j2', 'j3']
        label = df.loc[:data_num - 1, 'std'].to_numpy().reshape(-1, 1)
        
        # use original data
        train_data_num = math.floor(len(data_list) * 0.8)
        if(self.train == 0):
            # return data_list[:train_data_num], label[:train_data_num]
            return data_list, label
        elif(self.train == 1):
            return data_list[train_data_num:], label[train_data_num:]
        elif(self.train == 2):
            return data_list

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.root, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        if((self.train == 0) or (self.train == 1)):
            return tensor_image, self.y[idx, :]
        else:
            return tensor_image
