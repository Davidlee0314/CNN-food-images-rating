import os

import torch 
import torch.nn as nn
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from configs import configs

class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x,1)

def seed_torch(seed=2020):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transform():
    data_transforms = {
        'imagenet': transforms.Compose([
            transforms.Resize(configs.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'tr': transforms.Compose([
            transforms.RandomResizedCrop(configs.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(configs.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def train_mean(dataloader, model, optimizer, criterion):
    model.train() 
    avg_loss = 0.
    for idx, (imgs, imgs_imgnet, y) in enumerate(tqdm(dataloader)):
        imgs_train, imgs_imgnet, y = imgs.to(configs.device), imgs_imgnet.to(configs.device), y.to(configs.device)
        optimizer.zero_grad()
        y_pred = model(imgs_train, imgs_imgnet)
        loss = criterion(y_pred, y.float())
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    return avg_loss

def train_std(dataloader, model, optimizer, criterion):
    model.train() 
    avg_loss = 0.
    for idx, (imgs, y) in enumerate(tqdm(dataloader)):
        imgs_train, y = imgs.to(configs.device), y.to(configs.device)
        optimizer.zero_grad()
        y_pred = model(imgs_train)
        loss = criterion(y_pred, y.float())
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    return avg_loss