import torch.nn as nn
import torch
import torchvision
from efficientnet_pytorch import EfficientNet

from utils import Flatten

class Mean_Model(nn.Module):
    def __init__(self):
        super(Mean_Model, self).__init__()
        d = torch.load('./pretrained_models/food-101-train-epoch-10.pth')
        self.body_3 = EfficientNet.from_name("efficientnet-b4")
        self.body_3.load_state_dict(d)

        self.body_1 = torchvision.models.mobilenet_v2(pretrained=False)
        self.body_1 = nn.Sequential(*(list(self.body_1.children())[:-1]))

        self.body_2 = torchvision.models.mobilenet_v2(pretrained=True)
        self.body_2 = nn.Sequential(*(list(self.body_2.children())[:-1]))

        self.mid = nn.Sequential(
            nn.MaxPool2d(7),
            Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(4352, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        
        # freeze model weight
        for name, param in self.body_2.named_parameters():
            param.requires_grad = False
        for name, param in self.body_3.named_parameters():
            param.requires_grad = False


    def forward(self, x_96, x_img):
        x_1 = self.body_1(x_96)
        x_1 = self.mid(x_1)
        x_2 = self.body_2(x_img)
        x_2 = self.mid(x_2)
        x_3 = self.body_3.extract_features(x_96)
        x_3 = self.mid(x_3)
        x = torch.cat((x_1, x_2, x_3), 1)
        x = self.head(x)
        return x


class Std_Model(nn.Module):
    def __init__(self):
        super(Std_Model, self).__init__()

        self.body = torchvision.models.mobilenet_v2(pretrained=True)
        self.body = nn.Sequential(*(list(self.body.children())[:-1]))
        self.head = nn.Sequential(
            nn.MaxPool2d(7),
            Flatten(), 
            nn.Linear(1280, 1)
        )
        
        # freeze model weight
        count = 0
        for name, param in self.body.named_parameters():
            count += 1
            if(count < 100):
                param.requires_grad = False

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x
