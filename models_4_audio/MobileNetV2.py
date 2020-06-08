# from __future__ import print_function, division

# from torch.utils.data import Dataset, DataLoader
# import torchvision
# from torchvision import transforms, utils

# import os

# from skimage import io, transform
# import numpy as np
# import matplotlib as plt

# import argparse
# import os



# import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable

# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# import sklearn.utils
import torchvision
from torchvision.models import mobilenet_v2


# class Model(nn.Module):
    
#     def __init__(self, args):
#         super(Model, self).__init__()

#         self.backbone_model: DenseNet = torchvision.models.densenet121(pretrained=True)

#         self.fc1 = nn.Linear(in_features=self.backbone_model.classifier.in_features, out_features=args.classes_amount) # muzikas instrumentu klases

#     def forward(self, x):

#         out = self.backbone_model.features.forward(x)
#         out = F.adaptive_avg_pool2d(out, output_size=(1,1))
#         out = out.view(out.size(0), -1)
#         out = self.fc1.forward(out)

#         out = torch.softmax(out, dim=1)
        
#         return out


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        
        self.layer1 = nn.Sequential(
            
            nn.Conv2d(in_channels = 1, out_channels = 16, 
                kernel_size = 1, 
                padding = args.padding, stride=args.stride),
            nn.Dropout(0.1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            
        )
        
        self.layer2 = nn.Sequential(
            
            nn.Conv2d(in_channels = 16, out_channels = 32, 
                kernel_size = 1, 
                padding = args.padding, stride=args.stride),
            nn.Dropout(0.1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            
        )

        self.backbone_model = torchvision.models.mobilenet_v2(pretrained=True)
        
                

        #self.fc1 = nn.Linear(in_features=self.backbone_model.fc.in_features, out_features=args.classes_amount)
        

        


    def forward(self, x):
        
        x = x.repeat(1,3,1,1)
        out = self.backbone_model.forward(x)

        out = torch.softmax(out, dim=1)

        return out