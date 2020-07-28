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



def get_output_size(conv2d: nn.Conv2d, input_size):
    
    output_size = (input_size - conv2d.kernel_size[0] + 2*conv2d.padding[0]) / conv2d.stride[0] + 1
    
    return output_size

class ResNetBlock(nn.Module):
    
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        
       
        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                out_channels=in_channels,
                padding=1,
                stride=1,
                kernel_size = kernel_size
            ),
            
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(
                in_channels,
                out_channels=in_channels,
                padding=1,
                stride=1,
                kernel_size = kernel_size
            ),
            
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(
                in_channels,
                out_channels=in_channels,
                padding=1,
                stride=1,
                kernel_size = kernel_size
            ),
            
            nn.Dropout(0.1),
            nn.BatchNorm2d(num_features=in_channels)
  
            
            
            
        
        )
        
    def forward(self, x):
        out = self.convs.forward(x)
        out = out + x
        out = F.relu(out)
        
        #out =  F.batch_norm(out, running_mean=out.running_mean, running_var=x.running_mean, training=True, momentum=0.9)
        
        return out
    
class ResNetBottleNeck(nn.Module):
    
    def __init__(self, in_channels, out_channels,kernel_size):
        super().__init__()
        
        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                out_channels=in_channels,
                padding=1,
                stride=1,
                kernel_size = kernel_size
            ),
            
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(
                in_channels,
                out_channels=in_channels,
                padding=1,
                stride=1,
                kernel_size = kernel_size
            ),
            
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(
                in_channels,
                out_channels=out_channels,
                padding=1,
                stride=1,
                kernel_size = kernel_size
            ),
            
            #nn.Dropout(0.1),
            nn.BatchNorm2d(num_features=out_channels)

            
         
            
        
        )
        
        self.conv_skip = torch.nn.Conv2d(
            in_channels,
            out_channels=out_channels,
            padding = 0,
            stride=1,
            kernel_size=1      
            )
        
         
        
        
    def forward(self, x):
        out = self.convs.forward(x)
        out_skip = self.conv_skip.forward(x)
        out = out + out_skip
        out = F.relu(out)
        
        
        return out
    
    
class Model(nn.Module):
    
    def __init__(self, args):
        super(Model, self).__init__()
        
        #input_size = 28 #W, H
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, 
                kernel_size = args.kernel_size, 
                padding = args.padding, stride=args.stride),
            #nn.Dropout(0.1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
      
        )
        
        self.resBlock1 = ResNetBlock(in_channels=16, kernel_size = 3)
        self.resBlock2 = ResNetBottleNeck(in_channels=16, out_channels = 24, kernel_size = 3)
        
        #input_size = get_output_size(next(iter(self.layer1.children())), input_size)
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=64,
                kernel_size=args.kernel_size,
                padding=args.padding, stride=args.stride),
            #nn.Dropout(0.1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1)
            
        )
        
        #input_size = get_output_size(next(iter(self.layer2.children())), input_size)

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, 
                kernel_size=round(args.kernel_size/2),
                padding=args.padding, stride=args.stride),
            #nn.Dropout(0.1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1)
            #torch.nn.Dropout(0.1)
        )
        
        #input_size = get_output_size(next(iter(self.layer3.children())), input_size)
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=round(args.kernel_size/3),
                      padding=args.padding, stride=args.stride),
            #nn.Dropout(0.1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1)
            #torch.nn.Dropout(0.1)
        )
        

     
        
        self.lin_layer1 = nn.Sequential(
            nn.Linear(in_features=256,out_features=128),
            #nn.Dropout(0.1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()

        )
        
        self.lin_layer2 = nn.Sequential(
            nn.Linear(in_features=128,out_features=args.classes_amount),
            #nn.Dropout(0.1),
            nn.BatchNorm1d(num_features=args.classes_amount),
            nn.ReLU()
            #torch.nn.Dropout(0.1)
        )
        
        
        #input_size = get_output_size(next(iter(self.layer4.children())), input_size)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        
        #in_features=24*round(input_size)**2,
        
    def forward(self, x):
        out = self.layer1.forward(x)
        
        out = self.resBlock1.forward(out)
        out = self.resBlock2.forward(out)
        
        out = self.layer2.forward(out)
        out = self.layer3.forward(out)
        out = self.layer4.forward(out)
        

        
        

        
        out = self.adaptive_pool(out)
        
        out = out.view(out.size(0), -1)

        out = self.lin_layer1.forward(out)
        out = self.lin_layer2.forward(out)
    
        
        out = torch.softmax(out, dim=1)
        
        return out