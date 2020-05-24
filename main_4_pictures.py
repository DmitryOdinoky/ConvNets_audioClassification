from __future__ import print_function, division

import datetime

from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils

# import os
import torch
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os



import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import sklearn.utils

import warnings
warnings.filterwarnings("ignore")

plt.ion()  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#%%

parser = argparse.ArgumentParser(description='Keras Fashion MNIST Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--model", type=str, default='model_v4_adaptive', help="blahblah")


parser.add_argument('--log-dir', default='./logs',
                    help='blahblah')

parser.add_argument('--batchsize', type=int, default=256,
                    help='input batch size for training')

parser.add_argument(
    '--batch_valid', type=int, default=256,
    help='blahblah')

parser.add_argument('--epochs', type=int, default=3,
                    help='number of epochs to train')

parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='learning rate for a single GPU')

parser.add_argument('--kernel_size', type=int, default=3,
                    help='kernel size for convolution')

parser.add_argument('--padding', type=int, default=1,
                    help='padding')

parser.add_argument('--stride', type=int, default=2,
                    help='stride')

# parser.add_argument(
#     '--decay', type=float, default=0.8,
#     help='LR decay every 10 epochs')
# parser.add_argument('--warmup-epochs', type=float, default=5,
#                     help='number of warmup epochs')
# parser.add_argument('--momentum', type=float, default=0.9,
#                     help='SGD momentum')
# parser.add_argument('--wd', type=float, default=0.000005,
#                     help='weight decay')

args = parser.parse_args()

# Checkpoints will be written in the log directory.
args.checkpoint_format = os.path.join(args.log_dir, 'checkpoint-{epoch}.h5')


#%%

this_time = str(datetime.datetime.now().time()).replace(':','-').replace('.','-')
this_date = str(datetime.datetime.now().date())
todays_date = this_date + '_'  + this_time[:-7] + '_' + str(args.model)

#%%
    
    
train_dataset = torchvision.datasets.FashionMNIST(
    "./data", download=True, 
    transform = transforms.Compose([
        transforms.ToTensor()
        
        ])
)


test_dataset = torchvision.datasets.FashionMNIST(
    "./data", download=True, train=False, transform = transforms.Compose([
        transforms.ToTensor()
        ])
    
)  


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    batch_size = args.batchsize)


test_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=args.batch_valid)



#%%

#torch.manual_seed(1234)



Model = getattr(__import__(f'models_4_pics.{args.model}', fromlist=['Model']), 'Model')
net = Model(args)
net.to(device)



#learning_rate =
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)


number_of_epochs = args.epochs

#np.random.seed(32) # set seed value so that the results are reproduceable

var_dict = {'iterationz': [],
            'train_loss': [],
            'test_loss': [],
            'train_accuracies': [],
            'test_accuracies': [],
            'train_f1_scores': [],
            'test_f1_scores': []
            }


counter = 0
stage = ''

for epoch in range(number_of_epochs):
    
    counter += 1
    print(f'Epoch #{counter} started')
    
    for loader in [train_loader, test_loader]:
        
        if loader == train_loader:
            stage = 'train'
        else:
            stage = 'test'
        
        loss_epoch = []
        accuracy_epoch = []
        f1_epoch = []
        
        helper = 0
        
        for batch in loader:
            
            helper += 1
            # print(helper)
            
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            train_X = images
            train_y = labels
            
            optimizer.zero_grad()
            y_prim = net.forward(train_X)
            
            class_count = y_prim.size(1)          
            tmp = torch.arange(class_count)          
            
            
            y = (train_y.unsqueeze(dim=1) == tmp).float()
            
            
            
            loss = torch.mean(-y*torch.log(y_prim))
            
            #loss = loss(y_prim.item(), labels)
            loss_epoch.append(loss.item())

            _, predict_y = torch.max(y_prim, 1)
            
            correct = 0
            total = 0
            
            for i in range(len(images)):
                act_label = torch.argmax(y_prim[i]) # act_label = 1 (index)
                pred_label = torch.argmax(y[i]) # pred_label = 1 (index)
            
                if(act_label == pred_label):
                    correct += 1
                total += 1
            
            accuracy = correct/total
            
            #accuracy = accuracy_score(train_y.data, predict_y.data)
            
            
            f1 = sklearn.metrics.f1_score(train_y.data, predict_y.data, average='macro')
            
            accuracy_epoch.append(accuracy)
            f1_epoch.append(f1.item())
            
            if not (helper % 10):
                print(f"Iteration: {helper}, Loss: {loss.item()}, Accuracy: {accuracy*100}%")
            
            
            
            if loader == train_loader:
                
                loss.backward()
                optimizer.step()
                
        if stage == 'train':
   
           var_dict[f'{stage}_loss'].append(np.average(loss_epoch))
           var_dict[f'{stage}_accuracies'].append(np.average(accuracy_epoch))
           var_dict[f'{stage}_f1_scores'].append(np.average(f1_epoch))
           
           
            
        else:
           

            var_dict[f'{stage}_loss'].append(np.average(loss_epoch))
            var_dict[f'{stage}_accuracies'].append(np.average(accuracy_epoch))
            var_dict[f'{stage}_f1_scores'].append(np.average(f1_epoch))
        
    var_dict['iterationz'].append(counter)
    


#%%    
    
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=False)

ax1.set_title('1: Loss.  2: F1. 3: Accuracy.')

sc1 = ax1.scatter(var_dict['iterationz'], var_dict['train_loss'])
sc2 = ax1.scatter(var_dict['iterationz'],  var_dict['test_loss'])


sc3 = ax2.scatter(var_dict['iterationz'], var_dict['train_f1_scores'])
sc4 = ax2.scatter(var_dict['iterationz'], var_dict['test_f1_scores'])

sc5 = ax3.scatter(var_dict['iterationz'], var_dict['train_accuracies'])
sc6 = ax3.scatter(var_dict['iterationz'], var_dict['test_accuracies'])


ax1.set_xticks([])
ax2.set_xticks([])


ax1.legend((sc1, sc2), ('test', 'train'), loc='upper right', shadow=True)

#%%

attributes = list(var_dict.keys())
index = var_dict['iterationz']

lst = []

for key in var_dict:
    
    lst.append(var_dict[key])
      

transposed = list(map(list, zip(*lst)))
        
df = pd.DataFrame(transposed ,index=index, columns=attributes)  
df = df.drop(columns=['iterationz'],axis=0)
  
df.to_excel(f'output_ver_{todays_date}.xlsx')






#%%


