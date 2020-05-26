from __future__ import print_function, division



import datetime

from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils

# import os

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

import torchnet as tnt

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import sklearn.utils

import warnings

from data_class import fsd_dataset

warnings.filterwarnings("ignore")

plt.ion()  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#%%

parser = argparse.ArgumentParser(description='Self-made audio dataset example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--model", type=str, default='model_v6_adaptive', help="model:model_v")


parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')

parser.add_argument('--batchsize_train', type=int, default=60,
                    help='input batch size for training')

parser.add_argument(
    '--batchsize_test', type=int, default=60,
    help='Steps per epoch during validation')

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



    
    
train_dataset = fsd_dataset(csv_file = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/-=2020=-/graduation_project/data_stuff/mini_dataset/instruments.csv',
                               path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/-=2020=-/graduation_project/data_stuff/mini_dataset/wavfiles/',
                               train = True)


test_dataset = fsd_dataset(csv_file = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/-=2020=-/graduation_project/data_stuff/mini_dataset/instruments.csv',
                               path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/-=2020=-/graduation_project/data_stuff/mini_dataset/wavfiles/',
                               train = False)

train_loader = torch.utils.data.DataLoader(train_dataset,
                            shuffle=True,
                            batch_size = args.batchsize_train)

test_loader = torch.utils.data.DataLoader(test_dataset,
                            shuffle=True,
                            batch_size = args.batchsize_test)



#%%

#torch.manual_seed(1234)



Model = getattr(__import__(f'models_4_audio.{args.model}', fromlist=['Model']), 'Model')
net = Model(args)
net.to(device)



#learning_rate =
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)


meter_loss = tnt.meter.AverageValueMeter()
classerr = tnt.meter.ClassErrorMeter(accuracy=True)
confusion_meter = tnt.meter.ConfusionMeter(10, normalized=True) # 10 means number of clases? 

def reset_meters():
    classerr.reset()
    meter_loss.reset()
    confusion_meter.reset()
    
    
def write_report(var_dict, string):
    
    path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/-=2020=-/graduation_project/ConvNets_audioClassification/reports_out/'
    
    postfix = string
    this_time = str(datetime.datetime.now().time()).replace(':','-').replace('.','-')
    this_date = str(datetime.datetime.now().date())
    todays_date = this_date + '_time_'  + this_time[:-7] + '_' + str(args.model)
    
    attributes = list(var_dict.keys())
    index = var_dict['iterationz']
    
    lst = []
    
    for key in var_dict:
    
        lst.append(var_dict[key])
      
    
    transposed = list(map(list, zip(*lst)))
    
    df = pd.DataFrame(transposed ,index=index, columns=attributes)  
    df = df.drop(columns=['iterationz'],axis=0)
      
    df.to_excel(f'output_ver_{todays_date}_{postfix}.xlsx')
    
    
    
    


number_of_epochs = args.epochs



var_dict = {'iterationz': [],
            'train_loss': [],
            'test_loss': [],
            'train_accuracies': [],
            'test_accuracies': [],
            'train_f1_scores': [],
            'test_f1_scores': []
            }

meters = {
            'train_loss': tnt.meter.AverageValueMeter(),
            'test_loss': tnt.meter.AverageValueMeter(),
            
            'train_APMeter': tnt.meter.APMeter(),
            'test_APMeter': tnt.meter.APMeter(),
            
            'train_mAPeter': tnt.meter.mAPMeter(),
            'test_mAPeter': tnt.meter.mAPMeter(),

            'train_confusion': tnt.meter.ConfusionMeter(10, normalized=True), # 10 means number of clases? ,
            'test_confusion': tnt.meter.ConfusionMeter(10, normalized=True)     
            }


counter = 0
stage = ''



for epoch in range(number_of_epochs):
    
    for key in meters.keys():
        meters[key].reset()
        
              
    counter += 1
    print(f'Epoch #{counter} started')
    
    
    iter_epoch = 0
    
    for loader in [train_loader, test_loader]:
        
        if loader == train_loader:
            stage = 'train'
        else:
            stage = 'test'
            
        var_dict_epoch = {
            'iterationz': [] ,
            'loss_epoch': [] ,
            'accuracy_epoch': [] ,
            'f1_epoch': []
            
            }    
        
        # counter_epoch = []
        # loss_epoch = []
        # accuracy_epoch = []
        # f1_epoch = []
        
        helper = 0
        
        for batch in loader:
            
            helper += 1
             
            iter_epoch += 1
            
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
            var_dict_epoch['loss_epoch'].append(loss.item())

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
            
            var_dict_epoch['accuracy_epoch'].append(accuracy)
            var_dict_epoch['f1_epoch'].append(f1.item())
            var_dict_epoch['iterationz'].append(iter_epoch)
           
            
            
            if not (helper % 1):
                print(f"Iteration: {helper}, Loss: {loss.item()}, Accuracy: {accuracy*100}%")
            
            
            
            if loader == train_loader:
                
                loss.backward()
                optimizer.step()
                
        
                
        if stage == 'train':
   
           var_dict[f'{stage}_loss'].append(np.average(var_dict_epoch['loss_epoch']))
           var_dict[f'{stage}_accuracies'].append(np.average(var_dict_epoch['accuracy_epoch']))
           var_dict[f'{stage}_f1_scores'].append(np.average(var_dict_epoch['f1_epoch']))
           
           #meters[f'{stage}_loss'].add(np.median(tonumpy(var_dict_epoch['loss'])))
             

                                       
        else:
           

            var_dict[f'{stage}_loss'].append(np.average(var_dict_epoch['loss_epoch']))
            var_dict[f'{stage}_accuracies'].append(np.average(var_dict_epoch['accuracy_epoch']))
            var_dict[f'{stage}_f1_scores'].append(np.average(var_dict_epoch['f1_epoch']))
 
            #meters[f'{stage}_loss'].add(np.median(tonumpy(var_dict_epoch['loss'])))

        write_report(var_dict_epoch, f'per_epoch_{stage}')
            
        
        
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

write_report(var_dict, 'general')



