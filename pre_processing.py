import pandas as pd
import sklearn, sklearn.utils
import torch
import os
import json
    
import numpy as np


from tqdm import tqdm

import torch.utils


import librosa
import functools
import operator
import mmap


#%% fssdfdf

# csv_file = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.meta/test_post_competition.csv'
# path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.audio_test/'

# frame = pd.read_csv(csv_file)

# frame = frame.sample(frac=0.15, replace=True, random_state=1).reset_index()
# frame = frame.drop(['index'], axis=1)

# frame.to_excel('testing_mini_dataset.xlsx')

#%%

csv_file = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/-=2020=-/graduation_project/data_stuff/mini_dataset/train_mini_dataset.csv'

dataset_frame = pd.read_csv(csv_file)

path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.audio_train/'


mmap_path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/-=2020=-/graduation_project/data_stuff/mmaps/'

labels = dataset_frame.label.unique()
numeric_labels = np.arange(len(labels))
#indexes = labels.index.tolist()
dictionary = dict(zip(labels, numeric_labels))

dataset_frame['text_label'] = dataset_frame.label       
dataset_frame.label = [dictionary[item] for item in dataset_frame.label] 




all_samples = []
metadata_array = []
spectrogram_array = []

time_window = 140

n_fft = 512
hop_length = 256

sample_count = 0

window_length = 50
overlap_length = 24

n_mels = 64
    

fnames = dataset_frame['fname']





# def get_class_for_file(path_wav):
#     class_idx = 0
#     label_name = 'guitar'
#     return (class_idx, label_name)

def get_spectra_frames(path_wav):
    
    S, sr = librosa.load(path + f, sr=44100)
    S = librosa.util.normalize(S)
    
    # = librosa.resample(S, sr, 16000)
           # hop_length = 512
    
#    category = dataset_frame.loc[dataset_frame.fname == f, 'label']
    
    split_points = librosa.effects.split(S, top_db=160, frame_length=n_fft, hop_length=hop_length)
    S_cleaned = []
 
    for piece in split_points:
 
        S_cleaned.append(S[piece[0]:piece[1]])
 
    
    S = np.array(functools.reduce(operator.iconcat, S_cleaned, []))

    
    S = librosa.stft(S, n_fft=n_fft, hop_length=hop_length)
 
    #S = librosa.feature.mfcc(S, sr=44100, n_mfcc=30)
    
    #S = librosa.feature.melspectrogram(S, sr=sr, n_mels=n_mels,fmax=20000)           
    
    
    S = np.log(abs(S) + 1e-20)
    
    
    S_max = S.max()
    S_min = S.min()
    
    S -=S_min
    S /= (S_max - S_min) # 0..1

    result = []
    
    for idx in range(0, S.shape[1]-window_length, overlap_length):
        
        result.append(S[:, idx : idx+window_length])
    
    return result


for f in tqdm(fnames):
   sample_count += len(get_spectra_frames(f))
   frames = np.array(get_spectra_frames(f))
   
   


   
#%% dsfdsfds

    
shape_memmap = (sample_count, hop_length+1, window_length)

mmap_name = 'train_mini_data'

mmap = np.memmap(mmap_path + mmap_name + '.mmap', dtype=np.float32, mode='w+', shape=shape_memmap)

json_desc = {}
json_desc['shape_memmap'] = list(shape_memmap)
json_desc['class_idxs'] = []
json_desc['class_labels'] = []

idx_mmap = 0

for f in tqdm(fnames):
    
    frames = np.array(get_spectra_frames(f))

    mmap[idx_mmap:idx_mmap+len(frames),:,:] = frames
    idx_mmap += len(frames)
    
    class_idx = int(dataset_frame.loc[dataset_frame['fname'] == f,'label'].iloc[0])
    label_name = dataset_frame.loc[dataset_frame['fname'] == f,'text_label'].iloc[0]
    
    json_desc['class_idxs'] += [class_idx] * len(frames)
    json_desc['class_labels'] += [label_name] * len(frames)
    

mmap.flush()

with open(mmap_path + mmap_name + '.json', 'w') as fp:
    json.dump(json_desc, fp)
    

        
#%%    
    

def get_spectra_frames2(path_wav):
    
    S, sr = librosa.load(path + f, sr=44100)
    S = librosa.util.normalize(S)
    
    # = librosa.resample(S, sr, 16000)
           # hop_length = 512
    
#    category = dataset_frame.loc[dataset_frame.fname == f, 'label']
    
    split_points = librosa.effects.split(S, top_db=60, frame_length=n_fft, hop_length=hop_length)
    S_cleaned = []
 
    for piece in split_points:
 
        S_cleaned.append(S[piece[0]:piece[1]])
 
    
    S = np.array(functools.reduce(operator.iconcat, S_cleaned, []))

    
    S = librosa.stft(S, n_fft=n_fft, hop_length=hop_length)
 
    #S = librosa.feature.mfcc(S, sr=44100, n_mfcc=30)
    
    #S = librosa.feature.melspectrogram(S, sr=sr, n_mels=n_mels,fmax=20000)           
    
    
    S = np.log(abs(S) + 1e-20)
    
    
    S_max = S.max()
    S_min = S.min()
    
    S -=S_min
    S /= (S_max - S_min) # 0..1

    result = []
    
    for idx in range(0, S.shape[1]-window_length, overlap_length):
        
        result.append(S[:, idx : idx+window_length])
    
    return result


   
frames = np.array(get_spectra_frames2(f))

#%%

     