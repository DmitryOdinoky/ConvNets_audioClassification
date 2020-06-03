import pandas as pd
import sklearn, sklearn.utils
import torch
import os


    
import numpy as np
import mmap

from tqdm import tqdm


import librosa

import functools
import operator

#%%

csv_file = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/-=2020=-/graduation_project/data_stuff/mini_dataset/testing_mini_dataset.csv'
frame = pd.read_csv(csv_file)
path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.audio_test/'
target_path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/-=2020=-/graduation_project/data_stuff/mini_dataset/'

fnames = frame['fname']
hop_length = 512
n_fft = 1023

os.listdir(target_path)

for f in fnames:
   
    S, sr = librosa.load(path + f, sr=44100)

    
    S, index = librosa.effects.trim(S, top_db=30, frame_length=n_fft, hop_length=hop_length)
    
    
    
    #S = librosa.stft(S, n_fft=n_fft, hop_length=hop_length)
    
    #S = librosa.feature.mfcc(S, sr=44100, n_mfcc=20)
    
    S = librosa.feature.melspectrogram(S, sr=sr, n_mels=24,fmax=20000)           
    #S = librosa.power_to_db(abs(S),ref=np.max,top_db=120)
    
    #S = librosa.amplitude_to_db(abs(S),ref=np.max,top_db=120)
    
    S = abs(S)
    
    S_max = S.max()
    S_min = S.min()
    
    S -= S_min
    S /= (S_max - S_min) # 0..1
       
    
    
    # X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
    # X = librosa.feature.melspectrogram(x, sr=fs, n_mels=128,fmax=20000)
    # S = librosa.amplitude_to_db(abs(X),ref=np.max,top_db=120)
    
    
    
    librosa.output.write_wav(target_path + f, S, sr, norm=False)
        

     
     
     