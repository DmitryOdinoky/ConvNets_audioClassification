import pandas as pd
import sklearn, sklearn.utils
import torch
    
import numpy as np
import mmap

from tqdm import tqdm


import librosa


import functools
import operator
            
            
            
class fsd_dataset(object):
    
    def __init__(self, csv_file, path, train = True):

        self.if_train = True
      
        self.dataset_frame = pd.read_csv(csv_file).dropna(axis=0, how='all')
        
        #self.dataset_frame = self.dataset_frame.drop(['index','manually_verified','freesound_id','license'], axis=1)
        #self.dataset_frame = self.dataset_frame.drop(['index'], axis=1)
      
        #self.dataset_frame_mod = self.dataset_frame.copy(deep=True)
        
        self.labels = self.dataset_frame.label.unique()
        self.numeric_labels = np.arange(len(self.labels))
        #self.indexes = self.labels.index.tolist()
        dictionary = dict(zip(self.labels, self.numeric_labels))        
        self.dataset_frame.label = [dictionary[item] for item in self.dataset_frame.label] 


        self.all_samples = []
        
        self.metadata_array = []
        self.spectrogram_array = []
      
        
        self.path = path

        #self.time_window = 140
        
        self.hop_length = 256
        self.n_fft = 2048
        
        # self.window_length = 50
        # self.overlap_length = 24
        self.window_length = 200
        self.overlap_length = 180
                
        self.downsample = 16000
        
        self.n_mels = 63
        self.n_mfcc = 63
        
        for file in tqdm(self.dataset_frame['fname']):
           S, sr = librosa.load(self.path + file, sr=44100, mono=True)
           
           S = librosa.resample(S, sr, self.downsample)
           
           sr = self.downsample
           
           #S = librosa.util.normalize(S)
           
           S_max = S.max()
           S_min = S.min()          
           S -=S_min
           S /= (S_max - S_min) # 0..1
          

           category = self.dataset_frame.loc[self.dataset_frame.fname == file, 'label']
           
           split_points = librosa.effects.split(S, top_db=80, frame_length=self.n_fft, hop_length=self.hop_length)
           
           S_cleaned = []
         
           for piece in split_points:
         
               S_cleaned.append(S[piece[0]:piece[1]])
         
            
           S = np.array(functools.reduce(operator.iconcat, S_cleaned, []))
           
           for i in range(0,4):
               S = np.concatenate((S,S),axis=0)
           
  
           
           #S, index = librosa.effects.trim(S, top_db=60, frame_length=self.n_fft, hop_length=self.hop_length)
           #S = librosa.stft(S, n_fft=self.n_fft, hop_length=self.hop_length)
           
           S = librosa.feature.melspectrogram(S, sr=sr, n_mels=self.n_mels,fmax=sr/2)  
           #S = librosa.feature.mfcc(S, sr=sr, n_mfcc=self.n_mfcc)
           
          
           
           #S = librosa.power_to_db(abs(S),ref=np.max,top_db=120)
           
           #S = librosa.amplitude_to_db(abs(S),ref=np.max,top_db=120)
           
           S = np.log(abs(S) + 1e-20)
           
           #S = abs(S)
           
           S_max = S.max()
           S_min = S.min()          
           S -=S_min
           S /= (S_max - S_min) # 0..1
          
           
           
           # X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
           # X = librosa.feature.melspectrogram(x, sr=fs, n_mels=128,fmax=20000)
           # S = librosa.amplitude_to_db(abs(X),ref=np.max,top_db=120)
           
           samples = []
           categories = []
          
           
           for idx in range(0, S.shape[1]-self.window_length, self.overlap_length):
               
               A = S[:, idx : idx+self.window_length]
               
               samples.append(A)    
               categories.append(category.iloc[0])
        
           #samples = functools.reduce(operator.iconcat, samples, [])
           
           
           
           
           self.spectrogram_array.append(samples)
           self.metadata_array.append(categories)
           
           self.all_samples.append((samples,categories))
           
           #self.spectrogram_array.append(S[:,0:80])
           #self.metadata_array.append(category.iloc[0])   
        
        self.spectrogram_array = functools.reduce(operator.iconcat, self.spectrogram_array, [])
        self.metadata_array = functools.reduce(operator.iconcat, self.metadata_array, [])
    
        
    def __len__(self):
        return len(self.spectrogram_array)  
    
    def __getitem__(self, idx):
       if torch.is_tensor(idx):
           idx = idx.tolist()
       
       #spec, class_idx = self.all_samples[idx]
    
       #x = torch.Tensor(self.spectrogram_array[idx]).unsqueeze(0)
       #y = torch.Tensor(np.expand_dims(self.metadata_array[idx], axis=0))
       
       x = torch.Tensor(self.spectrogram_array[idx]).unsqueeze(0)
       y = torch.Tensor(np.expand_dims(self.metadata_array[idx], axis=0))
        
       #x = torch.Tensor(spectrogram_array).transpose(1,2).unsqueeze(1)
       #y = torch.Tensor(np.expand_dims(metadata_array, axis=1))
       
       return x, y








