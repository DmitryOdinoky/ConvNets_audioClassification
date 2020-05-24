import pandas as pd
import sklearn, sklearn.utils
import torch
    
import numpy as np
import mmap

from tqdm import tqdm


import librosa


class simple_Dataset(object):
    
    def __init__(self, csv_file, batch_size, path, train = True):

        self.if_train = True
        self.batch_size = batch_size
        self.dataset_frame = sklearn.utils.shuffle(pd.read_csv(csv_file))
        self.dataset_frame = self.dataset_frame.reset_index()
        #self.dataset_frame = self.dataset_frame.drop(['index','manually_verified','freesound_id','license'], axis=1)
        self.dataset_frame = self.dataset_frame.drop(['index','usage','freesound_id','license'], axis=1)
      
        self.dataset_frame_mod = self.dataset_frame.copy(deep=True)
        
        self.labels = self.dataset_frame_mod.label.unique()
        self.numeric_labels = np.arange(len(self.labels))
        #self.indexes = self.labels.index.tolist()
        
        dictionary = dict(zip(self.labels, self.numeric_labels))
        
        self.dataset_frame_mod.label = [dictionary[item] for item in self.dataset_frame_mod.label] 
        
        self.metadata_array = []
        self.spectrogram_array = []
      
        
        self.batchez_metadata = []
        self.pseudo_memaps = []
        
        self.path = path
        
      
        
        
        for i in range(0, len(self.dataset_frame_mod), self.batch_size):
            self.batchez_metadata.append(self.dataset_frame_mod[i:i+self.batch_size])


                
    def make_pseudo_memmaps(self):
        
        for file in tqdm(self.dataset_frame['fname']):
           x, sr = librosa.load(self.path + file, sr=44100, mono=True)
           #hop_length = 512
           hop_length = 512
           n_fft = 1023
           category = self.dataset_frame.loc[self.dataset_frame.fname == file, 'label']
           
           S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
           S = librosa.amplitude_to_db(abs(S),ref=np.max,top_db=120)
           S, index = librosa.effects.trim(S,top_db=70, frame_length=1024, hop_length=512)
           
           self.spectrogram_array.append(S[:,0:75])
           # X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
           # X = librosa.feature.melspectrogram(x, sr=fs, n_mels=128,fmax=20000)
           # S = librosa.amplitude_to_db(abs(X),ref=np.max,top_db=120)
           
           self.metadata_array.append({'instrument':f'{category}'})    
        
        for batch in self.batchez_metadata:
            
            mmap_batches = []
            
            for row in batch.T:
  
                mmap_batches.append(self.spectrogram_array[row])
                        
            self.pseudo_memaps.append(np.asarray(mmap_batches))
            
            
            
class fsd_dataset(object):
    
    def __init__(self, csv_file, path, train = True):

        self.if_train = True
      
        self.dataset_frame = pd.read_csv(csv_file)
        
        #self.dataset_frame = self.dataset_frame.drop(['index','manually_verified','freesound_id','license'], axis=1)
        #self.dataset_frame = self.dataset_frame.drop(['index'], axis=1)
      
        #self.dataset_frame_mod = self.dataset_frame.copy(deep=True)
        
        self.labels = self.dataset_frame.label.unique()
        self.numeric_labels = np.arange(len(self.labels))
        #self.indexes = self.labels.index.tolist()
        
        dictionary = dict(zip(self.labels, self.numeric_labels))
        
        self.dataset_frame.label = [dictionary[item] for item in self.dataset_frame.label] 


        
        self.metadata_array = []
        self.spectrogram_array = []
      
        
        self.path = path


                

        
        for file in tqdm(self.dataset_frame['fname']):
           x, sr = librosa.load(self.path + file, sr=44100, mono=True)
           # hop_length = 512
           hop_length = 512
           n_fft = 1023
           category = self.dataset_frame.loc[self.dataset_frame.fname == file, 'label']
           
           S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
           S = librosa.amplitude_to_db(abs(S),ref=np.max,top_db=120)
           
           self.spectrogram_array.append(S[:,0:80])
           # X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
           # X = librosa.feature.melspectrogram(x, sr=fs, n_mels=128,fmax=20000)
           # S = librosa.amplitude_to_db(abs(X),ref=np.max,top_db=120)
           
           self.metadata_array.append(category.iloc[0])   

        
       
            
    

        
    def __len__(self):
        return len(self.dataset_frame)  
    
    def __getitem__(self, idx):
       if torch.is_tensor(idx):
           idx = idx.tolist()
       
    
       x = torch.Tensor(self.spectrogram_array[idx]).unsqueeze(0)
       y = torch.Tensor(np.expand_dims(self.metadata_array[idx], axis=0))
       
       return x, y
    
