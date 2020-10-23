import pandas as pd
import sklearn, sklearn.utils
import torch
import matplotlib.pyplot as plt

from scipy.io import wavfile
from python_speech_features import mfcc
    
import numpy as np
# import mmap

from tqdm import tqdm


import librosa
import librosa.display
import json

import functools
import operator

import os




            
            
class fsd_dataset(object):
    
    def __init__(self, csv_file, path, mmap_path, json_path, train = True):

        self.train = train
      
        self.dataset_frame = pd.read_csv(csv_file).dropna(axis=0, how='all')
        
        #self.dataset_frame = self.dataset_frame.drop(['index','manually_verified','freesound_id','license'], axis=1)
        #self.dataset_frame = self.dataset_frame.drop(['index'], axis=1)
      
        #self.dataset_frame_mod = self.dataset_frame.copy(deep=True)
        
        self.labels = self.dataset_frame.label.unique()

        self.all_samples = []
        
        self.metadata_array = []
        self.spectrogram_array = []
        

      
        
        self.path = path
        self.mmap_path = mmap_path
        self.json_path = json_path

        #self.time_window = 140
        
        self.hop_length = 256
        self.n_fft = 1103

        self.samplerate = 44100

        self.window_length = 12000 # depending on sample length
        self.overlap_length = 6000

        self.n_mels = 26
        self.n_mfcc = 13
        
        self.sample_count = 0

        self.samples = []

        self.y_by_label = {}
        self.y_counts = {}
        
        
        if not os.listdir(self.mmap_path):
            
            print("Directory is empty")
            
            for file in tqdm(self.dataset_frame['fname']):
                
    
    
                # S, sr = librosa.load(self.path + file, sr=self.samplerate, mono=True)
                #raw = librosa.resample(raw, sr, self.samplerate)
                rate, S2 = wavfile.read(self.path + file)
               
                
                label = self.dataset_frame.loc[self.dataset_frame.fname == file, 'label'].values[0]
                
                
                
                # split_points = librosa.effects.split(raw, top_db=80, frame_length=self.n_fft, hop_length=self.hop_length)
                
                # S_cleaned = []
                            
                # for piece in split_points:
                 
                #     S_cleaned.append(raw[piece[0]:piece[1]])
             
                # S = np.array(functools.reduce(operator.iconcat, S_cleaned, []))
               
                # for i in range(0,1):
                #     S = np.concatenate((S,S),axis=0)
                    
                    
    
    
    
                for idx in range(0, S2.shape[0]-self.window_length, self.overlap_length):
                    
                    # sample = S[idx : idx+self.window_length]
                    sample2 = S2[idx : idx+self.window_length]
                    
                    # x_mel = librosa.feature.melspectrogram(sample, 
                    #                         sr=self.samplerate,
                    #                         n_fft=self.n_fft,
                    #                         hop_length=self.hop_length
                                     
                    #                       )
                    
                    
                    
                    # x = librosa.feature.mfcc(
                    #     S = x_mel,
                    #     sr=self.samplerate,
                    #     n_mfcc=round((self.n_mels)),
                    #     n_fft=round((self.n_fft)),
                    #     hop_length=self.hop_length
                    # )
    
    
                    x2 = mfcc(sample2, rate, numcep=self.n_mfcc, nfilt=self.n_mels, nfft = self.n_fft).T                
                    
                    # x = librosa.feature.mfcc(
                    #     sample,
                    #     sr=self.samplerate,
                    #     n_mfcc=self.n_mels,
                    #     n_fft=self.n_fft,
                    #     hop_length=self.hop_length
                    # )
                    
    
                    
                    # x_max = x.max()
                    # x_min = x.min()
                    # x -= x_min
                    # x /= (x_max - x_min) # 0..1
                    # x -= 0.5
                    # x *= 2.0 # -1 .. 1
                    
                    
                    x_max = x2.max()
                    x_min = x2.min()
                    x2 -= x_min
                    x2 /= (x_max - x_min) # 0..1
                    x2 -= 0.5
                    x2 *= 2.0 # -1 .. 1    
                    
                    # x = np.log(abs(x) + 1e-20)
                    x2 = np.log(abs(x2) + 1e-20)
                    
                    self.sample_count += len(x2)
                    
                          
    
                    if label not in self.y_by_label:
                       
                        self.y_by_label[label] = len(self.y_by_label)
    
                    y = self.y_by_label[label]
    
                    if y not in self.y_counts:
                        self.y_counts[y] = 0
                    
                    if self.y_counts[y] < 50:
                        
                        self.samples.append((x2, y))
                        self.y_counts[y] += 1
                        
                    # plt.imshow(x)
                    # plt.title(str(y))
                    # plt.show()
                    
                    plt.imshow(x2)
                    plt.title(str(y) + '_1')
                    plt.show()
                                
            print(f'len(self.samples): {len(self.samples)}')
            print('self.y_counts:')
            for key, y in self.y_by_label.items():
                print(f'{key}: {self.y_counts[y]}')

            self.y_weights = torch.zeros((len(self.y_counts.keys()),), dtype=torch.float)
            for key in self.y_counts.keys():
                self.y_weights[key] = 1.0 - self.y_counts[key] / len(self.samples)                
                        
                        

                        
            
            shape_memmap = (self.sample_count, self.n_mfcc, self.window_length)
            # mmap = np.memmap(mmap_path + 'dataset.mmap', dtype=np.float32, mode='w+', shape=shape_memmap)
            
            # json_desc = {}
            # json_desc['shape_memmap'] = list(shape_memmap)
            # json_desc['class_idxs'] = []
            # json_desc['class_labels'] = []

            # idx_mmap = 0
            
            # for thing in self.samples:
                

            
            #     # mmap[idx_mmap:idx_mmap+len(thing),:,:] = thing
            #     # idx_mmap += len(thing)
                
            #     class_idx = int(self.dataset_frame.loc[self.dataset_frame['fname'] == thing,'label'].iloc[0])
            #     label_name = self.dataset_frame.loc[self.dataset_frame['fname'] == thing,'text_label'].iloc[0]
                
            #     json_desc['class_idxs'] += [class_idx] * len(thing)
            #     json_desc['class_labels'] += [label_name] * len(thing)
                
            
            # # mmap.flush()
            
            # with open(mmap_path + 'memmap.json', 'w') as fp:
            #     json.dump(json_desc, fp)                        



                        
                        # librosa.display.specshow(x, x_axis='time')
                        # plt.title(str(y))
                        # plt.colorbar()
                        # plt.show()
                    
                    
                # S = librosa.stft(
                #     raw,
                #     sr=self.samplerate,
                #     n_fft=self.n_fft,
                #     hop_length=self.hop_length
                # )
    
    
                
               
                
    
    
                # S_max = S.max()
                # S_min = S.min()
                # S -= S_min
                # S /= (S_max - S_min) # 0..1
                # S -= 0.5
                # S *= 2.0 # -1 .. 1
    
    
    
                #print(f'file: {file} label: {label} counts: {S.shape[1]} / {self.window_length}')
                # if len(self.samples) > 64:
                #     break            
        
        
        else:    
            print("Directory is not empty")
            
            exit
        
        
        
        




    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return np.expand_dims(x, axis=0).astype(np.float32), y








