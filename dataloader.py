from torch.utils.data import DataLoader
from torch.utils import data as torchData
import torch
import librosa
import os
import random 
import numpy as np

path_dir = "./data_split/"

class AudioLoader(torchData.Dataset):
    
    def __init__(self, inPath, isShuffle = False):
        
        files = os.listdir(inPath)
        
        files = [f for f in files if f[-4:] == ".wav"]
        
#        files = [f for f in files if (f.find("wood",18) != -1 or f.find("metal",18) != -1)]
        files = [f for f in files if (f.find("ceramic",18) != -1 or f.find("metal",18) != -1)]
        
        if isShuffle:
            random.shuffle(files)
        
        self.inPath = inPath
        self.isShuffle = isShuffle
        self.len = len(files)  
        self.files = files
        print("# of wav files : ", self.len)
    
    def __getitem__(self, idx):
        
        y, _ = librosa.load(self.inPath+"/"+self.files[idx], 16000)
        y = torch.from_numpy(y)
        return y

    def __len__(self):
        return self.len

    def __iter__(self):
        return iter(range(self.len))
