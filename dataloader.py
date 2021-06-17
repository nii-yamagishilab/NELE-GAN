# coding=utf-8

import os

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import scipy.io as scio
from audio_util import *
#import pdb

power_law = (1/6) # power-law compression parameter

def toTorch(x):
    return torch.from_numpy(x.astype(np.float32))

class Generator_train_dataset(Dataset):
    def __init__(self, file_list, noise_path, rir_path=None):
        self.file_list = file_list
        self.noise_path = noise_path
        self.rir_path = rir_path
        self.target_score = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        self.target_qua = np.asarray([1.0, 1.0], dtype=np.float32)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        p = power_law

        filename = self.file_list[idx].split('/')[-1]
        clean_wav, sr = librosa.load(self.file_list[idx], sr=None)
        assert sr==16000
        noise_wav, sr = librosa.load(self.noise_path + filename, sr=None)
        assert sr==16000

        clean_band, clean_mag, clean_phase = Sp_and_phase_Speech(clean_wav, power=p, Normalization=True)
        noise_band, noise_mag, noise_phase = Sp_and_phase_Noise(noise_wav, power=p, Normalization=True)

        return clean_band, clean_mag, clean_phase, noise_band, noise_mag, noise_phase, self.target_score, self.target_qua, filename

class Discriminator_train_dataset(Dataset):
    def __init__(self, file_list, noise_path, clean_path, rir_path=None):
        self.file_list = file_list
        self.noise_path = noise_path
        self.clean_path = clean_path
        self.rir_path = rir_path 

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):
        p = power_law
        score_filepath = self.file_list[idx].split(',')

        enhance_wav, sr = librosa.load(score_filepath[5], sr=None)
        assert sr==16000
        enhance_band, enhance_mag, enhance_phase = Sp_and_phase_Speech(enhance_wav, power=p, Normalization=True)
        #pdb.set_trace()
        f = self.file_list[idx].split('/')[-1]
        if '@' in f:
            f = f.split('@')[0] + '.wav'

        noise_wav, sr = librosa.load(self.noise_path+f, sr=None)
        assert sr==16000
        noise_band, noise_mag, noise_phase = Sp_and_phase_Noise(noise_wav, power=p, Normalization=True)

        clean_wav, sr = librosa.load(self.clean_path+f, sr=None)
        assert sr==16000
        clean_band, clean_mag, clean_phase = Sp_and_phase_Speech(clean_wav, power=p, Normalization=True)

        True_score = np.asarray([float(score_filepath[0]), float(score_filepath[1]), float(score_filepath[2])], dtype=np.float32)
        True_score_Qua = np.asarray([float(score_filepath[3]), float(score_filepath[4])], dtype=np.float32)
        
        # change to [40, T]
        noise_band, clean_band, enhance_band = noise_band.T, clean_band.T, enhance_band.T

        noise_band = noise_band.reshape(1, 64, noise_band.shape[1])
        clean_band = clean_band.reshape(1, 64, clean_band.shape[1])
        enhance_band = enhance_band.reshape(1, 64, enhance_band.shape[1])

        return np.concatenate((enhance_band,noise_band,clean_band),axis=0), np.concatenate((enhance_band,clean_band),axis=0), True_score, True_score_Qua
    
def create_dataloader(filelist, noise_path, clean_path=None, rir_path=None, loader='G'):
    if loader=='G':
        return DataLoader(dataset=Generator_train_dataset(filelist, noise_path, rir_path),
                          batch_size=1,
                          shuffle=True,
                          num_workers=8,
                          drop_last=True)
    elif loader=='D':
        return DataLoader(dataset=Discriminator_train_dataset(filelist, noise_path, clean_path, rir_path),
                          batch_size=1,
                          shuffle=True,
                          num_workers=8,
                          drop_last=True)
    else:
        raise Exception("No such dataloader type!")