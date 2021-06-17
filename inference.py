# coding=utf-8

from joblib import Parallel, delayed
import shutil
import scipy.io
import librosa
import os
import time  
import numpy as np
import numpy.matlib
import random
import subprocess
import glob

import torch
import torch.nn as nn
from audio_util import *
from pystoi.stoi import stoi
from model import Generator_Conv1D_cLN, Discriminator
from dataloader import *
from tqdm import tqdm
import soundfile as sf
import pdb

random.seed(666)

# 1st: SIIB 2nd: HASPI 
TargetMetric='siib&haspi&estoi' # It can be either 'SIIB' or 'ESTOI' or 'HASPI' or xxx for now. Of course, it can be any arbitary metric of interest.
Target_score=np.asarray([1.0,1.0]) # Target metric scores you want generator to generate. 


output_path='./output_wav'
pt_dir = './chkpt'
GAN_epoch = 500
num_of_sampling = 300 #300
num_of_valid_sample = 960 #960
batch_size=1
fs = 16000
p_power = (1/6) # or 0.15
inv_p = 6

creatdir(output_path)
#########################  Training data #######################
# You should replace the addresses to your own
RIR_path = '/home/smg/haoyuli/datasets/Harvard_SI/RIR_DB16k/'

# Data_shuffle
######################### validation data #########################
# You should replace the addresses to your own
print('Reading path of validation data...')
Test_Noise_path ='/home/smg/haoyuli/datasets/Harvard_SI/Test/Noise/'
Test_Clean_path = '/home/smg/haoyuli/datasets/Harvard_SI/Test/Clean/'
Generator_Test_paths = get_filepaths('/home/smg/haoyuli/datasets/Harvard_SI/Test/Clean/') 

assert len(Generator_Test_paths)==720
# Data_shuffle
random.shuffle(Generator_Test_paths)
################################################################


G = Generator_Conv1D_cLN().cuda()
D = Discriminator().cuda()
MSELoss = nn.MSELoss().cuda()

Test_HASPI = []
Test_ESTOI = []
Test_SIIB = []
Test_PESQ = []
Test_VISQOL = []

chkpt_path = '/home/smg/haoyuli/SI-Extend/NELE-GAN/trained_model/chkpt_GD.pt'
G.load_state_dict(torch.load(chkpt_path)['enhance-model'])


print('Evaluate G by validation data ...')
Test_enhanced_Name = []
utterance = 0
G.eval()
with torch.no_grad():
    for i, path in enumerate(Generator_Test_paths[0:num_of_valid_sample]):
        print(i)
        S = path.split('/')
        wave_name = S[-1]

        clean_wav,sr = librosa.load(path, sr=None)
        assert sr==16000
        noise_wav,sr = librosa.load(Test_Noise_path+wave_name, sr=None)
        assert sr==16000

        clean_band, clean_mag, clean_phase = Sp_and_phase_Speech(clean_wav, power=p_power, Normalization=True)
        noise_band, noise_mag, noise_phase = Sp_and_phase_Noise(noise_wav, power=p_power, Normalization=True)
        
        clean_in = clean_band.reshape(1,clean_band.shape[0],-1)
        clean_in = torch.from_numpy(clean_in).cuda()
        noise_in = noise_band.reshape(1,noise_band.shape[0],-1)
        noise_in = torch.from_numpy(noise_in).cuda()
        # Energy normalization

        mask = G(clean_in, noise_in)
        clean_power = torch.pow(clean_in, inv_p)

        beta_2 = torch.sum(clean_power) / torch.sum(mask*clean_power)
        # beta_2 = torch.sum(clean_power, dim=2) / torch.sum(mask*clean_power, dim=2)
        # beta_2 = beta_2.unsqueeze(2)

        mask = mask * beta_2 # normed alpha2
        mask = mask.detach().cpu().squeeze(0).numpy()
        enh_wav = SP_to_wav(mask, clean_mag, clean_phase) # enh_wav has approximately same energy as input, but not strictlly same since calculation error
        enh_wav = enh_wav / rms(enh_wav) * 0.030 # Normalize enh_wav into exactly same energy level as input (here rms is set to 0.03 because we pre-normalized input speech into 0.03 RMS)

        enhanced_name=output_path+"/"+ wave_name[:-4]+'@1.wav'

    
        # # librosa.output.write_wav(enhanced_name, enh_wav, fs)
        sf.write(enhanced_name, enh_wav, fs,'PCM_16')
        # utterance+=1
        Test_enhanced_Name.append(enhanced_name) 

Test_enhanced_Name_Backup = Test_enhanced_Name

for noise in ['Cafeteria', 'AirportAnnouncement']:
    Test_enhanced_Name = [x for x in Test_enhanced_Name_Backup if noise in x]
    # Calculate True HASPI
    test_HASPI = read_batch_HASPI(Test_Clean_path, Test_Noise_path, Test_enhanced_Name, norm=False)
    Test_HASPI.append(np.mean(test_HASPI))

    # Calculate True ESTOI
    test_ESTOI = read_batch_STOI(Test_Clean_path, Test_Noise_path, Test_enhanced_Name, norm=False)
    Test_ESTOI.append(np.mean(test_ESTOI))

    # Calculate True SIIB
    test_SIIB = read_batch_SIIB(Test_Clean_path, Test_Noise_path, Test_enhanced_Name, norm=False)
    Test_SIIB.append(np.mean(test_SIIB))

    # Calculate True PESQ
    test_PESQ = read_batch_PESQ(Test_Clean_path, Test_enhanced_Name, norm=False)
    Test_PESQ.append(np.mean(test_PESQ))

    # Calculate True VISQOL
    test_VISQOL = read_batch_VISQOL(Test_Clean_path, Test_enhanced_Name, norm=False)
    Test_VISQOL.append(np.mean(test_VISQOL))
    print(noise+':')
    print('SIIB is %.3f, HASPI is %.3f, ESTOI is %.3f, PESQ is %.3f, VISQOL is %.3f\n'%(np.mean(test_SIIB), np.mean(test_HASPI), np.mean(test_ESTOI), np.mean(test_PESQ), np.mean(test_VISQOL)))
    print('======')

