# coding=utf-8

from joblib import Parallel, delayed
import shutil
import scipy.io
from scipy.signal import lfilter
import librosa
import os
import time  
import numpy as np
import numpy.matlib
import random
import subprocess

from intel import SIIB_Wrapper_harvard, SIIB_Wrapper_raw_harvard, HASPI_Wrapper_harvard, HASPI_Wrapper_raw_harvard, ESTOI_Wrapper_harvard, ESTOI_Wrapper_raw_harvard, PESQ_Wrapper_harvard, PESQ_Wrapper_raw_harvard
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

tag = 'ESTOI'
chkpt_path = '/home/smg/haoyuli/SI-Extend/trained_model_RT/chkpt_ESTOI.pt'

fs = 16000
p_power = (1/6) # or 0.15
inv_p = 6

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
Generator_Test_paths = [x for x in Generator_Test_paths if 'AIR_stairway' in x] # important!!
assert len(Generator_Test_paths)==720


G = Generator_Conv1D_cLN().cuda()
D = Discriminator().cuda()
MSELoss = nn.MSELoss().cuda()

G.load_state_dict(torch.load(chkpt_path)['enhance-model'])


print('Evaluate G by validation data ...')
Test_enhanced_Name = []
utterance = 0
G.eval()

with torch.no_grad():

    for rev_type in ['NO_rev', 'MIRD_610', 'AIR_stairway21']:
        Test_HASPI = []
        Test_ESTOI = []
        Test_SIIB = []
        Test_PESQ = []
        Test_VISQOL = []
        Test_powerratio = []

        output_path = '/home/smg/haoyuli/SI-Extend/results_RT/' + rev_type + '/' + tag

        for path in tqdm(Generator_Test_paths):
            S = path.split('/')
            wave_name = S[-1]
            idx = int(wave_name.split('_')[2].split('#')[0])   

            clean_wav,sr = librosa.load(path, sr=None)
            assert sr==16000
            noise_wav,sr = librosa.load(Test_Noise_path+wave_name, sr=None)
            assert sr==16000

            enhanced_name = output_path + "/" + wave_name[:-4] + '@' + tag + '.wav'

            if tag not in ['clean','ssdrc']:
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
                mask = mask * beta_2 # normed alpha2
                mask = mask.detach().cpu().squeeze(0).numpy()
                enh_wav = SP_to_wav(mask, clean_mag, clean_phase)
                Test_powerratio.append(1/beta_2.item())
                enh_wav = enh_wav/rms(enh_wav)* 0.03
            elif tag == 'clean':
                enh_wav = clean_wav
                enh_wav = enh_wav/rms(enh_wav)* 0.03
            elif tag == 'ssdrc':
                enh_wav, sr = librosa.load('/home/smg/haoyuli/datasets/Harvard_SI/Test/SSDRC/'+wave_name[:-4]+'@1.wav', sr=None)
                enh_wav = enh_wav/rms(enh_wav)* 0.03

            minL = min(len(enh_wav),len(noise_wav))
            enh_wav = enh_wav[:minL]
            noise_wav = noise_wav[:minL]
            clean_wav = clean_wav[:minL]

            zero_pads = np.zeros(round(0.5*fs))

            if rev_type == 'NO_rev':
                mixed_wav = enh_wav + noise_wav 
                mixed_wav = clip(mixed_wav)
                clean_a = clean_wav
                enhanced_name = enhanced_name.replace('AIR_stairway21_1%912%', 'NONREV')

            elif rev_type == 'MIRD_610':
                rirname = 'MIRD_610_1%588%.wav'
                rir_wav, _ = librosa.load(RIR_path + rirname, sr=None)
                b = np.argmax(rir_wav)
                tau = 32
                N = b + tau
                h_direct = np.hstack([rir_wav[:N], np.zeros(len(rir_wav)-N)])
                direct = lfilter(h_direct, [1], clean_wav)
                direct = direct/rms(direct)*0.03
                direct_wav = clip(direct)
                clean_a = direct_wav[b:]

                reverb_enhan = lfilter(rir_wav, [1], enh_wav)
                reverb_enhan = reverb_enhan / rms(reverb_enhan) * 0.03 # reverb enhanced speech
                reverb_enhan = clip(reverb_enhan)
                # reverb_a = reverb[b:]
                reverb_enhan_a = reverb_enhan[b:]
                noise_a = noise_wav[b:]
                mixed_wav = reverb_enhan_a + noise_a
                mixed_wav = clip(mixed_wav)
                enhanced_name = enhanced_name.replace('AIR_stairway21_1%912%', 'MIRD')


            elif rev_type == 'AIR_stairway21':
                rirname = 'AIR_stairway21_1%912%.wav'
                rir_wav, _ = librosa.load(RIR_path + rirname, sr=None)
                b = np.argmax(rir_wav)
                tau = 32
                N = b + tau
                h_direct = np.hstack([rir_wav[:N], np.zeros(len(rir_wav)-N)])
                direct = lfilter(h_direct, [1], clean_wav)
                direct = direct/rms(direct)*0.03
                direct_wav = clip(direct)
                clean_a = direct_wav[b:]

                reverb_enhan = lfilter(rir_wav, [1], enh_wav)
                reverb_enhan = reverb_enhan / rms(reverb_enhan) * 0.03 # reverb enhanced speech
                reverb_enhan = clip(reverb_enhan)
                # reverb_a = reverb[b:]
                reverb_enhan_a = reverb_enhan[b:]
                noise_a = noise_wav[b:]
                mixed_wav = reverb_enhan_a + noise_a
                mixed_wav = clip(mixed_wav)
                enhanced_name = enhanced_name.replace('AIR_stairway21_1%912%', 'AIR')
            
            Test_SIIB.append(SIIB_Wrapper_raw_harvard(clean_a, mixed_wav, fs))
            Test_HASPI.append(HASPI_Wrapper_raw_harvard(clean_a, mixed_wav, fs))
            Test_ESTOI.append(ESTOI_Wrapper_raw_harvard(clean_a, mixed_wav, fs))
            # save audio
            mixed_wav = np.concatenate([zero_pads, mixed_wav, zero_pads])
            # sf.write(enhanced_name, mixed_wav, fs,'PCM_16') # mixed_wav


        assert(len(Test_SIIB)==720)
        assert(len(Test_HASPI)==720)
        assert(len(Test_ESTOI)==720)
        # assert(len(Test_powerratio)==720)
        # np.mean(Test_powerratio)
        # np.percentile(Test_powerratio, [10, 25, 50, 75, 90])

        print(tag)
        print(rev_type)
        print('SIIB is %.3f, HASPI is %.3f, ESTOI is %.3f\n'%(np.mean(Test_SIIB), np.mean(Test_HASPI), np.mean(Test_ESTOI)))
        # print('Power ratio is %.3f \n'%(np.mean(Test_powerratio)))
        print('=======')

