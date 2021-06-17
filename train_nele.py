# coding=utf-8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

import torch
import torch.nn as nn
from audio_util import *
from pystoi.stoi import stoi
from model import Generator_Conv1D_cLN, Discriminator, Discriminator_Quality
from dataloader import *
from tqdm import tqdm
import soundfile as sf
import pdb

random.seed(666)

TargetMetric='siib&haspi&estoi' 


output_path='./output'
pt_dir = './chkpt'
GAN_epoch = 500
num_of_sampling = 300 # 300, in each epoch randonly sample only 300 for training
num_of_valid_sample = 480 # number of validation audio
batch_size = 1
fs = 16000 # sampling rate
p_power = (1/6) # power-law compression
inv_p = 6 # inverse of p_power

weight_qua = 0.0

creatdir(pt_dir)
creatdir(output_path)
#########################  Training data #######################
# You should replace data path to your own
print('Reading path of training data...')
RIR_path = '/home/smg/haoyuli/datasets/Harvard_SI/RIR_DB16k/'
Train_Noise_path = '/home/smg/haoyuli/datasets/Harvard_SI/Train/Noise/'
Train_Clean_path = '/home/smg/haoyuli/datasets/Harvard_SI/Train/Clean/'
Train_Enhan_path = '/home/smg/haoyuli/datasets/Harvard_SI/Train/MultiEnh/' # contains pre-enhanced speech examples y_hat shown in Eqs.(5) and (6). In our paper, we use SSDRC to generate them
Generator_Train_paths = get_filepaths('/home/smg/haoyuli/datasets/Harvard_SI/Train/Clean/')

# Data_shuffle
random.shuffle(Generator_Train_paths)
######################### validation data #########################
# You should replace the addresses to your own
print('Reading path of validation data...')
Test_Noise_path ='/home/smg/haoyuli/datasets/Harvard_SI/Test/Noise/'
Test_Clean_path = '/home/smg/haoyuli/datasets/Harvard_SI/Test/Clean/'
Generator_Test_paths = get_filepaths('/home/smg/haoyuli/datasets/Harvard_SI/Test/Clean/') 
Generator_Test_paths = [x for x in Generator_Test_paths if 'AIR_stairway' in x]

assert len(Generator_Test_paths)==720 # Just for double-check data size, change it to fit in you dataset
# Data_shuffle
random.shuffle(Generator_Test_paths)
################################################################

G = Generator_Conv1D_cLN().cuda()
D = Discriminator().cuda()
D_Qua = Discriminator_Quality().cuda()

# Load trained model
# chkpt_path = '/home/smg/haoyuli/SI-Extend/NELE-GAN/trained_model/chkpt_GD.pt'
# save_model = torch.load(chkpt_path)['enhance-model']
# model_dict = G.state_dict()
# state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
# model_dict.update(state_dict)
# G.load_state_dict(model_dict)

# D.load_state_dict(torch.load(chkpt_path)['intel-model'])
# D_Qua.load_state_dict(torch.load(chkpt_path)['quality-model'])
# print('Load Chkpt Finished')

MSELoss = nn.MSELoss().cuda()

optimizer_g = torch.optim.Adam(G.parameters(), lr=5e-4)
optimizer_d = torch.optim.Adam(D.parameters(), lr=2.5e-4)
optimizer_dqua = torch.optim.Adam(D_Qua.parameters(), lr=2.5e-4)


Test_HASPI = []
Test_ESTOI = []
Test_SIIB = []
Test_PESQ = []
Test_VISQOL = []

Previous_Discriminator_training_list = []
shutil.rmtree(output_path)

step_g = 0
step_d = 0

cuda_device = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
# OCCUPY_MEM = occumpy_mem(cuda_device, 0.5) # occupy 50% memory

for gan_epoch in np.arange(1, GAN_epoch+1):

    # Prepare directories
    creatdir(output_path+"/epoch"+str(gan_epoch))
    creatdir(output_path+"/epoch"+str(gan_epoch)+"/"+"Test_epoch"+str(gan_epoch))
    creatdir(output_path+'/For_discriminator_training')
    creatdir(output_path+'/temp')

    # random sample some training data  
    random.shuffle(Generator_Train_paths)
    genloader = create_dataloader(Generator_Train_paths[0:round(1*num_of_sampling)],Train_Noise_path)

    if gan_epoch>=2:
        print('Generator training (with discriminator fixed)...')
        for clean_band, clean_mag, clean_phase, noise_band, noise_mag, noise_phase, target, target_qua, filename in tqdm(genloader):
            clean_band = clean_band.cuda()
            noise_band = noise_band.cuda()
            target = target.cuda()
            target_qua = target_qua.cuda()

            mask = G(clean_band, noise_band) # outout mask is actually alpha^2 shown in paper, which should be applied to power spectrum 

            # Do utterance-level energy normalization
            clean_power = torch.pow(clean_band.detach(), inv_p)
            beta_2 = torch.sum(clean_power) / torch.sum(mask*clean_power)
            # Comment parts are frame-level energy normalization Eq.(10) in paper
            # beta_2 = torch.sum(clean_power, dim=2) / torch.sum(mask*clean_power, dim=2)
            # beta_2 = beta_2.unsqueeze(2)
            beta_p = beta_2 ** p_power

            enh_band = clean_band * torch.pow(mask, p_power) * beta_p
            ref_band = clean_band.detach()

            enh_band = enh_band.view(1,1,enh_band.shape[1],enh_band.shape[2]).transpose(2,3).contiguous()
            noise_band = noise_band.view(1,1,noise_band.shape[1],noise_band.shape[2]).transpose(2,3).contiguous()
            ref_band = ref_band.view(1,1,ref_band.shape[1],ref_band.shape[2]).transpose(2,3).contiguous()
            d_inputs = torch.cat((enh_band,noise_band,ref_band),dim=1)
            # d_inputs_qua = torch.cat((enh_band, ref_band),dim=1)

            score = D(d_inputs)
            # score_qua = D_Qua(d_inputs_qua)

            loss = MSELoss(score, target) # + weight_qua * MSELoss(score_qua, target_qua)
            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()
            step_g += 1


    # Evaluate the performance of generator in a validation set.
    interval_epoch = 1
    if gan_epoch % interval_epoch == 0: 
        print('Evaluate G by validation data ...')
        Test_enhanced_Name = []
        utterance = 0
        G.eval()
        with torch.no_grad():
            for i, path in enumerate(Generator_Test_paths[0:num_of_valid_sample]):
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

                mask = G(clean_in, noise_in)
                clean_power = torch.pow(clean_in, inv_p)
                beta_2 = torch.sum(clean_power) / torch.sum(mask*clean_power)
                # beta_2 = torch.sum(clean_power, dim=2) / torch.sum(mask*clean_power, dim=2)
                # beta_2 = beta_2.unsqueeze(2)
                mask = mask * beta_2 # normed alpha2
                mask = mask.detach().cpu().squeeze(0).numpy()
                enh_wav = SP_to_wav(mask, clean_mag, clean_phase)

                if utterance<20: # Only seperatly save the firt 20 utterance for listening comparision 
                    enhanced_name=output_path+"/epoch"+str(gan_epoch)+"/"+"Test_epoch"+str(gan_epoch)+"/"+ wave_name[0:-4]+"@"+str(gan_epoch)+wave_name[-4:]
                else:
                    enhanced_name=output_path+"/temp"+"/"+ wave_name[0:-4]+"@"+str(gan_epoch)+wave_name[-4:]
            
                sf.write(enhanced_name, enh_wav, fs,'PCM_16')
                utterance+=1      
                Test_enhanced_Name.append(enhanced_name) 
                #print(i)
        G.train()

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

        with open('./log.txt','a') as f:
	        f.write('SIIB is %.3f, HASPI is %.3f, ESTOI is %.3f, PESQ is %.3f, VISQOL is %.3f, EPOCH:%d \n'%(np.mean(test_SIIB), np.mean(test_HASPI), np.mean(test_ESTOI), 0, 0, gan_epoch))
        # Plot learning curves
        plt.figure(1)
        plt.plot(range(1,gan_epoch+1,interval_epoch),Test_HASPI,'b',label='ValidHASPI')
        plt.xlim([1,gan_epoch])
        plt.xlabel('GAN_epoch')
        plt.ylabel('HASPI')
        plt.grid(True)
        plt.show()
        plt.savefig('Test_HASPI.png', dpi=150)
        
        plt.figure(2)
        plt.plot(range(1,gan_epoch+1,interval_epoch),Test_SIIB,'r',label='ValidSIIB')
        plt.xlim([1,gan_epoch])
        plt.xlabel('GAN_epoch')
        plt.ylabel('SIIB')
        plt.grid(True)
        plt.show()
        plt.savefig('Test_SIIB.png', dpi=150)

        plt.figure(3)
        plt.plot(range(1,gan_epoch+1,interval_epoch),Test_ESTOI,'b',label='ValidESTOI')
        plt.xlim([1,gan_epoch])
        plt.xlabel('GAN_epoch')
        plt.ylabel('ESTOI')
        plt.grid(True)
        plt.show()
        plt.savefig('Test_ESTOI.png', dpi=150)

        plt.figure(3)
        plt.plot(range(1,gan_epoch+1,interval_epoch),Test_PESQ,'b',label='ValidPESQ')
        plt.xlim([1,gan_epoch])
        plt.xlabel('GAN_epoch')
        plt.ylabel('PESQ')
        plt.grid(True)
        plt.show()
        plt.savefig('Test_PESQ.png', dpi=150)

        plt.figure(3)
        plt.plot(range(1,gan_epoch+1,interval_epoch),Test_VISQOL,'b',label='ValidVISQOL')
        plt.xlim([1,gan_epoch])
        plt.xlabel('GAN_epoch')
        plt.ylabel('VISQOL')
        plt.grid(True)
        plt.show()
        plt.savefig('Test_VISQOL.png', dpi=150)

    # save the current enhancement model
    save_path = os.path.join(pt_dir, 'chkpt_%d.pt' % gan_epoch)
    torch.save({
        'enhance-model': G.state_dict(),
        'intel-model': D.state_dict(),
    }, save_path)

    print('Sample training data for discriminator training...')
    D_paths = Generator_Train_paths[0:num_of_sampling]

    Enhanced_name = []

    G.eval()
    # Generate samples for discriminator training
    with torch.no_grad():
        for path in D_paths:
            S = path.split('/')
            wave_name = S[-1]
            clean_wav, sr = librosa.load(path, sr=fs)
            assert sr==16000
            noise_wav, _ = librosa.load(Train_Noise_path+wave_name, sr=fs)

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
            enh_wav = SP_to_wav(mask, clean_mag, clean_phase)

            enhanced_name=output_path+"/For_discriminator_training/"+ wave_name[0:-4]+"@"+str(gan_epoch)+wave_name[-4:]
            sf.write(enhanced_name, enh_wav, fs,'PCM_16')
            Enhanced_name.append(enhanced_name)

    G.train()

    if TargetMetric=='siib&haspi&estoi':
        # Calculate True SIIB score
        train_SIIB = read_batch_SIIB(Train_Clean_path, Train_Noise_path, Enhanced_name)
        train_HASPI = read_batch_HASPI(Train_Clean_path, Train_Noise_path, Enhanced_name)
        train_ESTOI = read_batch_STOI(Train_Clean_path, Train_Noise_path, Enhanced_name)
        train_PESQ = read_batch_PESQ(Train_Clean_path, Enhanced_name)
        train_VISQOL = read_batch_VISQOL(Train_Clean_path, Enhanced_name)


        train_SIIB = List_concat_5scores(train_SIIB, train_HASPI, train_ESTOI, train_PESQ, train_VISQOL) # SIIB, HASPI, ESTOI, PESQ, VISQOL
        current_sampling_list=List_concat(train_SIIB, Enhanced_name) # This list is used to train discriminator.

        # DRC_Enhanced_name = [Train_Enhan_path+'Train_'+S.split('/')[-1].split('_')[-1].split('@')[0]+'.wav' for S in Enhanced_name]
        DRC_Enhanced_name = [Train_Enhan_path+S.split('/')[-1].split('@')[0]+'.wav' for S in Enhanced_name]
        #pdb.set_trace()
        train_SIIB_DRC = read_batch_SIIB_DRC(Train_Clean_path, Train_Noise_path, DRC_Enhanced_name)
        train_HASPI_DRC = read_batch_HASPI_DRC(Train_Clean_path, Train_Noise_path, DRC_Enhanced_name)
        train_ESTOI_DRC = read_batch_STOI_DRC(Train_Clean_path, Train_Noise_path, DRC_Enhanced_name)
        train_PESQ_DRC = read_batch_PESQ_DRC(Train_Clean_path, DRC_Enhanced_name)
        train_VISQOL_DRC = read_batch_VISQOL_DRC(Train_Clean_path, DRC_Enhanced_name)

        train_SIIB_DRC = List_concat_5scores(train_SIIB_DRC, train_HASPI_DRC, train_ESTOI_DRC, train_PESQ_DRC, train_VISQOL_DRC) # SIIB, HASPI, ESTOI
        Co_DRC_list = List_concat(train_SIIB_DRC, DRC_Enhanced_name)

    print("Discriminator training...")
    # Training for current list
    Current_Discriminator_training_list = current_sampling_list+Co_DRC_list

    random.shuffle(Current_Discriminator_training_list)
    disloader = create_dataloader(Current_Discriminator_training_list, Train_Noise_path, Train_Clean_path, loader='D')

    for x, x_qua, target, target_qua in tqdm(disloader):
        x = x.cuda()
        x_qua = x_qua.cuda()
        target = target.cuda()
        target_qua = target_qua.cuda()
        score = D(x)
        score_qua = D_Qua(x_qua)

        loss = MSELoss(score, target)
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        loss_qua = MSELoss(score_qua, target_qua)
        optimizer_dqua.zero_grad()
        loss_qua.backward()
        optimizer_dqua.step()

        step_d += 1
        #if step_d % 1000 ==0:
        #    print('Step %d: Loss in D training is %.3f'%(step_d,loss.item()))
    

    ## Training for current list + Previous list (like replay buffer in RL, optional)
    random.shuffle(Previous_Discriminator_training_list)

    Total_Discriminator_training_list=Previous_Discriminator_training_list[0:len(Previous_Discriminator_training_list)//30]+Current_Discriminator_training_list # Discriminator_Train_list is the list used for pretraining.
    random.shuffle(Total_Discriminator_training_list)

    disloader_past = create_dataloader(Total_Discriminator_training_list, Train_Noise_path, Train_Clean_path, loader='D')

    for x, x_qua, target, target_qua in tqdm(disloader_past):
        x = x.cuda()
        x_qua = x_qua.cuda()
        target = target.cuda()
        target_qua = target_qua.cuda()
        score = D(x)
        score_qua = D_Qua(x_qua)

        loss = MSELoss(score, target)
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        loss_qua = MSELoss(score_qua, target_qua)
        optimizer_dqua.zero_grad()
        loss_qua.backward()
        optimizer_dqua.step()

        step_d += 1
        #if step_d % 1000 ==0:
        #    print('Step %d: Loss in D training is %.3f'%(step_d,loss.item()))
        
    # Update the history list
    Previous_Discriminator_training_list=Previous_Discriminator_training_list+Current_Discriminator_training_list 
    
    # Training current list again
    for x, x_qua, target, target_qua in tqdm(disloader):
        x = x.cuda()
        x_qua = x_qua.cuda()
        target = target.cuda()
        target_qua = target_qua.cuda()
        score = D(x)
        score_qua = D_Qua(x_qua)

        loss = MSELoss(score, target)
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        loss_qua = MSELoss(score_qua, target_qua)
        optimizer_dqua.zero_grad()
        loss_qua.backward()
        optimizer_dqua.step()

        step_d += 1
        #if step_d % 1000 ==0:
        #    print('Step %d: Loss in D training is %.3f'%(step_d,loss.item()))
    
    shutil.rmtree(output_path+'/temp')
    print('Epoch %d Finished' % gan_epoch)

print('Finished')
