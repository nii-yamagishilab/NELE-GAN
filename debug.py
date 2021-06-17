#coding=utf-8
from audio_util import *
import numpy as np
import librosa
from dataloader import create_dataloader
from model import Generator_Conv1D, Generator_Conv1D_cLN, Discriminator, Discriminator_Union

G = Generator_Conv1D().cuda()
D = Discriminator_Union()


# save and lod
G2 = Generator_Conv1D_cLN().cuda()
chkpt_path = '/home/smg/haoyuli/SI-Extend/trained_model_RT/chkpt_Quality_UL.pt'
save_model = torch.load(chkpt_path)['enhance-model']
torch.save(save_model, 'G.pt')
model_dict = G2.state_dict()
state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
model_dict.update(state_dict)
G2.load_state_dict(model_dict)
#
# num_of_sampling = 300
# Generator_Train_paths = get_filepaths('/home/smg/haoyuli/datasets/Harvard_SI/Train/Clean/')
# Train_Noise_path = '/home/smg/haoyuli/datasets/Harvard_SI/Train/Noise/'

# genloader = create_dataloader(Generator_Train_paths[0:round(1*num_of_sampling)], Train_Noise_path, loader='G')
# genloader = iter(genloader)

# clean_band, clean_mag, clean_phase, noise_band, noise_mag, noise_phase, target_scores, filename = genloader.next()
# tmp = G(clean_band, noise_band)

# clean, sr = librosa.load('/home/smg/haoyuli/datasets/Harvard_SI/Train/Clean/m_hvd_173#AIR_lecture2_1%871%#Babble#-7.wav', sr=None)
# assert sr==16000
# noise, sr = librosa.load('/home/smg/haoyuli/datasets/Harvard_SI/Train/Noise/m_hvd_173#AIR_lecture2_1%871%#Babble#-7.wav', sr=None)
# assert sr==16000
# enhance, sr = librosa.load('/home/smg/haoyuli/datasets/Harvard_SI/Train/MultiEnh/m_hvd_173#AIR_lecture2_1%871%#Babble#-7.wav', sr=None)
# assert sr==16000

# X = STFT(clean)
# E = STFT(enhance)

# N = STFT(noise)

# Xband = compute_band_E(np.abs(X.T))
# estPSD = NoisePSD(N)
# Nband = compute_band_E(np.sqrt(estPSD.T))

num_of_sampling = 300
Generator_Train_paths = get_filepaths('/home/smg/haoyuli/datasets/Harvard_SI/Train/Clean/')
Train_Noise_path = '/home/smg/haoyuli/datasets/Harvard_SI/Train/Noise/'

genloader = create_dataloader(Generator_Train_paths[0:round(1*num_of_sampling)], Train_Noise_path, loader='G')
genloader = iter(genloader)

# clean_band, clean_mag, clean_phase, noise_band, noise_mag, noise_phase, target_scores, filename = genloader.next()
# clean, sr = librosa.load('/home/smg/haoyuli/datasets/Harvard_SI/Train/Clean/'+filename[0], sr=None)
# assert sr==16000
# enhance, sr = librosa.load('/home/smg/haoyuli/datasets/Harvard_SI/Train/MultiEnh/'+filename[0], sr=None)
# assert sr==16000
# X = STFT(clean)
# E = STFT(enhance)

# Xband = compute_band_E(np.abs(X.T))
# Eband = compute_band_E(np.abs(E.T))

# alpha2 = Eband / (Xband+1e-8)
# np.percentile(alpha2, [2,5,10,20,30,40,50,60,70,80,90,95, 98])
# clean_mag = clean_mag.detach().cpu().squeeze(0).numpy()
# clean_phase = clean_phase.detach().cpu().squeeze(0).numpy()
# enhan = SP_to_wav(alpha2, clean_mag, clean_phase)
# librosa.output.write_wav('enhan_lowfq_norm.wav',enhan,16000)

pdb.set_trace()