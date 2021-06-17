from imcra import imcra, imcra_est
import numpy as np
import librosa


mixed, sr = librosa.load('/home/smg/haoyuli/SiibGAN/database/Test/Noise/Train_721.wav',sr=16000)
mixed, sr = librosa.load('/home/smg/haoyuli/Cafeteria_1.wav',sr=16000)
MIXED = librosa.stft(mixed, 512, 256, 512)

N_EST = imcra_est(nfft=512)
gthPSD = np.abs(MIXED)**2
estPSD = N_EST.estimate(MIXED)

import scipy.io as scio
scio.savemat('/home/smg/haoyuli/gthPSD.mat', {'gth':gthPSD})
scio.savemat('/home/smg/haoyuli/estPSD.mat', {'est':estPSD})
