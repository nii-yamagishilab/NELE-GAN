#coding=utf-8
import librosa
import numpy as np
import torch
import scipy
from scipy.signal import get_window
from pysiib import SIIB
from pystoi.stoi import stoi
import os 
from joblib import Parallel, delayed
from noise_est.imcra import imcra, imcra_est
from intel import SIIB_Wrapper_harvard, SIIB_Wrapper_raw_harvard, HASPI_Wrapper_harvard, HASPI_Wrapper_raw_harvard, ESTOI_Wrapper_harvard, ESTOI_Wrapper_raw_harvard, PESQ_Wrapper_harvard, PESQ_Wrapper_raw_harvard
from datetime import datetime
import pandas
import pdb




# cutoff=62.5Hz
# This is an approximation of 64 ERB-scaled bands for 16kHz speech
# Please generate new ERB scale if use different number of ERB filters or speech of sampling rate 
gmtband = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 30, 32, 34, 36, 38, 41, 43, 46, 49, 52, 55, 58, 62, 66, 70, 74, 79, 83, 88, 93, 99, 105, 111, 117, 124, 131, 139, 147, 156, 165, 174, 184, 195, 206, 218, 230, 243, 257]
NB_BANDS = 64
win_hamm = None
fs = 16000


# This can be optimized by using a in-advance prepared 257X64 matrix
def compute_band_E(X):
    # X: magnitude of spectrogram (T x 257)
    global gmtband
    global NB_BANDS
    T, D = X.shape[0],X.shape[1]
    OUT = np.zeros((T,NB_BANDS),dtype=np.float32)

    for iT in range(T):
        sumE = np.zeros(NB_BANDS)

        for i in range(NB_BANDS-1):
            band_size = gmtband[i+1] - gmtband[i]

            for j in range(band_size):
                frac = float(j) / band_size
                tmp = X[iT,gmtband[i]+j]**2
                sumE[i] += (1-frac) * tmp
                sumE[i+1] += frac * tmp

        OUT[iT,:]=sumE
    return OUT


def STFT(x, nfft=512,nw=512,nm=256):
    # global win_hamm
    # if win_hamm is None:
    #     win_hamm = build_win_hamm(nw,nm)
    X = librosa.stft(x, n_fft=nfft, hop_length=nm, win_length=nw)
    return X

def ISTFT(X, nw=512,nm=256):
    # global win_hamm
    # if win_hamm is None:
    #     win_hamm = build_win_hamm(nw,nm)
    x = librosa.istft(X, hop_length=nm, win_length=nw)
    return x

def clip(x):
    if np.max(x)>=1 or np.min(x)<-1:
        print('Overflow occured!')
        small = 0.05
        while np.max(x)>=1 or np.min(x)<-1:
            x = x / (1.0+small)
            small = small + 0.05
    return x

def Resyn(X, alpha):
    # X 257xT
    # alpfa Tx40 amplifier factor for energy (alpha^2)
    T = alpha.shape[0]
    gain = np.zeros([257,T])
    # alpha = np.sqrt(alpha)

    for t in range(T):
        g = interp_band_gain(alpha[t,:])
        gain[:,t] = np.sqrt(g)

    Xnew = gain * X # complex spectrogram 
    Xnorm = Xnew
    # Xnorm = Xnew / np.sum(np.abs(Xnew)**2) * np.sum(np.abs(X)**2)
    return ISTFT(Xnorm)


def interp_band_gain(bandE):
    global gmtband
    global NB_BANDS
    FREQ_SIZE = 257
    
    g = np.ones(FREQ_SIZE)

    for i in range(NB_BANDS-1):
        band_size = gmtband[i+1] - gmtband[i]
        for j in range(band_size):
            frac = float(j) / band_size
            g[gmtband[i]+j] = (1-frac)*bandE[i] + frac * bandE[i+1]
    
    # simply remove low-frequency noises
    g[0] = 1e-4
    g[1] = 1e-4
    g[256] = 1e-2
    return g

# Use IMCRA to estimate noise PSD
def NoisePSD(MIXED, nfft=512):
    N_EST = imcra_est(nfft=nfft)
    estPSD = N_EST.estimate(MIXED)
    # return is noise PSD
    return estPSD


def read_STOI(clean_root, noise_root, enhanced_file, norm):
    f=enhanced_file.split('/')[-1]
    # wave_name=f.split('_')[-1].split('@')[0]
    if '@' in f:
        wave_name = f.split('@')[0]
    else:
        wave_name = f[:-4]
    
    # enhanced_file = enhanced_file[:-6]+'.wav'
    # clean_wav, sr    = librosa.load(clean_root+'Train_'+wave_name+'.wav', sr=fs)
    clean_wav, sr    = librosa.load(clean_root+wave_name+'.wav', sr=fs)
    assert sr==16000
    noise_wav,_    = librosa.load(noise_root+wave_name+'.wav', sr=fs)     
    enhanced_wav,_ = librosa.load(enhanced_file, sr=fs)
    minL = min(len(clean_wav),len(enhanced_wav))
    clean_wav = clean_wav[:minL]
    noise_wav = noise_wav[:minL]
    enhanced_wav = enhanced_wav[:minL]
    if norm:
        stoi_score = ESTOI_Wrapper_harvard(clean_wav, enhanced_wav + noise_wav, fs)
    else:
        stoi_score = ESTOI_Wrapper_raw_harvard(clean_wav, enhanced_wav + noise_wav, fs)
    return stoi_score
    
# Parallel computing for accelerating    
def read_batch_STOI(clean_root, noise_root, enhanced_list, norm=True):
    stoi_score = Parallel(n_jobs=32)(delayed(read_STOI)(clean_root, noise_root, en, norm) for en in enhanced_list)
    return stoi_score

def read_SIIB(clean_root, noise_root, enhanced_file, norm):
    f=enhanced_file.split('/')[-1]
    # wave_name=f.split('_')[-1].split('@')[0]
    if '@' in f:
        wave_name = f.split('@')[0]
    else:
        wave_name = f[:-4]
    
    # enhanced_file = enhanced_file[:-6]+'.wav'
    clean_wav,sr    = librosa.load(clean_root+wave_name+'.wav', sr=fs) 
    assert sr==16000    
    enhanced_wav,_ = librosa.load(enhanced_file, sr=fs)
    noise_wav,_    = librosa.load(noise_root+wave_name+'.wav', sr=fs)     
    minL = min(len(clean_wav),len(enhanced_wav))
    clean_wav = clean_wav[:minL]
    noise_wav = noise_wav[:minL]
    enhanced_wav = enhanced_wav[:minL]
    if norm:
        siib_score = SIIB_Wrapper_harvard(clean_wav, enhanced_wav + noise_wav, fs)
    else:
        siib_score = SIIB_Wrapper_raw_harvard(clean_wav, enhanced_wav + noise_wav, fs)
    return siib_score
    
# Parallel computing for accelerating    
def read_batch_SIIB(clean_root, noise_root, enhanced_list, norm=True):
    siib_score = Parallel(n_jobs=32)(delayed(read_SIIB)(clean_root, noise_root, en, norm) for en in enhanced_list)
    return siib_score

def read_HASPI(clean_root, noise_root, enhanced_file, norm):
    f=enhanced_file.split('/')[-1]
    # wave_name=f.split('_')[-1].split('@')[0]
    if '@' in f:
        wave_name = f.split('@')[0]
    else:
        wave_name = f[:-4]
    
    # enhanced_file = enhanced_file[:-6]+'.wav'
    clean_wav,sr    = librosa.load(clean_root+wave_name+'.wav', sr=fs) 
    assert sr==16000    
    enhanced_wav,_ = librosa.load(enhanced_file, sr=fs)
    noise_wav,_    = librosa.load(noise_root+wave_name+'.wav', sr=fs)     
    minL = min(len(clean_wav),len(enhanced_wav))
    clean_wav = clean_wav[:minL]
    noise_wav = noise_wav[:minL]
    enhanced_wav = enhanced_wav[:minL]
    if norm:
        haspi_score = HASPI_Wrapper_harvard(clean_wav, enhanced_wav + noise_wav, fs)
    else:
        haspi_score = HASPI_Wrapper_raw_harvard(clean_wav, enhanced_wav + noise_wav, fs)
    return haspi_score
    
# Parallel computing for accelerating    
def read_batch_HASPI(clean_root, noise_root, enhanced_list, norm=True):
    haspi_score = Parallel(n_jobs=32)(delayed(read_HASPI)(clean_root, noise_root, en, norm) for en in enhanced_list)
    return haspi_score

def read_PESQ(clean_root, enhanced_file, norm):
    f=enhanced_file.split('/')[-1]
    # wave_name=f.split('_')[-1].split('@')[0]
    if '@' in f:
        wave_name = f.split('@')[0]
    else:
        wave_name = f[:-4]
    
    # enhanced_file = enhanced_file[:-6]+'.wav'
    clean_wav,sr    = librosa.load(clean_root+wave_name+'.wav', sr=fs) 
    assert sr==16000    
    enhanced_wav,_ = librosa.load(enhanced_file, sr=fs)
    minL = min(len(clean_wav),len(enhanced_wav))
    clean_wav = clean_wav[:minL]
    enhanced_wav = enhanced_wav[:minL]
    if norm:
        pesq_score = PESQ_Wrapper_harvard(clean_wav, enhanced_wav, fs)
    else:
        pesq_score = PESQ_Wrapper_raw_harvard(clean_wav, enhanced_wav, fs)
    return pesq_score

def read_batch_PESQ(clean_root, enhanced_list, norm=True):
    pesq_score = Parallel(n_jobs=32)(delayed(read_PESQ)(clean_root, en, norm) for en in enhanced_list)
    return pesq_score

# A very dirty method to wrapper ViSQOL C programming in python
# I run ViSQOL exec program and write results to a file, then extract results by pandas
def read_batch_VISQOL(clean_root, enhanced_list, norm=True):
    time = datetime.now().strftime('%d-%H-%M-%S')
    program = '/home/smg/haoyuli/repos/visqol/bin/visqol' # visqol exec program
    model_path = '/home/smg/haoyuli/repos/visqol/model/libsvm_nu_svr_model.txt' # visqol model path
    input_csv = '/tmp/hyli-input-'+time+'.csv'
    result_csv = '/tmp/hyli-result-'+time+'.csv'
    with open(input_csv,'w') as f:
        f.write('reference,degraded\n')
        for enfile in enhanced_list:
            filename = enfile.split('/')[-1]
            if '@' in filename:
                wave_name = filename.split('@')[0]
            else:
                wave_name = filename[:-4]

            clean_path = clean_root+wave_name+'.wav'
            enhanced_path = enfile
            f.write(clean_path+','+enhanced_path+'\n')

    cmd = '%s --use_speech_mode --similarity_to_quality_model %s --batch_input_csv %s --results_csv %s  > nohup' % (program, model_path, input_csv, result_csv)
    ret = os.system(cmd)
    # ret = Parallel(n_jobs=1)(delayed(os.system)(cmd)) 
    assert ret==0
    result = pandas.read_csv(result_csv, sep=',')
    moslqo = list(result['moslqo'])
    assert len(moslqo) == len(enhanced_list)
    if norm:
        a = -2.5
        b = 2.2
        func = lambda x: 1/(1+np.exp(a *(x - b)))  
        scores = [func(x) for x in moslqo]
    else:
        scores = moslqo
    return scores

def read_STOI_DRC(clean_root, noise_root, enhanced_file):
    f=enhanced_file.split('/')[-1]
    wave_name = f
    clean_wav,_    = librosa.load(clean_root+wave_name, sr=fs)     
    enhanced_wav,_ = librosa.load(enhanced_file, sr=fs)
    noise_wav,_    = librosa.load(noise_root+wave_name, sr=fs)     
    minL = min(len(clean_wav),len(enhanced_wav))
    clean_wav = clean_wav[:minL]
    noise_wav = noise_wav[:minL]
    enhanced_wav = enhanced_wav[:minL]
    
    stoi_score = ESTOI_Wrapper_harvard(clean_wav, enhanced_wav + noise_wav, fs)
    return stoi_score

def read_batch_STOI_DRC(clean_root, noise_root, enhanced_list):
    stoi_score = Parallel(n_jobs=32)(delayed(read_STOI_DRC)(clean_root, noise_root, en) for en in enhanced_list)
    return stoi_score

def read_SIIB_DRC(clean_root, noise_root, enhanced_file):
    f=enhanced_file.split('/')[-1]
    wave_name = f
    clean_wav, sr    = librosa.load(clean_root+wave_name, sr=fs)     
    assert sr==16000
    enhanced_wav,_ = librosa.load(enhanced_file, sr=fs)
    noise_wav,_    = librosa.load(noise_root+wave_name, sr=fs)     
    minL = min(len(clean_wav),len(enhanced_wav))
    clean_wav = clean_wav[:minL]
    noise_wav = noise_wav[:minL]
    enhanced_wav = enhanced_wav[:minL]
    
    siib_score = SIIB_Wrapper_harvard(clean_wav, enhanced_wav + noise_wav, fs)  
    return siib_score

def read_batch_SIIB_DRC(clean_root, noise_root, enhanced_list):
    siib_score = Parallel(n_jobs=32)(delayed(read_SIIB_DRC)(clean_root, noise_root, en) for en in enhanced_list)
    return siib_score

def read_HASPI_DRC(clean_root, noise_root, enhanced_file):
    f=enhanced_file.split('/')[-1]
    wave_name = f
    clean_wav, sr    = librosa.load(clean_root+wave_name, sr=fs)     
    assert sr==16000
    enhanced_wav,_ = librosa.load(enhanced_file, sr=fs)
    noise_wav,_    = librosa.load(noise_root+wave_name, sr=fs)     
    minL = min(len(clean_wav),len(enhanced_wav))
    clean_wav = clean_wav[:minL]
    noise_wav = noise_wav[:minL]
    enhanced_wav = enhanced_wav[:minL]
    
    haspi_score = HASPI_Wrapper_harvard(clean_wav, enhanced_wav + noise_wav, fs)  
    return haspi_score

def read_batch_HASPI_DRC(clean_root, noise_root, enhanced_list):
    haspi_score = Parallel(n_jobs=32)(delayed(read_HASPI_DRC)(clean_root, noise_root, en) for en in enhanced_list)
    return haspi_score

def read_PESQ_DRC(clean_root, enhanced_file):
    f=enhanced_file.split('/')[-1]
    wave_name = f
    clean_wav, sr    = librosa.load(clean_root+wave_name, sr=fs)     
    assert sr==16000
    enhanced_wav,_ = librosa.load(enhanced_file, sr=fs)
    minL = min(len(clean_wav),len(enhanced_wav))
    clean_wav = clean_wav[:minL]
    enhanced_wav = enhanced_wav[:minL]
    
    pesq_score = PESQ_Wrapper_harvard(clean_wav, enhanced_wav, fs)  
    return pesq_score

def read_batch_PESQ_DRC(clean_root, enhanced_list):
    pesq_score = Parallel(n_jobs=32)(delayed(read_PESQ_DRC)(clean_root, en) for en in enhanced_list)
    return pesq_score

def read_batch_VISQOL_DRC(clean_root, enhanced_list):
    time = datetime.now().strftime('%d-%H-%M-%S')
    program = '/home/smg/haoyuli/repos/visqol/bin/visqol'
    model_path = '/home/smg/haoyuli/repos/visqol/model/libsvm_nu_svr_model.txt'
    input_csv = '/tmp/hyli-input-'+time+'.csv'
    result_csv = '/tmp/hyli-result-'+time+'.csv'
    with open(input_csv,'w') as f:
        f.write('reference,degraded\n')
        for enfile in enhanced_list:
            wave_name = enfile.split('/')[-1]
            clean_path = clean_root+wave_name
            enhanced_path = enfile
            f.write(clean_path+','+enhanced_path+'\n')

    cmd = '%s --use_speech_mode --similarity_to_quality_model %s --batch_input_csv %s --results_csv %s  > nohup' % (program, model_path, input_csv, result_csv)
    ret = os.system(cmd)
    assert ret==0
    result = pandas.read_csv(result_csv, sep=',')
    moslqo = list(result['moslqo'])
    assert len(moslqo) == len(enhanced_list)
    a = -2.5
    b = 2.2
    func = lambda x: 1/(1+np.exp(a *(x - b)))  
    scores = [func(x) for x in moslqo]
    return scores


def List_concat(score, enhanced_list):
    concat_list=[]
    for i in range(len(score)):
        concat_list.append(str(score[i])+','+enhanced_list[i]) 
    return concat_list

def List_concat_score(score, score2):
    concat_list=[]
    for i in range(len(score)):
        concat_list.append(str(score[i])+','+str(score2[i])) 
    return concat_list

def List_concat_3scores(score1, score2, score3):
    concat_list=[]
    for i in range(len(score1)):
        concat_list.append(str(score1[i])+','+str(score2[i])+','+str(score3[i])) 
    return concat_list

def List_concat_5scores(score1, score2, score3, score4, score5):
    concat_list=[]
    for i in range(len(score1)):
        concat_list.append(str(score1[i])+','+str(score2[i])+','+str(score3[i])+','+str(score4[i])+','+str(score5[i])) 
    return concat_list

def creatdir(directory):    
    if not os.path.exists(directory):
        os.makedirs(directory) 

def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path
     
def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            if '.wav' in filename:
                # Join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


def Sp_and_phase_Speech(signal, power, Normalization=True):        
    #signal_length = signal.shape[0]
    n_fft = 512
    
    F = STFT(signal)
    
    mag = np.abs(F)
    phase = np.angle(F)
    
    bandE = compute_band_E(mag.T) # Transpose for pytorch LSTM processing format
    if Normalization:
        bandE = bandE ** power
    else:
        print('No normalization for func: Sp_and_phase_Clean')

    return bandE, mag, phase #, signal_length

def Sp_and_phase_Noise(signal, power, Normalization=True):        
    #signal_length = signal.shape[0]
    n_fft = 512
    #y_pad = librosa.util.fix_length(signal, signal_length + n_fft // 2)
    
    F = STFT(signal)
    estPSD = NoisePSD(F)
    estPSD = estPSD.T
    bandE = compute_band_E(np.sqrt(estPSD))
    if Normalization:
        bandE = bandE ** power
    else:
        print('No normalization for func: Sp_and_phase_Noise')

    mag = np.abs(F)
    phase = np.angle(F)

    return bandE, mag, phase #, signal_length

def SP_to_wav(alpha2, mag, phase, signal_length=None):
    Complex_Mag = np.multiply(mag, np.exp(1j*phase))
    result = Resyn(Complex_Mag, alpha2)
    return result  

def rms(x):
    return np.sqrt(np.mean(x**2))




def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device, ratio):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * ratio)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    return x