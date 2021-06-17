import math
import numpy as np
from scipy.signal import get_window
from pysiib import SIIB
from scipy.fftpack import fft
from scipy.signal import get_window
from pyHASPI.pyhaspi2 import haspi_v2
from pystoi.stoi import stoi
from pypesq import pesq

# Wrapper function here adopts logistic compression to convert original scores into range [0, 1]
# You may attempt your own logistic parameters to fit in your data

EPS = np.finfo(np.float64).eps

def framing(x, window_length, window_shift, window):
    """
    Args:
        x: (Samples,)
        window_length:
        window_shift:
        window:
    Returns:
        y: (num_frame, window_length)
    """
    slen = x.shape[-1]
    if slen < window_length + 1:
        z = [(0, 0) for _ in range(x.ndim - 1)]
        x = np.pad(x, z + [(0, window_length + 1 - slen)], mode='constant')
    shape = x.shape[:-1] + (x.shape[-1] - window_length, window_length)
    strides = x.strides + (x.strides[-1],)
    y = np.lib.stride_tricks.as_strided(
        x, shape=shape, strides=strides)[..., ::window_shift, :]
    w = get_window(window, window_length)[None, :]
    return y * w

def get_vad(x, window_length, window_shift, window, delta_db):
    """
    Args:
        x: Time domain (Sample,)
    """
    # returns the indices of voice active frames
    x_frame = framing(x, window_length, window_shift, window)
    # compute the power (dB) of each frame
    x_dB = 10 * np.log10((x_frame ** 2).mean(axis=1) + EPS)

    # find the 99.9 percentile
    ind = int(round(len(x_dB) * 0.999) - 1)
    max_x = np.partition(x_dB, ind)[ind]
    return x_dB > (max_x - delta_db)

def stft(x, window_length, window_shift, window):
    frames = framing(x, window_length, window_shift, window=window)
    return fft(frames, n=window_length, axis=-1)[:, :window_length // 2 + 1]


def SIIB_Wrapper_raw_harvard(x,y,fs):
    minL = min(len(x),len(y))
    x = x[:minL]
    y = y[:minL]

    window_length=400
    window_shift=200
    window='hanning'
    R = 1 / window_shift * fs  
    x_hat = stft(x, window_length, window_shift, window).T
    x_hat = x_hat.real ** 2 + x_hat.imag ** 2
    vad_index_x = get_vad(x, window_length=400, window_shift=200, window='hanning', delta_db=40)
    x_hat = x_hat[:, vad_index_x]

    # check that the duration (after removing silence) is at least 20 s
    if x_hat.shape[1] / R < 20:
        M = int(np.floor(25 / (x_hat.shape[1] / R)))
        x = np.hstack([x]*M)
        y = np.hstack([y]*M)
    
    return SIIB(x,y,fs,gauss=True)

def SIIB_Wrapper_harvard(x,y,fs):
    minL = min(len(x),len(y))
    x = x[:minL]
    y = y[:minL]

    window_length=400
    window_shift=200
    window='hanning'
    R = 1 / window_shift * fs  
    x_hat = stft(x, window_length, window_shift, window).T
    x_hat = x_hat.real ** 2 + x_hat.imag ** 2
    vad_index_x = get_vad(x, window_length=400, window_shift=200, window='hanning', delta_db=40)
    x_hat = x_hat[:, vad_index_x]

    # check that the duration (after removing silence) is at least 20 s
    if x_hat.shape[1] / R < 20:
        M = int(np.floor(25 / (x_hat.shape[1] / R)))
        x = np.hstack([x]*M)
        y = np.hstack([y]*M)
    
    #return SIIB(x,y,fs,gauss=True)
    return mapping_SIIB_harvard(SIIB(x,y,fs,gauss=True))

def mapping_SIIB_harvard(x):
    a = -0.06
    b = 32
    y = 1/(1+np.exp(a*(x-b)))
    return y

def HASPI_Wrapper_harvard(x,y,fs):
    score, _ = haspi_v2(x, fs, y, fs)
    return mapping_HASPI_harvard(score)

def HASPI_Wrapper_raw_harvard(x,y,fs):
    score, _ = haspi_v2(x, fs, y, fs)
    return score

def mapping_HASPI_harvard(x):
    a = -0.95
    b = 2.8
    y = 1/(1+np.exp(a*(x - b)))
    return y

def ESTOI_Wrapper_raw_harvard(x,y,fs):
    minL = min(len(x),len(y))
    x = x[:minL]
    y = y[:minL]
    estoi_score= stoi(x, y, fs, extended=True)
    return estoi_score

def ESTOI_Wrapper_harvard(x,y,fs):
    minL = min(len(x),len(y))
    x = x[:minL]
    y = y[:minL]
    estoi_score= stoi(x, y, fs, extended=True)
    return mapping_ESTOI_harvard(estoi_score)

def mapping_ESTOI_harvard(x):
    a = -8.0
    b = 0.25
    y = 1/(1+np.exp(a *(x - b)))
    return y

def PESQ_Wrapper_raw_harvard(ref, deg, fs):
    minL = min(len(ref), len(deg))
    ref = ref[:minL]
    deg = deg[:minL]
    pesq_score = pesq(ref, deg, fs)
    return pesq_score

def PESQ_Wrapper_harvard(ref, deg, fs):
    minL = min(len(ref), len(deg))
    ref = ref[:minL]
    deg = deg[:minL]
    pesq_score = pesq(ref, deg, fs)
    return mapping_PESQ_harvard(pesq_score)

def mapping_PESQ_harvard(x):
    a = -1.5
    b = 2.5
    y = 1/(1+np.exp(a *(x - b)))
    return y

if __name__ == "__main__":
    import glob
    import librosa
    import pdb
    import os
    import scipy
    from audio_util import rms
    from scipy.signal import lfilter
    noise_path = '/home/smg/haoyuli/datasets/Harvard_SI/Train/Noise/'
    enhan_path = '/home/smg/haoyuli/datasets/Harvard_SI/Train/MultiEnh/'
    rir_path = '/home/smg/haoyuli/datasets/Harvard_SI/RIR_DB16k/'
    filenames = glob.glob('/home/smg/haoyuli/datasets/Harvard_SI/Train/Clean/*.wav')
    np.random.shuffle(filenames)
    np.random.shuffle(filenames)
    np.random.shuffle(filenames)

    estoi_before = []
    estoi_after = []
    estoi_rev_before = []
    estoi_rev_after = []

    haspi_before = []
    haspi_after = []
    haspi_rev_before = []
    haspi_rev_after = []

    siib_before = []
    siib_after = []
    siib_rev_before = []
    siib_rev_after = []

    for i in range(1500):
        clean_file = filenames[i]
        basename = os.path.basename(clean_file)
        rirname = basename.split('#')[1]+'.wav'
        clean, sr = librosa.load(clean_file,sr=None)
        assert sr==16000
        noise, sr = librosa.load(noise_path+basename,sr=None)
        assert sr==16000
        enhan, sr = librosa.load(enhan_path+basename,sr=None)
        assert sr==16000
        rir, sr = librosa.load(rir_path+rirname,sr=None)
        assert sr==16000

        b = np.argmax(rir)
        tau = 32
        N = b + tau
        h_direct = np.hstack([rir[:N], np.zeros(len(rir)-N)])

        reverb = lfilter(rir, [1], clean)
        reverb = reverb/rms(reverb)*0.03
        reverb_enhan = lfilter(rir, [1], enhan)
        reverb_enhan = reverb_enhan/rms(reverb_enhan)*0.03

        direct = lfilter(h_direct, [1], clean)
        direct = direct/rms(direct)*0.03

        direct_a = direct[b:]
        reverb_a = reverb[b:]
        reverb_enhan_a = reverb_enhan[b:]
        noise_a = noise[b:]
        clean_a = clean[:-b]

        estoi_before.append(ESTOI_Wrapper_harvard(clean, clean+noise, sr))
        estoi_after.append(ESTOI_Wrapper_harvard(clean, enhan+noise, sr))
        estoi_rev_before.append(ESTOI_Wrapper_harvard(direct_a, reverb_a+noise_a, sr))
        estoi_rev_after.append(ESTOI_Wrapper_harvard(direct_a, reverb_enhan_a+noise_a, sr))

        haspi_before.append(HASPI_Wrapper_harvard(clean, clean+noise, sr))
        haspi_after.append(HASPI_Wrapper_harvard(clean, enhan+noise, sr))
        haspi_rev_before.append(HASPI_Wrapper_harvard(direct_a, reverb_a+noise_a, sr))
        haspi_rev_after.append(HASPI_Wrapper_harvard(direct_a, reverb_enhan_a+noise_a, sr))

        siib_before.append(SIIB_Wrapper_harvard(clean, clean+noise, sr))
        siib_after.append(SIIB_Wrapper_harvard(clean, enhan+noise, sr))
        siib_rev_before.append(SIIB_Wrapper_harvard(direct_a, reverb_a+noise_a, sr))
        siib_rev_after.append(SIIB_Wrapper_harvard(direct_a, reverb_enhan_a+noise_a, sr))

        if i % 10 == 0:
            print('%d finished'%i)


    pdb.set_trace()
    estoi_before = np.array(estoi_before)
    estoi_after = np.array(estoi_after)
    estoi_rev_before = np.array(estoi_rev_before)
    estoi_rev_after = np.array(estoi_rev_after)

    haspi_before = np.array(haspi_before)
    haspi_after = np.array(haspi_after)
    haspi_rev_before = np.array(haspi_rev_before)
    haspi_rev_after = np.array(haspi_rev_after)

    siib_before = np.array(siib_before)
    siib_after = np.array(siib_after)
    siib_rev_before = np.array(siib_rev_before)
    siib_rev_after = np.array(siib_rev_after)

    np.save('./metric_norm/estoi_before.npy',     estoi_before)
    np.save('./metric_norm/estoi_after.npy',      estoi_after)
    np.save('./metric_norm/estoi_rev_before.npy', estoi_rev_before)
    np.save('./metric_norm/estoi_rev_after.npy',  estoi_rev_after)

    np.save('./metric_norm/haspi_before.npy',     haspi_before)
    np.save('./metric_norm/haspi_after.npy',      haspi_after)
    np.save('./metric_norm/haspi_rev_before.npy', haspi_rev_before)
    np.save('./metric_norm/haspi_rev_after.npy',  haspi_rev_after)

    np.save('./metric_norm/siib_before.npy',     siib_before)
    np.save('./metric_norm/siib_after.npy',      siib_after)
    np.save('./metric_norm/siib_rev_before.npy', siib_rev_before)
    np.save('./metric_norm/siib_rev_after.npy',  siib_rev_after)
    pdb.set_trace()