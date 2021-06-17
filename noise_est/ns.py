import numpy as np
import scipy.special

def preemphasis(x, coef=0.97):
    '''
    Pre-emphasis of time domain signal "x" [samples, channel] aplying a FIR
    filter of weigth "coef". This should mimic HTKs pre-emphasis.
    '''

    # Stereo case
    if len(x.shape) > 1:
        x[1:, :] -= coef * x[:-1, :]
        x[0, :]  *= 1-coef
    else:
        x[1:] -= coef * x[:-1]
        x[0]  *= 1-coef

    return x

def framing(x, windowsize, shift=None):
    '''
    Framing of time domain signal into a matrix frames 
    Input: x           [samples, channel] ndarray Time domain signal  (only 
                       one channel accepted).
    Input: windowsize  int Size of analysis window 
    Input: shift       int Shift of the analysis window
    Output: x_framed   [windowsize, L] complex ndarray STFT of 2*K+1 freq. bins 
                       (upper symmetric part discarded) and 
                       L = (len(x)-windowsize)/shift frames. Reminder discarded
    '''

    # Defaul 50%
    if not shift:
        shift = windowsize/2

    # Compute number of frames
    L        = (len(x)-windowsize)/shift
    # Initialize framed signal
    x_framed = np.zeros([windowsize, L])

    # For each frame (tried indices version, its slower)
    for l in np.arange(0, L):
        x_framed[:, l] = x[l*shift:(l*shift+windowsize)]

    # What to do with the remainder?
#    if L*shift + windowsize < len(x):
#        pass

    return x_framed

def iframing(x_framed, windowsize, shift):
    '''
    Inverse of the framing function
    '''

    # Get number of frames
    L = x_framed.shape[1]
    # Initialize time domain signal
    x = np.zeros([windowsize + L*shift], float)
    # For each frame
    for l in range(0, L):
        # Add the overlapping window in the coresponding time positions
        x[range(l*shift, l*shift + windowsize)] += x_framed[:, l]

    return x

def stft(x, windowsize, shift=None, nfft=None, winfunc='hamming'):
    '''
    Short-time Fourier transform (STFT) of time domain signal 
    Input: x           [samples, channel] ndarray Time domain signal  (only 
                       one channel accepted).
    Input: windowsize  int Size of analysis window 
    Input: shift       int Shift of the analysis window
    Input: nfft        int Number of FFT bins 
    Input: winfunc     Type of windowing function (see below)
    Output: X          [K, L] complex ndarray STFT of 2*K+1 freq. bins 
                       (upper symmetric part discarded) and 
                       L = (len(x)-windowsize)/shift frames. Reminder discarded
    '''

    # Defaults
    if not shift:
        shift = windowsize/2
    if not nfft:
        nfft = windowsize 
    # Framing
    x_framed   = framing(x, windowsize, shift)
    # Apply window function
    if winfunc   == 'hamming':
        x_framed = x_framed*np.hamming(windowsize)[:, None]
    elif winfunc == 'hanning':
        x_framed = x_framed*np.hanning(windowsize)[:, None]
    elif winfunc == 'rectangular':
        pass
    else:
        raise ValueError("Unknown windowing function %s" % winfunc)
    # STFT (bins under half freq.)
    X = np.fft.fft(x_framed, nfft, 0)[:(nfft/2 + 1), :]

    return X

def istft(X, windowsize, shift, nfft):
    '''
    Inverse of the stft function
    '''

    # Make sure it is numpy array
    I          = X.shape[0]
    # Recompose with lower conjugate part
    tilde_X    = np.concatenate((X[:I, :], np.flipud(np.conj(X[:(I-1), :]))), 0)
    # STFT (bins under half freq.)
    x_f        = np.real(np.fft.ifft(tilde_X, nfft, 0)[:windowsize, :])
    # Invert DFT and zero padding
    # Overlap Add
    x          = iframing(x_f, windowsize, shift)

    return x

#
# STFT SPEECH ENHANCEMENT
#

def MMSE_LSA(mu_XcY, Lambda_XcY):
    '''
    Minimum Mean Square Error Log-spectral Amplitude Estimator (MMSE-LSA) 
    derived from The posterior distribution associated to the Wiener filter
    
    Input: mu_XcY      Mean of the Wiener filter in STFT domain
    Input: Lambda_XcY  Minimum Mean Square Error (MSE) of the Wiener estimate
    '''
    # Floor MSE
    nu = ((np.absolute(mu_XcY)**2)/Lambda_XcY)   # Rice SNR
    return mu_XcY*np.exp(0.5*expint(nu))

def MMSE_PSD(mu_XcY, Lambda_XcY):
    '''
    Minimum Mean Square Error Squared Amplitude Estimator (MMSE-PSD) 
    derived from The posterior distribution associated to the Wiener filter
    
    Input: mu_XcY      Mean of the Wiener filter in STFT domain
    Input: Lambda_XcY  Minimum Mean Square Error (MSE) of the Wiener estimate
    '''
    return np.absolute(mu_XcY)**2 + Lambda_XcY

def MMSE_STSA(mu_XcY, Lambda_XcY):
    '''
    Minimum Mean Square Error Squared Amplitude Estimator (MMSE-PSD) 
    derived from The posterior distribution associated to the Wiener filter
    
    Input: mu_XcY      Mean of the Wiener filter in STFT domain
    Input: Lambda_XcY  Minimum Mean Square Error (MSE) of the Wiener estimate
    '''

    # Posterior SNR
    nu           = (np.absolute(mu_XcY)**2)/Lambda_XcY               # Rice SNR
    # Get size
    [K, L]        = nu.shape

    # Wiener approximation for very high nu values
    STSA           = np.ones(nu.shape)*(0 + 0*1j)
    STSA[nu>=1300] = mu_XcY[nu>=1300]
    # Conventional computation
    ok_nu          = nu[nu<1300]
    ok_Lambda_XcY  = Lambda_XcY[nu<1300]
    I0             = scipy.special.iv(0, ok_nu/2)
    I1             = scipy.special.iv(1, ok_nu/2)
    STSA[nu<1300]  = (scipy.special.gamma(1.5)*np.sqrt(ok_Lambda_XcY)
                      *np.exp(-ok_nu/2)*((1+ok_nu)*I0 + ok_nu*I1))
    if np.any(np.isnan(STSA.ravel())):
        print("ERROR: Bessel function failed")
        exit(1)

    return STSA 

def SegSNR(x, d,windowsize, shift):
    '''
    Compute segmental signal to noise ratio after
    S. Quackenbush, T. Barnwell, and M. Clements. Objective Measures of Speech
    Quality. Prentice-Hall, 1988.
    To scale noise up to a given SegSNR multiply by
    alpha = 10**((current_SegSNR - desired_SegSNR)/(20))
    '''

    # Make sure it is numpy array
    windowsize = np.array(windowsize) # 400
    shift      = np.array(shift)      # 160

    # Compute number of frames
    L          = (len(x)-windowsize)/shift+1

    SegSNR = 0
    # For each frame
    for l in range(0, L):
        idx                 = range(0, windowsize) + l*shift
        noise_frame_energy  = np.sum(d[idx]**2)
        speech_frame_energy = np.sum(x[idx]**2)
        inst_SegSNR         = np.log10(speech_frame_energy/noise_frame_energy)
        SegSNR              = (l*SegSNR + inst_SegSNR)/(l+1)

    return 10*SegSNR

def expint(nu):
    '''
    Approximate computation of the exponential integral using R. Martin's 
    piecewise  approximation initialization
    '''
    expi = np.zeros(nu.shape)
    # Piecewise approximation
    expi[nu<0.1]                         = -2.31*np.log10(nu[nu<0.1]) - 0.6
    expi[np.logical_and(nu>=0.1, nu<=1)] = (
        -1.544*np.log10(nu[np.logical_and(nu>=0.1, nu<=1)]) + 0.166)
    expi[nu>0.1]                         = np.power(10, -0.52*nu[nu>0.1]-0.26)

    return expi