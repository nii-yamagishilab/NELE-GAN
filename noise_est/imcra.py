#!/usr/bin/python
'''
Improved Minima Controlled Recursive Averaging (IMCRA) single channel
noise estmation after
    [1] Israel Cohen, Noise Spectrum estimation in Adverse Environments:
    Improved Minima Controlled Recursive Averaging. IEEE. Trans. Acoust.
    Speech Signal Process. VOL. 11, NO. 5, Sep 2003.
Ramon F. Astudillo Feb2015
'''

import numpy as np
import sys
import os
# Add the path of the toolbox root


# from ns import MMSE_LSA
# For debugging purposes
#import ipdb 
#np.seterr(divide='ignore',invalid='raise')

def post_speech_prob(Y_l, q, Gamma, xi):
    '''
    Posterior speech probability given prior speech absence and the complex
    Gaussian model of speech distortion
    Input: Y_l   [K, 1] STFT frame
    Input: q     [K, 1] a priori speech presence 
    Input: Gamma [K, 1] A posteriori SNR 
    Input: xi    [K, 1] A priori SNR
    
    '''
    nu       = Gamma*xi/(1+xi)
    p        = np.zeros(Y_l.shape)
    p[q < 1] = 1./(1+(q[q < 1]/(1-q[q < 1]))*(1+xi[q < 1])*np.exp(-nu[q < 1]))

    return p



def sym_hanning(n):
    '''
    Same Hanning as the matlab default
    '''
    # to float
    n = float(n)
    if np.mod(n, 2) == 0:
        # Even length window
        half = n/2;
    else:
        # Odd length window
        half = (n+1)/2;
    w    = .5*(1 - np.cos(2*np.pi*np.arange(1,half+1)/(n+1)))
    return np.concatenate((w, w[:-1]));


# Default buffer size
L_MAX = 1000

class imcra_se():
    '''
    Simple class for enhancement using IMCRA 
    '''
    def __init__(self, nfft, Lambda_D=None, Bmin=3.2, alpha =0.92, xi_min=10**(-25./20), IS=10):

        # Decision directed smoothing factor
        self.alpha  = alpha
        # Decision directed a priori SNR floor
        self.xi_min = xi_min
        self.nfft = int(nfft/2+1)
        #
        self.store             = {}
        self.store['Lambda_D'] = np.zeros((self.nfft, L_MAX))
        self.store['p']        = np.zeros((self.nfft, L_MAX))
        self.store['xi']       = np.zeros((self.nfft, L_MAX))
        self.store['MSE']       = np.zeros((self.nfft, L_MAX))
        self.l                 = 0

        # IMCRA initial background segment (frames) 
        # Initialization
        self.imcra = imcra(nfft, IS=IS, Bmin=Bmin)
        self.G     = 1
        self.p     = np.zeros([self.nfft, 1])

        # Initial noise estimate
        if Lambda_D is None:
            self.Lambda_D = 1e-6*np.ones([self.nfft, 1])
        else:
            self.Lambda_D = Lambda_D

    def update(self, Y):

        hat_X    = np.zeros(Y.shape, dtype=complex)
        G        = self.G
        Gamma    = G
        Lambda_D = self.Lambda_D 
        p        = self.p 
        K, L     = Y.shape

        if self.l + L > L_MAX:

            # If maximum size surpassed, add a new batch of zeroes
            self.store['Lambda_D'] = np.concatenate((self.store['Lambda_D'], 
                                                     np.zeros((K, L_MAX))), 1) 
            self.store['p']        = np.concatenate((self.store['p'], 
                                                     np.zeros((K, L_MAX))), 1)
            self.store['xi']        = np.concatenate((self.store['xi'], 
                                                     np.zeros((K, L_MAX))), 1)
            self.store['MSE']        = np.concatenate((self.store['MSE'], 
                                                      np.zeros((K, L_MAX))), 1)

        for l in np.arange(0, L):
        
            # A priori SNR, stationary parameter estimate (uses last Gamma)   
            xi_G              = (G**2)*Gamma
            # A posterori SNR
            Gamma             = (np.abs(Y[:, l:l+1])**2)/Lambda_D
            # A priori SNR, maximum likelihood estimate 
            xi_ML             = Gamma - 1
            xi_ML[xi_ML<1e-6] = 1e-6      
            # Decision directed rule
            xi                = self.alpha*xi_G + (1-self.alpha)*xi_ML
            # Flooring
            xi[xi<self.xi_min] = self.xi_min 

            # MMSE-LSA
            # Get Wiener gain
            G               = xi/(1 + xi) 
            hat_X[:, l:l+1] = MMSE_LSA(G*Y[:, l:l+1], G*Lambda_D)
            # Residual MSE of Wiener filter
            MSE             = G*Lambda_D
        
            # TODO: Store additional variables if solicited
            self.store['Lambda_D'][:, self.l:self.l+1]  = Lambda_D 
            self.store['p'][:, self.l:self.l+1]         = p
            self.store['xi'][:, self.l:self.l+1]        = xi
            self.store['MSE'][:, self.l:self.l+1]       = MSE 
            self.l                                     += 1

            # IMCRA noise estimate and posterior speech probability for the
            # next iteration
            Lambda_D, p = self.imcra.update(Y[:, l:l+1], Gamma, xi)

        # Keep these for the next iteration
        self.G        = G
        self.Lambda_D = Lambda_D[:, -2:]
        self.p        = p

        return hat_X

    def get_param(self, param_list):
        '''
        Return stored parameters
        '''
        val_list = []
        for par in param_list:
            if par not in self.store:
                raise(ValueError, "Parameter %s was not recorded" % par)
            val_list.append(self.store[par][:, :self.l]) 
        return val_list

class imcra():
    '''
    IMCRA class
    '''

    def __init__(self, nfft, Bmin=None, **kwargs):

        # IMCRA DEFAULT CONFIGURATION
        # Consult [1] on how to set the parameters
        # Number of initial frames of the signal assumed to be noise
        self.IS                 = 15
        # How many adjacent bins (before and after bin k) are used to smooth
        # spectrogram in frecuency (S_f[k,l])
        self.w                  = 1
        # Adaptation rate for the spectrogram smoothing in time
        self.alpha_s            = 0.9
        # Adaptation rate for speech probability dependent time smoothing
        # paramater
        self.alpha_d            = 0.85
        # U spectrogram values are stored to perform minimum tracking
        self.U                  = 8 # 8
        # Each V frames minimum tracking will take place
        self.V                  = 15 # 15
        # Number of frames before the actual frame considered in minimum
        # traking. (Irrelevant its included in Bmin)
        self.D                  = self.U*self.V
        # Significance levels
        # Significance level for the first VAD
        self.epsilon            = 0.01
        # Significance level for the second VAD
        self.epsilon1           = 0.05

        # Overload defaults
        for par in kwargs.keys():
            # If parameter is one of the defaults, overload it
            if par in self.__dict__:
                setattr(self, par, kwargs[par])
            else:
                raise ValueError("%s is not an imcra parameter" % par)

        # If the standard significance levels and smoothing parameters not used
        # we need to recompute everything
        if ((self.epsilon == 0.01) & (self.epsilon1 == 0.05)
            & (self.alpha_s == 0.9) & (self.w == 1)):
            # VAD is attained through hypothesis test assuming distributions
            # for ratios related to the minimum statistics, these are the
            # parameteres
            # Treshold to achieve 0.01 significance in first VAD speech absence
            # hypothesis test
            self.Gamma0          = 4.6
            # Treshold to achieve 0.05 significance in second VAD speech
            # presence hypothesis test
            self.Gamma1          = 3
            # Treshold to achieve 0.05 significance in second VAD speech
            # presence hypothesis test
            self.zeta0           = 1.67
            # Noise variance estimate bias for speech absence (depends only
            # on the last three parameters though [1, Eq. 31])
            self.beta            = 1.47

        else:
            # Note: This STILL assumes that we use the posterior probability
            # as defined by [1, Eq. 7], see [1, App. II]
            # This will need SCIPY!
            import scipy.stats.chi2 as chi2
            # Approx degrees of freedom of Eq. 20, see App. II
            mu          = (1+self.alpha_s)/(1-self.alha_s)*(1 + 0.7*w)
            # Treshold to achieve 0.01 significance in first VAD speech absence
            # hypothesis test
            self.Gamma0 = -np.log(epsilon)
            self.zeta0  = chi2.ppf(1-epsilon,mu)/mu
            self.Gamma1 = -np.log(epsilon1)
            # Treshold to achieve 0.05 significance in second VAD speech
            # presence hypothesis test [1, Eq. 31])
            self.beta   = ((gamma1 - 1 -np.exp(-1) + np.exp(-gamma1))/
                           (gamma1 -1 - 3*np.exp(-1)
                            + (gamma1+2)*np.exp(-gamma1)))

        # Check for smoothed spectrogram bias set
        if Bmin is None:
            import warnings
            warnings.warn("Minimum statistic bias Bmin is not set. Run "
                          "imcra.setBmin(), until then 2.1 used.",
                          DeprecationWarning)
            self.Bmin = 2.1   # This is for my config!
        else:
            self.Bmin = Bmin

        self.K = int(nfft/2+1)     # Number of frequency bins under Nyquist
        self.l = -1           # Set frame index. -1 is no frame processed
        self.j = 0            # Set counter for the buffer update
        self.u = 0            # Set copunter for buffer last filled position

        # To do the smoothing in frequency fast, we create indices to expand
        # each frame (column) to a matrix with that column on the center and
        # neighbouring frequencies on each row. Thay way we can smooth with
        # One single loop
        # Create matrix of indices
        self.sm_idx = np.arange(0,self.K)[:,None] + np.arange(-self.w,self.w+1)[None,:]
        # Create coresponding matrix of hamming windows
        self.sm_win = np.tile(sym_hanning(2*self.w+1),(self.K,1))
        # Ignore indices out of bounds
        self.sm_win[self.sm_idx<0]        = 0
        self.sm_win[self.sm_idx>self.K-1] = 0
        self.sm_idx[self.sm_idx<0]        = 0
        self.sm_idx[self.sm_idx>self.K-1] = self.K-1
        # Normalize
        self.sm_win = self.sm_win/np.sum(self.sm_win, 1, keepdims=True)

        # BUFFERS
        #  They will be propperly initialized when the first frame is processed

        #  Smoothed Spectrogram first iteration
        self.S             = np.zeros([self.K, 1])
        #  Smoothed Spectrogram minimum first iteration
        self.Smin          = np.zeros([self.K, 1])
        #  Smoothed Spectrogram second iteration
        self.tilde_S       = np.zeros([self.K, 1])
        #   Smoothed Spectrogram second iteration
        self.tilde_Smin    = np.zeros([self.K, 1])
        #  Smoothed Spectrogram minimum running minimum
        self.Smin_sw       = np.zeros([self.K, 1])
        #  Second smoothed Spectrogram minimum running minimum
        self.tilde_Smin_sw = np.zeros([self.K, 1])
        #  Smoothed Spectrogram minimum first iteration store buffer
        self.Storing       = np.zeros([self.K, self.U])
        #  Smoothed Spectrogram minimum second iteration store buffer
        self.tilde_Storing = np.zeros([self.K, self.U])
        #  Biased noise variance estimate
        self.ov_Lambda_D   = np.zeros([self.K, 1])
        #  Unbiased noise variance estimate
        self.Lambda_D      = np.zeros([self.K, 1])
        #  A posteriori speech presence probability
        self.q             = np.ones([self.K, 1])
        #  A posteriori speech presence probability
        self.p             = np.zeros([self.K, 1])
        #  Speech presence prob upbound
        self.p_upthr       = 0.9


    def reset(self, nfft=512, Bmin=3.2):
        self.__init__(self, nfft, Bmin)

    def setBmin(self, N):
        '''
        Computes Bmin given the stft of a white noise signal as obtained e.g
            n    = np.random.randn(1e5)
            N    = stft(n, windowsize, shift, nfft)
            Bmin = imcra.computeBmin(N)
        '''
        # SANITY CHECK: Enough samples
        if N.shape[1] < 3*self.U*self.V:
            raise ValueError("Not enough samples, pick a langer white noise signal")

        Bmin = np.zeros(N.shape)
        for l in np.arange(0,N.shape[1]):
            # Init
            if l == 0:
                self.init_params(N[:,l:l+1])

            # Get minimum statistics
            Bmin     = (np.abs(N[:,l:l+1])**2)/self.Smin                                          # [3,eq.18]
            #zeta[:,l:l+1]            = imcra.S/(imcra.Bmin*imcra.Smin)                           # [3,eq.21]
            #tilde_Gamma_min[:,l:l+1] = (np.abs(Y[:,l:l+1])**2)/(imcra.Bmin*imcra.tilde_Smin)     # [3,eq.18]
            #tilde_zeta[:,l:l+1]      = imcra.S/(imcra.Bmin*imcra.tilde_Smin)                     # [3,eq.21]

        # Compute mean as average
        self.Bmin = np.mean(Bmin)

        return self.Bmin

    '''
    Fast frequency smoothing
    '''
    def fsmooth(self, P_l):
        return np.sum(self.sm_win*P_l[self.sm_idx][:, :, 0], 1)[:,None]

    def init_params(self,Y_l):

        #  Smoothed spectrograms

        #  Smoothed Spectrogram first iteration
        self.S             = self.fsmooth(np.abs(Y_l)**2)
        #  Smoothed Spectrogram second iteration
        self.tilde_S       = self.S.copy()
        #  Smoothed Spectrogram minimum first iteration
        self.Smin          = self.S.copy()
        #  Smoothed Spectrogram minimum first second iteration
        self.tilde_Smin    = self.S.copy()
        #  Smoothed Spectrogram minimum running minimum
        self.Smin_sw       = self.S.copy()
        #  Second smoothed Spectrogram minimum running minimum
        self.tilde_Smin_sw = self.S.copy()
        #  Other parameters

        #  Biased noise variance estimate
        self.ov_Lambda_D   = np.abs(Y_l)**2
        #  Unbiased noise variance estimate
        self.Lambda_D      = self.ov_Lambda_D
        #  A posteriori speech presence probability
        self.p             = np.zeros(Y_l.shape)

    def update(self, Y_l, Gamma, xi):

        '''
        This calls the components of IMCRA as in the original paper. These are
    
        A priori speech absence probability estimator
            A posteriori speech probability estimator
        Probabilistic recursive smoothing
    
        For the initialization period (only noise assumed) it uses normal smoothing
        '''
        # Increase frame counter
        self.l += 1

        # If in first frame, initialize buffers with observed frame
        if self.l == 0:
            # print('first frame!')
            self.init_params(Y_l)

        # If in initialization segment, update noise stats only
        # Note: Keep in mind that IS might be zero
        if self.l < self.IS:

            # Frequency smoothing  [3,eq.14]
            Sf            = self.fsmooth(np.abs(Y_l)**2)                                         
            # Frequency and time smoothing  [3,eqs.15]
            self.S        = self.alpha_s*self.S + (1-self.alpha_s)*Sf                 
            # Update running minimum
            self.Smin     = np.min(np.concatenate((self.Smin,self.S),1),1)[:,None]
            self.Smin_sw  = np.min(np.concatenate((self.Smin_sw,self.S),1),1)[:,None]
            # Compute smoothed spectrogram for p = 0
            self.Lambda_D = self.alpha_d*self.Lambda_D + (1-self.alpha_d)*np.abs(Y_l)**2
            # Set a priori background probability to one
            self.q[:]     = 1
            # Set a posteriori speech probability to zero
            self.p[:]     = 0

        else:

            # FIRST MINIMA CONTROLLED VAD
            # This provides a rough VAD to eliminate relatively strong speech
            # components towards the second power spectrum estimation
            Sf           = self.fsmooth(np.abs(Y_l)**2)                                         # [3,eq.14]
            # Time smoothing
            self.S       = self.alpha_s*self.S+(1-self.alpha_s)*Sf                    # [3,eq.15]
            # update running minimum
            self.Smin     = np.min(np.concatenate((self.Smin,self.S),1),1)[:,None]
            self.Smin_sw  = np.min(np.concatenate((self.Smin_sw,self.S),1),1)[:,None]
            # Indicator function for VAD
            Gamma_min     = (np.abs(Y_l)**2)/(self.Bmin*self.Smin)                    # [3,eq.18]
            zeta          = self.S/(self.Bmin*self.Smin)                             # [3,eq.21]
            I             = np.zeros([self.K, 1])
            I[(Gamma_min < self.Gamma0 ) & (zeta < self.zeta0)] = 1                   # [3,eq.21]

            # SECOND MINIMA CONTROLLED VAD
            # This provides the speech probability needed to compute the final
            # noise estimation. The hard VAD index I, computed in the previous
            # estimation, is here used to exclude strong speech components.
            norm                  = self.fsmooth(I)
            self.tilde_Sf         = self.fsmooth(I*np.abs(Y_l)**2)
            self.tilde_Sf[norm>0] = self.tilde_Sf[norm>0]/norm[norm>0]
            # Time smoothing
            self.tilde_S       = self.alpha_s*self.tilde_S+(1-self.alpha_s)*self.tilde_Sf                 # [3,eq.27]
            # Update running minimum
            self.tilde_Smin     = np.min(np.concatenate((self.tilde_Smin,self.tilde_S),1),1)[:,None]      # [3,eq.26]
            self.tilde_Smin_sw  = np.min(np.concatenate((self.tilde_Smin_sw,self.tilde_S),1),1)[:,None]   # [3,eq.27]
            # A PRIORI SPEECH ABSENCE
            tilde_Gamma_min  = (np.abs(Y_l)**2)/(self.Bmin*self.tilde_Smin)
            tilde_zeta       = self.S/(self.Bmin*self.tilde_Smin)                      # [3,eq.28]
            # Speech absence
            self.q                                                     = np.zeros(Y_l.shape)
            self.q[(tilde_Gamma_min <= 1) & (tilde_zeta < self.zeta0)] = 1                        # [3,eq.29]
            self.q[(1 < tilde_Gamma_min) & (tilde_Gamma_min < self.Gamma1) & (tilde_zeta < self.zeta0)] = (self.Gamma1 - tilde_Gamma_min[(1 < tilde_Gamma_min) & (tilde_Gamma_min < self.Gamma1) & (tilde_zeta < self.zeta0)])/(self.Gamma1-1)  # [3,Eq.29]
            # self.q[:]     = 1

            # A POSTERIORI SPEECH PROBABILITY
            self.p           = post_speech_prob(Y_l,self.q,Gamma,xi)
            self.p[self.p>self.p_upthr]     = self.p_upthr

            # PROBABILITY DRIVEN RECURSIVE SMOOTHING
            # Smoothing parameter
            tilde_alpha_d    = self.alpha_d+(1-self.alpha_d)*self.p                                     # [3,eq.11]
            # UPDATE NOISE SPECTRUM ESTIMATE
            self.ov_Lambda_D = tilde_alpha_d*self.ov_Lambda_D + (1-tilde_alpha_d)*np.abs(Y_l)**2     # [3,eq.10]
            # Bias correction
            self.Lambda_D    = self.beta*self.ov_Lambda_D                                             # [3,eq.12]

            # UPDATE MINIMUM TRACKING
            self.j += 1
            if self.j == self.V:

                # Minimum tracking for the first estimation
                if self.u < self.U:
                    self.Storing[:,self.u:self.u+1] = self.Smin_sw

                else:
                    self.Storing                    = np.roll(self.Storing,-1,axis=1)
                    self.Storing[:,-1:]             = self.Smin_sw

                # Set Smin to minimum
                self.Smin = np.min(self.Storing[:,:self.u+1],1)[:,None]
                # Let Smin_sw = S
                self.Smin_sw = self.S

                # Minimum traking for the second estimation
                if self.u < self.U:
                    self.tilde_Storing[:,self.u:self.u+1] = self.tilde_Smin_sw
                else:
                    self.tilde_Storing                    = np.roll(self.tilde_Storing,-1,axis=1)
                    self.tilde_Storing[:,-1:]             = self.tilde_Smin_sw

                # Set Smin to minimum
                self.tilde_Smin    = np.min(self.tilde_Storing[:,:self.u+1],1)[:,None]
                # Let Smin_sw = tilde_S
                self.tilde_Smin_sw = self.tilde_S
                # reset counter
                self.j = 0
                # Increase counter of buffers
                self.u                 += 1


        return [self.Lambda_D, self.p]


class imcra_est():
    '''
    Simple class for enhancement using IMCRA 
    '''
    def __init__(self, nfft, Lambda_D=None, Bmin=3.2, alpha =0.92, xi_min=10**(-25./20), IS=15):
        self.L_MAX = 2000
        # Decision directed smoothing factor
        self.alpha  = alpha
        # Decision directed a priori SNR floor
        self.xi_min = xi_min
        self.nfft = int(nfft/2+1)
        #
        self.store             = {}
        self.store['Lambda_D'] = np.zeros((self.nfft, self.L_MAX))
        self.store['p']        = np.zeros((self.nfft, self.L_MAX))
        self.store['xi']       = np.zeros((self.nfft, self.L_MAX))
        self.store['MSE']       = np.zeros((self.nfft, self.L_MAX))
        self.l                 = 0

        # IMCRA initial background segment (frames) 
        # Initialization
        self.imcra = imcra(nfft, IS=IS, Bmin=Bmin)
        self.G     = 1
        self.p     = np.zeros([self.nfft, 1])

        # Initial noise estimate
        if Lambda_D is None:
            self.Lambda_D = 1e-6*np.ones([self.nfft, 1])
        else:
            self.Lambda_D = Lambda_D

    def reset(self,nfft=512,Bmin=3.2):
        self.imcra.reset(nfft, Bmin)

    def estimate(self, Y):

        N_PSD    = np.zeros(Y.shape, dtype=np.float32)
        G        = self.G
        Gamma    = G
        Lambda_D = self.Lambda_D 
        p        = self.p 
        K, L     = Y.shape

        if self.l + L > self.L_MAX:

            # If maximum size surpassed, add a new batch of zeroes
            self.store['Lambda_D'] = np.concatenate((self.store['Lambda_D'], 
                                                     np.zeros((K, self.L_MAX))), 1) 
            self.store['p']        = np.concatenate((self.store['p'], 
                                                     np.zeros((K, self.L_MAX))), 1)
            self.store['xi']        = np.concatenate((self.store['xi'], 
                                                     np.zeros((K, self.L_MAX))), 1)


        for l in np.arange(0, L):
        
            # A priori SNR, stationary parameter estimate (uses last Gamma)   
            xi_G              = (G**2)*Gamma
            # A posterori SNR
            Gamma             = (np.abs(Y[:, l:l+1])**2)/Lambda_D
            # A priori SNR, maximum likelihood estimate 
            xi_ML             = Gamma - 1
            xi_ML[xi_ML<1e-6] = 1e-6      
            # Decision directed rule
            xi                = self.alpha*xi_G + (1-self.alpha)*xi_ML
            # Flooring
            xi[xi<self.xi_min] = self.xi_min 

            # MMSE-LSA
            # Get Wiener gain
            G               = xi/(1 + xi) 
            # Residual MSE of Wiener filter
        
            # TODO: Store additional variables if solicited
            self.store['Lambda_D'][:, self.l:self.l+1]  = Lambda_D 
            self.store['p'][:, self.l:self.l+1]         = p
            self.store['xi'][:, self.l:self.l+1]        = xi
            self.l                                     += 1

            # IMCRA noise estimate and posterior speech probability for the
            # next iteration
            Lambda_D, p = self.imcra.update(Y[:, l:l+1], Gamma, xi)
            # if np.mean(p) > 0.1:
            #     print('Frame: %d Prob: %.4lf'%(l, np.mean(p)))
            N_PSD[:, l:l+1] = Lambda_D

        # Keep these for the next iteration
        self.G        = G
        self.Lambda_D = Lambda_D[:, -2:]
        self.p        = p

        return N_PSD