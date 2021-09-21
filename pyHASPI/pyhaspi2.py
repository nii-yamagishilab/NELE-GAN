'''
This is an unofficial Python implementation of Hearing-Aid Speech Perception Index (HASPI)
Usage:
    from pyhaspi2 import haspi_v2
    haspi(x,fx,y,fy, HL = np.zeros(6))
    haspi_v2(x, fx, y, fy, HL = np.zeros(6))  

    Where x is clean reference speech, fx is sampling rate of x, and y is degraded speech, and fy is sampling rate of y. HL is 6-dim vectors represents hearing loss at the 6 audiometric frequencies, with 0 normal.

Differences between original Matlab implementation by Prof. James M.Kates:
Differences between the original Matlab implementation by Prof. James M.Kates:
    1). In our implementation, we did not apply built-in alignment, so please make sure x and y are already aligned before using this function
    2). In our implementation of HASPI version 2, we use sigmoid weight approximation model shown in Eq.(1) in [2] instead of neural network weights
    3). We internally normalized both x and y into rms=1, and SPL level was set to 65 dB
    4). We only confirmed output results are correct in the case of HL=[0,0,0,0,0,0] (as normal hearing condition). Although HL vector can be set arbitrary in the function, we cannot guarantee results are correct since lack of extensive test.

Reference:
    [1]. Kates, James M., and Kathryn H. Arehart. "The Hearing-Aid Speech Perception Index (HASPI)." Speech Communication 65 (2014): 75-93.
    [2]. Kates, James M., and Kathryn H. Arehart. "The Hearing-Aid Speech Perception Index (HASPI) Version 2." Speech Communication (2020).
'''

import numpy as np
import librosa
import scipy
from scipy.signal import lfilter, group_delay
from numba import jit
from scipy.interpolate import interp1d
import pdb
import time


def hasqi_v2(x, fx, y, fy, HL=np.zeros(6)):
    L = min(len(x), len(y))
    x = x[:L]
    y = y[:L]
    rms_x = np.sqrt(np.sum(x**2) / L)
    rms_y = np.sqrt(np.sum(y**2) / L)
    x = x / rms_x
    y = y / rms_y
    Level1 = 65
    eq = 2
    # HASQI_v2(x,fx,y,fy,HL,eq,Level1);
    xenv, xBM, yenv, yBM, xSL, ySL, fsamp = eb_EarModel(x,fx,y,fy,HL,eq,Level1)
    segsize = 16
    xdB = eb_EnvSmooth(xenv, segsize, fsamp)
    ydB = eb_EnvSmooth(yenv, segsize, fsamp)

    thr = 2.5
    addnoise = 0.0
    CepCorr, xy = eb_melcor(xdB, ydB, thr, addnoise)
    dloud, dnorm, dslope = eb_SpectDiff(xSL, ySL)
    
    segcov = 16
    sigcov,sigMSx,sigMSy = eb_BMcovary(xBM,yBM,segcov,fsamp)
    thr = 2.5
    avecov, syncov = eb_AveCovary2(sigcov, sigMSx, thr)
    BMsync5 = syncov[4]

    d = dloud[1]
    d = d / 2.5
    d = 1.0 - d
    d = np.clip(d, a_min=0, a_max=1)
    Dloud = d

    d = dslope[1]
    d = 1.0 - d
    d = np.clip(d, a_min=0, a_max=1)
    Dslope = d 

    Nonlin = (CepCorr**2) * BMsync5
    Linear = 0.579*Dloud + 0.421*Dslope
    Combined = Nonlin * Linear
    raw = [CepCorr, BMsync5, Dloud, Dslope]
    return Combined,Nonlin,Linear,raw

def haspi_v2(x, fx, y, fy, HL = np.zeros(6)):
    Level1 = 65
    L = min(len(x), len(y))
    x = x[:L]
    y = y[:L]
    rms_x = np.sqrt(np.sum(x**2) / L)
    rms_y = np.sqrt(np.sum(y**2) / L)
    x = x / rms_x
    y = y / rms_y

    itype = 0 # Intelligibility model
    xenv, xBM, yenv, yBM, xSL, ySL, fsamp = eb_EarModel(x,fx,y,fy,HL,itype,Level1)

    fLP=320
    fsub=8*fLP
    xLP, yLP = ebm_EnvFilt(xenv,yenv,fLP,fsub,fsamp)    

    # Compute the cepstral coefficients as a function of subsampled time
    nbasis=6 # Use 6 basis functions
    thr=2.5 # Silence threshold in dB SL
    dither=0.1 # Dither in dB RMS to add to envelope signals
    xcep, ycep = ebm_CepCoef(xLP,yLP,thr,dither,nbasis)
    # Cepstral coefficients filtered at each modulation rate
    # Band center frequencies [2, 6, 10, 16, 25, 40, 64, 100, 160, 256] Hz
    # Band edges [0, 4, 8, 12.5, 20.5, 30.5, 52.4, 78.1, 128, 200, 328] Hz
    xmod,ymod,cfmod = ebm_ModFilt(xcep,ycep,fsub)
    aveCM = ebm_ModCorr(xmod, ymod)
    weights = np.array([1.361, 1.521, 1.164, 0.492, 0.436, 0.690, 1.142, 0.816, 1.576, 2.269])
    Intel = np.sum(weights*aveCM)
    raw = aveCM

    return Intel, raw 

def haspi(x, fx, y, fy, HL = np.zeros(6), alpha = -1.0):
    Level1 = 65
    L = min(len(x), len(y))
    x = x[:L]
    y = y[:L]
    rms_x = np.sqrt(np.sum(x**2) / L)
    rms_y = np.sqrt(np.sum(y**2) / L)
    x = x / rms_x
    y = y / rms_y

    itype = 0 # Intelligibility model
    xenv, xBM, yenv, yBM, xSL, ySL, fsamp = eb_EarModel(x,fx,y,fy,HL,itype,Level1)

    # Smooth the envelope outputs: 125 Hz sub-sampling rate
    segsize = 16 # Averaging segment size in msec
    xdB = eb_EnvSmooth(xenv, segsize, fsamp)
    ydB = eb_EnvSmooth(yenv, segsize, fsamp)

    # Mel cepstrum correlation using smoothed envelopes
    # m1=ave of coefficients 2-6
    # xy=vector of coefficients 1-6
    thr=2.5  # Silence threshold: sum across bands, dB above aud threshold
    addnoise=0.0 # Additive noise in dB SL to condition cross-covariances
    CepCorr, xy = eb_melcor(xdB, ydB, thr, addnoise)

    # Temporal fine structure correlation measurements
    # Compute the time-frequency segment covariances
    segcov=16 # Segment size for the covariance calculation
    sigcov,sigMSx,sigMSy = eb_BMcovary(xBM,yBM,segcov,fsamp)

    # Three-level signal segment covariance
    # cov3 vector:   [low, mid, high] intensity region average, uniform weights
    # covSII vector: [low, mid, high] with SII frequency band weights
    cov3,covSII = eb_3LevelCovary(sigcov,sigMSx,thr)

    # Intelligibility prediction
    # Combine the cepstral correlation and three-level covariance
    bias = -9.047
    wgtcep = 14.816
    wgtcov = np.array([0, 0, 4.616]) # [low, mid, high]
    arg = bias + wgtcep * CepCorr + np.sum(wgtcov * cov3)

    # Logsig transformation
    Intel = 1.0 / (1.0 + np.exp(alpha * arg)) # Logistic (logsig) function, default alpha = -1.0

    # Raw data
    raw = np.concatenate((np.array([CepCorr]), cov3))

    return Intel, raw 



def eb_AveCovary2(sigcov, sigMSx, thr):
    nchan = sigcov.shape[0]
    cfreq = eb_CenterFreq(nchan)
    p = np.array([1, 3, 5, 5, 5, 5])
    fcut = 1000*np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    fsync = np.zeros([6, nchan])
    for n in range(6):
        fc2p = fcut[n]**(2*p[n])
        freq2p = cfreq ** (2*p[n])
        fsync[n, :] = np.sqrt(fc2p/(fc2p+freq2p))
    
    sigRMS = np.sqrt(sigMSx)
    sigLinear = 10**(sigRMS/20)
    xsum = np.sum(sigLinear, 0) / nchan
    xsum = 20*np.log10(xsum)
    index = np.where(xsum > thr)[0]
    nseg = len(index)
    if nseg<=1:
        return 0, 0
    
    sigcov = sigcov[:, index]
    sigRMS = sigRMS[:, index]
    weight = np.zeros([nchan, nseg])
    wsync1 = np.zeros([nchan, nseg])
    wsync2 = np.zeros([nchan, nseg])
    wsync3 = np.zeros([nchan, nseg])
    wsync4 = np.zeros([nchan, nseg])
    wsync5 = np.zeros([nchan, nseg])
    wsync6 = np.zeros([nchan, nseg])

    for k in range(nchan):
        for n in range(nseg):
            if sigRMS[k,n] > thr:
                weight[k,n] = 1
                wsync1[k,n] = fsync[0, k]
                wsync2[k,n] = fsync[1, k]
                wsync3[k,n] = fsync[2, k]
                wsync4[k,n] = fsync[3, k]
                wsync5[k,n] = fsync[4, k]
                wsync6[k,n] = fsync[5, k]

    csum = np.sum(weight*sigcov)
    wsum = np.sum(weight)
    fsum = np.zeros(6)
    ssum = np.zeros(6)
    fsum[0] = np.sum(wsync1*sigcov)
    ssum[0] = np.sum(wsync1)
    fsum[1] = np.sum(wsync2*sigcov)
    ssum[1] = np.sum(wsync2)
    fsum[2] = np.sum(wsync3*sigcov)
    ssum[2] = np.sum(wsync3)
    fsum[3] = np.sum(wsync4*sigcov)
    ssum[3] = np.sum(wsync4)
    fsum[4] = np.sum(wsync5*sigcov)
    ssum[4] = np.sum(wsync5)
    fsum[5] = np.sum(wsync6*sigcov)
    ssum[5] = np.sum(wsync6)

    if wsum < 1:
        return 0, fsum/ssum
    else:
        return csum/wsum, fsum/ssum

def eb_SpectDiff(xSL, ySL):
    nbands = len(xSL)
    x = 10**(xSL/20)
    y = 10**(ySL/20)
    xsum = np.sum(x)
    x = x / xsum
    ysum = np.sum(y)
    y = y / ysum
    dloud = np.zeros(3)
    d = x-y
    dloud[0] = np.sum(np.abs(d))
    dloud[1] = nbands*np.std(d)
    dloud[2] = np.max(np.abs(d))

    dnorm = np.zeros(3)
    d = (x-y) / (x+y)
    dnorm[0] = np.sum(np.abs(d))
    dnorm[1] = nbands*np.std(d)
    dnorm[2] = np.max(np.abs(d))

    dslope = np.zeros(3)
    dx = (x[1:] - x[0:-1])
    dy = (y[1:] - y[0:-1])
    d = dx - dy
    dslope[0] = np.sum(np.abs(d))
    dslope[1] = nbands*np.std(d)
    dslope[2] = np.max(np.abs(d))
    return dloud, dnorm, dslope


def ebm_ModCorr(Xmod,Ymod):
    nchan = len(Xmod)
    nmod = len(Xmod[0])
    small = 1.0e-30
    CM = np.zeros([nchan,nmod])
    for m in range(nmod):
        for j in range(nchan):
            xj = Xmod[j][m]
            xj = xj - np.mean(xj)
            xsum = np.sum(xj**2)
            yj = Ymod[j][m]
            yj = yj - np.mean(yj)
            ysum = np.sum(yj**2)
            if xsum < small or ysum < small:
                CM[j,m] = 0
            else:
                CM[j,m] = np.abs(np.sum(xj*yj)) / np.sqrt(xsum*ysum)
    
    aveCM = np.mean(CM[1:6], axis=0) # Average over basis functions 2 - 6
    return aveCM

def ebm_ModFilt(Xenv,Yenv,fsub):
    nsamp, nchan = Xenv.shape[0], Xenv.shape[1]
    cf = np.array([2, 6, 10, 16, 25, 40, 64, 100, 160, 256])
    nmod = len(cf)
    edge = np.zeros(nmod+1)
    edge[0] = 0
    edge[1] = 4
    edge[2] = 8
    for k in range(3, nmod+1):
        # Log spacing for remaining constant-Q modulation filters
        edge[k] = cf[k-1]**2 / edge[k-1]

    # Allowable filters based on envelope subsampling rate
    fNyq = 0.5*fsub
    index = np.where(edge < fNyq)[0]
    edge = edge[index] # Filter upper band edges less than Nyquist rate
    nmod = len(edge) - 1
    cf = cf[0:nmod]
    t0 = 0.24
    t = np.zeros(nmod)
    t[0] = t0
    t[1] = t0
    t[2:nmod] = t0*cf[2]/cf[2:nmod] # Constant-Q filters above 10 Hz
    nfir = 2*np.floor(t*fsub/2)
    nhalf = nfir/2
    # Design the family of lowpass windows
    b = []
    for k in range(nmod):
        tmp = np.hanning(int(nfir[k])+1)
        tmp = tmp / np.sum(tmp)
        b.append(tmp)
    co = []
    si = []
    n = np.arange(1, nsamp+1)
    for k in range(nmod):
        if k==0:
            co.append(1)
            si.append(0)
        else:
            tmp = np.sqrt(2)*np.cos(np.pi*n*cf[k]/fNyq)
            co.append(tmp)
            tmp = np.sqrt(2)*np.sin(np.pi*n*cf[k]/fNyq)
            si.append(tmp)
    
    Xmod = [[None]*nmod for _ in range(nchan)]
    Ymod = [[None]*nmod for _ in range(nchan)]
    for k in range(nmod):
        bk = b[k]
        nh = int(nhalf[k])
        c = co[k]
        s = si[k]
        for m in range(nchan):
            x = Xenv[:, m]
            u = np.convolve((x*c-1j*x*s), bk)
            u = u[nh:nh+nsamp]
            xfilt = np.real(u)*c - np.imag(u)*s
            Xmod[m][k] = xfilt

            y = Yenv[:, m]
            v = np.convolve((y*c-1j*y*s), bk)
            v = v[nh:nh+nsamp]
            yfilt = np.real(v)*c - np.imag(v)*s
            Ymod[m][k] = yfilt
    
    return Xmod, Ymod, cf


def ebm_CepCoef(xdB,ydB,thrCep,thrNerve,nbasis):
    nbands=xdB.shape[1]
    freq = np.arange(0, nbasis)
    k = np.arange(0, nbands)
    cepm = np.zeros([nbands, nbasis])
    for nb in range(nbasis):
        basis = np.cos(freq[nb]*np.pi*k/(nbands-1))
        cepm[:, nb] = basis / np.linalg.norm(basis)

    # Find the segments that lie sufficiently above the quiescent rate
    xLinear = np.power(10, (xdB/20)) # Convert envelope dB to linear (specific loudness)
    xsum = np.sum(xLinear,axis=1) / nbands # Proportional to loudness in sones
    xsum = 20*np.log10(xsum) # Convert back to dB (loudness in phons)
    index = np.where(xsum > thrCep)[0]
    nsamp=len(index) # Number of segments above threshold
    if nsamp <= 1:
        raise Exception('Function ebm_CepCoef: Signal below threshold')
    xdB = xdB[index,:]
    ydB = ydB[index,:]
    # Add low-level noise to provide IHC firing jitter
    noise = thrNerve * np.random.randn(xdB.shape[0], xdB.shape[1])
    xdB = xdB + noise
    noise = thrNerve * np.random.randn(ydB.shape[0], ydB.shape[1])
    ydB = ydB + noise
    xcep = np.matmul(xdB, cepm)
    ycep = np.matmul(ydB, cepm)
    for n in range(nbasis):
        x = xcep[:,n]
        x = x - np.mean(x)
        xcep[:,n] = x
        y = ycep[:,n]
        y = y - np.mean(y)
        ycep[:,n] = y
    return xcep, ycep


def ebm_EnvFilt(xdB,ydB,fcut,fsub,fsamp):
    if fsub > fsamp:
        raise Exception('Error in ebm_EnvFilt: Subsampling rate too high')
    if fcut > 0.5 * fsub:
        raise Exception('Error in ebm_EnvFilt: LP cutoff frequency too high')

    nrow, ncol = xdB.shape[0], xdB.shape[1]
    if ncol > nrow:
        xdB = xdB.T
        ydB = ydB.T
    nsamp, nbands = xdB.shape[0], xdB.shape[1]
    # Compute the lowpass filter length in samples to give -3 dB at fcut Hz
    tfilt=1000*(1/fcut)
    tfilt=0.7*tfilt
    nfilt=round(0.001*tfilt*fsamp)
    nhalf = int(nfilt//2)
    nfilt = 2*nhalf

    # Design the FIR LP filter using a von Hann window to ensure that there are
    # no negative envelope values
    benv = np.hanning(nfilt)
    benv = benv / np.sum(benv)
    # LP filter for the envelopes at fsamp
    xenv = np.zeros([nfilt+nsamp-1,nbands])
    yenv = np.zeros([nfilt+nsamp-1,nbands])
    for n in range(nbands):
        xenv[:,n] = np.convolve(xdB[:,n], benv)
        yenv[:,n] = np.convolve(ydB[:,n], benv)
    xenv = xenv[nhalf:nhalf+nsamp,:]
    yenv = yenv[nhalf:nhalf+nsamp,:]
    
    # Subsample the LP filtered envelopes
    space = int(fsamp//fsub)
    index = np.arange(0, nsamp, space)
    xLP = xenv[index,:]
    yLP = yenv[index,:]
    return xLP, yLP



def eb_3LevelCovary(sigcov,sigMSx,thr):
    cov3 = np.zeros(3)
    covSII = np.zeros(3)
    nbands = sigcov.shape[0]
    # Find the segments that lie sufficiently above the threshold.
    sigRMS = np.sqrt(sigMSx)
    sigLinear = np.power(10, (sigRMS/20))
    xsum = np.sum(sigLinear, axis=0) / nbands
    xsum = 20*np.log10(xsum)
    index = np.where(xsum > thr)[0]
    nseg = len(index)
    if nseg <= 1:
        raise Exception('Function eb_3LevelCovary: Signal below threshold, outputs set to 0.')
    # Critical band center frequencies in Hz
    cfSII = [150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000, 8500]
    # cfSII = np.array(cfSII)
    # Weights for the Speech Intelligibility Index calculation
    wgtSII=[.0103, .0261, .0419, .0577, .0577, .0577, .0577, .0577, .0577, .0577, .0577, .0577, .0577, .0577, .0577, .0577, .0577, .0460, .0343, .0226, .0110]
    cfreq=eb_CenterFreq(nbands)
    fsamp=24000
    cfSII = [0] + cfSII + [fsamp]
    wgtSII = [0] + wgtSII + [0]
    CubicSpl = interp1d(np.array(cfSII), np.array(wgtSII), kind='cubic')
    wfreq = CubicSpl(cfreq)
    wfreq[0] = 0.0
    wfreq[1] = 0.0
    wfreq = wfreq / np.sum(wfreq)
    sigcov=sigcov[:,index]
    sigRMS=sigRMS[:,index]
    xsum = xsum[index]
    # Histogram of the segment intensities in phons
    dBmin = np.min(xsum)
    dBmax = np.max(xsum)
    dBstep = 0.5 # Bin width is 0.5 dB
    bins = np.arange(dBmin, dBmax+dBstep, dBstep)
    bin_pinf = 1e8
    bin_ninf = -1e8
    bin_max = np.concatenate((bins[1:], np.array([bin_pinf])))
    bin_tmp = (bins+bin_max) / 2
    bin_tmp = np.concatenate((np.array([bin_ninf]),bin_tmp))
    xhist, _ = np.histogram(xsum, bin_tmp)
    bincenters = bins
    nbins = len(xhist)

    # Compute the cumulative histogram
    xcum=np.zeros(nbins)
    xcum[0] = xhist[0]
    for k in range(1, nbins):
        xcum[k] = xcum[k-1] + xhist[k]
    
    xcum = xcum / xcum[nbins-1]
    # Find the boundaries for the lower, middle, and upper thirds
    edge=np.zeros(2)
    for n in range(nbins):
        if xcum[n] < 0.333:
            edge[0] = bincenters[n]
        if xcum[n] < 0.667:
            edge[1] = bincenters[n]

    # Assign segments to the lower, middle, and upper thirds
    low = np.where(xsum < edge[0])[0] # Segment indices for lower third
    tmp1 = (xsum >= edge[0])
    tmp2 = (xsum < edge[1])
    mid = np.where(tmp1*tmp2)[0]
    up = np.where(xsum>=edge[1])[0]

    # Compute the time-frequency weights. The weight=1 if a segment in a
    # frequency band is above threshold, and weight=0 if at or below threshold.
    weight = np.zeros([nbands, nseg])
    weight[sigRMS > thr] = 1
    sigcov = weight * sigcov # Apply the weights
    # Average the covariance across segment levels as a function of frequency
    cov_ave = np.zeros(nbands)
    cov_ave_SII = np.zeros(nbands)

    # Low-level segments
    s=sigcov[:,low] # Segment covariances for the low intensity
    w=weight[:,low] # Low intensity time-freq segments above threshold
    ssum = np.sum(s,axis=1) # Sum over the low-intensity segments in each frequency band
    wsum = np.sum(w,axis=1) # Sum of the above-threshold weights in each frequency band
    ncount=0
    wgtsum=0
    for n in range(nbands):
        if wsum[n]==0:
            cov_ave[n] = 0
            cov_ave_SII[n] = 0
        else:
            cov_ave[n] = ssum[n] / wsum[n]
            cov_ave_SII[n] = cov_ave[n] * wfreq[n]
            wgtsum = wgtsum + wfreq[n]
            ncount = ncount + 1
    cov3[0] = np.sum(cov_ave) / ncount
    covSII[0] = np.sum(cov_ave_SII) / wgtsum
    # Mid-level segments
    s=sigcov[:,mid] # Segment covariances for the mid intensity
    w=weight[:,mid] # Mid intensity time-freq segments above threshold
    ssum = np.sum(s,axis=1) # Sum over the mid-intensity segments in each frequency band
    wsum = np.sum(w,axis=1) # Sum of the above-threshold weights in each frequency band
    ncount=0
    wgtsum=0
    for n in range(nbands):
        if wsum[n]==0:
            cov_ave[n] = 0
            cov_ave_SII[n] = 0
        else:
            cov_ave[n] = ssum[n] / wsum[n]
            cov_ave_SII[n] = cov_ave[n] * wfreq[n]
            wgtsum = wgtsum + wfreq[n]
            ncount = ncount + 1
    cov3[1] = np.sum(cov_ave) / ncount
    covSII[1] = np.sum(cov_ave_SII) / wgtsum
    # High-level segments
    s=sigcov[:,up] # Segment covariances for the high intensity
    w=weight[:,up] # High intensity time-freq segments above threshold
    ssum = np.sum(s,axis=1) # Sum over the high-intensity segments in each frequency band
    wsum = np.sum(w,axis=1) # Sum of the above-threshold weights in each frequency band
    ncount=0
    wgtsum=0
    for n in range(nbands):
        if wsum[n]==0:
            cov_ave[n] = 0
            cov_ave_SII[n] = 0
        else:
            cov_ave[n] = ssum[n] / wsum[n]
            cov_ave_SII[n] = cov_ave[n] * wfreq[n]
            wgtsum = wgtsum + wfreq[n]
            ncount = ncount + 1
    cov3[2] = np.sum(cov_ave) / ncount
    covSII[2] = np.sum(cov_ave_SII) / wgtsum
    return cov3, covSII


def eb_BMcovary(xBM,yBM,segsize,fsamp):
    small=1.0e-30

    # Lag for computing the cross-covariance
    lagsize=1.0 # Lag (+/-) in msec
    maxlag=round(lagsize*(0.001*fsamp)) # Lag in samples

    nwin=round(segsize*(0.001*fsamp))
    test=nwin - 2*np.floor(nwin/2)
    if test>0:
        nwin = nwin + 1
    window=np.hanning(nwin) # Raised cosine von Hann window

    # TODO: translate this matlab code: wincorr=1./xcorr(window,window,maxlag);
    wincorr = np.array([0.00714486118736300, 0.00712980452227938, 0.00711541940039143, 0.00710170170277407, 0.00708864751353705, 0.00707625311737229, 0.00706451499720106, 0.00705342983191828, 0.00704299449423103, 0.00703320604858884, 0.00702406174920299, 0.00701555903815223, 0.00700769554357245, 0.00700046907792766, 0.00699387763635983, 0.00698791939511504, 0.00698259271004346, 0.00697789611517068, 0.00697382832133786, 0.00697038821490824, 0.00696757485653739, 0.00696538748000482, 0.00696382549110426, 0.00696288846658999, 0.00696257615317668, 0.00696288846658999, 0.00696382549110427, 0.00696538748000482, 0.00696757485653739, 0.00697038821490824, 0.00697382832133786, 0.00697789611517068, 0.00698259271004346, 0.00698791939511504, 0.00699387763635983, 0.00700046907792766, 0.00700769554357245, 0.00701555903815223, 0.00702406174920299, 0.00703320604858884, 0.00704299449423103, 0.00705342983191828, 0.00706451499720106, 0.00707625311737229, 0.00708864751353705, 0.00710170170277407, 0.00711541940039143, 0.00712980452227939, 0.00714486118736301])
    winsum2 = 1.0/np.sum(window**2)

    nhalf=nwin // 2
    halfwindow = window[nhalf:]

    # TODO: translate this matlab code: halfcorr=1./xcorr(halfwindow,halfwindow,maxlag);
    halfcorr = np.array([0.0171564012932667, 0.0169783665111901, 0.0168048368330894, 0.0166356935491691, 0.0164708221425357, 0.0163101121243555, 0.0161534568765699, 0.0160007535017831, 0.0158519026799599, 0.0157068085315890, 0.0155653784869908, 0.0154275231614608, 0.0152931562359640, 0.0151621943431025, 0.0150345569581033, 0.0149101662945780, 0.0147889472048270, 0.0146708270844672, 0.0145557357811784, 0.0144436055073713, 0.0143343707565927, 0.0142279682234914, 0.0141243367271791, 0.0140234171378276, 0.0139251523063533, 0.0140234171378276, 0.0141243367271791, 0.0142279682234914, 0.0143343707565927, 0.0144436055073713, 0.0145557357811784, 0.0146708270844672, 0.0147889472048270, 0.0149101662945780, 0.0150345569581033, 0.0151621943431025, 0.0152931562359640, 0.0154275231614608, 0.0155653784869908, 0.0157068085315891, 0.0158519026799599, 0.0160007535017831, 0.0161534568765699, 0.0163101121243555, 0.0164708221425357, 0.0166356935491691, 0.0168048368330894, 0.0169783665111901, 0.0171564012932667])
    halfsum2 = 1.0 / np.sum(halfwindow**2) # MS sum normalization, first segment

    nchan=xBM.shape[0]
    npts=xBM.shape[1]
    nseg = 1 + np.floor(npts/nwin) + np.floor((npts-nwin/2)/nwin)
    nseg = int(nseg)

    sigMSx = np.zeros(nchan*nseg)
    sigMSy = np.zeros(nchan*nseg)
    sigcov = np.zeros(nchan*nseg)

    for k in range(nchan):
        x = xBM[k,:]
        y = yBM[k,:]
        nstart = 0
        segx = x[nstart:nhalf] * halfwindow
        segy = y[nstart:nhalf] * halfwindow
        segx = segx - np.mean(segx)
        segy = segy - np.mean(segy)
        MSx = np.sum(segx**2) * halfsum2
        MSy = np.sum(segy**2) * halfsum2
        my_xcorr = np.correlate(segx,segy,'full')
        if len(my_xcorr) > 2*maxlag+1:
            start = (len(my_xcorr) - (2*maxlag+1)) // 2
            my_xcorr = my_xcorr[start: start+(2*maxlag+1)]

        Mxy = np.max(np.abs(my_xcorr * halfcorr))
        if (MSx > small) and (MSy > small):
            sigcov[k*nseg+0] = Mxy / np.sqrt(MSx*MSy)
        else:
            sigcov[k*nseg+0] = 0.0
        sigMSx[k*nseg+0] = MSx
        sigMSy[k*nseg+0] = MSy
        
        for n in range(1, nseg-1):
            nstart = nstart + nhalf
            nstop = nstart + nwin
            segx = x[nstart:nstop] * window
            segy = y[nstart:nstop] * window
            segx = segx - np.mean(segx)
            segy = segy - np.mean(segy)
            MSx = np.sum(segx**2) * winsum2
            MSy = np.sum(segy**2) * winsum2
            my_xcorr = np.correlate(segx,segy,'full')
            if len(my_xcorr) > 2*maxlag+1:
                start = (len(my_xcorr) - (2*maxlag+1)) // 2
                my_xcorr = my_xcorr[start: start+(2*maxlag+1)]

            Mxy = np.max(np.abs(my_xcorr * wincorr))
            if (MSx > small) and (MSy > small):
                sigcov[k*nseg+n] = Mxy / np.sqrt(MSx*MSy)
            else:
                sigcov[k*nseg+n] = 0.0
            sigMSx[k*nseg+n] = MSx
            sigMSy[k*nseg+n] = MSy
        
        nstart = nstart + nhalf
        nstop = nstart + nhalf
        segx=x[nstart:nstop]*window[:nhalf]
        segy=y[nstart:nstop]*window[:nhalf]
        segx = segx - np.mean(segx)
        segy = segy - np.mean(segy)
        MSx = np.sum(segx**2) * halfsum2
        MSy = np.sum(segy**2) * halfsum2
        my_xcorr = np.correlate(segx,segy,'full')
        if len(my_xcorr) > 2*maxlag+1:
            start = (len(my_xcorr) - (2*maxlag+1)) // 2
            my_xcorr = my_xcorr[start: start+(2*maxlag+1)]

        Mxy = np.max(np.abs(my_xcorr * halfcorr))
        if (MSx > small) and (MSy > small):
            sigcov[k*nseg+nseg-1] = Mxy / np.sqrt(MSx*MSy)
        else:
            sigcov[k*nseg+nseg-1] = 0.0
        sigMSx[k*nseg+nseg-1] = MSx
        sigMSy[k*nseg+nseg-1] = MSy

    # Limit the cross-covariance to lie between 0 and 1
    sigcov = np.clip(sigcov, a_min=0, a_max=1)
    # Adjust the BM magnitude to correspond to the envelope in dB SL
    sigMSx=2.0*sigMSx
    sigMSy=2.0*sigMSy

    sigcov = sigcov.reshape(nchan, nseg)
    sigMSx = sigMSx.reshape(nchan, nseg)
    sigMSy = sigMSy.reshape(nchan, nseg)

    return sigcov,sigMSx,sigMSy


# def my_xcorr(x, y, lags):
#     xy = np.correlate(x,y,'full')
#     if len(xy) > 2*lags+1:
#         start = (len(xy) - (2*lags+1)) // 2
#         xy = xy[start: start+(2*lags+1)]
#     elif len(xy) < 2*lags+1:
#         pads = (2*lags+1 - len(xy))//2
#         xy = np.concatenate((np.zeros(pads), xy, np.zeros(pads)))
#     return xy


@jit(nopython=True)
def eb_EnvSmooth(env,segsize,fsamp):
    nwin=round(segsize*(0.001*fsamp))
    test=nwin - 2*np.floor(nwin/2)
    if test>0:
        nwin = nwin + 1
    window=np.hanning(nwin) # Raised cosine von Hann window
    wsum=np.sum(window) # Sum for normalization
    nhalf=nwin//2
    halfwindow = window[nhalf:]
    halfsum = np.sum(halfwindow)
    # Number of segments and assign the matrix storage
    nchan = env.shape[0]
    npts = env.shape[1]
    nseg = 1 + np.floor(npts/nwin) + np.floor((npts-nwin/2)/nwin)
    nseg = int(nseg)
    smooth = np.zeros(nchan*nseg)
    for k in range(nchan):
        r = env[k, :]
        nstart = 0
        smooth[k*nseg + 0]= np.sum(r[nstart:nhalf] * halfwindow) / halfsum
        for n in range(1, nseg-1):
            nstart = nstart + nhalf
            nstop = nstart + nwin
            smooth[k*nseg+n] = np.sum(r[nstart:nstop] * window) / wsum
        nstart = nstart + nhalf
        nstop = nstart + nhalf
        smooth[k*nseg+nseg-1] = np.sum( r[nstart:nstop] * window[:nhalf] ) / halfsum
    
    smooth = smooth.reshape(nchan, nseg)
    return smooth


def eb_melcor(x,y,thr,addnoise):
    nbands = x.shape[0]
    nbasis=6 # Number of cepstral coefficients to be used
    freq = np.arange(0, nbasis)
    k = np.arange(0, nbands)
    cepm = np.zeros([nbands,nbasis])
    for nb in range(nbasis):
        basis = np.cos(freq[nb]*np.pi*k/(nbands-1))
        cepm[:, nb] = basis / np.linalg.norm(basis)

    # Find the segments that lie sufficiently above the quiescent rate
    xLinear = np.power(10, (x/20)) # Convert envelope dB to linear (specific loudness)
    xsum = np.sum(xLinear,axis=0) / nbands # Proportional to loudness in sones
    xsum = 20*np.log10(xsum) # Convert back to dB (loudness in phons)
    index = np.where(xsum > thr)[0]
    nsamp=len(index) # Number of segments above threshold
    if nsamp <= 1:
        raise Exception('Function eb_melcor: Signal below threshold, outputs set to 0.')
    x=x[:,index]
    y=y[:,index]
    if addnoise != 0.0:
        print('add noise in eb_melcor func!')
        x = x + addnoise * np.random.randn(x.shape[0], x.shape[1])
        y = y + addnoise * np.random.randn(y.shape[0], y.shape[1])
    xcep = np.zeros([nbasis,nsamp]) #Input
    ycep = np.zeros([nbasis,nsamp])  #Output
    for n in range(nsamp):
        for k in range(nbasis):
            xcep[k,n] = np.sum( x[:,n] * cepm[:,k]  )
            ycep[k,n] = np.sum( y[:,n] * cepm[:,k]  )

    for k in range(nbasis):
        xcep[k, :] = xcep[k, :] - np.mean(xcep[k, :])
        ycep[k, :] = ycep[k, :] - np.mean(ycep[k, :])

    xy=np.zeros(nbasis)
    small=1.0e-30
    for k in range(nbasis):
        xsum = np.sum( xcep[k,:]**2 )
        ysum = np.sum( ycep[k,:]**2 )
        if (xsum < small) or (ysum < small):
            xy[k] = 0.0
        else:
            xy[k] = np.abs( np.sum(xcep[k,:]*ycep[k,:]) / np.sqrt(xsum*ysum)  )
    m1 = np.sum(xy[1:]) / (nbasis-1)
    return m1, xy

def eb_CenterFreq(nchan, shift=None, placeholder=None):
    # Parameters for the filter bank
    lowFreq=80.0 # Lowest center frequency
    highFreq=8000.0 # Highest center frequency

    # Moore and Glasberg ERB values
    EarQ = 9.26449
    minBW = 24.7

    if shift and placeholder:
        k=1
        A=165.4
        a=2.1
        xLow = (1/a) * np.log10(k + (lowFreq/A))
        xHigh = (1/a) * np.log10(k + (highFreq/A))
        xLow=xLow*(1 + shift)
        xHigh=xHigh*(1 + shift)
        lowFreq = A * (np.power(10,(a*xLow)) - k)
        highFreq = A * (np.power(10,(a*xHigh)) - k)

    cf = -(EarQ * minBW) + np.exp(  np.arange(1, nchan) * (-np.log(highFreq + EarQ*minBW) + \
    np.log(lowFreq + EarQ*minBW))/(nchan-1)) * (highFreq + EarQ*minBW)
    cf = np.concatenate((np.array([highFreq]), cf)) # Last center frequency is set to highFreq
    cf = np.flipud(cf) # Reorder to put the low frequencies first
    return cf

def eb_LossParameters(HL, cfreq):
    aud = [250.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0]
    nfilt = len(cfreq)
    fv = [cfreq[0]] + aud + [cfreq[-1]]
    # TODO: simply use numpy interp to replace matlab interp1, might not be accurate!!!
    loss = np.interp(cfreq, fv, np.concatenate((np.array([HL[0]]),HL,np.array([HL[-1]]))))
    loss[loss<0] = 0.0
    # ---
    CR = np.zeros(nfilt)
    for i in range(nfilt):
        CR[i] = 1.25 + 2.25 * i / (nfilt-1)
    maxOHC = 70 * (1 - (1/CR))
    thrOHC=1.25*maxOHC
    attnOHC=np.zeros(nfilt) # Default is 0 dB attenuation
    attnIHC=np.zeros(nfilt)
    for i in range(nfilt):
        if loss[i] < thrOHC[i]:
            attnOHC[i]=0.8*loss[i]
            attnIHC[i]=0.2*loss[i]
        else:
            attnOHC[i]=0.8*thrOHC[i] # Maximum OHC attenuation
            attnIHC[i]=0.2*thrOHC[i] + (loss[i]-thrOHC[i])
    BW = np.ones(nfilt)
    BW = BW + (attnOHC / 50.0) + 2.0 * (attnOHC / 50.0)**6

    lowknee = attnOHC + 30 #  Lower kneepoint
    upamp = 30 + 70/CR # Output level for an input of 100 dB SPL
    CR = (100-lowknee)/(upamp + attnOHC - lowknee) # OHC loss Compression ratio
    return attnOHC,BW,lowknee,CR,attnIHC
        

def eb_Resamp24kHz(x,fsampx):
    fsamp = 24000
    if fsampx == fsamp:
        y = x
    elif fsampx < fsamp:
        y = librosa.resample(x, orig_sr=fsampx, target_sr=fsamp)
        xRMS = np.sqrt(np.mean(x**2))
        yRMS = np.sqrt(np.mean(y**2))
        y = (xRMS/yRMS) * y
    else:
        raise NotImplementedError
    return y, fsamp


def eb_InputAlign(x24, y24):
    nsamp = 24000
    raise NotImplementedError
    # Align x24 and y24
    return x24, y24

def eb_NALR(HL,nfir,fsamp):
    raise NotImplementedError

def eb_MiddleEar(x, fsamp):
    assert fsamp==24000
    bLP = np.array([0.434173751206302, 0.434173751206302])
    aLP = np.array([1.000000000000000, -0.131652497587396])
    y = lfilter(bLP, aLP, x)
    bHP = np.array([0.937260390269893, -1.874520780539785, 0.937260390269893])
    aHP = np.array([1.000000000000000, -1.870580640735279, 0.878460920344291])
    xout = lfilter(bHP, aHP, y)
    return xout

@jit(nopython=True)
def eb_CosSinCF(L, fs, cf):
    tpt = 2 * np.pi / fs
    npts = L
    cn=np.cos(tpt*cf)
    sn=np.sin(tpt*cf)
    coscf=np.zeros(npts)
    sincf=np.zeros(npts)
    cold=1
    sold=0
    coscf[0]=cold
    sincf[0]=sold
    for n in range(1, npts):
        arg=cold*cn + sold*sn
        sold=sold*cn - cold*sn
        cold=arg
        coscf[n]=cold
        sincf[n]=sold
    return coscf, sincf

def eb_GammatoneBM2(x,BWx,y,BWy,fs,cf):
    earQ=9.26449
    minBW=24.7
    ERB=minBW + (cf/earQ)
    # L = min(len(x),len(y))
    # x = x[:L]
    # y = y[:L]
    tpt = 2 * np.pi / fs
    tptBW=BWx*tpt*ERB*1.019
    a=np.exp(-tptBW)
    a1=4.0*a
    a2=-6.0*a*a
    a3=4.0*a*a*a
    a4=-a*a*a*a
    a5=4.0*a*a
    gain=2.0*(1-a1-a2-a3-a4)/(1+a1+a5)

    npts = len(x)
    coscf, sincf = eb_CosSinCF(npts, fs, cf)
    # cn=np.cos(tpt*cf)
    # sn=np.sin(tpt*cf)
    # coscf=np.zeros(npts)
    # sincf=np.zeros(npts)
    # cold=1
    # sold=0
    # coscf[0]=cold
    # sincf[0]=sold
    # for n in range(1, npts):
    #     arg=cold*cn + sold*sn
    #     sold=sold*cn - cold*sn
    #     cold=arg
    #     coscf[n]=cold
    #     sincf[n]=sold
    
    ureal = lfilter([1, a1, a5],[1, -a1, -a2, -a3, -a4], x * coscf)
    uimag = lfilter([1, a1, a5],[1, -a1, -a2, -a3, -a4], x * sincf)
    BMx=gain*(ureal*coscf + uimag*sincf)
    envx=gain*np.sqrt(ureal*ureal + uimag*uimag)

    tptBW=BWy*tpt*ERB*1.019
    a=np.exp(-tptBW)
    a1=4.0*a
    a2=-6.0*a*a
    a3=4.0*a*a*a
    a4=-a*a*a*a
    a5=4.0*a*a
    gain=2.0*(1-a1-a2-a3-a4)/(1+a1+a5)

    ureal = lfilter([1, a1, a5],[1, -a1, -a2, -a3, -a4], y * coscf)
    uimag = lfilter([1, a1, a5],[1, -a1, -a2, -a3, -a4], y * sincf)
    BMy = gain*(ureal*coscf + uimag*sincf)
    envy = gain * np.sqrt(ureal*ureal + uimag*uimag)
    return envx,BMx,envy,BMy

def eb_GammatoneEnv2(x,BWx,y,BWy,fs,cf):
    earQ=9.26449
    minBW=24.7
    ERB=minBW + (cf/earQ)
    # L = min(len(x),len(y))
    # x = x[:L]
    # y = y[:L]
    tpt = 2 * np.pi / fs
    tptBW=BWx*tpt*ERB*1.019
    a=np.exp(-tptBW)
    a1=4.0*a
    a2=-6.0*a*a
    a3=4.0*a*a*a
    a4=-a*a*a*a
    a5=4.0*a*a
    gain=2.0*(1-a1-a2-a3-a4)/(1+a1+a5)

    npts = len(x)
    coscf, sincf = eb_CosSinCF(npts, fs, cf)
    # cn=np.cos(tpt*cf)
    # sn=np.sin(tpt*cf)
    # coscf=np.zeros(npts)
    # sincf=np.zeros(npts)
    # cold=1
    # sold=0
    # coscf[0]=cold
    # sincf[0]=sold
    # for n in range(1, npts):
    #     arg=cold*cn + sold*sn
    #     sold=sold*cn - cold*sn
    #     cold=arg
    #     coscf[n]=cold
    #     sincf[n]=sold
    
    ureal = lfilter([1, a1, a5],[1, -a1, -a2, -a3, -a4], x * coscf)
    uimag = lfilter([1, a1, a5],[1, -a1, -a2, -a3, -a4], x * sincf)
    envx = gain * np.sqrt(ureal*ureal + uimag*uimag)

    tptBW=BWy*tpt*ERB*1.019
    a=np.exp(-tptBW)
    a1=4.0*a
    a2=-6.0*a*a
    a3=4.0*a*a*a
    a4=-a*a*a*a
    a5=4.0*a*a
    gain=2.0*(1-a1-a2-a3-a4)/(1+a1+a5)

    ureal = lfilter([1, a1, a5],[1, -a1, -a2, -a3, -a4], y * coscf)
    uimag = lfilter([1, a1, a5],[1, -a1, -a2, -a3, -a4], y * sincf)
    envy = gain * np.sqrt(ureal*ureal + uimag*uimag)

    return envx, envy


def eb_BWadjust(control,BWmin,BWmax,Level1):
    cRMS=np.sqrt(np.mean(control**2)) # RMS signal intensity
    cdB=20*np.log10(cRMS) + Level1 # Convert to dB SPL
    if cdB < 50:
        BW = BWmin
    elif cdB > 100:
        BW=BWmax
    else:
        BW=BWmin + ((cdB-50)/50)*(BWmax-BWmin)
    return BW

def eb_EnvCompressBM(envsig,bm,control,attnOHC,thrLow,CR,fsamp,Level1):
    thrHigh=100.0
    small=1.0e-30

    logenv = np.clip(control,a_min=small,a_max=None) # Don't want to take logarithm of zero or neg
    logenv = Level1 + 20*np.log10(logenv)
    logenv = np.clip(logenv, a_min=thrLow, a_max = thrHigh)

    gain = -attnOHC - (logenv - thrLow)*(1 - (1/CR))
    gain=np.power(10, (gain/20))
    flp=800
    b = [0.095107983402496, 0.095107983402496]
    a = [1.000000000000000, -0.809784033195007]
    gain = lfilter(b, a, gain)

    y = gain * envsig
    b = gain * bm
    return y, b

@jit(nopython=True)
def eb_EnvAlign(x,y):
    fsamp=24000 # Sampling rate in Hz
    range_=100 # Range in msec for the xcorr calculation
    lags=round(0.001*range_*fsamp) # Range in samples
    npts=len(x)
    lags = min(lags,npts)  # Use min of lags, length of the sequence
    xy = np.convolve(x, y[::-1])
    # xy = np.correlate(x,y,'full')
    if len(xy) > 2*lags-1:
        start = (len(xy) - (2*lags-1)) // 2
        xy = xy[start: start+(2*lags-1)]
    elif len(xy) < 2*lags-1:
        pads = (2*lags-1 - len(xy))//2
        xy = np.concatenate((np.zeros(pads), xy, np.zeros(pads)))
    
    location = np.argmax(xy) + 1
    delay = lags - location
    if delay > 0:
        y=np.concatenate((y[delay:npts+1], np.zeros(delay)))
    elif delay < 0:
        delay=-delay
        y = np.concatenate( (np.zeros(delay), y[:npts-delay]) )
    
    return y


@jit(nopython=True)
def eb_IHCadapt(xdB,xBM,delta,fsamp):
    dsmall=1.0001
    if delta < dsmall:
        delta = dsmall
    # Initialize the adaptation time constants
    tau1=2 # Rapid adaptation in msec
    tau2=60 # Short-term adaptation in msec
    tau1=0.001*tau1
    tau2=0.001*tau2 # Convert to seconds

    # Equivalent circuit parameters
    T=1/fsamp # Sampling period
    R1=1/delta
    R2=0.5*(1 - R1)
    R3=R2
    C1=tau1*(R1 + R2)/(R1*R2)
    C2=tau2/((R1 + R2)*R3)

    # Intermediate values used for the voltage update matrix inversion
    a11=R1 + R2 + R1*R2*(C1/T)
    a12=-R1
    a21=-R3
    a22=R2 + R3 + R2*R3*(C2/T)
    denom=1.0/(a11*a22 - a21*a12)
    # Additional intermediate values
    R1inv=1.0/R1
    R12C1=R1*R2*(C1/T)
    R23C2=R2*R3*(C2/T)

    # Initialize the outputs and state of the equivalent circuit
    nsamp=len(xdB)
    gain=np.ones(xdB.shape) # Gain vector to apply to the BM motion, default is 1
    ydB=np.zeros(xdB.shape) # Assign storage
    V1=0.0
    V2=0.0
    small=1.0e-30
    for n in range(nsamp):
        V0=xdB[n]
        b1=V0*R2 + R12C1*V1
        b2=R23C2*V2
        V1=denom*(a22*b1 - a12*b2)
        V2=denom*(-a21*b1 + a11*b2)
        out=(V0 - V1)*R1inv
        out = max(out, 0.0) # Envelope can not drop below threshold
        ydB[n]=out # Envelope with IHC adaptation
        gain[n]=(out + small)/(V0 + small) # Avoid division by zero
    
    # gain = (ydB + small) / (xdB + small)
    yBM = gain * xBM
    return ydB, yBM

def eb_EnvSL2(env,bm,attnIHC,Level1):
    small=1.0e-30 
    y=Level1 - attnIHC + 20*np.log10(env + small)
    y[y<0] = 0.0

    #Convert the linear BM motion to have a dB SL envelope
    gain=(y + small) / (env + small) # Gain that converted the env from lin to dB SPL
    b=gain*bm # Apply gain to BM motion
    return y, b
    

def eb_BMaddnoise(x,thr,Level1):
    gn = 10**((thr - Level1)/20.0)
    noise = gn*np.random.randn(len(x))
    y = x + noise
    return y


def eb_GroupDelayComp(xenv,BW,cfreq,fsamp):
    nchan = len(BW)
    # Filter ERB from Moore and Glasberg (1983)
    earQ=9.26449
    minBW=24.7
    ERB=minBW + (cfreq/earQ)

    # Initialize the gamatone filter coefficients
    tpt=2*np.pi/fsamp
    tptBW=tpt*1.019*BW*ERB
    a=np.exp(-tptBW)
    a1=4.0*a
    a2=-6.0*a*a
    a3=4.0*a*a*a
    a4=-a*a*a*a
    a5=4.0*a*a

    gd = np.zeros(nchan)
    for n in range(nchan):
        _, gd[n] = group_delay(( [1, a1[n], a5[n]] , [1, -a1[n], -a2[n], -a3[n], -a4[n]] ), w = 1 )
    gd = np.round(gd)
    gmin = np.min(gd)
    gd=gd - gmin
    gmax=np.max(gd)
    correct = gmax - gd

    yenv=np.zeros(xenv.shape)
    for n in range(nchan):
        r = xenv[n,:]
        npts=len(r)
        r = np.concatenate( ( np.zeros(int(correct[n])), r[0:npts-int(correct[n])] ) )
        yenv[n, :] = r

    return yenv

    

def eb_aveSL(env,control,attnOHC,thrLow,CR,attnIHC,Level1):
    thrHigh=100.0
    # Convert the control to dB SPL
    small=1.0e-30
    logenv = np.clip(control, a_min=small, a_max=None)
    logenv = Level1 + 20*np.log10(logenv)
    logenv = np.clip(logenv, a_min=thrLow, a_max=thrHigh)

    # Compute the compression gain in dB
    gain = -attnOHC - (logenv - thrLow) * (1 - (1/CR)) # Gain in dB

    # Convert the signal envelope to dB SPL
    logenv = np.clip(env, a_min=small, a_max=None)
    logenv = Level1 + 20*np.log10(logenv)
    logenv = np.clip(logenv, a_min=0, a_max=None)
    xdB = logenv + gain - attnIHC # Apply gain to the log spectrum
    xdB = np.clip(xdB, a_min=0.0, a_max=None) # dB SL
    return xdB


def eb_EarModel(x,fx,y,fy,HL,itype,Level1):
    # assert itype==0
    nchan = 32 # Use 32 auditory frequency bands
    mdelay = 1 # Compensate for the gammatone group delay
    cfreq = eb_CenterFreq(nchan) # Center frequencies on an ERB scale
    attnOHCy,BWminy,lowkneey,CRy,attnIHCy = eb_LossParameters(HL, cfreq)

    if itype==0:
        HLx=0*HL
    else:
        HLx=HL
    
    attnOHCx,BWminx,lowkneex,CRx,attnIHCx = eb_LossParameters(HLx, cfreq)
    HLmax = 100*np.ones(6)
    shift = 0.02 # Basal shift of 0.02 of the basilar membrane length
    cfreq1 = eb_CenterFreq(nchan, shift=shift) # Center frequencies for the control
    _, BW1, _, _, _ = eb_LossParameters(HLmax,cfreq1) # Maximum BW for the control
    x24, _ = eb_Resamp24kHz(x, fx)
    y24, fsamp = eb_Resamp24kHz(y, fy)
    # Bulk broadband signal alignment
    # x24, y24 = eb_InputAlign(x24, y24)
    nsamp = len(x24)
    if itype==1:
        nfir = 140
        nalr, _ = eb_NALR(HL,nfir,fsamp)
        x24 = np.convolve(x24, nalr)
        x24=x24[nfir:nfir+nsamp]

    # Cochlear model
    # Middle ear
    xmid = eb_MiddleEar(x24, fsamp)
    ymid = eb_MiddleEar(y24, fsamp)

    # Initialize storage
    # Reference and processed envelopes and BM motion
    xdB = np.zeros([nchan, nsamp])
    ydB = np.zeros([nchan, nsamp])
    xBM = np.zeros([nchan, nsamp])
    yBM = np.zeros([nchan, nsamp])
    xave = np.zeros(nchan)
    yave = np.zeros(nchan)
    xcave = np.zeros(nchan)
    ycave = np.zeros(nchan)
    BWx = np.zeros(nchan)
    BWy = np.zeros(nchan)

    for n in range(nchan):
        xcontrol, ycontrol = eb_GammatoneEnv2(xmid,BW1[n],ymid,BW1[n],fsamp,cfreq1[n])
        # Adjust the auditory filter bandwidths for the average signal level
        BWx[n]=eb_BWadjust(xcontrol,BWminx[n],BW1[n],Level1)  # Reference
        BWy[n]=eb_BWadjust(ycontrol,BWminy[n],BW1[n],Level1)  # Processed
        # Envelopes and BM motion of the reference and processed signals
        xenv,xbm,yenv,ybm = eb_GammatoneBM2(xmid,BWx[n],ymid,BWy[n],fsamp,cfreq[n])
        xave[n]=np.sqrt(np.mean(xenv**2)) # Ave signal mag in each band
        yave[n]=np.sqrt(np.mean(yenv**2))
        xcave[n]=np.sqrt(np.mean(xcontrol**2)) # Ave control signal
        ycave[n]=np.sqrt(np.mean(ycontrol**2))
        # Cochlear compression for the signal envelopes and BM motion
        xc,xb = eb_EnvCompressBM(xenv,xbm,xcontrol,attnOHCx[n],lowkneex[n], CRx[n],fsamp,Level1)
        yc,yb = eb_EnvCompressBM(yenv,ybm,ycontrol,attnOHCy[n],lowkneey[n], CRy[n],fsamp,Level1)

        # Correct for the delay between the reference and output
        # TODO: speed up Align function
        # yc=eb_EnvAlign(xc,yc)
        # yb=eb_EnvAlign(xb,yb)
        # Convert the compressed envelopes and BM vibration envelopes to dB SL

        xc,xb = eb_EnvSL2(xc,xb,attnIHCx[n],Level1)
        yc,yb = eb_EnvSL2(yc,yb,attnIHCy[n],Level1)
        # Apply the IHC rapid and short-term adaptation
        delta=2.0
        xdB_tmp,xb=eb_IHCadapt(xc,xb,delta,fsamp)
        xdB[n,:] = xdB_tmp
        ydB_tmp,yb=eb_IHCadapt(yc,yb,delta,fsamp)
        ydB[n,:] = ydB_tmp
        # Additive noise level to give the auditory threshold
        IHCthr=-10.0
        xBM_tmp = eb_BMaddnoise(xb,IHCthr,Level1)
        xBM[n,:] = xBM_tmp
        yBM_tmp = eb_BMaddnoise(yb,IHCthr,Level1)
        yBM[n,:] = yBM_tmp
        # print('finished: ',n)
    
    if mdelay > 0:
        xdB=eb_GroupDelayComp(xdB,BWx,cfreq,fsamp)
        ydB=eb_GroupDelayComp(ydB,BWx,cfreq,fsamp)
        xBM=eb_GroupDelayComp(xBM,BWx,cfreq,fsamp)
        yBM=eb_GroupDelayComp(yBM,BWx,cfreq,fsamp)
    
    # Convert average gammatone outputs to dB SL
    xSL = eb_aveSL(xave,xcave,attnOHCx,lowkneex,CRx,attnIHCx,Level1)
    ySL = eb_aveSL(yave,ycave,attnOHCy,lowkneey,CRy,attnIHCy,Level1)

    return xdB, xBM, ydB, yBM, xSL, ySL, fsamp




if __name__ == "__main__":
    x, fx = librosa.load('sig_clean.wav', sr=None)
    y, fy = librosa.load('sig_out.wav', sr=None)
    # Combined,Nonlin,Linear,raw = hasqi_v2(x,fx,y,fy)
    # pdb.set_trace()
    start = time.time()
    for i in range(50):
        Combined,Nonlin,Linear,raw = hasqi_v2(x,fx,y,fy)
    end = time.time()
    print((end-start)/50)

    # [score, raw] = haspi_v2(x,fx,y,fy)
    # print(score)
    # print(raw)
    # cProfile.run('haspi_v2(x,fx,y,fy)')