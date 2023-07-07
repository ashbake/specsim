##############################################################
# General functions for calc_snr_max
# ##############################################################

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy import signal

all = {'integrate','gaussian', 'define_lsf', 'vac_to_stand', 'setup_band', 'resample'}




def integrate(x,y):
    """
    Integrate y wrt x
    """
    return trapz(y,x=x)

def gaussian(x, shift, sig):
    ' Return normalized gaussian with mean shift and var = sig^2 '
    return np.exp(-.5*((x - shift)/sig)**2)/(sig * np.sqrt(2*np.pi))


def define_lsf(v,res):
    """
    define gaussian in pixel elements to convolve resolved spectrum with to get rightish resolution
    """
    dlam  = np.median(v)/res
    fwhm  = dlam/np.mean(np.diff(v)) # desired lambda spacing over current lambda spacing resolved to give sigma in array elements
    sigma = fwhm/2.634 # FWHM is dl/l but feed sigma    
    x = np.arange(sigma*10)
    gaussian = (1./sigma/np.sqrt(2*np.pi)) * np.exp(-0.5*( (x - 0.5*len(x))/sigma)**2 )

    return gaussian

def degrade_spec(x,y,res):
    """
    given wavelength, flux array, and resolving power R, return  spectrum at that R
    """
    lsf      = define_lsf(x,res=res)
    y_lowres = np.convolve(y,lsf,mode='same')

    return y_lowres

def tophat(x,l0,lf,throughput):
    ion = np.where((x > l0) & (x<lf))[0]
    bandpass = np.zeros_like(x)
    bandpass[ion] = throughput
    return bandpass


def vac_to_stand(wave_vac):
    """Convert vacuum wavelength (Ang) to standard wavelength in air since we're
    doing ground based stuff. 

	https://idlastro.gsfc.nasa.gov/ftp/pro/astro/vactoair.pro
    Equation from Prieto 2011 Apogee technical note
    and equation and parametersfrom Cidor 1996
    
    inputs: 
    -------
    wave_fac: 1D array, wavelength [A]

    outputs:
    -------
    
    """
    # eqn
    sigma2= (1e4/wave_vac)**2.
    fact = 1. +  5.792105e-2/(238.0185 - sigma2) + \
                            1.67917e-3/( 57.362 - sigma2)
                            
    # return l0 / n which equals lamda
    return wave_vac/fact


def setup_band(x, x0=0, sig=0.3, eta=1):
    """
    give step function

    inputs:
    ------
    x0
    sig
    eta
    """
    y = np.zeros_like(x)

    ifill = np.where((x > x0-sig/2) & (x < x0 + sig/2))[0]
    y[ifill] = eta

    return y

def rebin(x,y,nbin=3, eta=1):
    """
    resample using convolution

    x: wavelength array in nm
    y: y values evaluated at x

    nbin: - integer; numbers of pixels to combine
    eta: factor to multiply y by,default 1
    """
    tophat  = eta * np.ones(nbin) # do i need to pad this?

    int_spec_oversample    = signal.fftconvolve(y,tophat,mode='same') # dlam integrating factor

    int_lam  = x[::nbin] # 
    int_spec = int_spec_oversample[::nbin]

    return int_lam, int_spec


def resample(x,y,sig=0.3, dx=0, eta=1,mode='variable'):
    """
    resample using convolution

    x: wavelength array in nm
    y_in/y_out: two y arrays (evaluated at x) to resample, units in spectral density (e.g. photons/nm)

    sig in nanometers - width of bin, default 0.3nm
    dx - offset for taking first bin, defaul 0
    eta 0-1 for efficiency (amplitude of bin) default 1
    
    modes: slow, fast
    slow more accurate (maybe?), fast uses fft

    slow method uses trapz so slightly more accurate, i think? both return similar flux values

    """
    if mode=='fast':
        dlam    = np.median(np.diff(x)) # nm per pixel, most accurate if x is uniformly sampled in wavelength
        if sig <= dlam: raise ValueError('Sigma value is smaller than the sampling of the provided wavelength array')
        nsamp   = int(sig / dlam)     # width of tophat
        tophat  = eta * np.ones(nsamp) # do i need to pad this?

        int_spec_oversample    = dlam * signal.fftconvolve(y,tophat,mode='same') # dlam integrating factor
        
        int_lam  = x[int(nsamp/2 + dx/dlam):][::nsamp] # shift over by dx/dlam (npoints) before taking every nsamp point
        int_spec =  int_spec_oversample[int(nsamp/2 + dx/dlam):][::nsamp]

    if mode=='variable':
        # mode to take variable res element
        dlam    = np.median(np.diff(x)) # nm per pixel, most accurate if x is uniformly sampled in wavelength
        if np.min(sig) <= dlam: raise ValueError('Sigma value is smaller than the sampling of the provided wavelength array')
        nsamp   = sig // dlam    # width of tophat
        nsamp = nsamp.astype('int')
        
        nsamp_unique = np.unique(nsamp)
        int_lam=np.array([])
        int_spec=np.array([])
        for n in nsamp_unique:
            isub = np.where(nsamp==n)[0]
            tophat  = eta * np.ones(n) # do i need to pad this?
            int_spec_oversample    = dlam * signal.fftconvolve(y[isub],tophat,mode='same') # dlam integrating factor
            xnew = x[isub][::n]
            ynew = int_spec_oversample[::n]
            int_lam   = np.concatenate((int_lam,xnew))
            int_spec  = np.concatenate((int_spec,ynew))

    elif mode=='slow':
        i=0
        int_lam, int_spec  = [], []
        # step through and integrate each segment
        while i*sig/2 + dx< np.max(x)-sig/2 - np.min(x): # check
            xcent    = np.min(x) + dx + i*sig/2
            tophat   = setup_band(x, x0=xcent, sig=sig, eta=eta) # eta throughput of whole system
            int_spec.append(integrate(x,tophat * y))
            int_lam.append(xcent)
            i += 1

    if mode=='pixels':
        """
        reample by binning pixels, sig in pixels now
        """
        nsamp = int(sig)
        tophat  = eta * np.ones(int(nsamp)) # do i need to pad this?

        int_spec_oversample    = signal.fftconvolve(y,tophat,mode='same') # dlam integrating factor
        
        int_lam  = x[int(nsamp//2):][::nsamp] # shift over by dx/dlam (npoints) before taking every nsamp point
        int_spec =  int_spec_oversample[int(nsamp//2):][::nsamp]

    return int_lam, int_spec

