##############################################################
# General functions for calc_snr_max
###############################################################

import numpy as np
from scipy.interpolate import interp1d
from scipy import signal

all = {'integrate','gaussian', 'define_lsf', 'vac_to_stand', 'setup_band', 'resample'}


def integrate(x,y):
    """
    Integrate y with respect to x using the trapezoidal rule.

    inputs
    ------
    x - array
        independent variable (e.g. wavelength)
    y - array
        dependent variable to integrate, evaluated at x

    output
    ------
    integral - float
        definite integral of y over the range of x (trapezoidal approximation)
    """
    return np.trapz(y,x=x)

def gaussian(x, shift, sig):
    """
    Return a normalized (unit-area) gaussian probability density function
    evaluated at x, with mean `shift` and variance sig^2.

    Parameters
    ----------
    x : array or float
        Input dependent variable array (points at which to evaluate the gaussian)
    shift : float
        Mean (center) of the gaussian distribution, same units as x
    sig : float
        Standard deviation (sigma) of the gaussian, same units as x

    Returns
    -------
    g : array or float
        Normalized gaussian values evaluated at x, with peak amplitude
        1/(sig*sqrt(2*pi)) such that the integral over x equals 1
    """
    return np.exp(-.5*((x - shift)/sig)**2)/(sig * np.sqrt(2*np.pi))


def define_lsf(v,res):
    """
    define gaussian in pixel elements to convolve resolved spectrum with to get rightish resolution

    inputs
    ------
    v - wavelength array [nm], 1D array
        wavelength grid the LSF will be applied to; must be sampled finely
        enough that the resulting gaussian kernel spans at least ~20 pixels
    res - resolving power [int or float]
        spectral resolving power R = lambda / delta_lambda to represent

    outputs
    -------
    gaussian - [1D array]
        array of a gaussian with a sigma spanning N pixels where
        N times the wavelength sampling gives a FWHM matching res
    """
    dlam  = np.median(v)/res
    fwhm  = dlam/np.mean(np.diff(v)) # desired lambda spacing over current lambda spacing resolved to give sigma in array elements
    sigma = fwhm/2.634 # FWHM is dl/l but feed sigma    
    x = np.arange(sigma*10) # sigma is defined in pixels so this will be an integer
    gaussian = (1./sigma/np.sqrt(2*np.pi)) * np.exp(-0.5*( (x - 0.5*len(x))/sigma)**2 )

    if len(gaussian)<20:
        raise(ValueError('Wavelength sampling too coarse for requestion resolving power - resample wavelength grid finer or lower resolution'))
    
    return gaussian

def degrade_spec(x,y,res):
    """
    Degrade a spectrum to a lower resolving power by convolving it with a
    gaussian line spread function (LSF) matched to the requested resolution.

    inputs
    ------
    x - array
        wavelength array [nm]
    y - array
        flux array evaluated at x
    res - float or int
        resolving power (R) to degrade the spectrum to

    output
    ------
    y_lowres - array
        y convolved with the gaussian LSF for the requested resolving power,
        res (same length and sampling as input y)
    """
    lsf      = define_lsf(x,res=res)
    y_lowres = np.convolve(y,lsf,mode='same')

    return y_lowres

def tophat(x,l0,lf,throughput):
    """
    Return a tophat (rectangular) bandpass over x

    inputs
    ------
    x - array
        wavelength array
    l0 - float
        lower bound of the bandpass
    lf - float
        upper bound of the bandpass
    throughput - float
        constant value assigned to the bandpass between l0 and lf

    output
    ------
    bandpass - array
        array same shape as x, equal to throughput within (l0, lf) and 0 elsewhere
    """
    ion = np.where((x > l0) & (x<lf))[0]
    bandpass = np.zeros_like(x)
    bandpass[ion] = throughput
    return bandpass


def vac_to_stand(wave_vac):
    """Convert vacuum wavelength to standard (air) wavelength, since we're
    doing ground based observations where air wavelengths are appropriate.

	https://idlastro.gsfc.nasa.gov/ftp/pro/astro/vactoair.pro
    Equation from Prieto 2011 Apogee technical note
    and equation and parametersfrom Cidor 1996

    inputs:
    -------
    wave_vac: 1D array, vacuum wavelength [Angstrom]

    outputs:
    -------
    wave_air: 1D array, wavelength converted to standard air [Angstrom]
        computed as wave_vac / n, where n is the index of refraction of
        air given by the Cidor 1996 dispersion formula
    """
    # eqn
    sigma2= (1e4/wave_vac)**2.
    fact = 1. +  5.792105e-2/(238.0185 - sigma2) + \
                            1.67917e-3/( 57.362 - sigma2)
                            
    # return l0 / n which equals lamda
    return wave_vac/fact


def setup_band(x, x0=0, sig=0.3, eta=1):
    """
    Generate a tophat (step function) bandpass centered on x0, evaluated
    over the array x.

    inputs:
    ------
    x - array
        independent variable array (e.g. wavelength [nm]) the bandpass is evaluated over
    x0 - float
        center of the bandpass, same units as x. default 0
    sig - float
        full width of the bandpass, same units as x. default 0.3
    eta - float
        amplitude (e.g. throughput, 0-1) assigned within the bandpass. default 1

    outputs:
    -------
    y : array
        array same shape as x, equal to eta within (x0-sig/2, x0+sig/2) and 0 elsewhere
    """
    y = np.zeros_like(x)

    ifill = np.where((x > x0-sig/2) & (x < x0 + sig/2))[0]
    y[ifill] = eta

    return y

def rebin(x,y,nbin=3, eta=1):
    """
    Resample (bin down) a spectrum by a fixed integer number of pixels,
    using a boxcar (tophat) convolution followed by decimation.

    inputs
    ------
    x - array
        wavelength array [nm]
    y - array
        y values (e.g. flux) evaluated at x
    nbin - int
        number of pixels to combine into each output bin. default 3
    eta - float
        factor to multiply y by (e.g. throughput). default 1

    outputs
    -------
    int_lam - array
        wavelength array [nm], downsampled by taking every nbin-th value of x
    int_spec - array
        y convolved with a boxcar of width nbin and scaled by eta,
        downsampled by taking every nbin-th value
    """
    tophat  = eta * np.ones(nbin) # do i need to pad this?

    int_spec_oversample    = signal.fftconvolve(y,tophat,mode='same') # dlam integrating factor

    int_lam  = x[::nbin] # 
    int_spec = int_spec_oversample[::nbin]

    return int_lam, int_spec


def resample(x,y,sig=0.3, dx=0, eta=1,mode='variable'):
    """
    Resample (bin) a spectrum onto coarser wavelength sampling by convolving
    with a tophat kernel of width sig and decimating, using one of several
    methods.

    inputs
    ------
    x - array
        wavelength array [nm]
    y - array
        spectrum array (evaluated at x) to resample, units in spectral
        density (e.g. photons/nm)
    sig - float or array
        width of the resample bin(s) [nm], default 0.3nm. Can be a scalar
        (modes 'fast'/'slow'/'pixels') or an array matching x (mode
        'variable', for a spectrally-varying resolution element such as a
        constant-R grid)
    dx - float
        offset [nm] for the location of the first bin, default 0
    eta - float
        efficiency/throughput (0-1), amplitude scaling applied to the
        binned flux, default 1
    mode - str
        resampling method to use:
        'fast' - FFT convolution with a fixed-width tophat; requires x to
            be uniformly sampled and sig to be a scalar
        'variable' - like 'fast' but allows sig to vary per pixel (e.g. for
            constant resolving power), grouping pixels by integer bin width
        'variable_smooth' - not yet implemented (no-op)
        'slow' - steps through x and integrates each tophat-weighted
            segment with trapz; slightly more accurate but slower
        'pixels' - like 'fast' but sig is interpreted in units of pixels
            rather than nm

    outputs
    -------
    int_lam - array
        resampled wavelength array [nm], one value per output bin
    int_spec - array
        resampled spectrum, integrated/convolved flux in each bin
    """
    if mode=='fast':
        dlam    = np.median(np.diff(x)) # nm per pixel, most accurate if x is uniformly sampled in wavelength
        if sig <= dlam: raise ValueError('Sigma value is smaller than the sampling of the provided wavelength array')
        nsamp   = int(sig / dlam)     # width of tophat
        tophat  = eta * np.ones(nsamp) # do i need to pad this?

        # make nans zero
        inan = np.where(np.isnan(y))[0]
        y[inan] = 0
        
        # do the FFT
        int_spec_oversample    = dlam * signal.fftconvolve(y,tophat,mode='same') # dlam integrating factor
        int_spec_oversample[inan] = np.nan

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

    if mode=='variable_smooth':
        # sampling smoothly varies over spectrum
        # works similar to variable except tophat has fractional values
        # at edges to split boundary pixel flux appropriately
        pass

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

