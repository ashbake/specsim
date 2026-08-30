##############################################################
# General functions for calc_snr_max
###############################################################

import numpy as np
from scipy.interpolate import interp1d
from scipy import interpolate, signal

SPEEDOFLIGHT = 2.998e8  # m/s
GAUSSCONST = (2. * (2. * np.log(2))**0.5)  # FWHM = GAUSSCONST * sigma

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

def lsf_rotate(deltav,vsini,epsilon=0.6):
    '''
    Computes vsini rotation kernel.
    Based on the IDL routine LSF_ROTATE.PRO, which implements the
    analytic rotational broadening profile of Gray, D. F. 1992,
    "The Observation and Analysis of Stellar Photospheres".

    Parameters
    ----------
    deltav : float
        Velocity sampling for kernel (x-axis) [km/s]

    vsini : float
        Stellar vsini value [km/s]

    epsilon : float
        Limb darkening value (default is 0.6)

    Returns
    -------
    kernel : array
        Computed rotational broadening kernel profile, unitless, sampled
        on velgrid (suitable for use as a convolution kernel)

    velgrid : array
        x-values for kernel [km/s]

    '''
    # component calculations
    ep1 = 2.0*(1.0 - epsilon)
    ep2 = np.pi*epsilon/2.0
    ep3 = np.pi*(1.0 - epsilon/3.0)

    # make x-axis
    npts = np.ceil(2*vsini/deltav)
    if npts % 2 == 0:
        npts += 1
    nwid = np.floor(npts/2)
    x_vals = (np.arange(npts) - nwid) * deltav/vsini
    xvals_abs = np.abs(1.0 - x_vals**2)
    velgrid = xvals_abs*vsini

    # compute kernel
    kernel = (ep1*np.sqrt(xvals_abs) + ep2*xvals_abs)/ep3

    return kernel, velgrid

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



##############################################################
# AO: wavefront error -> Strehl ratio conversions
###############################################################

def calc_strehl_marechal(wfe,wavelength):
    """
    Compute Strehl ratio from wavefront error using the (simple) Marechal
    approximation

    inputs
    ------
    wfe : float or array [nm]
        wavefront error
    wavelength : float or array [nm]
        wavelength(s), grid or single number

    outputs
    -------
    strehl : float or array
        Strehl ratio at wavelength, same shape as wfe/wavelength (broadcast)
    """
    strehl = np.exp(-(2*np.pi*wfe/wavelength)**2)

    return strehl

def calc_strehl(wfe,wavelength):
    """
    Compute Strehl ratio from wavefront error using the extended Marechal
    equation - the function used by the rest of the code as of Nov 20th.
    See KOAN doc for info

    inputs
    ------
    wfe : float or array [nm]
        wavefront error
    wavelength : float or array [nm]
        wavelength(s), grid or single number

    outputs
    -------
    strehl : float or array
        Strehl ratio at wavelength, same shape as wfe/wavelength (broadcast)
    """
    marechal = 2*np.pi*wfe/wavelength
    strehl = np.exp(-(0.75 * (marechal + 0.2615))**2 + 0.05)

    return strehl

def tt_to_strehl(tt,lam,D):
    """
    convert tip tilt residuals in mas to strehl according to Rich's equation

    equation 4.60 from Hardy 1998 (adaptive optics for astronomy) matches this

    inputs
    ------
    tt : float or array [mas]
        tip tilt rms
    lam : float or array [nm]
        wavelength(s)
    D : float [m]
        telescope diameter

    outputs
    -------
    strehl_tt : float or array
        Strehl ratio due to residual tip/tilt error
    """
    tt_rad = tt * 1e-3/206265 # convert to radians from mas
    lam_m =lam * 1e-9
    bottom = 1 + np.pi**2/2*((tt_rad)/(lam_m/D))**2
    strehl_tt = 1/bottom

    #sig_D = 0.44* lam_m/D
    #1/(1 + tt_rad**2/sig_D**2) KAON1322 doc eq 5 method matches Richs eqn

    return strehl_tt


##############################################################
# Optics / atmosphere geometry
###############################################################

def calc_plate_scale(pixel_pitch, D=10, fratio=35):
    """
    Compute the on-sky plate scale of a detector given its pixel size and
    the telescope/beam geometry.

    inputs:
    -------
    pixel_pitch - float
        detector pixel pitch [um]

    D - float
        telescope diameter [m] (default 10)

    fratio - float
        beam focal ratio (f-number), unitless (default 35)

    return:
    -------
    platescale_arcsec_pix - float
        plate scale [arcsec/pixel]
    """
    platescale_arcsec_um = 206265 / fratio / (D * 10**6) #arc/um
    platescale_arcsec_pix = platescale_arcsec_um * pixel_pitch
    return platescale_arcsec_pix


def air_index_refraction(lam,p,t):
    """
    Compute the index of refraction of air at the given wavelength,
    pressure, and temperature, using the (modified) Edlen equation.

    https://iopscience.iop.org/article/10.1088/0026-1394/30/3/004/pdf
    edlen https://iopscience.iop.org/article/10.1088/0026-1394/2/2/002/pdf

    inputs:
    -------
    lam - float or array
        wavelength [nm]

    p - float
        air pressure [torr]

    t - float
        air temperature [celsius]

    returns:
    --------
    n - float or array
        index of refraction of air, unitless
    """
    sig = 10**7/lam * (1e-4) # 1e-4 cm/micron
    ns = 1 + (1/1e8) * (8342.13 + 2406030*(130 - sig**2)**(-1) + 15997*(38.9 -sig**2)**-1)
    n = 1 + (p * (ns -1)/ 720.775) * (1 + p*(0.817 - 0.0133*t)*(10**-6))/(1 + 0.0036610*t)
    return n


##############################################################
# Spectra: synthetic lines, doppler shifts, RV information content
###############################################################

def gaussian_fwhm(xarr, center, fwhm,A=1,B=0):
    '''
    Simple gaussian function, defined by center and FWHM

    Parameters
    ----------
    xarr : array
        Input dependant variable array
    center : float
        Center of gaussian distribution
    fwhm : float
        FWHM of gaussian desired
    A : float
        Amplitude of gaussian
    B : float
        Vertical offset of gaussian

    Returns
    -------
    gauss : array
        Computed gaussian values for xarr
    '''
    # gaussian function parameterized with FWHM
    gauss = A*np.exp(-0.5 * (xarr - center) ** 2. / (fwhm / GAUSSCONST) ** 2.) + B
    return gauss


def spec_make(wvl, weights, line_wvls, fwhms):
    '''
    Generate fake (normalized) spectrum of gaussian 'absorption' lines.

    Inputs:
    -------
    wvl : array
        Input wavelength array [nm]
    weights : array
        Line depths of specified lines (fractional, 0-1)
    line_wvls : array
        Line centers of features to be added [nm], same units as wvl
    fwhms : array
        FWHMs of lines specified [nm], same units as wvl

    Outputs:
    -------
    spec_out: array
         Final output absorption spectrum, normalized continuum at 1.0 with
         gaussian dips at each line_wvl of the given depth and fwhm
    '''

    # initialize array
    spec_out = np.zeros_like(wvl)

    # for each line wavelength, add a gaussian at the specified depth
    for weight, line_wvl, fwhm in zip(weights, line_wvls, fwhms):
        spec_out += (weight * gaussian_fwhm(wvl, line_wvl, fwhm))
    return 1. - spec_out


def spec_rv_noise_calc(wvl, spec, sigma_spec):
    '''
    Calculates photon-limited RV uncertainty of given spectrum in km/s

    Parameters
    ----------
    wvl : array
        Input wavelength array of spectrum [nm]
    spec : array
        Flux values of spectrum -- assumes only photon noise
    sigma_spec : array
        1-sigma flux uncertainty of spec (e.g. sqrt(counts) photon noise),
        same length/units as spec. Zero-valued entries are reset in-place
        to a large number (1e5) to avoid division by zero.

    Returns
    -------
    sigma_rv : float
        Computed photon-limited RV uncertainty [m/s]
    '''

    # calculate pixel optimal weights, follows Murphy et al. 2007
    wvl_m_ord = wvl * 1e-9 # convert wavelength values to meters

    # calculate noise (photon only, assume root N)
    sigma_spec[np.where(sigma_spec==0)[0]] = 100000

    # calculate slopes of spectrum
    #slopes = np.gradient(spec, wvl_m_ord)
    flux_interp = interpolate.InterpolatedUnivariateSpline(wvl_m_ord,spec, k=1)
    dflux = flux_interp.derivative()
    slopes = dflux(wvl_m_ord)

    # calculate weighted slopes, ignoring the edge pixels (breaks derivative)
    top = (wvl_m_ord[1:slopes.size - 1]**2.) * (slopes[1:slopes.size - 1]**2.)
    bottom = (sigma_spec[1:slopes.size - 1]**2.)
    w_ord = top / bottom

    # combined weighted slopes
    return SPEEDOFLIGHT / ((np.nansum(w_ord[1:-1]))**0.5) # m/s


def doppler(v):
    """
    Compute the (non-relativistic) Doppler shift factor for a given velocity.

    inputs
    ------
    v - float or array, [m/s]
        velocity for the shift

    output
    ------
    factor - float or array
        Doppler factor (1 + v/c); multiply a rest wavelength by this
        factor to get the Doppler-shifted wavelength
    """
    return (1.0 + (v / SPEEDOFLIGHT))


##############################################################
# Detector noise: per-term contributions and their quadrature sum
###############################################################

def sum_total_noise(flux,texp, nsamp, inst_bg, sky_bg, darknoise,readnoise,npix,speckle,noisefloor=None):
    """
    noise in 1 exposure

    inputs:
    --------
    flux - array [e-] 
        spectrum of star in units of electrons
    texp - float [seconds]
        exposure time, (0s,900s] (for one frame)
    nsamp - int
        number of samples in a ramp which will reduce read noise [1,inf] - 16 max for kpic
    inst_bg - array or float [e-/s]
        instrument background, if array should match sampling of flux
    sky_bg - array or float [e-/s]
        sky background, if array should match sampling of flux
    darknoise - float [e-/s/pix]
        dark noise of detector
    readnoise - float [e-/s]
        read noise of detector
    npix - float [pixels]
        number of pixels in cross dispersion of spectrum being combined into one 1D spectrum
    speckle - array [e-]
        counts from speckle leakage from star. should be zeroes if on axis
    noisefloor - float or None (default: None)
        noise cap to be applied. Defined relative to flux such that 1/noisecap is the max SNR allowed
    
    outputs:
    -------
    noise: array [e-]
        total noise sampled on flux grid
    """
    # shot noise - array w/ wavelength or integrated over band
    sig_flux = np.sqrt(np.abs(flux))

    # speckle noise
    speckle_noise = np.sqrt(speckle)
    post_processing_gain = 100. # reduction of speckle related systematics in software

    # background (instrument and sky) - array w/ wavelength matching flux array sampling or integrated over band
    sig_bg   = background_noise(inst_bg,sky_bg, texp)

    # read noise  - reduces by number of ramps, limit to 6 at best
    sig_read = read_noise(np.max((6,(readnoise/np.sqrt(nsamp)))), npix)
    
    # dark current - times time and pixels
    sig_dark = dark_noise(darknoise,npix,texp) #* get dark noise every sample
    
    noise    = np.sqrt(sig_flux **2 + sig_bg**2 + sig_read**2 + sig_dark**2 + speckle_noise**2 + (speckle/post_processing_gain)**2) 

    # cap the noise if a number is provided
    if noisefloor is not None:
        noise[np.where(noise < noisefloor)] = noisefloor * flux # noisecap is fraction of flux, 1/noisecap gives max SNR

    return noise


def background_noise(inst_bg,sky_bg, texp):
    """
    Compute the noise due to instrument and sky background photons

    inputs
    ------
    inst_bg - float/array [photons/sec/reduced pixel]
        the instrument background flux 
    sky_bg  -  float/array [photons/sec/reduced pixel]
        the sky background flux 
    texp    - float [seconds]
        the exposure time

    returns
    -------
    float [photons]
        the standard deviation noise of sky and instrument thermal background thermal
    """
    total_bg = texp * (inst_bg + sky_bg) # per reduced pixel already so dont need to include vertical pixel extent
    
    return np.sqrt(np.abs(total_bg) )


def read_noise(rn,npix):
    """
    Compute the total detector read noise contribution over npix pixels
    by adding the per-pixel read noise in quadrature (rn * sqrt(npix)).

    input:
    ------
    rn: [e-/pix]
        read noise per pixel (per read/ramp, already reduced by number
        of samples by the caller if applicable)
    npix [pix]
        number of pixels

    output:
    -------
    float [photons]
        the standard deviation of detector read noise over npix
    """
    return np.sqrt(npix * rn**2)


def dark_noise(darknoise,npix,texp):
    """
    Computes Poisson noise due to dark current, i.e. the standard
    deviation of the Poisson-distributed dark current counts accumulated
    over npix pixels during the exposure time (sqrt(darknoise * npix * texp)).

    input:
    ------
    darknoise: [e-/pix/s]
        dark current rate per pixel
    npix [pix]
        number of pixels
    texp [s]
        exposure time in seconds

    output:
    -------
    sig_dark [photons]
        the standard deviation of dark current photons over npix
    """
    sig_dark = np.sqrt(darknoise * npix * texp)
    return sig_dark
