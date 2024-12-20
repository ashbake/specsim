##############################################################
# Tools for loading spectra and scaling to correct magnitude
###############################################################

import numpy as np
from astropy.io import fits
from scipy import interpolate
import glob,os
from astropy.convolution import convolve

from specsim.functions import *

#all = {}



def load_phoenix(stelname,stelpath,wav_start=750,wav_end=780):
	"""
	load fits file stelname with stellar spectrum from phoenix 
	http://phoenix.astro.physik.uni-goettingen.de/?page_id=15
	
	return subarray 
	
	wav_start, wav_end specified in nm
	
	convert s from egs/s/cm2/cm to phot/cm2/s/nm using
	https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf
	"""
	# conversion factor

	f = fits.open(stelpath + stelname)
	spec = f[0].data / (1e8) # ergs/s/cm2/cm to ergs/s/cm2/Angstrom for conversion
	f.close()
	
	wave_file = os.path.join(stelpath + 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits') #assume wave in same folder
	f = fits.open(wave_file)
	lam = f[0].data # angstroms
	f.close()
	
	# Convert
	conversion_factor = 5.03*10**7 * lam #lam in angstrom here
	spec *= conversion_factor # phot/cm2/s/angstrom
	
	# Take subarray requested
	isub = np.where((lam > wav_start*10.0) & (lam < wav_end*10.0))[0]

	# Convert 
	return lam[isub]/10.0,spec[isub] * 10 * 100**2 #nm, phot/m2/s/nm

def load_filter(filter_path,family,band):
	"""
	"""
	filter_file    = glob.glob(filter_path + '*' + family + '*' + band + '.dat')[0]
	xraw, yraw     = np.loadtxt(filter_file).T # nm, transmission out of 1
	return xraw/10, yraw

def load_sonora(stelname,wav_start=750,wav_end=780):
	"""
	load sonora model file
	
	return subarray 
	
	wav_start, wav_end specified in nm
	
	convert s from erg/cm2/s/Hz to phot/cm2/s/nm using
	https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf

	wavelength loaded is microns high to low
	"""
	f = np.loadtxt(stelname,skiprows=2)

	lam  = 10000* f[:,0][::-1] #microns to angstroms, needed for conversiosn
	spec = f[:,1][::-1] # erg/cm2/s/Hz
	
	spec *= 3e18/(lam**2)# convert spec to erg/cm2/s/angstrom
	
	conversion_factor = 5.03*10**7 * lam #lam in angstrom here
	spec *= conversion_factor # phot/cm2/s/angstrom
	
	isub = np.where( (lam > wav_start*10.0) & (lam < wav_end*10.0))[0]

	return lam[isub]/10.0,spec[isub] * 10 * 100**2 #nm, phot/m2/s/nm (my fave)

def calc_nphot(dl_l, zp, mag):
	"""
	http://astroweb.case.edu/ssm/ASTR620/mags.html

	Values are all for a specific bandpass, can refer to table at link ^ for values
	for some bands. Function will return the photons per second per meter squared
	at the top of Earth atmosphere for an object of specified magnitude

	inputs:
	-------
	dl_l: float, delta lambda over lambda for the passband
	zp: float, flux at m=0 in Jansky
	mag: stellar magnitude

	outputs:
	--------
	photon flux
	"""
	phot_per_s_m2_per_Jy = 1.51*10**7 # convert to phot/s/m2 from Jansky

	return dl_l * zp * 10**(-0.4*mag) * phot_per_s_m2_per_Jy

def scale_stellar(filt,stelv,stels,mag):
	"""
	scale spectrum by magnitude
	inputs: 
	filt: so.filt object
	mag: magnitude in filter desired

	load new stellar to match bounds of filter since may not match working badnpass elsewhere
	"""
	if (np.min(filt.xraw) < np.min(stelv)) or (np.max(filt.xraw) > np.max(stelv)):
		raise Warning('Check that stellar model in scale_stellar extends past filter profile')
	
	filt_interp       =  interpolate.interp1d(filt.xraw,filt.yraw, bounds_error=False,fill_value=0)

	filtered_stellar   = stels * filt_interp(stelv)    # filter profile resampled to phoenix times phoenix flux density
	nphot_expected_0   = calc_nphot(filt.dl_l, filt.zp, mag)    # what's the integrated flux supposed to be in photons/m2/s?
	nphot_model        = integrate(stelv,filtered_stellar)            # what's the integrated flux now? in same units as ^
	
	return nphot_expected_0/nphot_model


def load_stellar_model(x,mag,teff,vsini,so,rv=0):
	"""
	Loads stellar model as sonora or phoenix based on temperature
	Then scales to the designated magnitude
	then broadens by vsini

	so only used for paths and filter information
	"""
	# wavelength bounds should incldue filter entirely
	l0,l1 = np.min((np.min(x),np.min(so.filt.xraw))),np.max((np.max(x),np.max(so.filt.xraw)))

	if teff < 2300: # sonora models arent sampled as well so use phoenix as low as can
		g    = '316' # mks units, np.log10(316 * 100)=4.5 to match what im holding for phoenix models.
		teff = str(int(teff))
		stel_file         = so.stel.sonora_folder + 'sp_t%sg%snc_m0.0' %(teff,g)
		vraw,sraw = load_sonora(stel_file,wav_start=l0,wav_end=l1)
		model             = 'sonora'
	else:
		teff = str(int(teff)).zfill(5)
		logg = '{:.2f}'.format(so.stel.logg)
		model             = 'phoenix' 
		stel_file         = 'lte%s-%s-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'%(teff,logg)
		vraw,sraw = load_phoenix(stel_file,so.stel.phoenix_folder,wav_start=l0, wav_end=l1) #phot/m2/s/nm
	

	# apply scaling factor to match filter zeropoint
	factor_0   = scale_stellar(so.filt,vraw,sraw,mag) # loads spectrum over selected filter and finds scaling to get correct magnitude
	
	# interpolate file onto x, apply factor
	tck_stel    = interpolate.splrep(vraw,sraw, k=2, s=0)
	s           = factor_0 * interpolate.splev(x,tck_stel,der=0,ext=1)
	
	#units = 'photons/s/m2/nm' # stellar spec is in photons/s/m2/nm

	# broaden star spectrum with rotation kernal
	SPEEDOFLIGHT   = 2.998e8 # m/s
	if vsini > 0:
		dwvl_mean = np.abs(np.nanmean(np.diff(x)))
		dvel_mean      = (dwvl_mean / np.nanmean(x)) * SPEEDOFLIGHT / 1e3 # average sampling in km/s
		vsini_kernel,_ = _lsf_rotate(dvel_mean,vsini,epsilon=0.6)
		flux_vsini     = convolve(s,vsini_kernel,normalize_kernel=True)  # photons / second / Ang
		s              = flux_vsini

	# Offset star by an RV (for CCF purposes to offset from tellurics)
	if rv!= 0:
		doppler_factor = (1.0 + ((rv * 1000) / SPEEDOFLIGHT)) # rv in km/s
		tck = interpolate.splrep(x*doppler_factor,s, k=3, s=0)
		shifted_spec = interpolate.splev(x,tck,der=0,ext=1)
	else: 
		shifted_spec=s.copy()

	# some negatives are created when interpolating, change these to zero
	ineg = np.where(shifted_spec<0)[0]
	shifted_spec[ineg] = 0
	
	return shifted_spec, vraw, sraw, model, stel_file, factor_0


def get_band_mag2(so,family,band,factor_0):
    """
    factor_0: scaling model to photons
    """
    x,y          = load_filter(so.filt.filter_path,family,band)
    filt_interp  = interpolate.interp1d(x, y, bounds_error=False,fill_value=0)
    dl_l         = np.mean(integrate(x,y)/x) # dlambda/lambda to account for spectral fraction
    
    # load stellar the multiply by scaling factor, factor_0, and filter. integrate
    if (np.min(x) < so.inst.l0) or (np.max(x) > so.inst.l1):
        if so.stel.model=='phoenix':
            vraw,sraw = load_phoenix(so.stel.stel_file,so.stel.phoenix_folder,wav_start=np.min(x), wav_end=np.max(x)) #phot/m2/s/nm
        elif so.stel.model=='sonora':
            vraw,sraw = load_sonora(so.stel.stel_file,wav_start=np.min(x), wav_end=np.max(x)) #phot/m2/s/nm
    else:
        vraw,sraw = so.stel.vraw, so.stel.sraw

    filtered_stel = factor_0 * sraw * filt_interp(vraw)
    flux = integrate(vraw,filtered_stel)    #phot/m2/s

    phot_per_s_m2_per_Jy = 1.51*10**7 # convert to phot/s/m2 from Jansky
    
    flux_Jy = flux/phot_per_s_m2_per_Jy/dl_l
    
    # get zps
    zps          = np.loadtxt(so.filt.zp_file,dtype=str).T
    izp          = np.where((zps[0]==family) & (zps[1]==band))[0]
    zp           = float(zps[2][izp])

    mag = -2.5*np.log10(flux_Jy/zp)

    return mag


def get_band_mag(so,vraw, sraw, model,stel_file,family,band,factor_0):
    """
    REDO TO NOT ASSUME SO!
    factor_0: scaling model to photons
    """
    xfilt,yfilt  = load_filter(so.filt.filter_path,family,band)
    filt_interp  = interpolate.interp1d(xfilt, yfilt, bounds_error=False,fill_value=0)
    dl_l         = np.mean(integrate(xfilt,yfilt)/xfilt) # dlambda/lambda to account for spectral fraction
    # load stellar the multiply by scaling factor, factor_0, and filter. integrate
    # reload if filter extends past currently loaded stellar model
    # if (np.min(xfilt) < np.min(vraw)) or (np.max(xfilt) > np.max(vraw)):
	# 	if model=='phoenix':
	#         vraw,sraw = load_phoenix(stel_file,so.stel.phoenix_folder,wav_start=np.min(xfilt), wav_end=np.max(xfilt)) #phot/m2/s/nm
	# 	elif model=='sonora':
	#         vraw,sraw = load_sonora(stel_file,wav_start=np.min(xfilt), wav_end=np.max(xfilt)) #phot/m2/s/nm
	#     print('Note had to reload stellar model for _get_band_mag')

    if (np.min(xfilt) < np.min(vraw)) or (np.max(xfilt) > np.max(vraw)):
        if model=='phoenix':
            vraw,sraw = load_phoenix(stel_file,so.stel.phoenix_folder,wav_start=np.min(xfilt), wav_end=np.max(xfilt)) #phot/m2/s/nm
            print('Note: had to reload Phoenix stellar model for _get_band_mag')
        elif model=='sonora':
            vraw,sraw = load_sonora(stel_file,wav_start=np.min(xfilt), wav_end=np.max(xfilt)) #phot/m2/s/nm
            print('Note: had to reload Sonora stellar model for _get_band_mag')

    filtered_stel = factor_0 * sraw * filt_interp(vraw)
    flux = integrate(vraw,filtered_stel)    #phot/m2/s

    phot_per_s_m2_per_Jy = 1.51*10**7 # convert to phot/s/m2 from Jansky
    
    flux_Jy = flux/phot_per_s_m2_per_Jy/dl_l
    
    # get zps
    zps          = np.loadtxt(so.filt.zp_file,dtype=str).T
    izp          = np.where((zps[0]==family) & (zps[1]==band))[0]
    zp           = float(zps[2][izp])
	
    mag = -2.5*np.log10(flux_Jy/zp)
	
    return mag



def _lsf_rotate(deltav,vsini,epsilon=0.6):
    '''
    Computes vsini rotation kernel.
    Based on the IDL routine LSF_ROTATE.PRO

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
        Computed kernel profile

    velgrid : float
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
