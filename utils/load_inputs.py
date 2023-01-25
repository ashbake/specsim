##############################################################
# Load variables into objects object
###############################################################

import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from astropy.io import fits
from scipy import interpolate
import sys,glob
from astropy.convolution import Gaussian1DKernel, convolve

from functions import *

__all__ = ['fill_data','load_phoenix']

def load_phoenix(stelname,wav_start=750,wav_end=780):
	"""
	load fits file stelname with stellar spectrum from phoenix 
	http://phoenix.astro.physik.uni-goettingen.de/?page_id=15
	
	return subarray 
	
	wav_start, wav_end specified in nm
	
	convert s from egs/s/cm2/cm to phot/cm2/s/nm using
	https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf
	"""
	
	# conversion factor

	f = fits.open(stelname)
	spec = f[0].data / (1e8) # ergs/s/cm2/cm to ergs/s/cm2/Angstrom for conversion
	f.close()
	
	path = stelname.split('/')
	f = fits.open(path[0] + '/' + path[1] + '/' + path[2] +'/' + \
					 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
	lam = f[0].data # angstroms
	f.close()
	
	# Convert
	conversion_factor = 5.03*10**7 * lam #lam in angstrom here
	spec *= conversion_factor # phot/cm2/s/angstrom
	
	# Take subarray requested
	isub = np.where( (lam > wav_start*10.0) & (lam < wav_end*10.0))[0]

	# Convert 
	return lam[isub]/10.0,spec[isub] * 10 * 100**2 #nm, phot/m2/s/nm

def load_sonora(stelname,wav_start=750,wav_end=780):
	"""
	load sonora model file
	
	return subarray 
	
	wav_start, wav_end specified in nm
	
	convert s from erg/cm2/s/Hz to phot/cm2/s/nm using
	https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf

	wavelenght loaded is microns high to low
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

def scale_stellar(so, mag):
	"""
	scale spectrum by magnitude
	inputs: 
	so: object with all variables
	mag: magnitude in filter desired

	load new stellar to match bounds of filter since may not match working badnpass elsewhere
	"""
	if so.stel.model=='phoenix':
		stelv,stels       =  load_phoenix(so.stel.stel_file,wav_start=np.min(so.filt.xraw), wav_end=np.max(so.filt.xraw)) #phot/m2/s/nm
	elif so.stel.model=='sonora':
		stelv,stels       =  load_sonora(so.stel.stel_file,wav_start=np.min(so.filt.xraw), wav_end=np.max(so.filt.xraw)) #phot/m2/s/nm

	filt_interp       =  interpolate.interp1d(so.filt.xraw, so.filt.yraw, bounds_error=False,fill_value=0)

	filtered_stellar   = stels * filt_interp(stelv)    # filter profile resampled to phoenix times phoenix flux density
	nphot_expected_0   = calc_nphot(so.filt.dl_l, so.filt.zp, mag)    # what's the integrated flux supposed to be in photons/m2/s?
	nphot_phoenix      = integrate(stelv,filtered_stellar)            # what's the integrated flux now? in same units as ^
	
	return nphot_expected_0/nphot_phoenix

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


class fill_data():
	""" 
	Load variables into storage object
	
	Inputs: so (storage object with user defined things loaded)
	Outputs: so (storage object with data and other stuff loaded)
	
	Edits
	-----
	Ashley - initial implementation Oct 26, 2018
	"""
	def __init__(self, so):		
		# define 
		self.x = np.arange(so.const.l0,so.const.l1,0.0005)
		self.hispec(so)
		self.filter(so)
		self.stellar(so)
		self.telluric(so)

	def hispec(self,so):
		###########
		# load hispec transmission
		xtemp, ytemp  = np.loadtxt(so.hispec.transmission_file,delimiter=',').T #microns!
		f = interp1d(xtemp*1000,ytemp,kind='linear', bounds_error=False, fill_value=0)
		
		so.hispec.xtransmit, so.hispec.ytransmit = self.x, f(self.x) 


	def stellar(self,so):
		"""
		loads stellar spectrum
		returns spectrum scaled to input V band mag 

		everything in nm
		"""
		# Part 1: load raw spectrum
		#
		if so.var.teff < 2300: # sonora models arent sampled as well so use phoenix as low as can
			g    = '316' # mks units, np.log10(316 * 100)=4.5 to match what im holding for phoenix models.
			teff = str(int(so.var.teff))
			so.stel.stel_file         = so.stel.sonora_folder + 'sp_t%sg%snc_m0.0' %(teff,g)
			so.stel.vraw,so.stel.sraw = load_sonora(so.stel.stel_file,wav_start=so.const.l0,wav_end=so.const.l1)
			so.stel.model             = 'sonora'
		else:
			teff = str(int(so.var.teff)).zfill(5)
			so.stel.model             = 'phoenix' 
			so.stel.stel_file         = so.stel.phoenix_folder + 'lte%s-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'%(teff)
			so.stel.vraw,so.stel.sraw = load_phoenix(so.stel.stel_file,wav_start=so.const.l0, wav_end=so.const.l1) #phot/m2/s/nm
		
		so.stel.v   = self.x
		tck_stel    = interpolate.splrep(so.stel.vraw,so.stel.sraw, k=2, s=0)
		so.stel.s   = interpolate.splev(self.x,tck_stel,der=0,ext=1)

		# apply scaling factor to match filter zeropoint
		so.stel.factor_0   = scale_stellar(so, so.var.mag) 
		so.stel.s   *= so.stel.factor_0 
		so.stel.units = 'photons/s/m2/nm' # stellar spec is in photons/s/m2/nm

		# broaden star spectrum with rotation kernal
		if so.stel.vsini > 0:
			dwvl_mean = np.abs(np.nanmean(np.diff(self.x)))
			SPEEDOFLIGHT = 2.998e8 # m/s
			dvel_mean = (dwvl_mean / np.nanmean(self.x)) * SPEEDOFLIGHT / 1e3 # average sampling in km/s
			vsini_kernel,_ = _lsf_rotate(dvel_mean,so.stel.vsini,epsilon=0.6)
			flux_vsini     = convolve(so.stel.s,vsini_kernel,normalize_kernel=True)  # photons / second / Ang
			so.stel.s      = flux_vsini
			#****TODO****
	
	def telluric(self,so):
		"""
		load tapas telluric file
		"""
		data      = fits.getdata(so.tel.telluric_file)
		_,ind  = np.unique(data['Wave/freq'],return_index=True)
		tck_tel   = interpolate.splrep(data['Wave/freq'][ind],data['Total'][ind], k=2, s=0)
		so.tel.v, so.tel.s = self.x, interpolate.splev(self.x,tck_tel,der=0,ext=1)

	def filter(self,so):
		"""
		load tracking band
		"""
		# read zeropoint file, get zp
		zps                     = np.loadtxt(so.filt.zp_file,dtype=str).T
		izp                     = np.where((zps[0]==so.filt.family) & (zps[1]==so.filt.band))[0]
		so.filt.zp              = np.float(zps[2][izp])

		# find filter file and load filter
		so.filt.filter_file         = glob.glob(so.filt.filter_path + '*' + so.filt.family + '*' +so.filt.band + '.dat')[0]
		so.filt.xraw, so.filt.yraw  = np.loadtxt(so.filt.filter_file).T # nm, transmission out of 1
		so.filt.xraw /= 10
		
		f                       = interpolate.interp1d(so.filt.xraw, so.filt.yraw, bounds_error=False,fill_value=0)
		so.filt.v, so.filt.s    = so.hispec.xtransmit, f(so.hispec.xtransmit)  #filter profile sampled at stellar

		so.filt.dl_l            = np.mean(integrate(so.filt.xraw, so.filt.yraw)/so.filt.xraw) # dlambda/lambda


def load_filter(so,family,band):
	"""
	"""
	filter_file    = glob.glob(so.filt.filter_path + '*' + family + '*' + band + '.dat')[0]
	xraw, yraw     = np.loadtxt(filter_file).T # nm, transmission out of 1
	return xraw/10, yraw

