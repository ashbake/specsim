##############################################################
# Load variables into objects object
# ##############################################################

import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from astropy.io import fits
from scipy import interpolate
import sys,glob,os
import pandas as pd
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt, correlate
import warnings
from scipy import signal



import throughput_tools# import pick_coupling, get_band_mag, get_base_throughput,grid_interp_coupling
from wfe_tools import get_tip_tilt_resid, get_HO_WFE
import obs_tools
import noise_tools
import astropy.units as u
from functions import *
from astropy import constants as consts
from scipy.ndimage.interpolation import shift
import wfe_tools

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
	wave_file = '/' + os.path.join(*stelname.split('/')[0:-1]) + '/' + \
					'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits' #assume wave in same folder
	f = fits.open(wave_file)
	lam = f[0].data # angstroms
	f.close()
	
	# Convert
	conversion_factor = 5.03*10**7 * lam #lam in angstrom here
	spec *= conversion_factor # phot/cm2/s/angstrom
	
	# Take subarray requested
	isub = np.where( (lam > wav_start*10.0) & (lam < wav_end*10.0))[0]

	# Convert 
	return lam[isub]/10.0,spec[isub] * 10 * 100**2 #nm, phot/m2/s/nm

def load_filter(filter_path,family,band):
	"""
	"""
	files = glob.glob(filter_path + '*' + family + '*' + band + '.dat')
	if not files:
	    raise FileNotFoundError(f"No file matches the pattern {filter_path}*{family}*{band}.dat")
	filter_file = files[0]
#	filter_file    = glob.glob(filter_path + '*' + family + '*' + band + '.dat')[0]
	xraw, yraw     = np.loadtxt(filter_file).T # nm, transmission out of 1
	return xraw/10, yraw

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


def get_band_mag(so,family,band,factor_0):
    """
    factor_0: scaling model to photons
    """
    x,y          = load_filter(so.filt.filter_path,family,band)
    filt_interp  = interpolate.interp1d(x, y, bounds_error=False,fill_value=0)
    dl_l         = np.mean(integrate(x,y)/x) # dlambda/lambda to account for spectral fraction
    
    # load stellar the multiply by scaling factor, factor_0, and filter. integrate
    if (np.min(x) < so.inst.l0) or (np.max(x) > so.inst.l1):
        if so.stel.model=='phoenix':
            vraw,sraw = load_phoenix(so.stel.stel_file,wav_start=np.min(x), wav_end=np.max(x)) #phot/m2/s/nm
        elif so.stel.model=='sonora':
            vraw,sraw = load_sonora(so.stel.stel_file,wav_start=np.min(x), wav_end=np.max(x)) #phot/m2/s/nm
    else:
        vraw,sraw = so.stel.vraw, so.stel.sraw

    filtered_stel = factor_0 * sraw * filt_interp(vraw)
    flux = integrate(vraw,filtered_stel)    #phot/m2/s

    phot_per_s_m2_per_Jy = 1.51*10**7 # convert to phot/s/m2 from Jansky
    
    flux_Jy = flux/phot_per_s_m2_per_Jy/dl_l
    
    # get zps
    zps                     = np.loadtxt(so.filt.zp_file,dtype=str).T
    izp                     = np.where((zps[0]==family) & (zps[1]==band))[0]
    zp                      = float(zps[2][izp])

    mag = -2.5*np.log10(flux_Jy/zp)

    return mag,x,y


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
	Ashley - changed variable names jan 26, 2023
	"""
	def __init__(self, so):		
		print("------FILLING OBJECT--------")
		# define x array to carry everywhere
		self.x = np.arange(so.inst.l0,so.inst.l1,0.0005)
		ind_1 = np.where((self.x>940)&(self.x<1090))[0]
		ind_2 = np.where((self.x>1100)&(self.x<1360))[0]
		ind_3 = np.where((self.x>1480)&(self.x<1820))[0]
		ind_4 = np.where((self.x>1950)&(self.x<2350))[0]
		ind_filter = np.array(ind_1.tolist()+ind_2.tolist()+ind_3.tolist()+ind_4.tolist())
		self.x_filter = self.x[ind_filter]
		# define bands here
		so.inst.y=[980,1100]
		so.inst.J=[1170,1327]
		so.inst.H=[1490,1780]
		so.inst.K=[1990,2460]

		# order of these matter
		if so.run.mode == 'etc_snr_off':
			self.filter(so)
			self.planet(so)
			self.stellar(so)
			self.telluric(so)
			self.ao(so)
			self.instrument(so)
			self.get_speckle_noise(so)
			self.etc_D(so)

		elif so.run.mode =='etc_snr_on':
			self.filter(so)
			self.stellar(so)
			self.telluric(so)
			self.ao(so)
			self.instrument(so)
			self.etc(so)

		elif so.run.mode =='snr_on':
			self.filter(so)
			self.stellar(so)
			self.telluric(so)
			self.ao(so)
			self.instrument(so)
			self.observe(so)

		elif so.run.mode =='snr_off':
			self.filter(so)
			self.planet(so)
			self.stellar(so)
			self.telluric(so)
			self.ao(so)
			self.instrument(so)
			self.observe(so)
			self.get_speckle_noise(so)
			self.observe_D(so)

		elif so.run.mode =='rv_on':
			self.filter(so)
			self.stellar(so)
			self.telluric(so)
			self.ao(so)
			self.instrument(so)
			self.observe(so)
			self.get_order_bounds(so)
			self.make_telluric_mask(so)
			self.get_rv_content(so)
			self.get_rv_precision(so)
		elif so.run.mode =='rv_off':
			self.filter(so)
			self.stellar(so)
			self.telluric(so)
			self.ao(so)
			self.instrument(so)
			self.observe(so)
			self.get_speckle_noise(so)
			self.get_order_bounds(so)
			self.make_telluric_mask(so)
			self.get_rv_content_D(so)
			self.get_rv_precision_D(so)

		elif so.run.mode =='ccf':
			self.filter(so)
			self.planet(so)
			self.stellar(so)
			self.telluric(so)
			self.ao(so)
			self.instrument(so)
			self.observe(so)
			self.get_speckle_noise(so)
			self.observe_D(so)
			self.ccf(so)
		elif so.run.mode =='etc_ccf':
			self.filter(so)
			self.planet(so)
			self.stellar(so)
			self.telluric(so)
			self.ao(so)
			self.instrument(so)
			self.observe(so)
			self.get_speckle_noise(so)
			self.observe_D(so)
			self.etc_ccf(so)

		
	def filter(self,so):
		"""
		load band for scaling stellar spectrum
		"""
		# read zeropoint file, get zp
		zps                     = np.loadtxt(so.filt.zp_file,dtype=str).T
		izp                     = np.where((zps[0]==so.filt.family) & (zps[1]==so.filt.band))[0]
		so.filt.zp              = float(zps[2][izp])

		# find filter file and load filter
		so.filt.filter_file         = glob.glob(so.filt.filter_path + '*' + so.filt.family + '*' +so.filt.band + '.dat')[0]
		so.filt.xraw, so.filt.yraw  = np.loadtxt(so.filt.filter_file).T # nm, transmission out of 1
		if np.max(so.filt.xraw)>5000: so.filt.xraw /= 10
		if np.max(so.filt.xraw) < 10: so.filt.xraw *= 1000
		
		f                       = interpolate.interp1d(so.filt.xraw, so.filt.yraw, bounds_error=False,fill_value=0)
		so.filt.v, so.filt.s    = self.x, f(self.x)  #filter profile sampled at stellar

		so.filt.dl_l                 = np.mean(integrate(so.filt.xraw, so.filt.yraw)/so.filt.xraw) # dlambda/lambda
		so.filt.center_wavelength    = integrate(so.filt.xraw,so.filt.yraw*so.filt.xraw)/integrate(so.filt.xraw,so.filt.yraw)
####
	def planet(self,so):
		"""
		loads planet spectrum
		returns spectrum scaled to input V band mag 

		everything in nm

		date of the change: Jul 12, 2023

        Huihao Zhang (zhang.12043@osu.edu)

		"""
		# Part 1: load raw spectrum
		#
		print('Planet Teff set to %s'%so.plan.teff)
		print('Planet %s band mag set to %s'%(so.filt.band,so.plan.mag))
	
		if so.plan.teff < 2300: # sonora models arent sampled as well so use phoenix as low as can
			g    = '316' # mks units, np.log10(316 * 100)=4.5 to match what im holding for phoenix models.
			pteff = str(int(so.plan.teff))
			so.stel.model             = 'sonora'
			so.stel.stel_file         = so.stel.sonora_folder + 'sp_t%sg%snc_m0.0' %(pteff,g)
			so.plan.vraw,so.plan.sraw = load_sonora(so.stel.stel_file,wav_start=so.inst.l0,wav_end=so.inst.l1)
		else:
			pteff = str(int(so.plan.teff)).zfill(5)
			so.stel.model             = 'phoenix' 
			so.stel.stel_file         = so.stel.phoenix_folder + 'lte%s-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'%(pteff)
			so.plan.vraw,so.plan.sraw = load_phoenix(so.stel.stel_file,wav_start=so.inst.l0, wav_end=so.inst.l1) #phot/m2/s/nm
		
		so.plan.v   = self.x
		tck_plan   = interpolate.splrep(so.plan.vraw,so.plan.sraw, k=2, s=0)
		so.plan.s   = interpolate.splev(self.x,tck_plan,der=0,ext=1)

		# apply scaling factor to match filter zeropoint
		so.plan.factor_0   = scale_stellar(so, so.plan.mag) 
		so.plan.s   *= so.plan.factor_0
		so.plan.units = 'photons/s/m2/nm' # stellar spec is in photons/s/m2/nm

		# broaden star spectrum with rotation kernal
		if so.plan.vsini > 0:
			dwvl_mean = np.abs(np.nanmean(np.diff(self.x)))
			SPEEDOFLIGHT = 2.998e8 # m/s
			dvel_mean = (dwvl_mean / np.nanmean(self.x)) * SPEEDOFLIGHT / 1e3 # average sampling in km/s
			vsini_kernel,_ = _lsf_rotate(dvel_mean,so.plan.vsini,epsilon=0.6)
			flux_vsini     = convolve(so.plan.s,vsini_kernel,normalize_kernel=True)  # photons / second / Ang
			so.plan.s      = flux_vsini       
		if so.plan.rv > 0:
			dvelocity =   (so.plan.v / 300000) * u.nm * consts.c / (so.plan.v * u.nm )
			rv_shift_resel = np.mean(so.plan.rv * u.km / u.s / dvelocity) * 1000*u.m/u.km
			spec_shifted = shift(so.plan.s,rv_shift_resel.value)
			so.plan.s      = spec_shifted 

	def stellar(self,so):
		"""
		loads stellar spectrum
		returns spectrum scaled to input V band mag 

		everything in nm
		"""
		# Part 1: load raw spectrum
		#
		print('Star Teff set to %s'%so.stel.teff)
		print('Star %s band mag set to %s'%(so.filt.band,so.stel.mag))
	
		if so.stel.teff < 2300: # sonora models arent sampled as well so use phoenix as low as can
			g    = '316' # mks units, np.log10(316 * 100)=4.5 to match what im holding for phoenix models.
			teff = str(int(so.stel.teff))
			so.stel.stel_file         = so.stel.sonora_folder + 'sp_t%sg%snc_m0.0' %(teff,g)
			so.stel.vraw,so.stel.sraw = load_sonora(so.stel.stel_file,wav_start=so.inst.l0,wav_end=so.inst.l1)
			so.stel.model             = 'sonora'
		else:
			teff = str(int(so.stel.teff)).zfill(5)
			so.stel.model             = 'phoenix' 
			so.stel.stel_file         = so.stel.phoenix_folder + 'lte%s-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'%(teff)
			so.stel.vraw,so.stel.sraw = load_phoenix(so.stel.stel_file,wav_start=so.inst.l0, wav_end=so.inst.l1) #phot/m2/s/nm
		
		so.stel.v   = self.x
		tck_stel    = interpolate.splrep(so.stel.vraw,so.stel.sraw, k=2, s=0)
		so.stel.s   = interpolate.splev(self.x,tck_stel,der=0,ext=1)

		# apply scaling factor to match filter zeropoint
		so.stel.factor_0   = scale_stellar(so, so.stel.mag) 
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
		if so.stel.rv > 0:
			dvelocity =   (so.stel.v / 300000) * u.nm * consts.c / (so.stel.v * u.nm )
			rv_shift_resel = np.mean(so.stel.rv * u.km / u.s / dvelocity) * 1000*u.m/u.km
			spec_shifted = shift(so.stel.s,rv_shift_resel.value)
			so.stel.s      = spec_shifted


####

####            
	def get_speckle_noise(self,so):
		'''
		Returns the contrast for a given list of separations.

		Inputs: 
		separations  - A list of separations at which to calculate the speckle noise in arcseconds [float list length n]. Assumes these are sorted. 
		ao_mag       - The magnitude in the ao band, here assumed to be I-band
		wvs          - A list of wavelengths in microns [float length m]
		telescope    - A psisim telescope object.

		Outputs: 
		get_speckle_noise - Either an array of length [n,1] if only one wavelength passed, or shape [n,m]

        date of the change: Jul 12, 2023

        Huihao Zhang (zhang.12043@osu.edu)
		based on "get_speckle_noise" in https://github.com/planetarysystemsimager/psisim/blob/kpic/psisim/instruments/modhis.py
		note that get_speckle_noise in PSIsim-hispec and PSIsim-modhis are the same
		'''

		#TODO: decide if PIAA will be optional via flag or permanent
		#TODO: add ADC residuals effect
		#TODO: @Max, why feed "filter", "star_spt" if not used. Why feed "telescope" if already available from self.telescope?

		#if self.mode != 'vfn':
		#    print("Warning: only 'vfn' mode has been confirmed")
		separations = (so.plan.ang_sep * u.mas).to(u.arcsec)
		ao_mag=so.ao.ho_wfe_mag
		mode=so.coron.mode
		wvs=so.inst.xtransmit/1000 * u.um
        
		if so.tel.seeing == 'good':
			so.coron.telescope_seeing = 0.6
		elif so.tel.seeing == 'bad':
			so.coron.telescope_seeing = 1.05
		else:
			so.coron.telescope_seeing=1.5
		if mode == "on-axis":
			return np.ones([np.size(separations),np.size(wvs)])

		if np.size(wvs) < 2:
			wvs = np.array(wvs)

		if mode == "off-axis":
			#-- Deal with nominal MODHIS mode (fiber centered on planet)
			#TODO: this was copied from KPIC instrument. Check if any mods are needed for MODHIS

			#Get the Strehl Ratio
			SR = so.inst.strehl

			p_law_kolmogorov = -11./3
			p_law_ao_coro_filter = so.coron.p_law_dh#-p_law_kolmogorov 

			r0 = 0.55e-6/((so.coron.telescope_seeing*u.arcsec).to(u.arcsecond).value/206265) * u.m #Easiest to ditch the seeing unit here. 

			#The AO control radius in units of lambda/D
			cutoff = so.coron.nactuators/2

			contrast = np.zeros([np.size(separations),np.size(wvs)])

			if np.size(separations) < 2:
				separations = np.array([separations.value])*separations.unit

			#Dimitri to put in references to this math
			r0_sc = r0 * (wvs/(0.55*u.micron))**(6./5)
			w_halo = so.inst.tel_diam * u.m / r0_sc

			for i,sep in enumerate(separations):
				ang_sep_resel_in = sep/206265/u.arcsecond*so.inst.tel_diam * u.m /wvs.to(u.m) #Convert separtiona from arcsec to units of lam/D

				f_halo = np.pi*(1-SR)*0.488/w_halo**2 * (1+11./6*(ang_sep_resel_in/w_halo)**2)**(-11/6.)

				contrast_at_cutoff = np.pi*(1-SR)*0.488/w_halo**2 * (1+11./6*(cutoff/w_halo)**2)**(-11/6.)
				#Fill in the contrast array
				contrast[i,:] = f_halo

				biggest_ang_sep = np.abs(ang_sep_resel_in - cutoff) == np.min(np.abs(ang_sep_resel_in - cutoff))

				contrast[i][ang_sep_resel_in < cutoff] = contrast_at_cutoff[ang_sep_resel_in < cutoff]*(ang_sep_resel_in[ang_sep_resel_in < cutoff]/cutoff)**p_law_ao_coro_filter

			#Apply the fiber contrast gain
			contrast /= so.coron.fiber_contrast_gain

			#Make sure nothing is greater than 1. 
			contrast[contrast>1] = 1.

			so.coron.contrast= contrast
			print("dynamic instrument contrast ready")

		else:
			raise ValueError("'%s' is a not a supported 'mode'" % (mode))
		
	def get_order_bounds(self,so):
		"""
		given array, return max and mean of snr per order
		"""
		if so.rv.line_spacing == 'None':
			line_spacing= None
		else :
			line_spacing=so.rv.line_spacing

		peak_spacing=so.rv.peak_spacing
		height=so.rv.height
		order_peaks	  = signal.find_peaks(so.inst.base_throughput_filter,height=height,distance=peak_spacing,prominence=0.01)
		ind_1 = np.where((so.stel.v>940)&(so.stel.v<1090))[0]
		ind_2 = np.where((so.stel.v>1100)&(so.stel.v<1360))[0]
		ind_3 = np.where((so.stel.v>1480)&(so.stel.v<1820))[0]
		ind_4 = np.where((so.stel.v>1950)&(so.stel.v<2350))[0]
		so.stel.ind_filter = np.array(ind_1.tolist()+ind_2.tolist()+ind_3.tolist()+ind_4.tolist())
		so.stel.v_filter = so.stel.v[so.stel.ind_filter]
		order_cen_lam	= so.stel.v_filter[order_peaks[0]]
		blaze_angle	  =  76
		order_indices	=[]
		for i,lam_cen in enumerate(order_cen_lam):
			if line_spacing == None: line_spacing_now = 0.02 if lam_cen < 1475 else 0.01
			else: line_spacing_now=line_spacing
			m = np.sin(blaze_angle*np.pi/180) * 2 * (1/line_spacing_now)/(lam_cen/1000)
			fsr  = lam_cen/m
			isub_test= np.where((so.stel.v[so.stel.ind_filter]> (lam_cen - fsr/2)) & (so.stel.v[so.stel.ind_filter] < (lam_cen+fsr/2))) #FINISH THIS
			#plt.plot(so.stel.v[isub_test],so.inst.ytransmit[isub_test],'k--')
			order_indices.append(np.where((so.obs.v[so.obs.ind_filter] > (lam_cen - 0.9*fsr/2)) & (so.obs.v[so.obs.ind_filter]  < (lam_cen+0.9*fsr/2)))[0])

		so.rv.order_cen_lam=order_cen_lam
		so.rv.order_indices=order_indices

	def make_telluric_mask(self,so):
		"""
		"""
		cutoff=so.rv.cutoff
		velocity_cutoff=int(so.rv.velocity_cutoff)
		water_only=so.rv.water_only ###  bool 
		telluric_spec = np.abs(so.tel.s/so.tel.rayleigh)**so.tel.airmass
		if water_only: telluric_spec = np.abs(so.tel.h2o)**so.tel.airmass #h2o only
		telluric_spec[np.where(np.isnan(telluric_spec))] = 0
		telluric_spec_lores = degrade_spec(so.stel.v, telluric_spec, so.inst.res)
		# resample onto v array
		filt_interp	 = interpolate.interp1d(so.stel.v_filter, telluric_spec_lores[so.stel.ind_filter], bounds_error=False,fill_value=0)
		ind_1 = np.where((so.obs.v>940)&(so.obs.v<1090))[0]
		ind_2 = np.where((so.obs.v>1100)&(so.obs.v<1360))[0]
		ind_3 = np.where((so.obs.v>1480)&(so.obs.v<1820))[0]
		ind_4 = np.where((so.obs.v>1950)&(so.obs.v<2350))[0]
		so.obs.ind_filter = np.array(ind_1.tolist()+ind_2.tolist()+ind_3.tolist()+ind_4.tolist())
		so.obs.v_filter = so.obs.v[so.obs.ind_filter]
		s_tel		 = filt_interp(so.obs.v_filter)/np.max(filt_interp(so.obs.v_filter))	# filter profile resampled to phoenix times phoenix flux density

		#cutoff = 0.01 # reject lines greater than 1% depth
		telluric_mask = np.ones_like(s_tel)
		telluric_mask[np.where(s_tel < (1-cutoff))[0]] = 0
		# avoid +/-5km/s  (5pix) around telluric
		for iroll in range(velocity_cutoff):
			telluric_mask[np.where(np.roll(s_tel,iroll) < (1-cutoff))[0]] = 0
			telluric_mask[np.where(np.roll(s_tel,-1*iroll) < (1-cutoff))[0]] = 0

		so.rv.telluric_mask=telluric_mask
		so.rv.s_tel=s_tel

	def get_rv_content(self,so):
		"""
		"""
		flux_interp = interpolate.InterpolatedUnivariateSpline(so.obs.v,so.obs.s, k=1)
		dflux = flux_interp.derivative()
		spec_deriv = dflux(so.obs.v_filter)
		sigma_ord = np.abs(so.obs.noise[so.obs.ind_filter]) #np.abs(s) ** 0.5 # np.abs(n)
		sigma_ord[np.where(sigma_ord ==0)] = 1e10
		all_w = (so.obs.v_filter ** 2.) * (spec_deriv ** 2.) / sigma_ord ** 2. # include read noise and dark here!!
		
		so.rv.all_w=all_w

	def get_rv_content_D(self,so):
		"""
		"""
		flux_interp = interpolate.InterpolatedUnivariateSpline(so.obs.v,so.obs.p, k=1)
		dflux = flux_interp.derivative()
		spec_deriv = dflux(so.obs.v_filter)
		sigma_ord = np.abs(so.obs.noise_p[so.obs.ind_filter]) #np.abs(s) ** 0.5 # np.abs(n)
		sigma_ord[np.where(sigma_ord ==0)] = 1e10
		all_w = (so.obs.v_filter ** 2.) * (spec_deriv ** 2.) / sigma_ord ** 2. # include read noise and dark here!!
		
		so.rv.all_w_p=all_w

	def get_rv_precision(self,so):
		SPEEDOFLIGHT = 2.998e8 # m/s
		all_w=so.rv.all_w
		order_cens=so.rv.order_cen_lam
		order_inds=so.rv.order_indices
		noise_floor=so.rv.rv_floor
		mask=so.rv.telluric_mask
		if np.any(mask==None):
			mask = np.ones_like(all_w)
		dv_vals = np.zeros_like(order_cens)
		for i,lam_cen in enumerate(order_cens):
			w_ord = all_w[order_inds[i]] * mask[order_inds[i]]
			dv_order  = SPEEDOFLIGHT / (np.nansum(w_ord[1:-1])**0.5) # m/s
			dv_vals[i]  = dv_order
		
		dv_tot  = np.sqrt(dv_vals**2 + noise_floor**2)
		dv_spec  = 1. / (np.nansum(1./dv_vals**2.))**0.5
		dv_spec_floor  = 1. / (np.nansum(1./dv_tot**2.))**0.5

		so.rv.dv_tot=dv_tot
		so.rv.dv_spec=dv_spec
		so.rv.dv_vals=dv_vals

	def get_rv_precision_D(self,so):
		SPEEDOFLIGHT = 2.998e8 # m/s
		all_w=so.rv.all_w_p
		order_cens=so.rv.order_cen_lam
		order_inds=so.rv.order_indices
		noise_floor=so.rv.rv_floor
		mask=so.rv.telluric_mask
		if np.any(mask==None):
			mask = np.ones_like(all_w)
		dv_vals = np.zeros_like(order_cens)
		for i,lam_cen in enumerate(order_cens):
			w_ord = all_w[order_inds[i]] * mask[order_inds[i]]
			dv_order  = SPEEDOFLIGHT / (np.nansum(w_ord[1:-1])**0.5) # m/s
			dv_vals[i]  = dv_order
		
		dv_tot  = np.sqrt(dv_vals**2 + noise_floor**2)
		dv_spec  = 1. / (np.nansum(1./dv_vals**2.))**0.5
		dv_spec_floor  = 1. / (np.nansum(1./dv_tot**2.))**0.5

		so.rv.dv_tot_p=dv_tot
		so.rv.dv_spec_p=dv_spec
		so.rv.dv_vals_p=dv_vals

####            

	def telluric(self,so):
		"""
		load tapas telluric file
		"""
		data      = fits.getdata(so.tel.telluric_file)
		pwv0      = fits.getheader(so.tel.telluric_file)['PWV']
		airmass0  = fits.getheader(so.tel.telluric_file)['AIRMASS']
		
		_,ind     = np.unique(data['Wave/freq'],return_index=True)
		tck_tel   = interpolate.splrep(data['Wave/freq'][ind],data['Total'][ind]**(so.tel.airmass/airmass0), k=2, s=0)
		so.tel.v, so.tel.s = self.x, interpolate.splev(self.x,tck_tel,der=0,ext=1)
		
		tck_tel    = interpolate.splrep(data['Wave/freq'][ind],data['H2O'][ind]**(so.tel.pwv * so.tel.airmass/pwv0/airmass0), k=2, s=0)
		so.tel.h2o = interpolate.splev(self.x,tck_tel,der=0,ext=1)

		tck_tel    = interpolate.splrep(data['Wave/freq'][ind],data['Rayleigh'][ind]**(so.tel.airmass/airmass0), k=2, s=0)
		so.tel.rayleigh = interpolate.splev(self.x,tck_tel,der=0,ext=1)

		tck_tel    = interpolate.splrep(data['Wave/freq'][ind],data['O3'][ind]**(so.tel.airmass/airmass0), k=2, s=0)
		so.tel.o3 = interpolate.splev(self.x,tck_tel,der=0,ext=1)

	def ao(self,so):
		if so.ao.mode == 'auto':
			print('auto ao mode')
			ho_wfe = []
			ao_ho_wfe_mag = []
			ao_ho_wfe_band = []
			tt_wfe_mag = []
			tt_wfe_band = []
			tt_wfe = []
			sr_tt = []
			sr_ho = []
			
			if so.ao.inst =='hispec':
				if so.ao.mag=='default': 
					factor_0 = so.stel.factor_0 # if mag is same as one loaded, dont change spectral mag
				else: 
					# scale to find factor_0 for new mag
					factor_0 = so.stel.factor_0 * 10**(0.4*so.ao.mag)
				if type(so.ao.ho_wfe_set) is str:
					f_full = pd.read_csv(so.ao.ho_wfe_set,header=[0,1])
					mags             = f_full['seeing'].values.T[0]
					f=f_full[['LGS_STRAP_45','SH','LGS_100J_45']]
					so.ao.modes = f.columns
				for i in range(len(f.columns)):
					wfes = f[f.columns[i][0]].values.T[0]
					ho_wfe_band= f.columns[i][1]
					ho_wfe_mag,x_test1,y_test1 = get_band_mag(so,'Johnson',ho_wfe_band,factor_0)
					f_howfe = interpolate.interp1d(mags,wfes, bounds_error=False,fill_value=10000)
					ao_ho_wfe     = float(f_howfe(ho_wfe_mag))
					strehl_ho = wfe_tools.calc_strehl(ao_ho_wfe,so.filt.center_wavelength)
					ho_wfe.append(ao_ho_wfe)
					ao_ho_wfe_band.append(ho_wfe_band)
					ao_ho_wfe_mag.append(ho_wfe_mag)
					sr_ho.append(strehl_ho)

				if type(so.ao.ttdynamic_set) is str:
					f_full = pd.read_csv(so.ao.ttdynamic_set,header=[0,1])
					mags             = f_full['seeing'].values.T[0]
					f=f_full[['LGS_STRAP_45','SH','LGS_100J_45']]
					so.ao.modes = f.columns
				for i in range(len(f.columns)):
					tts             = f[f.columns[i][0]].values.T[0]
					ttdynamic_band=f.columns[i][1] # this is the mag band wfe is defined in, must be more readable way..			
					ttdynamic_mag,x_test2,y_test2 = get_band_mag(so,'Johnson',ttdynamic_band,so.stel.factor_0) # get magnitude of star in appropriate band
					f_ttdynamic=  interpolate.interp1d(mags,tts, bounds_error=False,fill_value=10000)
					tt_dynamic     = float(f_ttdynamic(ttdynamic_mag))
					strehl_tt = wfe_tools.tt_to_strehl(tt_dynamic,so.filt.center_wavelength,so.inst.tel_diam)
					tt_wfe.append(tt_dynamic)
					tt_wfe_band.append(ttdynamic_band)
					tt_wfe_mag.append(ttdynamic_mag)
					sr_tt.append(strehl_tt)

				ind_auto_ao = np.where(np.array(sr_tt)*np.array(sr_ho) == np.max(np.array(sr_tt)*np.array(sr_ho)))[0][0]
				so.ao.sr_tt = sr_tt
				so.ao.sr_ho = sr_ho
				so.ao.mode=f.columns[ind_auto_ao][0]
				print("ao mode:", f.columns[ind_auto_ao][0], f.columns[ind_auto_ao][1])
				so.ao.tt_dynamic= tt_wfe[ind_auto_ao]
				so.ao.ho_wfe= ho_wfe[ind_auto_ao]
				so.ao.ho_wfe_mag=ao_ho_wfe_mag[ind_auto_ao]
				so.ao.ho_wfe_band=ao_ho_wfe_band[ind_auto_ao]
				so.ao.ttdynamic_mag=tt_wfe_mag[ind_auto_ao]
				so.ao.ttdynamic_band=tt_wfe_band[ind_auto_ao]
				print("tt:",so.ao.tt_dynamic)
				print("ho:",so.ao.ho_wfe)
				if so.ao.mode =='80J':
					so.ao.pywfs_dichroic = 1 - tophat(self.x,so.inst.J[0],so.inst.J[1],0.8)
				elif so.ao.mode =='80H':
					so.ao.pywfs_dichroic = 1 - tophat(self.x,so.inst.H[0],so.inst.H[1],0.8)
				elif so.ao.mode =='80JH':
					so.ao.pywfs_dichroic = 1 - tophat(self.x,so.inst.J[0],so.inst.H[1],0.8)
				elif so.ao.mode =='100JH':
					so.ao.pywfs_dichroic = 1 - tophat(self.x,so.inst.J[0],so.inst.H[1],1)
				elif so.ao.mode =='100K':
					so.ao.pywfs_dichroic = 1 - tophat(self.x,so.inst.K[0],so.inst.K[1],1)
				else:
					so.ao.pywfs_dichroic = np.ones_like(self.x)
								
			elif so.ao.inst =='modhis':
				ho_wfe = []
				ao_ho_wfe_mag = []
				ao_ho_wfe_band = []
				tt_wfe_mag = []
				tt_wfe_band = []
				tt_wfe = []
				sr_tt = []
				sr_ho = []
				if so.ao.mag=='default': 
					factor_0 = so.stel.factor_0 # if mag is same as one loaded, dont change spectral mag
				else: 
					# scale to find factor_0 for new mag
					factor_0 = so.stel.factor_0 * 10**(0.4*so.ao.mag)
				if type(so.ao.ho_wfe_set) is str:
					f_before = pd.read_csv(so.ao.ho_wfe_set,header=[0,1,2,3])
					mags             = f_before['seeing'].values.T[0]
					f = f_before[so.tel.seeing][str(int(so.tel.zenith))]
					so.ao.modes = f.columns
				for i in range(len(f.columns)):
					wfes = f[f.columns[i][0]].values.T[0]
					ho_wfe_band= f.columns[i][1]
					ho_wfe_mag,x_test3,y_test3 = get_band_mag(so,'Johnson',ho_wfe_band,so.stel.factor_0)
					f_howfe = interpolate.interp1d(mags,wfes, bounds_error=False,fill_value=10000)
					ao_ho_wfe     = float(f_howfe(ho_wfe_mag))
					strehl_ho = wfe_tools.calc_strehl(ao_ho_wfe,so.filt.center_wavelength)
					ho_wfe.append(ao_ho_wfe)
					ao_ho_wfe_band.append(ho_wfe_band)
					ao_ho_wfe_mag.append(ho_wfe_mag)
					sr_ho.append(strehl_ho)

				if type(so.ao.ttdynamic_set) is str:
					f_before = pd.read_csv(so.ao.ttdynamic_set,header=[0,1,2,3])
					mags            = f_before['seeing'].values.T[0]
					f = f_before[so.tel.seeing][str(int(so.tel.zenith))]
					so.ao.modes_tt  = f.columns # should match howfe..
				for i in range(len(f.columns)):
					tts             = f[f.columns[i][0]].values.T[0]
					ttdynamic_band=f.columns[i][1] # this is the mag band wfe is defined in, must be more readable way..			
					ttdynamic_mag,x_test4,y_test4= get_band_mag(so,'Johnson',ttdynamic_band,so.stel.factor_0) # get magnitude of star in appropriate band
					f_ttdynamic=  interpolate.interp1d(mags,tts, bounds_error=False,fill_value=10000)
					tt_dynamic     = float(f_ttdynamic(ttdynamic_mag))
					strehl_tt = wfe_tools.tt_to_strehl(tt_dynamic,so.filt.center_wavelength,so.inst.tel_diam)
					tt_wfe.append(tt_dynamic)
					tt_wfe_band.append(ttdynamic_band)
					tt_wfe_mag.append(ttdynamic_mag)
					sr_tt.append(strehl_tt)

				ind_auto_ao = np.where(np.array(sr_tt)*np.array(sr_ho) == np.max(np.array(sr_tt)*np.array(sr_ho)))[0][0]
				so.ao.mode=f.columns[ind_auto_ao][0]
				print("ao mode:", f.columns[ind_auto_ao][0], f.columns[ind_auto_ao][1])
				so.ao.tt_dynamic= tt_wfe[ind_auto_ao]
				so.ao.ho_wfe= ho_wfe[ind_auto_ao]
				so.ao.ho_wfe_mag=ao_ho_wfe_mag[ind_auto_ao]
				so.ao.ho_wfe_band=ao_ho_wfe_band[ind_auto_ao]
				so.ao.ttdynamic_mag=tt_wfe_mag[ind_auto_ao]
				so.ao.ttdynamic_band=tt_wfe_band[ind_auto_ao]
				print("tt:",so.ao.tt_dynamic)
				print("ho:",so.ao.ho_wfe)
				if so.ao.mode =='80J':
					so.ao.pywfs_dichroic = 1 - tophat(self.x,so.inst.J[0],so.inst.J[1],0.8)
				elif so.ao.mode =='80H':
					so.ao.pywfs_dichroic = 1 - tophat(self.x,so.inst.H[0],so.inst.H[1],0.8)
				elif so.ao.mode =='80JH':
					so.ao.pywfs_dichroic = 1 - tophat(self.x,so.inst.J[0],so.inst.H[1],0.8)
				elif so.ao.mode =='100JH':
					so.ao.pywfs_dichroic = 1 - tophat(self.x,so.inst.J[0],so.inst.H[1],1)
				elif so.ao.mode =='100K':
					so.ao.pywfs_dichroic = 1 - tophat(self.x,so.inst.K[0],so.inst.K[1],1)
				else:
					so.ao.pywfs_dichroic = np.ones_like(self.x)
			else:
				raise ValueError('instrument must be modhis or hispec')
		else:
			if so.ao.mag=='default': 
				factor_0 = so.stel.factor_0 # if mag is same as one loaded, dont change spectral mag
			else: 
				# scale to find factor_0 for new mag
				factor_0 = so.stel.factor_0 * 10**(0.4*so.ao.mag)

			if type(so.ao.ho_wfe_set) is str:
				f_before = pd.read_csv(so.ao.ho_wfe_set,header=[0,1,2,3])
				mags             = f_before['seeing'].values.T[0]
				f = f_before[so.tel.seeing][str(int(so.tel.zenith))]
				so.ao.modes = f.columns
				wfes             = f[so.ao.mode].values.T[0]
				so.ao.ho_wfe_band= f[so.ao.mode].columns[0] # this is the mag band wfe is defined in, must be more readable way..
				so.ao.ho_wfe_mag,x_test5,y_test5 = get_band_mag(so,'Johnson',so.ao.ho_wfe_band,factor_0) # get magnitude of star in appropriate band
				f_howfe          = interpolate.interp1d(mags,wfes, bounds_error=False,fill_value=10000)
				so.ao.ho_wfe     = float(f_howfe(so.ao.ho_wfe_mag))
				print('HO WFE %s mag is %s'%(so.ao.ho_wfe_band,so.ao.ho_wfe_mag))
			else:
				so.ao.ho_wfe = so.ao.ho_wfe_set

			if type(so.ao.ttdynamic_set) is str:
				f_before = pd.read_csv(so.ao.ttdynamic_set,header=[0,1,2,3])
				mags             = f_before['seeing'].values.T[0]
				f = f_before[so.tel.seeing][str(int(so.tel.zenith))]
				so.ao.modes_tt  = f.columns # should match howfe..
				tts             = f[so.ao.mode].values.T[0]
				so.ao.ttdynamic_band=f[so.ao.mode].columns[0] # this is the mag band wfe is defined in, must be more readable way..			
				so.ao.ttdynamic_mag,x_test6,y_test6 = get_band_mag(so,'Johnson',so.ao.ttdynamic_band,factor_0) # get magnitude of star in appropriate band
				f_ttdynamic=  interpolate.interp1d(mags,tts, bounds_error=False,fill_value=10000)
				so.ao.tt_dynamic     = float(f_ttdynamic(so.ao.ttdynamic_mag))
				print('Tip Tilt %s mag is %s'%(so.ao.ttdynamic_band,so.ao.ttdynamic_mag))
			else:
				so.ao.tt_dynamic = so.ao.ttdynamic_set

			print('AO mode: %s'%so.ao.mode)

			#so.ao.ho_wfe = get_HO_WFE(so.ao.v_mag,so.ao.mode) #old
			print('HO WFE is %s'%so.ao.ho_wfe)

			#so.ao.tt_dynamic = get_tip_tilt_resid(so.ao.v_mag,so.ao.mode)
			print('tt dynamic is %s'%so.ao.tt_dynamic)

			# consider throughput impact of ao here
			if so.ao.mode =='80J':
				so.ao.pywfs_dichroic = 1 - tophat(self.x,so.inst.J[0],so.inst.J[1],0.8)
			elif so.ao.mode =='80H':
				so.ao.pywfs_dichroic = 1 - tophat(self.x,so.inst.H[0],so.inst.H[1],0.8)
			elif so.ao.mode =='80JH':
				so.ao.pywfs_dichroic = 1 - tophat(self.x,so.inst.J[0],so.inst.H[1],0.8)
			elif so.ao.mode =='100JH':
				so.ao.pywfs_dichroic = 1 - tophat(self.x,so.inst.J[0],so.inst.H[1],1)
			elif so.ao.mode =='100K':
				so.ao.pywfs_dichroic = 1 - tophat(self.x,so.inst.K[0],so.inst.K[1],1)
			else:
				so.ao.pywfs_dichroic = np.ones_like(self.x)

	def instrument(self,so):
		###########
		# load hispec transmission
		#xtemp, ytemp  = np.loadtxt(so.inst.transmission_file,delimiter=',').T #microns!
		#f = interp1d(xtemp*1000,ytemp,kind='linear', bounds_error=False, fill_value=0)
		
		#so.inst.xtransmit, so.inst.ytransmit = self.x, f(self.x) 

		# save dlambda
		sig = so.stel.v/so.inst.res/so.inst.res_samp # lambda/res = dlambda, nm per pixel
		so.inst.sig=sig

		# THROUGHPUT
		try: # if config has transmission file, use it, otherwise load HISPEC version
			thput_x, thput_y = np.loadtxt(so.inst.transmission_file,delimiter=',').T
			if np.max(thput_x) < 5: thput_x*=1000
			tck_thput   = interpolate.splrep(thput_x,thput_y, k=2, s=0)
			so.inst.xtransmit   = self.x
			so.inst.ytransmit   = interpolate.splev(self.x,tck_thput,der=0,ext=1)
			so.inst.base_throughput = so.inst.ytransmit.copy() # store this here bc ya
			#add airmass calc for strehl for seeing limited instrument
		except:
			so.inst.base_throughput  = throughput_tools.get_base_throughput(self.x,datapath=so.inst.transmission_path) # everything except coupling
			so.inst.base_throughput_filter  = throughput_tools.get_base_throughput(self.x_filter,datapath=so.inst.transmission_path) # everything except coupling

			# interp grid
			try: so.inst.points
			except AttributeError: 
				out = throughput_tools.grid_interp_coupling(int(so.inst.pl_on),path=so.inst.transmission_path + 'coupling/',atm=int(so.inst.atm),adc=int(so.inst.adc))
				so.inst.grid_points, so.inst.grid_values = out[0],out[1:] #if PL, three values
			try:
				so.inst.coupling, so.inst.strehl,so.inst.rawcoup = throughput_tools.pick_coupling(self.x,so.ao.ho_wfe,so.ao.tt_static,so.ao.tt_dynamic,LO=so.ao.lo_wfe,PLon=so.inst.pl_on,points=so.inst.grid_points, values=so.inst.grid_values)
			except ValueError:
				# hack here bc tt dynamic often is out of bounds
				so.inst.coupling, so.inst.strehl,so.inst.rawcoup = throughput_tools.pick_coupling(self.x,so.ao.ho_wfe,so.ao.tt_static,20,LO=so.ao.lo_wfe,PLon=so.inst.pl_on,points=so.inst.grid_points, values=so.inst.grid_values)
				so.inst.notes = 'tt dynamic out of bounds! %smas' %so.ao.tt_dynamic

			so.inst.xtransmit = self.x
			so.inst.ytransmit = so.inst.base_throughput* so.inst.coupling * so.ao.pywfs_dichroic
            



	def observe(self,so):
		"""
		"""
		flux_per_sec_nm = so.stel.s  * so.inst.tel_area * so.inst.ytransmit * np.abs(so.tel.s)

		if so.obs.texp_frame_set=='default':
			max_ph_per_s  =  np.max(flux_per_sec_nm * so.inst.sig)
			if so.obs.texp < 900: 
				so.obs.texp_frame = np.min((so.obs.texp,so.inst.saturation/max_ph_per_s))
			else:
				so.obs.texp_frame = np.min((900,so.inst.saturation/max_ph_per_s))
			print('Texp per frame set to %s'%so.obs.texp_frame)
			so.obs.nframes = int(np.ceil(so.obs.texp/so.obs.texp_frame))
			print('Nframes set to %s'%so.obs.nframes)
		else:
			so.obs.texp_frame = so.obs.texp_frame_set
			so.obs.nframes = int(np.ceil(so.obs.texp/so.obs.texp_frame))
			print('Texp per frame set to %s'%so.obs.texp_frame)
			print('Nframes set to %s'%so.obs.nframes)
		
		# degrade to instrument resolution
		so.obs.flux_per_nm = flux_per_sec_nm * so.obs.texp_frame
		s_ccd_lores = degrade_spec(so.stel.v, so.obs.flux_per_nm, so.inst.res)

		# resample onto res element grid
		so.obs.v, so.obs.s_frame = resample(so.stel.v,s_ccd_lores,sig=so.inst.sig, dx=0, eta=1,mode='variable')
		so.obs.s_frame *=so.inst.extraction_frac # extraction fraction, reduce photons
		so.obs.s =  so.obs.s_frame * so.obs.nframes
		ind_1 = np.where((so.obs.v>940)&(so.obs.v<1090))[0]
		ind_2 = np.where((so.obs.v>1100)&(so.obs.v<1360))[0]
		ind_3 = np.where((so.obs.v>1480)&(so.obs.v<1820))[0]
		ind_4 = np.where((so.obs.v>1950)&(so.obs.v<2350))[0]
		so.obs.ind_filter = np.array(ind_1.tolist()+ind_2.tolist()+ind_3.tolist()+ind_4.tolist())
		so.obs.s_filter = so.obs.s[so.obs.ind_filter]

		# resample throughput for applying to sky background
		base_throughput_interp= interpolate.interp1d(so.inst.xtransmit,so.inst.ytransmit)

		# load background spectrum - sky is top of telescope and will be reduced by inst BASE throughput. Coupling already accounted for in solid angle of fiber. Does inst bkg need throughput applied?
		so.obs.sky_bg_ph    = base_throughput_interp(so.obs.v) * noise_tools.get_sky_bg(so.obs.v,so.tel.airmass,pwv=so.tel.pwv,skypath=so.tel.skypath)
		so.obs.inst_bg_ph   = noise_tools.get_inst_bg(so.obs.v,npix=so.inst.pix_vert,R=so.inst.res,diam=so.inst.tel_diam,area=so.inst.tel_area,datapath=so.inst.transmission_path)
		so.obs.sky_bg_ph_test = noise_tools.get_sky_bg(so.obs.v,so.tel.airmass,pwv=so.tel.pwv,skypath=so.tel.skypath)
		# calc noise
		if so.inst.pl_on: # 3 port lantern hack
			noise_frame_yJ  = np.sqrt(3) * noise_tools.sum_total_noise(so.obs.s_frame/3,so.obs.texp_frame, so.obs.nsamp,so.obs.inst_bg_ph, so.obs.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert)
			noise_frame     = noise_tools.sum_total_noise(so.obs.s_frame,so.obs.texp_frame, so.obs.nsamp,so.obs.inst_bg_ph, so.obs.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert)
			yJ_sub          = np.where(so.obs.v < 1400)[0]
			noise_frame[yJ_sub] = noise_frame_yJ[yJ_sub] # fill in yj with sqrt(3) times noise in PL case
		else:
			noise_frame  = noise_tools.sum_total_noise(so.obs.s_frame,so.obs.texp_frame, so.obs.nsamp,so.obs.inst_bg_ph, so.obs.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert)
		
		noise_frame[np.where(np.isnan(noise_frame))] = np.inf
		noise_frame[np.where(noise_frame==0)] = np.inf
		
		so.obs.noise_frame = noise_frame
		so.obs.noise = np.sqrt(so.obs.nframes)*noise_frame

		so.obs.noise_filter = so.obs.noise[so.obs.ind_filter]
		so.obs.snr = so.obs.s_filter/so.obs.noise_filter
		#so.obs.v_resamp, so.obs.snr_reselement = resample(so.obs.v,so.obs.snr,sig=so.inst.res_samp, dx=0, eta=1/np.sqrt(so.inst.res_samp),mode='pixels')
		#so.obs.v_resamp, so.obs.s_reselement = resample(so.obs.v,so.obs.s,sig=so.inst.res_samp, dx=0, eta=1/np.sqrt(so.inst.res_samp),mode='pixels')
		#so.obs.v_resamp, so.obs.noise_reselement = resample(so.obs.v,so.obs.noise,sig=so.inst.res_samp, dx=0, eta=1/np.sqrt(so.inst.res_samp),mode='pixels')

####
	def observe_D(self,so):
		"""
		direct imaging

		date of the change: Jul 12, 2023

        Huihao Zhang (zhang.12043@osu.edu)
		based on "simulate_observation" in https://github.com/planetarysystemsimager/psisim/blob/kpic/psisim/observation.py
		"""
		flux_per_sec_nm_s = so.stel.s  * so.inst.tel_area * so.inst.ytransmit * np.abs(so.tel.s)
		flux_per_sec_nm_p = so.plan.s  * so.inst.tel_area * so.inst.ytransmit * np.abs(so.tel.s)
		flux_per_sec_nm_p_nosky = so.plan.s  * so.inst.tel_area * so.inst.ytransmit

		so.obs.flux_per_sec_nm_p_before = flux_per_sec_nm_p
		so.obs.flux_per_sec_nm_s_before = flux_per_sec_nm_s
        
		if so.obs.texp_frame_set=='default':
			max_ph_per_s  =  np.max(flux_per_sec_nm_p * so.inst.sig)
			if so.obs.texp < 900: 
				so.obs.texp_frame = np.min((so.obs.texp,so.inst.saturation/max_ph_per_s))
			else:
				so.obs.texp_frame = np.min((900,so.inst.saturation/max_ph_per_s))
			print('Texp per frame set to %s'%so.obs.texp_frame)
			so.obs.nframes = int(np.ceil(so.obs.texp/so.obs.texp_frame))
			print('Nframes set to %s'%so.obs.nframes)
		else:
			so.obs.texp_frame = so.obs.texp_frame_set
			so.obs.nframes = int(np.ceil(so.obs.texp/so.obs.texp_frame))
			print('Texp per frame set to %s'%so.obs.texp_frame)
			print('Nframes set to %s'%so.obs.nframes)
		
		# degrade to instrument resolution
		so.obs.flux_per_nm_s = flux_per_sec_nm_s * so.obs.texp_frame
		s_ccd_lores = degrade_spec(so.stel.v, so.obs.flux_per_nm_s, so.inst.res)
		so.obs.s_ccd_lores = s_ccd_lores

		so.obs.flux_per_nm_p = flux_per_sec_nm_p * so.obs.texp_frame
		p_ccd_lores = degrade_spec(so.stel.v, so.obs.flux_per_nm_p, so.inst.res)
		so.obs.p_ccd_lores = p_ccd_lores

		so.obs.flux_per_nm_p_nosky = flux_per_sec_nm_p_nosky * so.obs.texp_frame
		p_nosky_ccd_lores = degrade_spec(so.stel.v, so.obs.flux_per_nm_p_nosky, so.inst.res)
		so.obs.p_nosky_ccd_lores = p_nosky_ccd_lores

        
		# resample onto res element grid
		so.obs.v, so.obs.s_frame = resample(so.stel.v,s_ccd_lores,sig=so.inst.sig, dx=0, eta=1,mode='variable')
		so.obs.s_frame *=so.inst.extraction_frac # extraction fraction, reduce photons
		so.obs.s =  so.obs.s_frame * so.obs.nframes
		so.obs.s_filter = so.obs.s[so.obs.ind_filter]
        
		so.obs.v, so.obs.p_frame = resample(so.stel.v,p_ccd_lores,sig=so.inst.sig, dx=0, eta=1,mode='variable')
		so.obs.p_frame *=so.inst.extraction_frac # extraction fraction, reduce photons
		so.obs.p =  so.obs.p_frame * so.obs.nframes
		so.obs.p_filter = so.obs.p[so.obs.ind_filter]


		so.obs.v, so.obs.p_nosky_frame = resample(so.stel.v,p_nosky_ccd_lores,sig=so.inst.sig, dx=0, eta=1,mode='variable')
		so.obs.p_nosky_frame *=so.inst.extraction_frac # extraction fraction, reduce photons
		so.obs.p_nosky =  so.obs.p_nosky_frame* so.obs.nframes
		so.obs.p_nosky_filter = so.obs.p_nosky[so.obs.ind_filter]

        
		# resample throughput for applying to sky background
		base_throughput_interp= interpolate.interp1d(so.inst.xtransmit,so.inst.ytransmit)
		instrument_contrast_interp= interpolate.interp1d(so.inst.xtransmit,so.coron.contrast)


		# load background spectrum - sky is top of telescope and will be reduced by inst BASE throughput. Coupling already accounted for in solid angle of fiber. Does inst bkg need throughput applied?
		so.coron.inst_contr = instrument_contrast_interp(so.obs.v)[0]
		so.obs.sky_bg_ph    = base_throughput_interp(so.obs.v) * noise_tools.get_sky_bg(so.obs.v,so.tel.airmass,pwv=so.tel.pwv,skypath=so.tel.skypath)
		so.obs.inst_bg_ph   = noise_tools.get_inst_bg(so.obs.v,npix=so.inst.pix_vert,R=so.inst.res,diam=so.inst.tel_diam,area=so.inst.tel_area,datapath=so.inst.transmission_path)

		sky_bg_tot = so.obs.sky_bg_ph * so.obs.texp_frame
		inst_bg_tot = so.obs.inst_bg_ph * so.obs.texp_frame
		# calc noise
		if so.inst.pl_on: # 3 port lantern hack
			noise_frame_yJ_p  = np.sqrt(3) * noise_tools.sum_total_noise_D(so.obs.p_frame/3,so.obs.s_frame/3,so.obs.texp_frame, so.obs.nsamp,so.obs.inst_bg_ph, so.obs.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert,so.coron.inst_contr)
			noise_frame_p     = noise_tools.sum_total_noise_D(so.obs.p_frame,so.obs.s_frame,so.obs.texp_frame, so.obs.nsamp,so.obs.inst_bg_ph, so.obs.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert,so.coron.inst_contr)
			yJ_sub_p          = np.where(so.obs.v < 1400)[0]
			noise_frame_p[yJ_sub_p] = noise_frame_yJ_p[yJ_sub_p] # fill in yj with sqrt(3) times noise in PL case

		else:
			noise_frame_p  = noise_tools.sum_total_noise_D(so.obs.p_frame,so.obs.s_frame,so.obs.texp_frame, so.obs.nsamp,so.obs.inst_bg_ph, so.obs.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert,so.coron.inst_contr)
		speckle_noise = so.obs.s_frame * so.coron.inst_contr
		so.obs.speckle_noise = speckle_noise * np.sqrt(so.obs.nframes)

#		so.obs.thermal =(so.obs.inst_bg_ph + so.obs.sky_bg_ph +  so.inst.darknoise * so.inst.pix_vert) * so.obs.texp_frame * so.obs.nframes
		so.obs.thermal =(so.obs.inst_bg_ph + so.obs.sky_bg_ph) * so.obs.texp_frame * so.obs.nframes
		so.obs.read_noise = np.max((3,(so.inst.readnoise/np.sqrt(so.inst.res_samp))))*np.sqrt(so.obs.nframes)
		so.obs.noise_frame_p=noise_frame_p
		noise_frame_p[np.where(np.isnan(noise_frame_p))] = np.inf
		noise_frame_p[np.where(noise_frame_p==0)] = np.inf

		so.obs.sky_bg_tot = sky_bg_tot
		so.obs.inst_bg_tot = inst_bg_tot

		sky_bg_tot[np.where(np.isnan(sky_bg_tot))] = np.inf
		sky_bg_tot[np.where(sky_bg_tot==0)] = np.inf

		inst_bg_tot[np.where(np.isnan(inst_bg_tot))] = np.inf
		inst_bg_tot[np.where(inst_bg_tot==0)] = np.inf		
		
		so.obs.sky_bg_tot = sky_bg_tot * np.sqrt(so.obs.nframes)
		so.obs.inst_bg_tot = inst_bg_tot * np.sqrt(so.obs.nframes)

		
		so.obs.noise_frame_p = noise_frame_p
		so.obs.noise_p = np.sqrt(so.obs.nframes)*noise_frame_p
		so.obs.noise_p_filter = so.obs.noise_p[so.obs.ind_filter]


		so.obs.p_snr = so.obs.p_filter/so.obs.noise_p_filter
		print('direct imaging ready')

	def etc(self,so):
		"""
		direct imaging

		date of the change: Jul 12, 2023

        Huihao Zhang (zhang.12043@osu.edu)
		based on "simulate_observation" in https://github.com/planetarysystemsimager/psisim/blob/kpic/psisim/observation.py
		"""
		flux_per_sec_nm_s = so.stel.s  * so.inst.tel_area * so.inst.ytransmit * np.abs(so.tel.s)
		
		# degrade to instrument resolution
		so.etc.flux_per_nm_s = flux_per_sec_nm_s * so.etc.texp_frame
		s_ccd_lores = degrade_spec(so.stel.v, so.etc.flux_per_nm_s, so.inst.res)
		so.etc.s_ccd_lores = s_ccd_lores

        
		# resample onto res element grid
		so.etc.v, so.etc.s_frame = resample(so.stel.v,s_ccd_lores,sig=so.inst.sig, dx=0, eta=1,mode='variable')
		so.etc.s_frame *=so.inst.extraction_frac # extraction fraction, reduce photons
        
        
		# resample throughput for applying to sky background
		base_throughput_interp= interpolate.interp1d(so.inst.xtransmit,so.inst.ytransmit)


		# load background spectrum - sky is top of telescope and will be reduced by inst BASE throughput. Coupling already accounted for in solid angle of fiber. Does inst bkg need throughput applied?
		so.etc.sky_bg_ph    = base_throughput_interp(so.etc.v) * noise_tools.get_sky_bg(so.etc.v,so.tel.airmass,pwv=so.tel.pwv,skypath=so.tel.skypath)
		so.etc.inst_bg_ph   = noise_tools.get_inst_bg(so.etc.v,npix=so.inst.pix_vert,R=so.inst.res,diam=so.inst.tel_diam,area=so.inst.tel_area,datapath=so.inst.transmission_path)
		# calc noise
		if so.inst.pl_on: # 3 port lantern hack
			noise_frame_yJ  = np.sqrt(3) * noise_tools.sum_total_noise(so.etc.s_frame/3,so.etc.texp_frame, so.obs.nsamp,so.etc.inst_bg_ph, so.etc.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert)
			noise_frame     = noise_tools.sum_total_noise(so.etc.s_frame,so.etc.texp_frame, so.obs.nsamp,so.etc.inst_bg_ph, so.etc.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert)
			yJ_sub          = np.where(so.etc.v < 1400)[0]
			noise_frame[yJ_sub] = noise_frame_yJ[yJ_sub] # fill in yj with sqrt(3) times noise in PL case
		else:
			noise_frame  = noise_tools.sum_total_noise(so.etc.s_frame,so.etc.texp_frame, so.obs.nsamp,so.etc.inst_bg_ph, so.etc.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert)

		so.etc.noise_frame_s = noise_frame

		s_frame = so.etc.s_frame 

		s_frame[np.where(np.isnan(s_frame))] = np.inf
		s_frame[np.where(s_frame==0)] = np.inf
		so.etc.s_frame = s_frame
		ind_1 = np.where((so.etc.v>940)&(so.etc.v<1090))[0]
		ind_2 = np.where((so.etc.v>1100)&(so.etc.v<1360))[0]
		ind_3 = np.where((so.etc.v>1480)&(so.etc.v<1820))[0]
		ind_4 = np.where((so.etc.v>1950)&(so.etc.v<2350))[0]
		so.obs.ind_filter = np.array(ind_1.tolist()+ind_2.tolist()+ind_3.tolist()+ind_4.tolist())

		so.etc.nframe_s = (so.etc.noise_frame_s / so.etc.s_frame)**2 * so.etc.SN**2
		so.etc.nframe_s_filter = so.etc.nframe_s[so.obs.ind_filter]
		so.etc.total_expt_s = so.etc.nframe_s * so.etc.texp_frame
		so.etc.total_expt_s = so.etc.total_expt_s[so.obs.ind_filter]
		
		
		print('ETC for off-axis mode ready')

	def etc_D(self,so):
		"""
		direct imaging

		date of the change: Jul 12, 2023

        Huihao Zhang (zhang.12043@osu.edu)
		based on "simulate_observation" in https://github.com/planetarysystemsimager/psisim/blob/kpic/psisim/observation.py
		"""
		flux_per_sec_nm_s = so.stel.s  * so.inst.tel_area * so.inst.ytransmit * np.abs(so.tel.s)
		flux_per_sec_nm_p = so.plan.s  * so.inst.tel_area * so.inst.ytransmit * np.abs(so.tel.s)
		so.etc.flux_per_sec_nm_p_before = flux_per_sec_nm_p
		so.etc.flux_per_sec_nm_s_before = flux_per_sec_nm_s
		
		# degrade to instrument resolution
		so.etc.flux_per_nm_s = flux_per_sec_nm_s * so.etc.texp_frame
		s_ccd_lores = degrade_spec(so.stel.v, so.etc.flux_per_nm_s, so.inst.res)
		so.etc.s_ccd_lores = s_ccd_lores

		so.etc.flux_per_nm_p = flux_per_sec_nm_p * so.etc.texp_frame
		p_ccd_lores = degrade_spec(so.stel.v, so.etc.flux_per_nm_p, so.inst.res)
		so.etc.p_ccd_lores = p_ccd_lores

        
		# resample onto res element grid
		so.etc.v, so.etc.s_frame = resample(so.stel.v,s_ccd_lores,sig=so.inst.sig, dx=0, eta=1,mode='variable')
		so.etc.s_frame *=so.inst.extraction_frac # extraction fraction, reduce photons
        
		so.etc.v, so.etc.p_frame = resample(so.stel.v,p_ccd_lores,sig=so.inst.sig, dx=0, eta=1,mode='variable')
		so.etc.p_frame *=so.inst.extraction_frac # extraction fraction, reduce photons
        
		# resample throughput for applying to sky background
		base_throughput_interp= interpolate.interp1d(so.inst.xtransmit,so.inst.ytransmit)
		instrument_contrast_interp= interpolate.interp1d(so.inst.xtransmit,so.coron.contrast)


		# load background spectrum - sky is top of telescope and will be reduced by inst BASE throughput. Coupling already accounted for in solid angle of fiber. Does inst bkg need throughput applied?
		so.coron.inst_contr = instrument_contrast_interp(so.etc.v)[0]
		so.etc.sky_bg_ph    = base_throughput_interp(so.etc.v) * noise_tools.get_sky_bg(so.etc.v,so.tel.airmass,pwv=so.tel.pwv,skypath=so.tel.skypath)
		so.etc.inst_bg_ph   = noise_tools.get_inst_bg(so.etc.v,npix=so.inst.pix_vert,R=so.inst.res,diam=so.inst.tel_diam,area=so.inst.tel_area,datapath=so.inst.transmission_path)
		# calc noise
		if so.inst.pl_on: # 3 port lantern hack
			noise_frame_yJ_p  = np.sqrt(3) * noise_tools.sum_total_noise_D(so.etc.p_frame/3,so.etc.s_frame/3,so.etc.texp_frame, so.obs.nsamp,so.etc.inst_bg_ph, so.etc.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert,so.coron.inst_contr)
			noise_frame_p     = noise_tools.sum_total_noise_D(so.etc.p_frame,so.etc.s_frame,so.etc.texp_frame, so.obs.nsamp,so.etc.inst_bg_ph, so.etc.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert,so.coron.inst_contr)
			yJ_sub_p          = np.where(so.etc.v < 1400)[0]
			noise_frame_p[yJ_sub_p] = noise_frame_yJ_p[yJ_sub_p] # fill in yj with sqrt(3) times noise in PL case
#so.etc.p_frame
		else:
			noise_frame_p  = noise_tools.sum_total_noise_D(so.etc.p_frame,so.etc.s_frame,so.etc.texp_frame, so.obs.nsamp,so.etc.inst_bg_ph, so.etc.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert,so.coron.inst_contr)

		if so.inst.pl_on: # 3 port lantern hack
			noise_frame_yJ  = np.sqrt(3) * noise_tools.sum_total_noise(so.etc.s_frame/3,so.etc.texp_frame, so.obs.nsamp,so.etc.inst_bg_ph, so.etc.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert)
			noise_frame     = noise_tools.sum_total_noise(so.etc.s_frame,so.etc.texp_frame, so.obs.nsamp,so.etc.inst_bg_ph, so.etc.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert)
			yJ_sub          = np.where(so.etc.v < 1400)[0]
			noise_frame[yJ_sub] = noise_frame_yJ[yJ_sub] # fill in yj with sqrt(3) times noise in PL case
		else:
			noise_frame  = noise_tools.sum_total_noise(so.etc.s_frame,so.etc.texp_frame, so.obs.nsamp,so.etc.inst_bg_ph, so.etc.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert)
		so.etc.noise_frame_s = noise_frame
		so.etc.noise_frame = noise_frame_p
		p_frame = so.etc.p_frame 

		p_frame[np.where(np.isnan(p_frame))] = np.inf
		p_frame[np.where(p_frame==0)] = np.inf
		

		s_frame = so.etc.s_frame 

		s_frame[np.where(np.isnan(s_frame))] = np.inf
		s_frame[np.where(s_frame==0)] = np.inf
		so.etc.s_frame = s_frame

		so.etc.p_frame = p_frame

		ind_1 = np.where((so.etc.v>940)&(so.etc.v<1090))[0]
		ind_2 = np.where((so.etc.v>1100)&(so.etc.v<1360))[0]
		ind_3 = np.where((so.etc.v>1480)&(so.etc.v<1820))[0]
		ind_4 = np.where((so.etc.v>1950)&(so.etc.v<2350))[0]
		so.obs.ind_filter = np.array(ind_1.tolist()+ind_2.tolist()+ind_3.tolist()+ind_4.tolist())

		so.etc.nframe = (so.etc.noise_frame / so.etc.p_frame)**2 * so.etc.SN**2
		so.etc.nframe_filter = so.etc.nframe[so.obs.ind_filter]
		so.etc.total_expt = so.etc.nframe * so.etc.texp_frame
		so.etc.total_expt_filter = so.etc.total_expt[so.obs.ind_filter]

		so.etc.nframe_s = (so.etc.noise_frame_s / so.etc.s_frame)**2 * so.etc.SN**2
		so.etc.nframe_s_filter = so.etc.nframe_s[so.obs.ind_filter]
		so.etc.total_expt_s = so.etc.nframe_s * so.etc.texp_frame
		so.etc.total_expt_s = so.etc.total_expt_s[so.obs.ind_filter]
		
		
		print('ETC for off-axis mode ready')

	def ccf(self,so):
		if so.etc.ccf=='open':
			#The photon flux at the object will be the stellar flux multipled by the contrast there: 
			# full_host_spectrum
			host_flux_at_obj = so.obs.s[so.obs.ind_filter] *so.obs.speckle_noise[so.obs.ind_filter]

			systematics = (so.etc.cal*(host_flux_at_obj+so.obs.thermal[so.obs.ind_filter]))**2 #Variance of systematics

			noise_plus_systematics = np.sqrt(so.obs.noise_p[so.obs.ind_filter]**2+systematics)
			sky_trans = np.interp(so.obs.v[so.obs.ind_filter],so.stel.v,so.tel.s)
			#Get the wavelength spacing
			dwvs = np.abs(so.obs.s[so.obs.ind_filter] - np.roll(so.obs.s[so.obs.ind_filter], 1))
			dwvs[0] = dwvs[1]
			dwv_mean = np.mean(dwvs)
			lsf_fwhm = (2.2* 10**(-5) * u.um/dwv_mean).decompose() #Get the lsf_fwhm in units of current wavelength spacing
			lsf_sigma = lsf_fwhm/(2*np.sqrt(2*np.log(2))) #Convert to sigma

			#Calculate the 
			sky_transmission_lsf = gaussian_filter(sky_trans,lsf_sigma.value)
			signal=so.obs.p[so.obs.ind_filter]
			model=so.obs.p_nosky[so.obs.ind_filter]
			total_noise=noise_plus_systematics
			sky_trans=sky_transmission_lsf
			systematics_residuals=so.etc.cal
			kernel_size=501
			norm_cutoff=0.8
			total_noise_var = (total_noise* u.ph)**2 
			bad_noise = np.isnan(total_noise_var)
			total_noise_var[bad_noise]=np.inf

			#Calculate some normalization factor
			#Dimitri to explain this better. 
			norm = ((1-systematics_residuals)*sky_trans)
			
			#Get a median-filtered version of your model spectrum
			model_medfilt = medfilt(model,kernel_size=kernel_size)
			#Subtract the median version from the original model, effectively high-pass filtering the model
			model_filt = model*u.ph-model_medfilt*u.ph
			model_filt[np.isnan(model_filt)] = 0.
			model_filt[norm<norm_cutoff] = 0.
			model_filt[bad_noise] = 0.

			#Divide out the sky transmision
			normed_signal = signal/norm
			#High-pass filter like with the model
			signal_medfilt = medfilt(normed_signal,kernel_size=kernel_size)
			signal_filt = normed_signal*u.ph-signal_medfilt*u.ph
			signal_filt[np.isnan(signal_filt)] = 0.
			signal_filt[norm<norm_cutoff] = 0.
			signal_filt[bad_noise] = 0.
			
			#Now the actual ccf_snr
			ccf_snr = np.sqrt((np.sum(signal_filt * model_filt/total_noise_var))**2 / np.sum(model_filt * model_filt/total_noise_var))
			so.obs.ccf_snr = ccf_snr
			print('CCF SNR ready')
		else:
			print('no ccf snr')

	def etc_ccf(self,so):
		warnings.warn('This function is incomplete at the moment. Double check all results for accuracy.')

		if so.etc.ccfetc == 'open':
			#Get the wavelength spacing
			# Compute total obs. time from instrument object
			sky_trans = np.interp(so.obs.v[so.obs.ind_filter],so.stel.v,so.tel.s)
			dwvs = np.abs(so.obs.s[so.obs.ind_filter] - np.roll(so.obs.s[so.obs.ind_filter], 1))
			dwvs[0] = dwvs[1]
			dwv_mean = np.mean(dwvs)
			lsf_fwhm = (2.2* 10**(-5) * u.um/dwv_mean).decompose() #Get the lsf_fwhm in units of current wavelength spacing
			lsf_sigma = lsf_fwhm/(2*np.sqrt(2*np.log(2))) #Convert to sigma

			#Calculate the 
			sky_transmission_lsf = gaussian_filter(sky_trans,lsf_sigma.value)
			obs_time = so.obs.texp

			host_flux_at_obj = so.obs.s[so.obs.ind_filter] *so.obs.speckle_noise[so.obs.ind_filter]

			systematics = (so.etc.cal*(host_flux_at_obj+so.obs.thermal[so.obs.ind_filter]))**2 #Variance of systematics
			signal = so.obs.p[so.obs.ind_filter]
			model = so.obs.p_nosky[so.obs.ind_filter]
			photon_noise = np.sqrt(so.obs.thermal[so.obs.ind_filter]+so.obs.p[so.obs.ind_filter]+so.obs.speckle_noise[so.obs.ind_filter])
			read_noise = so.obs.read_noise
			sky_trans=sky_transmission_lsf
			goal_ccf=so.etc.goal_ccf
			systematics_residuals=so.etc.cal
			kernel_size=501
			norm_cutoff=0.8


			#Calculate the 
			sky_transmission_lsf = gaussian_filter(sky_trans,lsf_sigma.value)

			# Remove time to get flux
			signal = signal / obs_time
			model  = model / obs_time

			#Get the noise variance
			total_noise_flux = (photon_noise**2 /obs_time) #+ (read_noise**2/instrument.n_exposures) #+ (systematics/ (obs_time**2))
			bad_noise = np.isnan(total_noise_flux)
			total_noise_flux[bad_noise]=np.inf

			#Calculate some normalization factor
			#Dimitri to explain this better. 
			norm = ((1-systematics_residuals)*sky_trans)

			#Get a median-filtered version of your model spectrum
			model_medfilt = medfilt(model,kernel_size=kernel_size)
			#Subtract the median version from the original model, effectively high-pass filtering the model
			model_filt = model-model_medfilt
			model_filt[np.isnan(model_filt)] = 0.
			model_filt[norm<norm_cutoff] = 0.
			model_filt[bad_noise] = 0.

			#Divide out the sky transmision
			normed_signal = signal/norm
			#High-pass filter like with the model
			signal_medfilt = medfilt(normed_signal,kernel_size=kernel_size)
			signal_filt = normed_signal-signal_medfilt
			signal_filt[np.isnan(signal_filt)] = 0.
			signal_filt[norm<norm_cutoff] = 0.
			signal_filt[bad_noise] = 0.

			#Now the actual ccf_snr
			min_exp_time = goal_ccf**2 / ((np.sum(signal_filt * model_filt/total_noise_flux))**2 / np.sum(model_filt * model_filt/total_noise_flux))
			so.obs.etc_ccf = min_exp_time
			print('ETC for ccf snr ready')
		else:
			print('no etc for ccf')
	

####

	def tracking(self,so):
		"""
		"""
		#pick guide camera - eventually settle on one and put params in config file!
		rn, pixel_pitch, qe_mod, dark,saturation = obs_tools.get_tracking_cam(camera=so.track.camera,x=self.x)
		so.track.pixel_pitch = pixel_pitch
		so.track.dark        = dark
		so.track.rn          = rn
		so.track.qe_mod      = qe_mod      # to switch cameras, wont need this later bc qe will match throughput model
		so.track.saturation  = saturation  # to switch cameras, wont need this later bc qe will match throughput model

		# load and store tracking camera throughput - file structure hard coded
		if type(so.track.transmission_file)==float:
			so.track.xtransmit,so.track.ytransmit = self.x, np.ones_like(self.x)*so.track.transmission_file * so.track.qe_mod
		else:
			xtemp, ytemp  = np.loadtxt(so.track.transmission_file,delimiter=',').T #microns!
			f = interp1d(xtemp*1000,ytemp,kind='linear', bounds_error=False, fill_value=0)
			so.track.xtransmit, so.track.ytransmit = self.x, f(self.x) * so.track.qe_mod 
	
		# get plate scale
		so.track.platescale = obs_tools.calc_plate_scale(so.track.pixel_pitch, D=so.inst.tel_diam, fratio=so.track.fratio)
		so.track.platescale_units = 'arcsec/pixel'

		# load tracking band
		bandpass, so.track.center_wavelength = obs_tools.get_tracking_band(self.x,so.track.band)
		so.track.bandpass = bandpass * so.ao.pywfs_dichroic

		# get fwhm (in pixels)
		so.track.fwhm = float(obs_tools.get_fwhm(so.ao.ho_wfe,so.ao.tt_dynamic,so.track.center_wavelength,so.inst.tel_diam,so.track.platescale,field_r=so.track.field_r,camera=so.track.camera,getall=False))
		so.track.fwhm_units = 'pixel'
		print('Tracking FWHM=%spix'%so.track.fwhm)
		
		so.track.strehl = np.exp(-(2*np.pi*so.ao.ho_wfe/so.track.center_wavelength)**2)

		# get sky background and instrument background, spec is ph/nm/s, fwhm must be in arcsec
		so.track.sky_bg_spec = noise_tools.get_sky_bg_tracking(self.x,so.track.fwhm*so.track.platescale,airmass=so.tel.airmass,pwv=so.tel.pwv,area=so.inst.tel_area,skypath=so.tel.skypath)
		so.track.sky_bg_ph   = so.track.texp * np.trapz(so.track.sky_bg_spec * so.track.bandpass * so.track.ytransmit,self.x) # sky bkg needs mult by throughput and bandpass profile

		so.track.inst_bg_spec = noise_tools.get_inst_bg_tracking(self.x,so.track.fwhm * so.track.platescale,area=76,datapath=so.inst.transmission_path)
		so.track.inst_bg_ph   = so.track.texp * np.trapz(so.track.inst_bg_spec * so.track.bandpass,self.x) # inst background needs multiplied by bandpass, inst throughput included in emissivities (i think)

		# get photons in band
		so.track.signal_spec = so.stel.s * so.track.texp *\
		 			so.inst.tel_area * so.track.ytransmit*\
		 			np.abs(so.tel.s)
	
		fac = 0.8 # amount of light approx under gaussian FWHM
		so.track.nphot = fac * so.track.strehl * np.trapz(so.track.signal_spec * so.track.bandpass,so.stel.v)
		print('Tracking photons: %s e-'%so.track.nphot)

		# get noise
		so.track.npix  = np.pi* (so.track.fwhm/2)**2 # only take noise in circle of diameter FWHM 
		so.track.noise = noise_tools.sum_total_noise(so.track.nphot,so.track.texp, 1, so.track.inst_bg_ph, so.track.sky_bg_ph,so.track.dark,so.track.rn,so.track.npix)
		print('Tracking noise: %s e-'%so.track.noise)
		so.track.snr = so.track.nphot/so.track.noise 
		
		# get centroid error, cap if saturated
		if so.track.nphot/so.track.npix > so.track.saturation:
			nphot_capped = so.inst.saturation * so.track.npix # cap nphot
			noise_capped = noise_tools.sum_total_noise(nphot_capped,so.track.texp, 1, so.track.inst_bg_ph, so.track.sky_bg_ph,so.track.dark,so.track.rn,so.track.npix)
			snr_capped   = nphot_capped/noise_capped
			so.track.centroid_err = (1/np.pi) * so.track.fwhm/snr_capped # same fwhm but snr is reduced to not saturate like if used an ND filter
			so.track.noise  = noise_capped
			so.track.snr    = snr_capped
			so.track.signal = nphot_capped
		else:
			so.track.centroid_err = (1/np.pi) * so.track.fwhm/so.track.snr

	def set_teff_aomode(self,so,temp,aomode,trackonly=False):
		"""
		given new temperature, relaod things as needed
		mode: 'track' or 'spec'
		"""
		so.stel.teff = temp
		so.ao.mode   = aomode
		self.stellar(so)
		self.ao(so)
		if not trackonly:
			self.instrument(so)
		self.tracking(so)
		self.observe(so)

	def set_teff_mag(self,so,temp,mag,star_only=False,trackonly=False):
		"""
		given new temperature, relaod things as needed
		mode: 'track' or 'spec'
		"""
		so.stel.teff  = temp
		so.stel.mag   = mag
		self.stellar(so)
		if not star_only:
			if trackonly:
				self.ao(so)
				self.instrument(so)
				self.tracking(so)
			else:
				self.ao(so)
				self.instrument(so)
				self.observe(so)

	def set_mag(self,so,mag,trackonly=False):
		"""
		given new temperature, relaod things as needed
		"""
		print('-----Reloading Stellar Magnitude-----')
		so.stel.mag = mag
		self.stellar(so)
		self.ao(so)
		if not trackonly:
			self.instrument(so)
		self.tracking(so)
		self.observe(so)

	def set_tracking_band_texp(self,so,band,texp):
		"""
		given new temperature, relaod things as needed
		"""
		print('-----Reloading Tracking Band and Exposure Time------')
		so.track.band = band
		so.track.texp = texp
		self.tracking(so)

	def set_ao_mode(self,so,mode,trackonly=False):
		"""
		given new temperature, relaod things as needed
		"""
		print('-----Reloading Stellar Magnitude-----')
		so.ao.mode = mode
		self.ao(so)
		if not trackonly:
			self.instrument(so)
		self.tracking(so)
		self.observe(so)

	def set_filter_band_mag(self,so,band,family,mag,trackonly=False):
		"""
		given new filter band, reload everything
		"""
		print('-----Reloading Filter Band Definition-----')
		so.filt.band = band
		so.filt.family = family
		so.stel.mag=mag
		self.filter(so)
		self.stellar(so)
		self.ao(so)
		if not trackonly:
			self.instrument(so)
			self.observe(so)
		self.tracking(so)
