##############################################################
# Load variables into storage object
###############################################################

import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from astropy.io import fits
from scipy import interpolate
import sys,glob,os
import pandas as pd
from astropy.convolution import convolve

from specsim import throughput_tools# import pick_coupling, get_band_mag, get_base_throughput,grid_interp_coupling
from specsim import obs_tools
from specsim import noise_tools
from specsim import wfe_tools 
from specsim import ccf_tools 

from specsim.functions import *

__all__ = ['fill_data','load_phoenix']

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


def _load_stellar_model(x,mag,teff,vsini,so,rv=0):
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


def _get_band_mag(so,vraw, sraw, model,stel_file,family,band,factor_0):
    """
    REDO TO NOT ASSUME THE STAR!!
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
	def __init__(self, so,track_on=False):		
		print("------FILLING OBJECT--------")
		# define x array to carry everywhere
		self.x = np.arange(so.inst.l0,so.inst.l1,0.0005)
		self.bands = {}
		self.bands['y'] = [980,1100]
		self.bands['J'] = [1170,1327]
		self.bands['H'] = [1490,1780]
		self.bands['K'] = [1990,2460]

		# define bands here
		# this should become deprecated
		so.inst.y=[980,1100]
		so.inst.J=[1170,1327]
		so.inst.H=[1490,1780]
		so.inst.K=[1990,2460]

		# order of these matter
		self.filter(so)
		self.stellar(so)
		self.telluric(so)
		self.ao(so)
		self.instrument(so)
		self.observe(so)

		# turn off tracking for now, not needed
		if track_on:
			self.tracking(so)
		self.track_on=track_on

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
		if np.max(so.filt.xraw) > 5000: so.filt.xraw /= 10
		if np.max(so.filt.xraw) < 10: so.filt.xraw *= 1000
		
		f                       = interpolate.interp1d(so.filt.xraw, so.filt.yraw, bounds_error=False,fill_value=0)
		so.filt.v, so.filt.s    = self.x, f(self.x)  #filter profile sampled at stellar

		so.filt.dl_l                 = np.mean(integrate(so.filt.xraw, so.filt.yraw)/so.filt.xraw) # dlambda/lambda
		so.filt.center_wavelength    = integrate(so.filt.xraw,so.filt.yraw*so.filt.xraw)/integrate(so.filt.xraw,so.filt.yraw)

	def stellar(self,so):
		"""
		loads stellar spectrum
		returns spectrum scaled to input V band mag 

		everything in nm
		"""
		# Part 1: load raw spectrum
		#
		print('Teff set to %s'%so.stel.teff)
		print('%s band mag set to %s'%(so.filt.band,so.stel.mag))
	
		# load on axis target
		so.stel.s, so.stel.vraw,so.stel.sraw,so.stel.model, so.stel.stel_file, so.stel.factor_0 = _load_stellar_model(self.x,so.stel.mag,so.stel.teff,so.stel.vsini,so,rv=so.stel.rv)
		# load companion if there is one (requires separation>0)
		if so.stel.pl_sep>0:
			so.stel.pl_s, _,_,so.stel.pl_model, so.stel.pl_stel_file, so.stel.pl_factor_0 = _load_stellar_model(self.x,so.stel.pl_mag,so.stel.pl_teff,so.stel.pl_vsini,so,rv=so.stel.rv)

		so.stel.v   = self.x
		so.stel.units = 'photons/s/m2/nm' # stellar spec is in photons/s/m2/nm

	def telluric(self,so):
		"""
		load tapas telluric file
		"""
		data      = fits.getdata(so.tel.telluric_file)
		pwv0      = fits.getheader(so.tel.telluric_file)['PWV']
		airmass0  = fits.getheader(so.tel.telluric_file)['AIRMASS']
		
		so.tel.airmass = 1/np.cos(np.pi * so.obs.zenith_angle / 180.)

		_,ind     = np.unique(data['Wave/freq'],return_index=True)
		#tck_tel   = interpolate.splrep(data['Wave/freq'][ind],data['Total'][ind]**(so.tel.airmass/airmass0), k=2, s=0)
		so.tel.v = self.x
		#so.tel.s = interpolate.splev(self.x,tck_tel,der=0,ext=1)
		
		tck_tel    = interpolate.splrep(data['Wave/freq'][ind],data['H2O'][ind]**(so.tel.pwv * so.tel.airmass/pwv0/airmass0), k=2, s=0)
		so.tel.h2o = interpolate.splev(self.x,tck_tel,der=0,ext=1)

		tck_tel    = interpolate.splrep(data['Wave/freq'][ind],data['Rayleigh'][ind]**(so.tel.airmass/airmass0), k=2, s=0)
		so.tel.rayleigh = interpolate.splev(self.x,tck_tel,der=0,ext=1)

		tck_tel    = interpolate.splrep(data['Wave/freq'][ind],data['O3'][ind]**(so.tel.airmass/airmass0), k=2, s=0)
		so.tel.o3  = interpolate.splev(self.x,tck_tel,der=0,ext=1)

		tck_tel    = interpolate.splrep(data['Wave/freq'][ind],data['O2'][ind]**(so.tel.airmass/airmass0), k=2, s=0)
		so.tel.o2  = interpolate.splev(self.x,tck_tel,der=0,ext=1)

		tck_tel    = interpolate.splrep(data['Wave/freq'][ind],data['N2'][ind]**(so.tel.airmass/airmass0), k=2, s=0)
		so.tel.n2  = interpolate.splev(self.x,tck_tel,der=0,ext=1)

		tck_tel    = interpolate.splrep(data['Wave/freq'][ind],data['CO'][ind]**(so.tel.airmass/airmass0), k=2, s=0)
		so.tel.co  = interpolate.splev(self.x,tck_tel,der=0,ext=1)

		tck_tel    = interpolate.splrep(data['Wave/freq'][ind],data['CH4'][ind]**(so.tel.airmass/airmass0), k=2, s=0)
		so.tel.ch4  = interpolate.splev(self.x,tck_tel,der=0,ext=1)
		
		tck_tel    = interpolate.splrep(data['Wave/freq'][ind],data['CO2'][ind]**(so.tel.airmass/airmass0), k=2, s=0)
		so.tel.co2  = interpolate.splev(self.x,tck_tel,der=0,ext=1)

		tck_tel    = interpolate.splrep(data['Wave/freq'][ind],data['N2O'][ind]**(so.tel.airmass/airmass0), k=2, s=0)
		so.tel.n2o  = interpolate.splev(self.x,tck_tel,der=0,ext=1)
		
		so.tel.s = so.tel.h2o * so.tel.rayleigh * so.tel.o3 *so.tel.o2*\
					so.tel.n2 * so.tel.co * so.tel.ch4 * so.tel.co2*\
					so.tel.n2o

		# seeing mapping
		if so.tel.seeing_set=='good': so.tel.seeing=0.6
		elif so.tel.seeing_set=='average': so.tel.seeing=0.8
		elif so.tel.seeing_set=='bad': so.tel.seeing=1.1
		else: print('seeing_set must be good, average, or bad')

	def ao(self,so):
		"""
		fill in ao info 
		"""
		if so.ao.teff=='default':
			vraw,sraw = so.stel.vraw, so.stel.sraw
			model     = so.stel.model
			stel_file = so.stel.stel_file
			if so.ao.mag=='default': 
				factor_0 = so.stel.factor_0
			else:
				# scale to find factor_0 for new mag if teff is the same
				factor_0 = so.stel.factor_0 * 10**(0.4*(so.stel.mag - so.ao.mag))
		else: # if new teff, load new model
			_, vraw, sraw, model, stel_file, factor_0 = _load_stellar_model(self.x,so.ao.mag,so.ao.teff,0,so)

		# now make getband mag take new stel file and factor 0

		if type(so.ao.ttdynamic_set) is not str or type(so.ao.ho_wfe_set) is not str:
			# set tt dynamic and ho wfe
			# requires either both to be text file or both to be floats
			so.ao.tt_dynamic = so.ao.ttdynamic_set
			so.ao.ho_wfe     = so.ao.ho_wfe_set
			if type(so.ao.ho_wfe) != type(so.ao.tt_dynamic): raise ValueError('HO WFE and TT Dynamic must *both* be set to float values or both to file paths to WFE files')
			so.ao.mode_chosen = 'User Defined'
			so.ao.band = 'N/A'
		else:
			# so.obs.zenith_angle = (180/np.pi) * np.arccos(1/so.tel.airmass) # if decide to take seeing
			data = wfe_tools.load_WFE(so.ao.ho_wfe_set, so.ao.ttdynamic_set, so.obs.zenith_angle, so.tel.seeing_set)
			ao_modes   = np.array(list(data.keys()))
			strehl, ho_wfes, tt_wfes, aomags = [], [], [],[]
			for ao_mode in ao_modes:
				# get magnitude in band the AO mode is defined in 
				wfe_mag  = _get_band_mag(so, vraw, sraw, model,stel_file,'Johnson',data[ao_mode]['band'],factor_0)
				#wfe_mag  = get_band_mag(so,'Johnson',data[ao_mode]['band'],factor_0)
				aomags.append(wfe_mag)
				# interpolate over WFEs and sample HO and TT at correct mag
				f_howfe    = interpolate.interp1d(data[ao_mode]['ho_mag'],data[ao_mode]['ho_wfe'], bounds_error=False,fill_value=10000)
				f_ttwfe    = interpolate.interp1d(data[ao_mode]['tt_mag'],data[ao_mode]['tt_wfe'], bounds_error=False,fill_value=10000)
				ho_wfe  = float(f_howfe(wfe_mag))
				tt_wfe  = float(f_ttwfe(wfe_mag))

				#compute strehl and save total
				strehl_ho = wfe_tools.calc_strehl_marechal(ho_wfe,so.filt.center_wavelength)
				strehl_tt = wfe_tools.tt_to_strehl(tt_wfe,so.filt.center_wavelength,so.inst.tel_diam)
				strehl.append(strehl_ho * strehl_tt)
				ho_wfes.append(ho_wfe)
				tt_wfes.append(tt_wfe)
				if 'PyWFS' in ao_mode:
					strehl[-1] *= 0 # hack to rid of pyramid mode for now

			so.ao.strehl_array = np.array(strehl)
			# if user wants the code to pick best mode:
			if so.ao.mode == 'auto' or so.ao.mode == 'Auto':
				print('Auto AO Mode')
				i_AO       = np.argmax(np.array(strehl))
			# if the user selected a specific mode:
			else:
				if so.ao.mode in ao_modes: 
					i_AO = np.where(so.ao.mode==ao_modes)[0][0]
				else:
					raise ValueError('AO mode chosen not a mode! Modes: auto or %s'%ao_modes)

			# store in object
			so.ao.mode_chosen   = ao_modes[i_AO]
			so.ao.ho_wfe        = ho_wfes[i_AO]
			so.ao.tt_dynamic    = tt_wfes[i_AO]
			so.ao.ao_mag        = aomags[i_AO]
			so.ao.strehl        = strehl[i_AO]
			so.ao.band          = data[so.ao.mode_chosen]['band']
			so.ao.ao_modes = ao_modes.copy()

			print('AO mag is %s in %s band for %sK AO star (%s=%s)'%(round(so.ao.ao_mag,2),so.ao.band, so.ao.teff,so.filt.band,so.ao.mag))
		# TODO: make name of mag in config to mag_set
		print('AO mode chosen: %s'%so.ao.mode_chosen)

		print('HO WFE is %s'%round(so.ao.ho_wfe))
	
		print('tt dynamic is %s'%round(so.ao.tt_dynamic,2))
		

		# consider throughput impact of ao mode here
		# dichroic gets applied to science
		# pywfs_dichroic gets applied to tracking
		"""
		if '100H' in so.ao.mode_chosen:
			so.ao.dichroic = 1 - tophat(self.x,so.inst.H[0],so.inst.H[1],1)
		elif '100J' in so.ao.mode_chosen:
			so.ao.dichroic = 1 - tophat(self.x,so.inst.J[0],so.inst.J[1],1)
		else:
			so.ao.dichroic = np.ones_like(self.x)
		"""
		# if pyramid,apply to tracking, otherwise LGS light 100J/H goes to tracking
		so.ao.dichroic = np.ones_like(self.x)
		if 'PyWFS' in so.ao.mode_chosen: 
			so.ao.pywfs_dichroic = so.ao.dichroic.copy()
		else:
			so.ao.pywfs_dichroic = np.ones_like(self.x)

	def instrument(self,so):
		###########
		# load order centers and widths 
		so.inst.order_cens, so.inst.order_widths  = ccf_tools.get_order_bounds(so.inst.order_bounds_file)

		# save dlambda
		so.inst.sig = so.stel.v/so.inst.res/so.inst.res_samp # lambda/res = dlambda, nm per pixel

		# THROUGHPUT
		try: # if config has transmission file, use it, otherwise load HISPEC version
			thput_x, thput_y = np.loadtxt(so.inst.transmission_file,delimiter=',').T
			if np.max(thput_x) < 5: thput_x*=1000
			tck_thput   = interpolate.splrep(thput_x,thput_y, k=2, s=0)
			so.inst.xtransmit   = self.x
			so.inst.ytransmit   = interpolate.splev(self.x,tck_thput,der=0,ext=1)
			so.inst.ytransmit   = np.where(so.inst.ytransmit < 0, 0, so.inst.ytransmit) # make negative throughput values to 0
			so.inst.base_throughput = so.inst.ytransmit.copy() # store this here bc ya
			#add airmass calc for strehl for seeing limited instruments?
			print('')
		except:
			so.inst.base_throughput,_  = throughput_tools.get_base_throughput(self.x,datapath=so.inst.transmission_path) # everything except coupling
			so.inst.base_throughput  = np.where(so.inst.base_throughput < 0, 0, so.inst.base_throughput) # make negative throughput values to 0

			# interp grid
			#try: so.inst.points
			#except AttributeError: 
			#	out = throughput_tools.grid_interp_coupling(int(so.inst.pl_on),path=so.inst.transmission_path + 'coupling/',atm=int(so.inst.atm),adc=int(so.inst.adc))
			#	so.inst.grid_points, so.inst.grid_values = out[0],out[1:] #if PL, three values
			#try:
			#	so.inst.coupling, so.inst.strehl = throughput_tools.pick_coupling(self.x,so.ao.ho_wfe,so.ao.tt_static,so.ao.tt_dynamic,LO=so.ao.lo_wfe,PLon=so.inst.pl_on,points=so.inst.grid_points, values=so.inst.grid_values)
			#except ValueError:
				# hack here bc tt dynamic often is out of bounds
			#	so.inst.coupling, so.inst.strehl = throughput_tools.pick_coupling(self.x,so.ao.ho_wfe,so.ao.tt_static,20,LO=so.ao.lo_wfe,PLon=so.inst.pl_on,points=so.inst.grid_points, values=so.inst.grid_values)
			#	so.inst.notes = 'tt dynamic out of bounds! %smas' %so.ao.tt_dynamic

            # load coupling (just round to nearest value instead of doing the interpolation above!)
			filename_skeleton = 'coupling/couplingEff_atm%s_adc%s_PL%s_defoc%snmRMS_LO%snmRMS_ttStatic%smas_ttDynamic%smasRMS.csv'
			tt_dynamic_rounded = np.round(2 * so.ao.tt_dynamic) / 2 # round to neared 0.5 because grid is sampled to 0.5mas
			lo_wfe_rounded = int(100*np.round(4*(so.ao.lo_wfe/100))/4) # round to nearest 25
			tt_static_rounded = np.round(so.ao.tt_static*2)/2
			if int(tt_static_rounded)==tt_static_rounded: tt_static_rounded  = int(tt_static_rounded)
			if int(tt_dynamic_rounded)==tt_dynamic_rounded: tt_dynamic_rounded  = int(tt_dynamic_rounded)
			defocus_rounded =  int(100*np.round(4*(so.ao.defocus/100))/4)
			
			# cap on tt dynamic
			if tt_dynamic_rounded < 20:
				so.inst.coupling_file = filename_skeleton%(int(so.inst.atm),int(so.inst.adc),int(so.inst.pl_on),defocus_rounded,lo_wfe_rounded,tt_static_rounded,tt_dynamic_rounded)
			else:
				so.inst.coupling_file = filename_skeleton%(int(so.inst.atm),int(so.inst.adc),int(so.inst.pl_on),defocus_rounded,lo_wfe_rounded,tt_static_rounded,19.5)

			# load and add coupling data
			f = pd.read_csv(so.inst.transmission_path+so.inst.coupling_file) # load file

			if so.inst.pl_on:
				coupling_data_raw = f['coupling_eff_mode1'] + f['coupling_eff_mode2'] + f['coupling_eff_mode3']
			else:
				coupling_data_raw = f['coupling_eff_mode1']

			# interpolate onto self.x
			finterp = interpolate.interp1d(1000*f['wavelength_um'].values,coupling_data_raw,bounds_error=False,fill_value=0)
			coupling_data = finterp(self.x)

			piaa_boost = 1.3 # based on Gary's sims, but needs updating because will be less for when Photonic lantern is being used
			so.ao.ho_strehl  = wfe_tools.calc_strehl_marechal(so.ao.ho_wfe,self.x)
			so.inst.coupling = coupling_data  * so.ao.ho_strehl * piaa_boost

			so.inst.xtransmit = self.x
			so.inst.ytransmit = so.inst.base_throughput* so.inst.coupling * so.ao.dichroic # pywfs not being considered typically so pywfs_dichroic is one here

	def observe(self,so):
		"""
		Computes the flux reaching the spectrometer sampled 
		in pixels and the noise spectrum to compute the 
		snr per pixel (so.obs.v,so.obs.snr) 
		and snr per resolution element (so.obs.v_res_element,so.obs.snr_res_element)
		"""
		# flux density is stellar flux * telescope area * instrument throughput * atmospheric absorption 
		# If planet separation is >0, compute for the planet also
		phot_per_sec_nm = so.stel.s * so.inst.tel_area * so.inst.ytransmit * np.abs(so.tel.s)
		if so.stel.pl_sep>0:
			phot_per_sec_nm_pl = so.stel.pl_s  * so.inst.tel_area * so.inst.ytransmit * np.abs(so.tel.s)
			try:
				contrast = noise_tools.get_MODHIS_contrast(so.ao.contrast_profile_path, so.ao.mode_chosen, so.tel.seeing, so.obs.zenith_angle, so.stel.mag, self.x, so.stel.pl_sep) # new version, specific to MODHIS
				print("Using new MODHIS contrast calculator with radial profile database.")
			except Exception as e:
				print(f"Error: {e}, using old contrast calculator with analytic method.")
				contrast = noise_tools.get_contrast(self.x,so.stel.pl_sep,so.inst.tel_diam,so.tel.seeing,so.ao.strehl) # old version
			
			# contrast1 = noise_tools.get_MODHIS_contrast(so.ao.contrast_profile_path, so.ao.mode_chosen, so.tel.seeing, so.obs.zenith_angle, so.stel.mag, self.x, so.stel.pl_sep) # new version, specific to MODHIS
			# contrast2 = noise_tools.get_contrast(self.x,so.stel.pl_sep,so.inst.tel_diam,so.tel.seeing,so.ao.strehl) # old version


		# Figure out the exposure time per frame to avoid saturation
		# Default case takes 900s as maximum frame exposure time length
		if so.obs.texp_frame_set=='default':
			if so.stel.pl_sep>0: # use estimated planet flux if off axis mode
				max_ph_per_s  =  np.max((phot_per_sec_nm_pl + contrast * phot_per_sec_nm) * so.inst.sig)
			else:
				max_ph_per_s  =  np.max(phot_per_sec_nm * so.inst.sig)
			if so.obs.texp < 900: 
				texp_frame_tmp = np.min((so.obs.texp,so.inst.saturation/max_ph_per_s))
			else:
				texp_frame_tmp = np.min((900,so.inst.saturation/max_ph_per_s))
			so.obs.nframes = int(np.ceil(so.obs.texp/texp_frame_tmp))
			print('Nframes set to %s'%so.obs.nframes)
			so.obs.texp_frame = np.round(so.obs.texp / so.obs.nframes,2)
			print('Texp per frame set to %s'%so.obs.texp_frame)
		# user defined exposure time per frame case:
		else:
			so.obs.texp_frame = so.obs.texp_frame_set
			so.obs.nframes = int(np.ceil(so.obs.texp/so.obs.texp_frame))
			print('Texp per frame set to %s'%so.obs.texp_frame)
			print('Nframes set to %s'%so.obs.nframes)
		
		# Degrade to instrument resolution after applying frame exposure time
		#
		so.obs.frame_phot_per_nm = phot_per_sec_nm * so.obs.texp_frame
		s_ccd_lores    = degrade_spec(so.stel.v, so.obs.frame_phot_per_nm, so.inst.res)
		
		if so.stel.pl_sep>0:
			so.obs.frame_phot_per_nm_pl = phot_per_sec_nm_pl * so.obs.texp_frame
			s_ccd_lores_pl = degrade_spec(so.stel.v, so.obs.frame_phot_per_nm_pl, so.inst.res)

		# Resample onto res element grid - new wavelength grid so.obs.v
		# 
		so.obs.v, so.obs.s_frame = resample(so.stel.v,s_ccd_lores,sig=np.mean(so.inst.sig), dx=0, eta=1,mode='fast')
		so.obs.s_frame    *=so.inst.extraction_frac # extraction fraction, reduce photons to mimic spectral extraction imperfection
		if so.stel.pl_sep>0:
			_, so.obs.s_frame_pl     = resample(so.stel.v,s_ccd_lores_pl,sig=np.mean(so.inst.sig), dx=0, eta=1,mode='fast')
			so.obs.s_frame_pl *= so.inst.extraction_frac # extraction fraction, reduce photons to mimic spectral extraction imperfection
		
			# interpolate contrast curve onto new low res array
			instrument_contrast_interp= interpolate.interp1d(so.inst.xtransmit,contrast)
			so.obs.contrast  = instrument_contrast_interp(so.obs.v)
		
			so.obs.s_frame = np.where(so.obs.s_frame < 0, 0, so.obs.s_frame)
			so.obs.speckle_frame = so.obs.contrast * so.obs.s_frame
		else:
			so.obs.speckle_frame = np.zeros_like(so.obs.s_frame)

		# Get total spectrum for all frames
		# save planet spectrum as main science spectrum
		s_star    =  so.obs.s_frame * so.obs.nframes
		if so.stel.pl_sep>0:
			so.obs.s =  so.obs.s_frame_pl * so.obs.nframes
			so.obs.s_star = s_star
		else:
			so.obs.s = s_star
		# Resample throughput for applying to sky background
		#
		base_throughput_interp = interpolate.interp1d(so.inst.xtransmit,so.inst.base_throughput)
		
		# Load background spectrum - sky is top of telescope and will be reduced by inst BASE throughput. Coupling already accounted for in solid angle of fiber. Does inst bkg need throughput applied?
		#
		so.obs.sky_bg_ph    = base_throughput_interp(so.obs.v) * noise_tools.get_sky_bg(so.obs.v,so.tel.airmass,pwv=so.tel.pwv,skypath=so.tel.skypath)
		so.obs.inst_bg_ph   = noise_tools.get_inst_bg(so.obs.v,npix=so.inst.pix_vert,R=so.inst.res,diam=so.inst.tel_diam,area=so.inst.tel_area,datapath=so.inst.transmission_path)
		
		# Calculate noise
		#
		if so.inst.pl_on: # 3 port lantern hack
			#need to figure out what to do for sky and inst bkg bc depends on coupling
			noise_frame_yJ  = np.sqrt(3) * noise_tools.sum_total_noise(so.obs.s_frame/3,so.obs.texp_frame, so.obs.nsamp,so.obs.inst_bg_ph/np.sqrt(3) , so.obs.sky_bg_ph/np.sqrt(3) , so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert,so.obs.speckle_frame) # flux split evenly over 3 traces for each of 3 PL outputs
			noise_frame     = noise_tools.sum_total_noise(so.obs.s_frame,so.obs.texp_frame, so.obs.nsamp,so.obs.inst_bg_ph, so.obs.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert,so.obs.speckle_frame)
			yJ_sub          = np.where(so.obs.v < 1400)[0]
			noise_frame[yJ_sub] = noise_frame_yJ[yJ_sub] # fill in yj with sqrt(3) times noise in PL case
		else:
			noise_frame  = noise_tools.sum_total_noise(so.obs.s_frame,so.obs.texp_frame, so.obs.nsamp,so.obs.inst_bg_ph, so.obs.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert,so.obs.speckle_frame)
		
		# Remove nans and 0s from noise frame, make these infinite
		#
		noise_frame[np.where(np.isnan(noise_frame))] = np.inf
		noise_frame[np.where(noise_frame==0)]        = np.inf
		
		# Combine noise in quadrature for all frames
		#
		so.obs.noise_frame = noise_frame
		so.obs.noise = np.sqrt(so.obs.nframes)*noise_frame

		# Compute snr and resample to get SNR per res element (assumes flux in the number of pixels spanning a res element (3 for hispec/modhis) combine in quadrature) 
		so.obs.snr = so.obs.s/so.obs.noise
		so.obs.v_res_element, so.obs.snr_res_element = resample(so.obs.v,so.obs.snr,sig=so.inst.res_samp, dx=0, eta=1/np.sqrt(so.inst.res_samp),mode='pixels')

		# compute median and max snr per order
		order_snrs_mean = []
		order_snrs_max  = []
		order_inds      = []
		for i,lam_cen in enumerate(so.inst.order_cens):
			order_ind   = np.where((so.obs.v_res_element > lam_cen - 0.9*so.inst.order_widths[i]/2) & (so.obs.v_res_element< lam_cen + 0.9*so.inst.order_widths[i]/2))[0]
			order_inds.append(order_ind)
			if np.nanmean(so.obs.snr_res_element[order_ind]) > 0.1:
				order_snrs_mean.append(np.nanmean(so.obs.snr_res_element[order_ind]))
				order_snrs_max.append(np.nanmax(so.obs.snr_res_element[order_ind]))
			else:
				order_snrs_mean.append(np.nan)
				order_snrs_max.append(np.nan)

		so.obs.snr_max_orders  = np.array(order_snrs_max)
		so.obs.snr_mean_orders = np.array(order_snrs_mean)
		so.obs.order_inds = order_inds

	def tracking(self,so):
		"""
		Gets the tracking centroid precision based 
		on the SNR and FWHM of the PSF
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
		so.track.fwhm  = float(obs_tools.get_fwhm(so.ao.ho_wfe,so.ao.tt_dynamic,so.track.center_wavelength,so.inst.tel_diam,so.track.platescale,field_r=so.track.field_r,camera=so.track.camera,getall=False,aberrations_file=so.track.aberrations_file))
		so.track.npix  = np.pi* (so.track.fwhm/2)**2 # only take noise in circle of diameter FWHM 
		so.track.fwhm_units = 'pixel'
		print('Tracking FWHM=%spix'%so.track.fwhm)
		
		so.track.strehl = wfe_tools.calc_strehl(so.ao.ho_wfe,so.track.center_wavelength)

		# get sky background and instrument background, spec is ph/nm/s
		# fwhm must be in arcsec 
		so.track.sky_bg_spec = noise_tools.get_sky_bg_tracking(self.x,so.track.fwhm*so.track.platescale,airmass=so.tel.airmass,pwv=so.tel.pwv,area=so.inst.tel_area,skypath=so.tel.skypath)
		so.track.sky_bg_ph   = np.trapz(so.track.sky_bg_spec * so.track.bandpass * so.track.ytransmit,self.x) # sky bkg needs mult by throughput and bandpass profile

		# get background spec (takes thermal emission from warm cryostat window)
		# units of ph/nm/s for spectrum and ph/s for inst_bg_ph
		so.track.inst_bg_spec, so.track.inst_bg_ph = noise_tools.get_inst_bg_tracking(self.x,so.track.pixel_pitch,so.track.npix,datapath=so.inst.transmission_path)

		# get photons in band
		so.track.signal_spec = so.stel.s * so.track.texp *\
		 			so.inst.tel_area * so.track.ytransmit*\
		 			np.abs(so.tel.s)
	
		fac = 0.5 # empirically the fraction of light approx under 2D gaussian of FWHM~4pix, which was used to get npix and matches expectation in toy centroiding model
		nphot = fac * so.track.strehl * np.trapz(so.track.signal_spec * so.track.bandpass,so.stel.v)
		#print('Tracking photons: %s e-'%so.track.nphot)

		# get noise
		so.track.noise = noise_tools.sum_total_noise(nphot,so.track.texp, 1, so.track.inst_bg_ph, so.track.sky_bg_ph,so.track.dark,so.track.rn,so.track.npix,0)
		print('Tracking noise: %s e-'%so.track.noise)
		
		# get centroid error, cap if saturated
		# peak of 2D Gaussian 4pix wide will be 1/10th of the flux in a 4pix diameter aperture (empirically derived)
		flux_in_peak = nphot/10 # nphot is already times 0.5 to give flux in a 4 pix diameter aperture
		if flux_in_peak > so.track.saturation:
			# pick an ND filter
			# compute OD of filter needed (neg neg computes ceiling)
			so.track.od = np.max((-1*round(-1 * np.log10(flux_in_peak/so.track.saturation),0),0))
			# Apply chosen nd filter
			so.track.signal = 10**(-1*so.track.od) * nphot # cap nphot
			so.track.noise  = noise_tools.sum_total_noise(so.track.signal,so.track.texp, 1, so.track.inst_bg_ph, so.track.sky_bg_ph,so.track.dark,so.track.rn,so.track.npix,0)
			# save things related to saturation
			so.track.saturation_flag = True
			so.track.nphot_nocap     = nphot
		else:
			so.track.od              = 0.0
			so.track.nphot_nocap     = nphot
			so.track.signal = nphot  # no blocking needed
			so.track.saturation_flag = False
		
		print('Tracking photons: %s e-'%so.track.signal)

		so.track.snr    = so.track.signal/so.track.noise
		# for centroid error, care about the SNR in the peak
		#signal_peak     = so.track.signal/20 # peak is 1/20th of Gaussian PSF flux assuming 4.1pix FWHM
		#noise_peak      = noise_tools.sum_total_noise(so.track.signal,so.track.texp, 1, so.track.inst_bg_ph, so.track.sky_bg_ph,so.track.dark,so.track.rn,1,0) # hack for noise for one pixel
		so.track.centroid_err = (1/np.pi) * so.track.fwhm/so.track.snr # same fwhm but snr is reduced to not saturate like if used an ND filter

	def compute_rv(self,so,telluric_cutoff=0.01,velocity_cutoff=20):
		"""
		"""
		# Create spectrum with continuum removed and tellurics removed
		# the noise spectrum will consider tellurics but shouldnt be in the spectrum for computing RV
		continuum = so.inst.ytransmit/np.max(so.inst.ytransmit)
		if so.stel.pl_sep>0:
			telcont_free_hires = so.obs.nframes * so.obs.frame_phot_per_nm_pl/continuum/np.abs(so.tel.s)			
		else:
			telcont_free_hires = so.obs.nframes * so.obs.frame_phot_per_nm/continuum/np.abs(so.tel.s)
		telcont_free_lores = degrade_spec(so.stel.v, telcont_free_hires, so.inst.res)
		v, telcont_free = resample(so.stel.v,telcont_free_lores,sig=np.mean(so.inst.sig), dx=0, eta=1,mode='fast')
		telcont_free[np.where(np.isnan(telcont_free))] = 0
		f_interp	 = interpolate.interp1d(v, telcont_free, bounds_error=False,fill_value=0)
		so.inst.s_telcont_free = f_interp(so.obs.v)

		# make telluric only spectrum, resample onto so.obs.v to match so.obs.s
		so.tel.rayleigh[so.tel.rayleigh==0] = np.inf
		telluric_spec = so.tel.s/so.tel.rayleigh #h2o only
		telluric_spec[np.where(np.isnan(telluric_spec))] = 0
		telluric_spec_lores = degrade_spec(so.stel.v, telluric_spec, so.inst.res)
		v, telluric_spec_lores_resamp = resample(so.stel.v,telluric_spec_lores,sig=np.mean(so.inst.sig), dx=0, eta=1,mode='fast')
		tel_interp	 = interpolate.interp1d(v, telluric_spec_lores_resamp, bounds_error=False,fill_value=0)
		s_tel		 = tel_interp(so.obs.v)/np.max(tel_interp(so.obs.v))	
		
		# run radial velocity precision
		so.obs.telluric_mask      = ccf_tools.make_telluric_mask(so.obs.v,s_tel,cutoff=telluric_cutoff,velocity_cutoff=velocity_cutoff)
		dv_tot,dv_spec,dv_vals	  = ccf_tools.get_rv_precision(so.obs.v,so.inst.s_telcont_free,so.obs.noise,so.inst.order_cens,so.inst.order_widths,noise_floor=so.inst.rv_floor,mask=so.obs.telluric_mask)

		so.obs.rv_order = dv_tot # per order rv with noise floor
		so.obs.rv_tot   = np.sqrt(dv_spec**2 + so.inst.rv_floor**2) # add noise floor

	def compute_etc(self,so,target_snr):
		# exposure time calculator
		snr_frame = np.sqrt(so.inst.res_samp) * so.obs.s_frame/so.obs.noise_frame # per resolution element
		# make 0s nans so doesnt blow up
		inan = np.where(snr_frame < 1)[0]
		snr_frame[inan] = np.nan

		so.obs.etc   = so.obs.texp_frame * (target_snr/snr_frame)**2  # texp per frame times nframes - per snr element
		so.obs.etc_order_max  = so.obs.texp_frame * (target_snr/(so.obs.snr_max_orders/so.obs.nframes))**2  # per order max 
		so.obs.etc_order_mean = so.obs.texp_frame * (target_snr/(so.obs.snr_mean_orders/so.obs.nframes))**2   # per 


	def compute_ccf_snr(self, so, model=None,systematics_residuals=0.01,kernel_size=201,norm_cutoff=0.8):
		'''
		Calculate the Cross-correlation function signal to noise ration with a matched filter

		Inputs:
		so          - so object, uses  signal, total_noise, sky_trans
		model       - Your model spectrum, default None and divides signal by telluric spec
		sky_trans   - The sky transmission
		systematics_residuals - A multiplicative factor that estimates the residual level of the host star spectrum and telluric lines in your signal (Default of 1%)
		kernel_size  - The default high-pass filter size.
		norm_cutoff  - A cutoff below which we don't calculate the ccf-snr

		references:
		-----------
		https://github.com/planetarysystemsimager/psisim/blob/kpic/psisim/signal.py
		https://arxiv.org/pdf/1909.07571.pdf
		https://arxiv.org/pdf/2305.19355.pdf
		'''
		# pull out per pixel signal and noise from so.obs
		signal = so.obs.s.copy()
		noise  = so.obs.noise.copy()

		#make telluirc spec sampled to obs.s
		so.tel.rayleigh[so.tel.rayleigh==0] = np.inf
		telluric_spec = so.tel.s/so.tel.rayleigh #h2o only
		telluric_spec[np.where(np.isnan(telluric_spec))] = 0
		telluric_spec_lores = degrade_spec(so.stel.v, telluric_spec, so.inst.res)
		filt_interp	 = interpolate.interp1d(so.stel.v, telluric_spec_lores, bounds_error=False,fill_value=0)
		sky_trans    = filt_interp(so.obs.v)/np.max(filt_interp(so.obs.v))	# filter profile resampled to phoenix times phoenix flux density

		#Get the noise variance
		total_noise_var = noise**2 
		bad_noise = np.isnan(total_noise_var)
		total_noise_var[bad_noise]=np.inf

		#Calculate some normalization factor
		norm = ((1-systematics_residuals)*sky_trans)

		#Get a median-filtered version of your model spectrum
		# smaller kernel size speeds up calculation, seems a little conservative (lower ccf snr out) bc doesnt smooth as well maybe
		if np.any(model==None): model = signal/sky_trans # default to this bc at R~100k this is good enough and adds simplicity
		model_medfilt = medfilt(model,kernel_size=kernel_size) # finds continuum of spectrum
		#Subtract the median version from the original model, effectively high-pass filtering the model
		model_filt = model - model_medfilt#*model.unit # leaves just high freq variations
		model_filt[np.isnan(model_filt)] = 0. # set nans to 0
		model_filt[norm<norm_cutoff] = 0.     # set deep tellurics to 0
		model_filt[bad_noise] = 0.            # set where noise is nan to 0

		#Divide out the sky transmision
		normed_signal = signal/norm
		#High-pass filter like with the model
		#signal_medfilt = medfilt(normed_signal,kernel_size=kernel_size)
		signal_filt = normed_signal - model_medfilt/np.max(norm)# subtract off model_medfilt instead to speed things up, gets very close
		signal_filt[np.isnan(signal_filt)] = 0.
		signal_filt[norm<norm_cutoff] = 0.
		signal_filt[bad_noise] = 0.

		#Now the actual ccf_snr
		so.obs.ccf_snr = np.sqrt((np.sum(signal_filt * model_filt/total_noise_var))**2 / np.sum(model_filt * model_filt/total_noise_var))
		# basically same thing for future me confused by not simplifying:
		#so.obs.ccf_snr = np.sqrt((np.sum((model_filt*model_filt/total_noise_var))))
		#so.obs.ccf_snr = np.sqrt((np.sum((signal_filt**2/total_noise_var))))
		# by band ccf snr
		sub_y = np.where(so.obs.v < 1100)[0]
		sub_J = np.where((so.obs.v > 1100) & (so.obs.v < 1327))[0]
		sub_H = np.where((so.obs.v > 1490) & (so.obs.v < 1780))[0]
		sub_K = np.where((so.obs.v > 1990) & (so.obs.v < 2460))[0]
		ccf_snr_y = np.sqrt((np.sum(signal_filt[sub_y] * model_filt[sub_y]/total_noise_var[sub_y]))**2 / np.sum(model_filt[sub_y] * model_filt[sub_y]/total_noise_var[sub_y]))
		ccf_snr_J = np.sqrt((np.sum(signal_filt[sub_J] * model_filt[sub_J]/total_noise_var[sub_J]))**2 / np.sum(model_filt[sub_J] * model_filt[sub_J]/total_noise_var[sub_J]))
		ccf_snr_H = np.sqrt((np.sum(signal_filt[sub_H] * model_filt[sub_H]/total_noise_var[sub_H]))**2 / np.sum(model_filt[sub_H] * model_filt[sub_H]/total_noise_var[sub_H]))
		ccf_snr_K = np.sqrt((np.sum(signal_filt[sub_K] * model_filt[sub_K]/total_noise_var[sub_K]))**2 / np.sum(model_filt[sub_K] * model_filt[sub_K]/total_noise_var[sub_K]))
		so.obs.ccf_snr_y= ccf_snr_y
		so.obs.ccf_snr_J= ccf_snr_J
		so.obs.ccf_snr_H= ccf_snr_H
		so.obs.ccf_snr_K= ccf_snr_K					

	def compute_ccf_snr_etc(self, so, goal_ccf, model=None,systematics_residuals=0.01,kernel_size=201,norm_cutoff=0.8):
		'''
		{    Calculate the time required to achieve a desired CCF SNR with a matched filter

		Inputs:
		so
		goal_ccf    - CCF SNR for which exposure time will be computed
		systematics_residuals - A multiplicative factor that estimates the residual level of the host star spectrum and telluric lines in your signal (Default of 1%)
		kernel_size  - The default high-pass filter size.
		norm_cutoff  - A cutoff below which we don't calculate the ccf-snr
		'''
		# TODO: This function does not account for systematics at the moment
		# To account for read_noise, we need to change how the number of frames is done in PSISIM
		# For systematics, we need to find a nice way to invert the CCF SNR equation when systematics are present
		#warnings.warn('ccf snr etc function is incomplete at the moment. Double check all results for accuracy.')
		
		#make telluirc spec sampled to obs.s
		so.tel.rayleigh[so.tel.rayleigh==0] = np.inf
		telluric_spec = so.tel.s/so.tel.rayleigh #h2o only
		telluric_spec[np.where(np.isnan(telluric_spec))] = 0
		telluric_spec_lores = degrade_spec(so.stel.v, telluric_spec, so.inst.res)
		filt_interp	 = interpolate.interp1d(so.stel.v, telluric_spec_lores, bounds_error=False,fill_value=0)
		sky_trans		 = filt_interp(so.obs.v)/np.max(filt_interp(so.obs.v))	# filter profile resampled to phoenix times phoenix flux density

		# Remove time to get flux
		signal = so.obs.s_frame
		model  = signal / sky_trans
		noise  = so.obs.noise_frame

		#Get the noise variance
		total_noise_flux = noise**2
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
		#signal_medfilt = medfilt(normed_signal,kernel_size=kernel_size)
		signal_filt = normed_signal-model_medfilt/np.max(norm)#signal_medfilt
		signal_filt[np.isnan(signal_filt)] = 0.
		signal_filt[norm<norm_cutoff] = 0.
		signal_filt[bad_noise] = 0.

		#Now the actual ccf_snr
		so.obs.ccf_snr_etc = so.obs.texp_frame *  goal_ccf**2 / ((np.sum(signal_filt * model_filt/total_noise_flux))**2 / np.sum(model_filt * model_filt/total_noise_flux))

		
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

	def set_teff_mag(self,so,temp,mag,staronly=False,trackonly=False):
		"""
		given new temperature, relaod things as needed
		mode: 'track' or 'spec'
		"""
		so.stel.teff  = temp
		so.stel.mag   = mag
		self.stellar(so)
		if not staronly:
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
		given new magnitude, relaod things as needed
		"""
		print('-----Reloading Stellar Magnitude-----')
		so.stel.mag = mag
		self.filter(so)
		self.stellar(so)
		self.ao(so)
		if not trackonly:
			self.instrument(so)
		if self.track_on:
			self.tracking(so)
		self.observe(so)

	def set_tracking_band_texp(self,so,band,texp):
		"""
		given new tracking band, relaod things as needed
		"""
		print('-----Reloading Tracking Band and Exposure Time------')
		so.track.band = band
		so.track.texp = texp
		self.tracking(so)

	def set_ao_mode(self,so,mode,trackonly=False):
		"""
		given new ao mode, reload things as needed
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


