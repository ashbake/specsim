##############################################################
# Load variables into objects object
###############################################################

import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from astropy.io import fits
from scipy import interpolate
import sys,glob,os
import pandas as pd
from astropy.convolution import Gaussian1DKernel, convolve

import throughput_tools# import pick_coupling, get_band_mag, get_base_throughput,grid_interp_coupling
from wfe_tools import get_tip_tilt_resid, get_HO_WFE
import obs_tools
import noise_tools

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
	def __init__(self, so):		
		print("------FILLING OBJECT--------")
		# define x array to carry everywhere
		self.x = np.arange(so.inst.l0,so.inst.l1,0.0005)
		# define bands here
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
		self.tracking(so)

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
		"""
		fill in ao info 
		"""
		# load ao information from ao file
		if so.ao.mag=='default': 
			factor_0 = so.stel.factor_0 # if mag is same as one loaded, dont change spectral mag
		else: 
			# scale to find factor_0 for new mag
			factor_0 = so.stel.factor_0 * 10**(0.4*so.ao.mag)

		if type(so.ao.ho_wfe_set) is str:
			f = pd.read_csv(so.ao.ho_wfe_set,header=[0,1])
			so.ao.modes = f.columns
			mags             = f['mag'].values.T[0]
			wfes             = f[so.ao.mode].values.T[0]
			so.ao.ho_wfe_band= f[so.ao.mode].columns[0] # this is the mag band wfe is defined in, must be more readable way..
			so.ao.ho_wfe_mag = get_band_mag(so,'Johnson',so.ao.ho_wfe_band,factor_0) # get magnitude of star in appropriate band
			f_howfe          = interpolate.interp1d(mags,wfes, bounds_error=False,fill_value=10000)
			so.ao.ho_wfe     = float(f_howfe(so.ao.ho_wfe_mag))
			print('HO WFE %s mag is %s'%(so.ao.ho_wfe_band,so.ao.ho_wfe_mag))
		else:
			so.ao.ho_wfe = so.ao.ho_wfe_set

		if type(so.ao.ttdynamic_set) is str:
			f = pd.read_csv(so.ao.ttdynamic_set,header=[0,1])
			so.ao.modes_tt  = f.columns # should match howfe..
			mags            = f['mag'].values.T[0]
			tts             = f[so.ao.mode].values.T[0]
			so.ao.ttdynamic_band=f[so.ao.mode].columns[0] # this is the mag band wfe is defined in, must be more readable way..			
			so.ao.ttdynamic_mag = get_band_mag(so,'Johnson',so.ao.ttdynamic_band,factor_0) # get magnitude of star in appropriate band
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
			
			# interp grid
			try: so.inst.points
			except AttributeError: 
				out = throughput_tools.grid_interp_coupling(int(so.inst.pl_on),path=so.inst.transmission_path + 'coupling/',atm=int(so.inst.atm),adc=int(so.inst.adc))
				so.inst.grid_points, so.inst.grid_values = out[0],out[1:] #if PL, three values
			try:
				so.inst.coupling, so.inst.strehl = throughput_tools.pick_coupling(self.x,so.ao.ho_wfe,so.ao.tt_static,so.ao.tt_dynamic,LO=so.ao.lo_wfe,PLon=so.inst.pl_on,points=so.inst.grid_points, values=so.inst.grid_values)
			except ValueError:
				# hack here bc tt dynamic often is out of bounds
				so.inst.coupling, so.inst.strehl = throughput_tools.pick_coupling(self.x,so.ao.ho_wfe,so.ao.tt_static,20,LO=so.ao.lo_wfe,PLon=so.inst.pl_on,points=so.inst.grid_points, values=so.inst.grid_values)
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

		# resample throughput for applying to sky background
		base_throughput_interp= interpolate.interp1d(so.inst.xtransmit,so.inst.base_throughput)

		# load background spectrum - sky is top of telescope and will be reduced by inst BASE throughput. Coupling already accounted for in solid angle of fiber. Does inst bkg need throughput applied?
		so.obs.sky_bg_ph    = base_throughput_interp(so.obs.v) * noise_tools.get_sky_bg(so.obs.v,so.tel.airmass,pwv=so.tel.pwv,skypath=so.tel.skypath)
		so.obs.inst_bg_ph   = noise_tools.get_inst_bg(so.obs.v,npix=so.inst.pix_vert,R=so.inst.res,diam=so.inst.tel_diam,area=so.inst.tel_area,datapath=so.inst.transmission_path)

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

		so.obs.snr = so.obs.s/so.obs.noise
		so.obs.v_resamp, so.obs.snr_reselement = resample(so.obs.v,so.obs.snr,sig=so.inst.res_samp, dx=0, eta=1/np.sqrt(so.inst.res_samp),mode='pixels')

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

	def set_teff_mag(self,so,temp,mag,star_only=False):
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


