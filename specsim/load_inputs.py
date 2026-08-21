##############################################################
# All computations happen here in fill_data class
###############################################################

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from astropy.io import fits
from scipy import interpolate
import glob
import pandas as pd


from specsim import throughput_tools, obs_tools, noise_tools, wfe_tools, ccf_tools, source_tools

from specsim.functions import *

__all__ = ['fill_data','load_phoenix']


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
		"""
		Top level driver: runs the full pipeline of methods below, in order,
		to fill the storage object with all derived spectra, throughput,
		and noise quantities needed for the SNR/ETC/RV calculations. This is
		normally the only entry point users call directly (e.g.
		`fill_data(so)`); the individual methods below are called from here
		and can also be re-run individually afterwards (see the `set_*`
		methods) to update only the parts of `so` that changed.

		Sets up a shared high-resolution wavelength grid (self.x, sampled at
		0.0005nm from so.inst.l0 to so.inst.l1) and band definitions
		(self.bands) used by all the methods below, stores yJHK band edges
		on so.inst.y/J/H/K, then calls, in order: filter, stellar, telluric,
		ao, instrument, observe. Depending on so.run.mode it then also calls
		compute_rv and compute_ccf_snr ('snr_on'/'snr_off'), compute_etc
		('etc_off'/'etc_on'), and compute_ccf_snr_etc ('etc_off'). If
		track_on is True, tracking is also computed. The call order matters:
		each method depends on so attributes filled in by the ones before it.

		inputs
		------
		so - storage object with user defined things loaded
		track_on - bool
			if True, also compute tracking camera quantities (calls self.tracking)

		output
		------
		self - fill_data instance; self.x (shared wavelength grid, nm) and
			self.bands (dict of yJHK band edges) are stored on the instance,
			and so is mutated in place by each of the called methods (see
			their individual docstrings for what each one sets)
		"""
		print("------FILLING OBJECT--------")
		# define x array to carry everywhere
		self.x = np.arange(so.inst.l0,so.inst.l1,0.0005)
		self.bands = {}
		self.bands['y'] = [980,1100]
		self.bands['J'] = [1170,1327]
		self.bands['H'] = [1490,1780]
		self.bands['K'] = [1990,2460]

		# define bands here
		# this should become deprecated - plot_tools.py uses it for now
		so.inst.y=self.bands['y'].copy()
		so.inst.J=self.bands['J'].copy()
		so.inst.H=self.bands['H'].copy()
		so.inst.K=self.bands['K'].copy()

		# order of these matter
		self.filter(so)
		self.stellar(so)
		self.telluric(so)
		self.ao(so)
		self.instrument(so)
		self.observe(so)
		if so.run.mode=='snr_on' or so.run.mode=='snr_off':
			self.compute_rv(so)
			self.compute_ccf_snr(so)
		if so.run.mode=='etc_off' or so.run.mode=='etc_on':
			self.compute_etc(so,so.obs.target_snr)
		if so.run.mode=='etc_off':
			self.compute_ccf_snr_etc(so,so.obs.target_ccf_snr)

		# turn off tracking for now, not needed
		if track_on:
			self.tracking(so)

		self.track_on=track_on

	def filter(self,so):
		"""
		Load the photometric filter bandpass (so.filt.band/family) that the
		stellar and companion magnitudes are defined in, together with its
		zeropoint. This filter curve is used later (in stellar) to scale
		the loaded model spectrum so that it matches the requested
		magnitude, and so.filt.center_wavelength/dl_l are used by ao and
		instrument for strehl and coupling calculations.

		inputs
		------
		so - storage object; reads so.filt.zp_file (zeropoint table),
			so.filt.family/band (which filter to select), and
			so.filt.filter_path (directory to search for the filter curve
			file)

		output
		------
		so.filt.zp - float, zeropoint flux for the selected band/family (Jy)
		so.filt.filter_file - str, path to the filter curve file found
		so.filt.xraw, so.filt.yraw - raw filter wavelength [nm] and
			transmission [0,1] as loaded from file
		so.filt.v, so.filt.s - filter transmission interpolated onto self.x
		so.filt.dl_l - float, mean dlambda/lambda of the filter bandpass
		so.filt.center_wavelength - float, transmission-weighted center
			wavelength of the filter [nm]
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
		Loads the on-axis stellar spectrum (PHOENIX model for
		so.stel.teff>=2300K, Sonora model otherwise), rotationally
		broadened by so.stel.vsini and scaled to match the magnitude
		so.stel.mag in the band loaded by filter() (so.filt.band/family).
		If a companion is present (so.stel.pl_sep>0, i.e. off-axis), also
		loads and scales a second spectrum for the companion using its own
		teff/mag/vsini. All wavelengths are in nm and both spectra are
		sampled onto the shared grid self.x.

		inputs
		------
		so - storage object; reads so.stel.teff [K], so.stel.mag [mag, in
			so.filt band], so.stel.vsini [km/s], so.stel.rv [km/s],
			so.stel.pl_sep [mas] (if >0, also loads companion using
			so.stel.pl_teff/pl_mag/pl_vsini), so.stel.phoenix_folder/
			sonora_folder, and so.filt.band (for the print statement)

		output
		------
		so.stel.s - array, stellar spectrum resampled onto self.x, scaled
			to match so.stel.mag [photons/s/m2/nm]
		so.stel.vraw, so.stel.sraw - raw (un-resampled) model wavelength/
			spectrum as loaded from the PHOENIX/Sonora file
		so.stel.model - str, which model family was used ('phoenix' or
			'sonora')
		so.stel.stel_file - str, path to the model file loaded
		so.stel.factor_0 - float, scale factor applied to sraw to match
			so.stel.mag (reused by ao() to rescale for a different AO star
			magnitude without reloading a model)
		so.stel.v - array, wavelength grid (= self.x) [nm]
		so.stel.units - str, units of so.stel.sraw ('photons/s/m2/nm')
		so.stel.pl_s, so.stel.pl_model, so.stel.pl_stel_file,
			so.stel.pl_factor_0 - companion equivalents of the above,
			only set if so.stel.pl_sep>0
		"""
		# Part 1: load raw spectrum
		#
		print('Teff set to %s'%so.stel.teff)
		print('%s band mag set to %s'%(so.filt.band,so.stel.mag))
	
		# load on axis target
		so.stel.s, so.stel.vraw,so.stel.sraw,so.stel.model, so.stel.stel_file, so.stel.factor_0 = source_tools.load_stellar_model(self.x,so.stel.mag,so.stel.teff,so.stel.vsini,so,rv=so.stel.rv)
		# load companion if there is one (requires separation>0)
		if so.stel.pl_sep>0:
			so.stel.pl_s, _,_,so.stel.pl_model, so.stel.pl_stel_file, so.stel.pl_factor_0 = source_tools.load_stellar_model(self.x,so.stel.pl_mag,so.stel.pl_teff,so.stel.pl_vsini,so,rv=so.stel.rv)

		so.stel.v   = self.x
		so.stel.units = 'photons/s/m2/nm' # stellar spec is in photons/s/m2/nm

	def telluric(self,so):
		"""
		Loads a TAPAS telluric transmission model (so.tel.telluric_file)
		and scales each molecular/scattering component (H2O, Rayleigh, O3,
		O2, N2, CO, CH4, CO2, N2O) from the file's reference PWV/airmass to
		the requested so.tel.pwv and the airmass implied by
		so.obs.zenith_angle, then multiplies them together to build the
		total telluric transmission spectrum used throughout the pipeline
		to attenuate the stellar spectrum and shape the noise/RV
		calculations. Also maps so.tel.seeing_set ('good'/'average'/'bad')
		to a numeric seeing value.

		inputs
		------
		so - storage object; reads so.tel.telluric_file (TAPAS fits file,
			with PWV/AIRMASS header keywords giving the file's reference
			conditions), so.tel.pwv [mm], so.obs.zenith_angle [deg], and
			so.tel.seeing_set ('good'/'average'/'bad')

		output
		------
		so.tel.airmass - float, airmass computed from so.obs.zenith_angle
		so.tel.v - array, wavelength grid (= self.x) [nm]
		so.tel.h2o, so.tel.rayleigh, so.tel.o3, so.tel.o2, so.tel.n2,
			so.tel.co, so.tel.ch4, so.tel.co2, so.tel.n2o - per-species
			transmission spectra [0,1], each scaled to so.tel.pwv/airmass
			and resampled onto self.x
		so.tel.s - array, total telluric transmission spectrum [0,1]
			(product of all species above)
		so.tel.seeing - float, numeric seeing [arcsec] corresponding to
			so.tel.seeing_set
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
		Determines the AO correction quality (high-order and tip-tilt
		wavefront error, and resulting Strehl) for the on-axis star. If
		so.ao.tt_dynamic/ho_wfe are already set by the user (both floats or
		both file paths), those user-defined values are used directly and
		so.ao.mode_chosen is set to 'User Defined'. Otherwise, WFE lookup
		tables (so.ao.ho_wfe_file/tt_dynamic_file) are loaded for every
		available AO mode as a function of guide-star magnitude, seeing,
		and zenith angle; the guide-star magnitude used to sample each mode
		is computed either from the on-axis star's spectrum (so.ao.teff/mag
		== 'default') or from a freshly loaded model at so.ao.teff/so.ao.mag
		otherwise. The Strehl for each candidate mode is computed from its
		HO WFE (Marechal approximation) and TT WFE, and either the mode
		with the highest Strehl is picked (so.ao.mode=='auto') or the
		user-requested mode is used. Also builds the AO dichroic
		transmission array (so.ao.dichroic) applied later to both the
		science and tracking light paths.

		inputs
		------
		so - storage object; reads so.ao.mode ('auto' or a specific mode
			name), so.ao.user_defined (bool - if True, so.ao.tt_dynamic/
			ho_wfe are used directly as overrides instead of being looked
			up from so.ao.mode), so.ao.tt_static [mas], so.ao.lo_wfe [nm],
			so.ao.defocus [nm], so.ao.teff [K] / so.ao.mag [mag] (or
			'default' to reuse the on-axis star), so.ao.ho_wfe_file/
			tt_dynamic_file, and so.obs.zenith_angle [deg], so.tel.seeing_set;
			also uses so.stel.vraw/sraw/model/factor_0/stel_file from
			stellar() and so.filt.center_wavelength from filter()

		output
		------
		so.ao.mode_chosen - str, AO mode selected ('User Defined' or one of
			so.ao.ao_modes)
		so.ao.ho_wfe - float, chosen high-order wavefront error [nm]
		so.ao.tt_dynamic - float, chosen dynamic tip-tilt error [mas]
		so.ao.ao_mag - float, guide-star magnitude in the AO mode's native band
		so.ao.strehl - float, total Strehl (HO x TT) for the chosen mode
		so.ao.strehl_array - array, Strehl computed for every candidate mode
		so.ao.band - str, photometric band the chosen AO mode is defined in
		so.ao.ao_modes - array, list of AO mode names loaded from file
		so.ao.dichroic - array, wavelength-dependent dichroic transmission
			[0,1] applied to both the science and tracking paths
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
			_, vraw, sraw, model, stel_file, factor_0 = source_tools.load_stellar_model(self.x,so.ao.mag,so.ao.teff,0,so)

		# now make getband mag take new stel file and factor 0

		if so.ao.user_defined:
			# set tt dynamic and ho wfe
			# requires either both to be text file or both to be floats
			if so.ao.ho_wfe is None or so.ao.tt_dynamic is None: raise ValueError('so.ao.user_defined=True requires so.ao.ho_wfe and so.ao.tt_dynamic to both be set')
			if type(so.ao.ho_wfe) != type(so.ao.tt_dynamic): raise ValueError('HO WFE and TT Dynamic must *both* be set to float values or both to file paths to WFE files')
			so.ao.mode_chosen = 'User Defined'
			so.ao.band = 'N/A'
		else:
			# load the files
			# so.obs.zenith_angle = (180/np.pi) * np.arccos(1/so.tel.airmass) # if decide to take seeing
			data = wfe_tools.load_WFE(so.ao.ho_wfe_file, so.ao.tt_dynamic_file, so.obs.zenith_angle, so.tel.seeing_set)
			ao_modes   = np.array(list(data.keys()))
			strehl, ho_wfes, tt_wfes, aomags = [], [], [],[]
			for ao_mode in ao_modes:
				# get magnitude in band the AO mode is defined in 
				wfe_mag  = source_tools.get_band_mag(so, vraw, sraw, model,stel_file,'Johnson',data[ao_mode]['band'],factor_0)
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
					raise ValueError('AO mode chosen not a mode! Modes: auto or %s. You picked: %s'%(ao_modes, so.ao.mode))

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
		# dichroic gets applied to both science and tracking
		if '100H' in so.ao.mode_chosen:
			so.ao.dichroic = 1 - tophat(self.x,so.inst.H[0],so.inst.H[1],1)
		elif '100J' in so.ao.mode_chosen:
			so.ao.dichroic = 1 - tophat(self.x,so.inst.J[0],so.inst.J[1],1)
		else:
			so.ao.dichroic = np.ones_like(self.x)

	def instrument(self,so):
		"""
		Loads the spectrograph echelle order geometry, the per-pixel
		wavelength sampling, and the total instrument throughput curve
		(base optical/detector throughput times fiber coupling efficiency
		times the AO dichroic from ao()). If so.inst.transmission_file is
		set, a user-supplied total throughput curve is loaded directly.
		Otherwise the base throughput is loaded from
		so.inst.transmission_path and combined with a fiber-coupling
		efficiency file selected by rounding so.ao.ho_wfe/tt_static/
		tt_dynamic/defocus and so.inst.pl_on (photonic lantern) onto the
		nearest precomputed grid point, scaled by the AO Strehl
		(so.ao.ho_strehl, computed here) and an empirical PIAA boost
		factor. Depends on ao() having already been run.

		inputs
		------
		so - storage object; reads so.inst.order_bounds_file,
			so.stel.v/so.inst.res/so.inst.res_samp (for pixel sampling),
			so.inst.transmission_file (optional user override) or
			so.inst.transmission_path/atm/adc/pl_on (to build the
			base throughput + coupling), and so.ao.ho_wfe/tt_static/
			tt_dynamic/defocus/dichroic from ao()

		output
		------
		so.inst.order_cens, so.inst.order_widths - arrays, center
			wavelength and width [nm] of each echelle order
		so.inst.sig - array, resolution element size (dlambda) [nm]
		so.inst.base_throughput - array, throughput excluding fiber
			coupling (or the full user-supplied throughput, if a custom
			transmission_file was used)
		so.inst.coupling - array, fiber coupling efficiency [0,1]
			(only set when not using a custom transmission_file)
		so.ao.ho_strehl - float, high-order Strehl used to scale coupling
			(only set when not using a custom transmission_file)
		so.inst.xtransmit, so.inst.ytransmit - arrays, wavelength grid and
			total instrument throughput [0,1] (base * coupling * dichroic)
		"""
		###########
		# load order centers and widths
		so.inst.order_cens, so.inst.order_widths  = ccf_tools.get_order_bounds(so.inst.order_bounds_file)

		# save dlambda
		so.inst.sig = so.stel.v/so.inst.res/so.inst.res_samp # lambda/res = dlambda, nm per pixel

		# THROUGHPUT
		try: # if config has transmission file, use it, otherwise load HISPEC version
			thput_x, thput_y = np.loadtxt(so.inst.transmission_file,delimiter=',').T
			if np.max(thput_x) < 5: thput_x*=1000 # convert to nanometers
			tck_thput   = interpolate.splrep(thput_x,thput_y, k=1, s=0)
			so.inst.xtransmit   = self.x
			so.inst.ytransmit   = interpolate.splev(self.x,tck_thput,der=0,ext=1)
			so.inst.ytransmit   = np.where(so.inst.ytransmit < 0, 0, so.inst.ytransmit) # make negative throughput values to 0
			so.inst.base_throughput = so.inst.ytransmit.copy() # store this here bc ya
			#add airmass calc for strehl for seeing limited instruments?
			print('Loaded Custom Transmission File')
		except:
			so.inst.base_throughput,_  = throughput_tools.get_base_throughput(self.x,datapath=so.inst.transmission_path) # everything except coupling
			so.inst.base_throughput    = np.where(so.inst.base_throughput < 0, 0, so.inst.base_throughput) # make negative throughput values to 0

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
			so.inst.ytransmit = so.inst.base_throughput* so.inst.coupling * so.ao.dichroic # pywfs not being considered typically so dichroic is one here

	def observe(self,so):
		"""
		Simulates the actual observation: computes the flux reaching the
		spectrometer (stellar spectrum x telescope area x instrument
		throughput x telluric transmission), picks the per-frame exposure
		time to avoid saturation (or uses a user-set value), degrades and
		resamples the spectrum onto the instrument's resolution/pixel
		grid, adds sky and instrument thermal background, and computes the
		total photon and read/dark noise per frame and across all frames.
		From that it derives the SNR spectrum per pixel (so.obs.v,
		so.obs.snr) and per resolution element (so.obs.v_res_element,
		so.obs.snr_res_element), plus max/mean SNR per echelle order.

		If so.stel.pl_sep>0 (off-axis companion), it additionally computes
		the companion flux and the stellar speckle contribution at the
		companion's separation (via so.ao.contrast_profile_path/MODHIS
		contrast calculator, falling back to an analytic contrast model),
		and so.obs.s/so.obs.snr then refer to the companion signal with the
		star's speckle halo as an added noise/background term. Depends on
		filter(), stellar(), telluric(), ao(), and instrument() having
		already been run.

		inputs
		------
		so - storage object; reads so.stel.s/v (and so.stel.pl_s if
			pl_sep>0), so.inst.tel_area [m2], so.inst.ytransmit,
			so.tel.s, so.obs.texp [s] / texp_frame_set [s or 'default'],
			so.inst.saturation [e-], so.inst.res, so.inst.sig [nm],
			so.obs.nsamp, so.inst.darknoise [e-/pix/s], so.inst.readnoise
			[e-], so.inst.pix_vert, so.inst.pl_on (photonic lantern),
			so.inst.extraction_frac, and (if pl_sep>0) so.stel.pl_sep [mas],
			so.inst.tel_diam [m], so.tel.seeing, so.obs.zenith_angle [deg],
			so.stel.mag

		output
		------
		so.obs.texp_frame [s], so.obs.nframes - chosen per-frame exposure
			time and number of frames to reach so.obs.texp
		so.obs.frame_phot_per_nm - array, stellar photon flux per frame
			[photons/s/nm-ish, pre-resample]
		so.obs.v, so.obs.s_frame_star, so.obs.s_frame - wavelength grid and
			per-frame spectrum resampled onto the instrument grid (s_frame
			is the companion spectrum if pl_sep>0, else equal to
			s_frame_star)
		so.obs.speckle_frame - array, per-frame stellar speckle halo flux
			at the companion location (zeros if on-axis)
		so.obs.s - array, total spectrum summed over all frames (main
			science spectrum used downstream)
		so.obs.sky_bg_ph, so.obs.inst_bg_ph - arrays, sky and instrument
			thermal background photons per frame
		so.obs.noise_frame, so.obs.noise - arrays, per-frame and
			all-frames-combined noise spectrum
		so.obs.snr - array, SNR per pixel (so.obs.s/so.obs.noise)
		so.obs.v_res_element, so.obs.snr_res_element - array, wavelength
			grid and SNR per resolution element
		so.obs.snr_max_orders, so.obs.snr_mean_orders, so.obs.order_inds,
			so.obs.order_cens - per-order SNR summary and pixel indices
		so.obs.ind_filter - array, indices of so.obs.v that fall within the
			yJ/HK detector passbands
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
				print(f"Warning: {e}, using old contrast calculator with analytic method.")
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
			# set text frame
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
			if so.obs.texp < so.obs.texp_frame_set:
				print('Exposure time is less than the set exposure time per frame, will set frame time to the total exposure time')
			so.obs.texp_frame = np.min((so.obs.texp_frame_set, so.obs.texp))
			so.obs.nframes = int(np.ceil(so.obs.texp/so.obs.texp_frame))
			print('Texp per frame set to user defined value %s'%so.obs.texp_frame)
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
		so.obs.v, so.obs.s_frame_star =  resample(so.stel.v,s_ccd_lores,sig=np.mean(so.inst.sig), dx=0, eta=1,mode='fast')
		so.obs.s_frame_star *= so.inst.extraction_frac
		# remove negatives from star spectrum
		so.obs.s_frame_star = np.where(so.obs.s_frame_star < 0, 0, so.obs.s_frame_star)
		if so.stel.pl_sep>0:
			_, so.obs.s_frame  = resample(so.stel.v,s_ccd_lores_pl,sig=np.mean(so.inst.sig), dx=0, eta=1,mode='fast')
			so.obs.s_frame *= so.inst.extraction_frac # extraction fraction, reduce photons to mimic spectral extraction imperfection
		
			# interpolate contrast curve onto new low res array
			instrument_contrast_interp= interpolate.interp1d(so.inst.xtransmit,contrast)
			so.obs.contrast  = instrument_contrast_interp(so.obs.v)
			# speckle is the star flux times contrast
			so.obs.speckle_frame = so.obs.contrast * so.obs.s_frame_star
		else: # sframe is the star when on axis,  speckle is zeros
			so.obs.s_frame = so.obs.s_frame_star
			so.obs.speckle_frame = np.zeros_like(so.obs.s_frame)

		# Get total spectrum for all frames
		# save planet spectrum as main science spectrum
		so.obs.s =  so.obs.s_frame * so.obs.nframes

		# Resample throughput for applying to sky background
		#
		base_throughput_interp = interpolate.interp1d(so.inst.xtransmit,so.inst.base_throughput)
		so.obs.ytransmit = base_throughput_interp(so.obs.v) # save throughput sampled to final spectrum
		
		# Load background spectrum - sky is top of telescope and will be reduced by inst BASE throughput. Coupling already accounted for in solid angle of fiber. Does inst bkg needs partial throughput applied - ignored for now to be conservative
		#
		so.obs.sky_bg_ph    = so.obs.ytransmit * noise_tools.get_sky_bg(so.obs.v,so.tel.airmass,pwv=so.tel.pwv,skypath=so.tel.sky_path)
		so.obs.inst_bg_ph   = noise_tools.get_inst_bg(so.obs.v,npix=so.inst.pix_vert,R=so.inst.res,diam=so.inst.tel_diam,area=so.inst.tel_area,datapath=so.inst.transmission_path)
		
		# Calculate noise
		#
		if so.inst.pl_on: # 3 port lantern hack
			# need to figure out what to do for sky and inst bkg bc depends on coupling
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
		#
		order_snrs_mean = []
		order_snrs_max  = []
		order_inds      = []
		for i,lam_cen in enumerate(so.inst.order_cens):
			order_ind   = np.where((so.obs.v_res_element > lam_cen - 0.9*so.inst.order_widths[i]/2) & (so.obs.v_res_element< lam_cen + 0.9*so.inst.order_widths[i]/2))[0]
			order_inds.append(order_ind)
			if np.nanmean(so.obs.snr_res_element[order_ind]) > 0.001:
				order_snrs_mean.append(np.nanmean(so.obs.snr_res_element[order_ind]))
				order_snrs_max.append(np.nanmax(so.obs.snr_res_element[order_ind]))
			else:
				order_snrs_mean.append(np.nan)
				order_snrs_max.append(np.nan)

		so.obs.snr_max_orders  = np.array(order_snrs_max)
		so.obs.snr_mean_orders = np.array(order_snrs_mean)
		so.obs.order_inds = order_inds
		so.obs.order_cens = so.inst.order_cens.copy() # nice to have this in obs too
		# define indices in passbands that actually fall on detectors (TODO should tweak these?)
		#
		ind_yj = np.where((so.obs.v>980)&(so.obs.v<1335))[0]
		ind_hk = np.where((so.obs.v>1480)&(so.obs.v<2450))[0]
		so.obs.ind_filter = np.array(ind_yj.tolist()+ind_hk.tolist())


	def tracking(self,so):
		"""
		Simulates the acquisition/tracking camera used for guiding: loads
		the tracking camera detector properties and throughput curve,
		computes the plate scale and PSF FWHM (from so.ao.ho_wfe/
		tt_dynamic via wfe_tools), the sky and instrument background seen
		by the camera, and the stellar photon signal integrated over the
		tracking band and PSF core. From the resulting SNR it derives the
		centroid error used to characterize tracking/guiding precision. If
		the peak flux would saturate the tracking detector, an equivalent
		neutral-density filter (so.track.od) is applied to cap the signal.
		Depends on stellar(), telluric(), and ao() having already been run.

		inputs
		------
		so - storage object; reads so.track.camera, so.track.
			transmission_file, so.track.fratio, so.track.band,
			so.track.field_r, so.track.aberrations_file, so.track.texp [s],
			so.inst.tel_diam [m], so.ao.ho_wfe [nm] / so.ao.tt_dynamic [mas]
			/ so.ao.dichroic, so.tel.airmass / so.tel.pwv /
			so.tel.sky_path / so.tel.s, so.inst.tel_area [m2],
			so.inst.transmission_path, and so.stel.s/v

		output
		------
		so.track.pixel_pitch, so.track.dark, so.track.rn, so.track.qe_mod,
			so.track.saturation - tracking camera detector properties
		so.track.xtransmit, so.track.ytransmit - arrays, tracking camera
			throughput [0,1]
		so.track.platescale [arcsec/pixel], so.track.center_wavelength [nm],
			so.track.bandpass - plate scale, tracking band center
			wavelength, and bandpass profile (with pyWFS dichroic applied)
		so.track.fwhm [pixel], so.track.npix, so.track.strehl - PSF FWHM,
			effective aperture size, and Strehl at the tracking wavelength
		so.track.sky_bg_spec/sky_bg_ph, so.track.inst_bg_spec/inst_bg_ph -
			sky and instrument background spectra/integrated photon rates
		so.track.signal_spec, so.track.signal, so.track.nphot_nocap - star
			signal spectrum and photon counts (capped/uncapped) in the
			tracking aperture
		so.track.od - float, neutral-density filter optical density applied
			if saturated (0 otherwise), and so.track.saturation_flag - bool
		so.track.noise - float, total noise [e-] in the tracking aperture
		so.track.snr, so.track.centroid_err - SNR and resulting centroid
			error [pixel] of the tracking measurement
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
		so.track.bandpass = bandpass * so.ao.dichroic

		# get fwhm (in pixels)
		so.track.fwhm  = float(obs_tools.get_fwhm(so.ao.ho_wfe,so.ao.tt_dynamic,so.track.center_wavelength,so.inst.tel_diam,so.track.platescale,field_r=so.track.field_r,camera=so.track.camera,getall=False,aberrations_file=so.track.aberrations_file))
		so.track.npix  = np.pi* (so.track.fwhm/2)**2 # only take noise in circle of diameter FWHM 
		so.track.fwhm_units = 'pixel'
		print('Tracking FWHM=%spix'%so.track.fwhm)
		
		so.track.strehl = wfe_tools.calc_strehl(so.ao.ho_wfe,so.track.center_wavelength)

		# get sky background and instrument background, spec is ph/nm/s
		# fwhm must be in arcsec 
		so.track.sky_bg_spec = noise_tools.get_sky_bg_tracking(self.x,so.track.fwhm*so.track.platescale,airmass=so.tel.airmass,pwv=so.tel.pwv,area=so.inst.tel_area,skypath=so.tel.sky_path)
		so.track.sky_bg_ph   = np.trapz(so.track.sky_bg_spec * so.track.bandpass * so.track.ytransmit,self.x) # sky bkg needs mult by throughput and bandpass profile

		# get background spec (takes thermal emission from warm cryostat window)
		# units of ph/nm/s for spectrum and ph/s for inst_bg_ph
		so.track.inst_bg_spec, so.track.inst_bg_ph = noise_tools.get_inst_bg_tracking(self.x,so.track.pixel_pitch,so.track.npix,datapath=so.inst.transmission_path)

		# get photons in band
		so.track.signal_spec = so.stel.s * so.track.texp *\
		 			so.inst.tel_area * so.track.ytransmit*\
		 			np.abs(so.tel.s)

		# fac is empirically the fraction of light approx under 2D gaussian of FWHM~4pix, 
		# which scales npix. This was tuned based on a actual toy centroiding model fit and gets results to match
		fac = 0.5 		
		nphot = fac * so.track.strehl * np.trapz(so.track.signal_spec * so.track.bandpass,so.stel.v)

		# get noise
		so.track.noise = noise_tools.sum_total_noise(nphot,so.track.texp, 1, so.track.inst_bg_ph, so.track.sky_bg_ph,so.track.dark,so.track.rn,so.track.npix,0)
		print(f'Tracking noise: {so.track.noise} e-')
		
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
			print(f'Tracking OD needed {so.track.od}, nphot capped to {so.track.signal} e-')
		else:
			so.track.od              = 0.0
			so.track.nphot_nocap     = nphot
			so.track.signal = nphot  # no blocking needed
			so.track.saturation_flag = False
			print(f'Tracking photons: {so.track.signal} e-')

		so.track.snr    = so.track.signal/so.track.noise
		# for centroid error, care about the SNR in the peak
		#signal_peak     = so.track.signal/20 # peak is 1/20th of Gaussian PSF flux assuming 4.1pix FWHM
		#noise_peak      = noise_tools.sum_total_noise(so.track.signal,so.track.texp, 1, so.track.inst_bg_ph, so.track.sky_bg_ph,so.track.dark,so.track.rn,1,0) # hack for noise for one pixel
		so.track.centroid_err = (1/np.pi) * so.track.fwhm/so.track.snr # same fwhm but snr is reduced to not saturate like if used an ND filter
		print(f'Tracking SNR: {so.track.snr}, centroid error: {so.track.centroid_err} pix')
		
	def compute_rv(self,so,telluric_cutoff=0.01,velocity_cutoff=30):
		"""
		Computes the achievable radial velocity precision for the
		observation. Builds a "telluric/continuum-free" version of the
		observed spectrum (throughput continuum and telluric absorption
		divided out) so the RV information content reflects only the
		stellar lines, builds a mask that excludes wavelengths near deep
		telluric lines (deeper than telluric_cutoff, masked out to
		+/-velocity_cutoff in velocity space), and passes these along with
		the noise spectrum to ccf_tools.get_rv_precision to get the RV
		uncertainty per order and for the full spectrum, adding the
		instrument/telluric systematic noise floor (so.inst.rv_floor) in
		quadrature. Depends on stellar(), telluric(), instrument(), and
		observe() having already been run.

		inputs
		------
		so - storage object; reads so.inst.ytransmit, so.obs.nframes,
			so.obs.frame_phot_per_nm (or frame_phot_per_nm_pl if
			pl_sep>0), so.tel.s, so.stel.v, so.inst.res/sig, so.obs.v,
			so.tel.rayleigh/o3, so.obs.noise, so.inst.order_cens/
			order_widths, and so.inst.rv_floor [m/s]
		telluric_cutoff - float
			telluric line depth (0-1) below which wavelengths start being
			masked out of the RV calculation
		velocity_cutoff - float, km/s
			velocity window around each masked telluric line to also
			exclude

		output
		------
		so.inst.s_telcont_free - array, stellar spectrum with instrument
			throughput continuum and telluric absorption removed, resampled
			onto so.obs.v
		so.obs.telluric_mask - array, boolean/weight mask excluding regions
			near deep telluric lines
		so.obs.rv_order - array, per-order RV precision including the
			instrument/telluric noise floor [m/s]
		so.obs.rv_tot - float, total RV precision across the full spectrum,
			including the noise floor [m/s]
		"""
		# Create spectrum with continuum removed and tellurics removed
		# the noise spectrum will consider tellurics but shouldnt be in the spectrum for computing RV
		continuum = 1 + 0*so.inst.ytransmit/np.max(so.inst.ytransmit) # quick hack to no longer continuum correct. this was messing things up
		if so.stel.pl_sep>0:
			telcont_free_hires = so.obs.nframes * so.obs.frame_phot_per_nm_pl/continuum/np.abs(so.tel.s)			
		else:
			telcont_free_hires = so.obs.nframes * so.obs.frame_phot_per_nm/continuum/np.abs(so.tel.s)
		
		# remove telurics
		telcont_free_lores = degrade_spec(so.stel.v, telcont_free_hires, so.inst.res)
		v, telcont_free    = resample(so.stel.v,telcont_free_lores,sig=np.mean(so.inst.sig), dx=0, eta=1,mode='fast')
		telcont_free[np.where(np.isnan(telcont_free))] = 0
		f_interp	       = interpolate.interp1d(v, telcont_free, bounds_error=False,fill_value=0)
		so.inst.s_telcont_free = f_interp(so.obs.v)

		# make telluric only spectrum, resample onto so.obs.v to match so.obs.s
		so.tel.rayleigh[so.tel.rayleigh==0] = np.inf
		telluric_spec = so.tel.s/so.tel.rayleigh/so.tel.o3 #no continuum altering things! 
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
		"""
		Exposure time calculator (ETC): given the per-frame SNR already
		computed by observe() (so.obs.s_frame/so.obs.noise_frame, scaled
		to a resolution element), scales by (target_snr/snr_frame)^2 to
		derive the total exposure time needed to reach target_snr, per
		pixel/resolution-element (so.obs.etc) and per order using both the
		per-order max and mean SNR (so.obs.etc_order_max/mean). Depends on
		observe() having already been run.

		inputs
		------
		so - storage object; reads so.inst.res_samp, so.obs.s_frame,
			so.obs.noise_frame, so.obs.texp_frame [s], so.obs.nframes,
			so.obs.snr_max_orders, so.obs.snr_mean_orders
		target_snr - float
			desired signal to noise ratio

		output
		------
		so.obs.etc - array, exposure time [s] needed at each wavelength/
			resolution element to reach target_snr
		so.obs.etc_order_max - array, exposure time [s] per order needed to
			reach target_snr at that order's max SNR wavelength
		so.obs.etc_order_mean - array, exposure time [s] per order needed
			to reach target_snr at that order's mean SNR
		"""
		# exposure time calculator
		snr_frame = np.sqrt(so.inst.res_samp) * so.obs.s_frame/so.obs.noise_frame # per resolution element
		# make 0s nans so doesnt blow up
		inan = np.where(snr_frame ==0)[0]
		snr_frame[inan] = np.nan

		# result is in seconds
		so.obs.etc   = so.obs.texp_frame * (target_snr/snr_frame)**2  # texp per frame times nframes - per snr element
		so.obs.etc_order_max  = so.obs.texp_frame * (target_snr/(so.obs.snr_max_orders/np.sqrt(so.obs.nframes)))**2  # per order max 
		so.obs.etc_order_mean = so.obs.texp_frame * (target_snr/(so.obs.snr_mean_orders/np.sqrt(so.obs.nframes)))**2   # per 

	def compute_ccf_snr(self, so, model=None,systematics_residuals=0.01,kernel_size=201,norm_cutoff=0.95):
		'''
		Calculates the cross-correlation function (CCF) signal-to-noise
		ratio using a matched-filter formalism, i.e. the SNR that would be
		obtained by cross-correlating the observed spectrum against a
		stellar/telluric template (as used for high-resolution
		spectroscopy detections/RV work), for the full so.obs.s spectrum
		and separately for the y/J/H/K bands. A telluric transmission
		spectrum is built from the H2O/Rayleigh components, the model (or
		signal/sky_trans if no model given) is median-filtered to remove
		the continuum leaving just high-frequency spectral features, deep
		telluric regions (below norm_cutoff) are excluded, and the matched
		filter equation is evaluated using the per-pixel noise variance.
		Depends on stellar(), telluric(), instrument(), and observe()
		having already been run.

		Inputs:
		so          - storage object; uses so.obs.s, so.obs.noise, so.obs.v,
		              so.tel.s, so.tel.rayleigh, so.stel.v, so.inst.res
		model       - Your model spectrum, default None and divides signal by telluric spec
		systematics_residuals - A multiplicative factor that estimates the residual level of the host star spectrum and telluric lines in your signal (Default of 1%)
		kernel_size  - The default high-pass filter size.
		norm_cutoff  - A cutoff below which we don't calculate the ccf-snr

		Output:
		so.obs.ccf_snr - float, matched-filter CCF SNR for the full spectrum
		so.obs.ccf_snr_y, so.obs.ccf_snr_J, so.obs.ccf_snr_H, so.obs.ccf_snr_K
		            - float, CCF SNR restricted to each photometric band

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

	def compute_ccf_snr_etc(self, so, goal_ccf, model=None,systematics_residuals=0.01,kernel_size=201,norm_cutoff=0.95):
		'''
		Calculates the exposure time required to achieve a desired CCF SNR
		(goal_ccf) with a matched filter. This is essentially a copy of
		compute_ccf_snr's matched-filter calculation but run on a single
		frame's signal/noise (so.obs.s_frame, so.obs.noise_frame) instead
		of the full multi-frame spectrum; the model here is always defined
		as signal/sky_trans (no user-supplied model option, unlike
		compute_ccf_snr). Since CCF SNR scales as sqrt(exposure time), the
		per-frame CCF SNR in each band is scaled by (goal_ccf/ccf_snr)^2 x
		so.obs.texp_frame to get the needed total exposure time, computed
		separately for the y/J/H/K bands. Does not currently account for
		systematics_residuals scaling with exposure time. Depends on
		stellar(), telluric(), instrument(), and observe() having already
		been run.

		Inputs:
		--------
		so - storage object; uses so.obs.s_frame, so.obs.noise_frame,
		     so.obs.v, so.tel.s, so.tel.rayleigh, so.stel.v, so.inst.res,
		     so.obs.texp_frame [s]
		goal_ccf    - CCF SNR for which exposure time will be computed
		systematics_residuals - A multiplicative factor that estimates the residual level of the host star spectrum and telluric lines in your signal (Default of 1%)
		kernel_size  - The default high-pass filter size.
		norm_cutoff  - A cutoff below which we don't calculate the ccf-snr

		Output:
		--------
		so.obs.etc_ccf_snr_y, so.obs.etc_ccf_snr_J, so.obs.etc_ccf_snr_H,
		so.obs.etc_ccf_snr_K - float, exposure time [s] needed in each band
		to reach goal_ccf
		'''
		# TODO: This function does not account for systematics at the moment
		# To account for read_noise, we need to change how the number of frames is done in PSISIM
		# For systematics, we need to find a nice way to invert the CCF SNR equation when systematics are present
		#warnings.warn('ccf snr etc function is incomplete at the moment. Double check all results for accuracy.')
		# Remove time to get flux
		signal = so.obs.s_frame.copy()
		noise  = so.obs.noise_frame.copy()

		#make telluirc spec sampled to obs.s
		so.tel.rayleigh[so.tel.rayleigh==0] = np.inf
		telluric_spec = so.tel.s/so.tel.rayleigh #h2o only
		telluric_spec[np.where(np.isnan(telluric_spec))] = 0
		telluric_spec_lores = degrade_spec(so.stel.v, telluric_spec, so.inst.res)
		filt_interp	 = interpolate.interp1d(so.stel.v, telluric_spec_lores, bounds_error=False,fill_value=0)
		sky_trans    = filt_interp(so.obs.v)/np.max(filt_interp(so.obs.v))	# filter profile resampled to phoenix times phoenix flux density

		# define model as signal with no sky, this is not ideal
		model  = signal / sky_trans

		#Get the noise variance
		total_noise_var = noise**2
		bad_noise = np.isnan(total_noise_var)
		total_noise_var[bad_noise]=np.inf

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
		
		sub_y = np.where(so.obs.v < 1100)[0]
		sub_J = np.where((so.obs.v > 1100) & (so.obs.v < 1327))[0]
		sub_H = np.where((so.obs.v > 1490) & (so.obs.v < 1780))[0]
		sub_K = np.where((so.obs.v > 1990) & (so.obs.v < 2460))[0]
		ccf_snr_y = np.sqrt((np.sum(signal_filt[sub_y] * model_filt[sub_y]/total_noise_var[sub_y]))**2 / np.sum(model_filt[sub_y] * model_filt[sub_y]/total_noise_var[sub_y]))
		ccf_snr_J = np.sqrt((np.sum(signal_filt[sub_J] * model_filt[sub_J]/total_noise_var[sub_J]))**2 / np.sum(model_filt[sub_J] * model_filt[sub_J]/total_noise_var[sub_J]))
		ccf_snr_H = np.sqrt((np.sum(signal_filt[sub_H] * model_filt[sub_H]/total_noise_var[sub_H]))**2 / np.sum(model_filt[sub_H] * model_filt[sub_H]/total_noise_var[sub_H]))
		ccf_snr_K = np.sqrt((np.sum(signal_filt[sub_K] * model_filt[sub_K]/total_noise_var[sub_K]))**2 / np.sum(model_filt[sub_K] * model_filt[sub_K]/total_noise_var[sub_K]))

		so.obs.etc_ccf_snr_y= so.obs.texp_frame *  goal_ccf**2 /ccf_snr_y**2
		so.obs.etc_ccf_snr_J= so.obs.texp_frame *  goal_ccf**2 /ccf_snr_J**2
		so.obs.etc_ccf_snr_H= so.obs.texp_frame *  goal_ccf**2 /ccf_snr_H**2
		so.obs.etc_ccf_snr_K= so.obs.texp_frame *  goal_ccf**2 /ccf_snr_K**2					


		
	def set_teff_aomode(self,so,temp,aomode,trackonly=False):
		"""
		Convenience re-loader: updates the star's effective temperature and
		the AO mode, then re-runs only the downstream methods that depend
		on those two values (stellar spectrum, AO Strehl/WFE, instrument
		throughput, tracking, and full observation), avoiding recomputing
		filter/telluric which are unaffected. Intended for use after the
		initial fill_data(so) call, e.g. when scanning over a grid of
		Teff/AO mode combinations.

		inputs
		------
		so - storage object (already filled by fill_data.__init__)
		temp - float, K
			new stellar effective temperature, written to so.stel.teff
		aomode - str
			new AO mode, written to so.ao.mode
		trackonly - bool
			if True, skip recomputing so.inst throughput (instrument());
			tracking() and observe() are always re-run regardless

		output
		------
		so - updated in place: so.stel.* (via stellar), so.ao.* (via ao),
			so.inst.* (via instrument, unless trackonly), so.track.* (via
			tracking), and so.obs.* (via observe)
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
		Convenience re-loader: updates the star's effective temperature and
		magnitude, then re-runs only the downstream methods needed to
		propagate that change, avoiding recomputing filter/telluric which
		are unaffected. Intended for use after the initial fill_data(so)
		call, e.g. when scanning over a grid of Teff/mag combinations.

		inputs
		------
		so - storage object (already filled by fill_data.__init__)
		temp - float, K
			new stellar effective temperature, written to so.stel.teff
		mag - float, mag (in so.filt band)
			new stellar magnitude, written to so.stel.mag
		staronly - bool
			if True, only reload the stellar spectrum (stellar()) and skip
			ao/instrument/tracking/observe entirely
		trackonly - bool, ignored if staronly is True
			if True, re-run ao+instrument+tracking (not observe); if False,
			re-run ao+instrument+observe (not tracking)

		output
		------
		so - updated in place: so.stel.* (via stellar), and, unless
			staronly, so.ao.*/so.inst.* (via ao/instrument) plus either
			so.track.* (via tracking, if trackonly) or so.obs.* (via
			observe, otherwise)
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
		Convenience re-loader: updates the stellar magnitude and re-runs
		filter, stellar, ao, (optionally) instrument, (optionally)
		tracking, and observe to propagate the change through the
		pipeline. Intended for use after the initial fill_data(so) call,
		e.g. when scanning over a grid of magnitudes.

		inputs
		------
		so - storage object (already filled by fill_data.__init__)
		mag - float, mag (in so.filt band)
			new stellar magnitude, written to so.stel.mag
		trackonly - bool
			if True, skip recomputing so.inst throughput (instrument());
			tracking() is re-run only if self.track_on, and observe() is
			always re-run regardless of trackonly

		output
		------
		so - updated in place: so.filt.* (via filter), so.stel.* (via
			stellar), so.ao.* (via ao), so.inst.* (via instrument, unless
			trackonly), so.track.* (via tracking, if self.track_on), and
			so.obs.* (via observe)
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
		Convenience re-loader: updates the tracking camera's photometric
		band and exposure time and re-runs only tracking() to propagate
		the change (nothing else in the pipeline depends on these two
		values). Intended for use after the initial fill_data(so) call.

		inputs
		------
		so - storage object (already filled by fill_data.__init__)
		band - str
			new tracking camera band, written to so.track.band (e.g.
			'JHgap','z','y','J','H','K')
		texp - float, s
			new tracking camera exposure time, written to so.track.texp

		output
		------
		so - updated in place: so.track.* (via tracking)
		"""
		print('-----Reloading Tracking Band and Exposure Time------')
		so.track.band = band
		so.track.texp = texp
		self.tracking(so)

	def set_ao_mode(self,so,mode,trackonly=False):
		"""
		Convenience re-loader: updates the AO mode and re-runs the
		downstream methods that depend on it (AO Strehl/WFE, instrument
		throughput, tracking, and the full observation). Intended for use
		after the initial fill_data(so) call, e.g. when scanning over AO
		modes.

		inputs
		------
		so - storage object (already filled by fill_data.__init__)
		mode - str
			new AO mode, written to so.ao.mode ('auto' or a specific mode
			name, see ao())
		trackonly - bool
			if True, skip recomputing so.inst throughput (instrument());
			tracking() and observe() are always re-run regardless

		output
		------
		so - updated in place: so.ao.* (via ao), so.inst.* (via instrument,
			unless trackonly), so.track.* (via tracking), and so.obs.*
			(via observe)
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
		Convenience re-loader: updates the photometric filter band/family
		that the stellar magnitude is defined in (plus the magnitude
		itself) and re-runs the full downstream chain so all the derived
		quantities are consistent with the new band. Intended for use
		after the initial fill_data(so) call, e.g. when switching which
		band a target's magnitude is quoted in.

		inputs
		------
		so - storage object (already filled by fill_data.__init__)
		band - str
			new filter band, written to so.filt.band (e.g. 'y','J','H','K')
		family - str
			new filter family, written to so.filt.family (e.g. '2mass','cfht')
		mag - float, mag (in the new band)
			new stellar magnitude, written to so.stel.mag
		trackonly - bool
			if True, skip recomputing so.inst throughput and so.obs (i.e.
			skip instrument() and observe()); tracking() is always re-run
			regardless of trackonly

		output
		------
		so - updated in place: so.filt.* (via filter), so.stel.* (via
			stellar), so.ao.* (via ao), so.inst.* and so.obs.* (via
			instrument/observe, unless trackonly), and so.track.* (via
			tracking)
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


