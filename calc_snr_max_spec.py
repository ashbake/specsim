# calc signal to noise
# max for the calcium H&K
import sys
import matplotlib
import os

import numpy as np
import matplotlib.pylab as plt

from astropy import units as u
from astropy import constants as c 
from astropy.io import fits
from astropy.table import Table
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu

import speclite.filters
from scipy.ndimage import gaussian_filter

from scipy import signal, interpolate
import pandas as pd
from scipy.interpolate import interpn

font = {'size'   : 14}
matplotlib.rc('font', **font)

sys.path.append('./utils/')
from objects import load_object
from load_inputs import fill_data, load_filter,load_phoenix, load_sonora
from functions import *
#from kpf_etc.etc import kpf_photon_noise_estimate, kpf_etc_rv, kpf_etc_snr

plt.ion()

def hispec_sim_spectrum(so,throughput):
	"""
	scale output from kpf scaling to Ca H&K throughput 
	"""
	s_ccd_hires = so.stel.s * so.var.exp_time * so.const.tel_area * throughput * np.abs(so.tel.s)**so.tel.airmass

	# convolve to lower res
	s_ccd_lores = degrade_spec(so.stel.v, s_ccd_hires, so.const.res_hispec)

	# resample onto res element grid
	lam0=1500#nm take as lam to calc dlambda, perhaps fix this to have diff res per order..
	sig = lam0/so.const.res_hispec/so.const.res_sampling # lambda/res = dlambda
	v_resamp, s_resamp = resample(so.stel.v,s_ccd_lores,sig=sig, dx=0, eta=1,mode='fast')

	return v_resamp, s_resamp

def load_photonic_lantern():
	"""
	load PL info like unitary matrices
	"""
	wavearr = np.linspace(970,1350,20)
	data = np.load('./data/throughput/photonic_lantern/unitary_matrices.npy')
	
	return wavearr,data

def tophat(x,l0,lf,throughput):
	ion = np.where((x > l0) & (x<lf))[0]
	bandpass = np.zeros_like(x)
	bandpass[ion] = throughput
	return bandpass

def get_band_mag(so,family,band,factor_0):
	"""
	factor_0: scaling model to photons
	"""
	x,y          = load_filter(so,family,band)
	filt_interp  =  interpolate.interp1d(x, y, bounds_error=False,fill_value=0)
	dl_l         =   np.mean(integrate(x,y)/x) # dlambda/lambda to account for spectral fraction
	
	# load stellar the multiply by scaling factor, factor_0, and filter. integrate
	if so.stel.model=='phoenix':
		vraw,sraw = load_phoenix(so.stel.stel_file,wav_start=np.min(x), wav_end=np.max(x)) #phot/m2/s/nm
	elif so.stel.model=='sonora':
		vraw,sraw = load_sonora(so.stel.stel_file,wav_start=np.min(x), wav_end=np.max(x)) #phot/m2/s/nm
	
	filtered_stel = factor_0 * sraw * filt_interp(vraw)
	flux = integrate(vraw,filtered_stel)    #phot/m2/s

	phot_per_s_m2_per_Jy = 1.51*10**7 # convert to phot/s/m2 from Jansky
	
	flux_Jy = flux/phot_per_s_m2_per_Jy/dl_l
	
	# get zps
	zps                     = np.loadtxt(so.filt.zp_file,dtype=str).T
	izp                     = np.where((zps[0]==family) & (zps[1]==band))[0]
	zp                      = np.float(zps[2][izp])

	mag = -2.5*np.log10(flux_Jy/zp)

	return mag

def get_tip_tilt_resid(Vmag, mode):
	"""
	load data from haka sims, spit out tip tilt
	"""
	modes = np.array(['K','SH','80J','80H','80JK','LGS'])# corresponding modes to match assumptions of text files 
	imode = np.where(modes==mode)[0]

	#load data file
	f = np.loadtxt('./data/WFE/HAKA/Kstar_tiptilt.txt').T
	vmags = f[0]
	tip_tilts = f[1:][imode] # switch to take best mode for observing case

	#interpolate
	f_tt = interpolate.interp1d(vmags,tip_tilts, bounds_error=False,fill_value=10000)

	return f_tt(Vmag)

def get_HO_WFE(Vmag, mode):
	"""
	load data from haka sims, spit out tip tilt
	"""
	modes = np.array(['K','SH','80J','80H','80JK','LGS'])# corresponding modes to match assumptions of text files 
	imode = np.where(modes==mode)[0][0]

	#load data file
	f = np.loadtxt('./data/WFE/HAKA/Kstar_HOwfe.txt').T
	vmags = f[0]
	wfes = f[1:][imode]

	#interpolate
	f_wfe = interpolate.interp1d(vmags,wfes, bounds_error=False,fill_value=10000)

	return f_wfe(Vmag)

def get_dyn_wfe(Rmag):
	"""
	***fix this - only return wfe that is important for centroiding based
	on josh's simulations*** maybe it's all but probs should take out tip/tilt
	
	# R mag inputed!
	https://www2.keck.hawaii.edu/optics/lgsao/performance.html

	should get this soon from rich
	"""
	mag_arr = np.linspace(0,20,21) #sampling to match following arrays
	# old haka - to update
	aowfe_ngs=np.array([133,134,134,134,135,137,140,144,153,162,182,209,253,311,398,510,671,960,1446,2238,3505])
	aowfe_lgs=np.array([229,229,229,229,229,229,230,230,231,231,232,235,238,247,255,281,301,345,434,601,889])

	f_ngs = interpolate.interp1d(mag_arr,aowfe_ngs, bounds_error=False,fill_value=10000)
	f_lgs = interpolate.interp1d(mag_arr,aowfe_lgs, bounds_error=False,fill_value=10000)

	wfe = np.min([f_ngs(Rmag), f_lgs(Rmag)])

	if Rmag>20:
		print("WARNING, Rmag outside bounds of WFE data")

	return wfe

def grid_interp_coupling(PLon):
	"""
	interpolate coupling files over their various parameters
	"""
	LOs = np.arange(0,125,25)
	ttStatics = np.arange(11)
	ttDynamics = np.arange(0,10,0.5)
	
	if PLon: 
		path_to_files     = './data/throughput/hispec_subsystems_11032022/coupling/couplingEff_wPL_202212014/'
		filename_skeleton = 'couplingEff_atm0_adc0_PL%s_defoc0nmRMS_LO%snmRMS_ttStatic%smas_ttDynamic%smasRMS.csv'
	else:
		path_to_files     = './data/throughput/hispec_subsystems_11032022/coupling/couplingEff_20221005/'
		filename_skeleton = 'couplingEff_atm0_adc0_defoc0nmRMS_LO%snmRMS_ttStatic%smas_ttDynamic%smasRMS.csv'

	# to dfine values, must open up each file. not sure if can deal w/ wavelength
	values_1 = np.zeros((len(LOs),len(ttStatics),len(ttDynamics),100))
	values_2 = np.zeros((len(LOs),len(ttStatics),len(ttDynamics),100))
	values_3 = np.zeros((len(LOs),len(ttStatics),len(ttDynamics),100))	
	for i,LO in enumerate(LOs):
		for j,ttStatic in enumerate(ttStatics):
			for k,ttDynamic in enumerate(ttDynamics):
				if round(ttDynamic)==ttDynamic: ttDynamic=round(ttDynamic)
				if PLon:
					f = pd.read_csv(path_to_files+filename_skeleton%(PLon,LO,ttStatic,ttDynamic))
					values_1[i,j,k,:]=f['coupling_eff_mode1'] #what to fill here?
					values_2[i,j,k,:]=f['coupling_eff_mode2'] #what to fill here?
					values_3[i,j,k,:]=f['coupling_eff_mode3'] #what to fill here?
				else:
					f = pd.read_csv(path_to_files+filename_skeleton%(LO,ttStatic,ttDynamic))
					values_1[i,j,k,:]=f['coupling_efficiency'] #what to fill here?

				#values_hk[i,j,k]=f['coupling_eff_mode1'][50] #what to fill here?
	
	points = (LOs, ttStatics, ttDynamics,f['wavelength_um'].values)

	if PLon:
		return points,values_1,values_2,values_3
	else:
		return points,values_1

def get_base_throughput(x,ploton=False):
	"""
	get throughput except leave out coupling

	also store emissivity
	"""
	datapath = './data/throughput/hispec_subsystems_11032022/'
	#plt.figure()
	for spec in ['red','blue']:
		if spec=='red':
			include = ['tel', 'ao', 'feicom', 'feired','fibred','rspec']#,'coupling']
		if spec=='blue':
			include = ['tel', 'ao', 'feicom', 'feiblue','fibblue','bspec']#,'coupling']

		for i in include:
			if i==include[0]:
				w,s = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
				#plt.plot(w,s,label=i)
			else:
				wtemp, stemp = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
				# interpolate onto s
				f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
				s*=f(w)
				#plt.plot(w,s,label=i)

		if spec=='red':
			isub = np.where(w > 1.4) 
			wred = w[isub]
			specred = s[isub]
		if spec=='blue':
			isub = np.where(w<1.4)
			specblue = s[isub]
			wblue = w[isub]
	
	w = np.concatenate([wblue,wred])
	s = np.concatenate([specblue,specred])

	# reinterpolate 
	if np.min(x) > 10:
		x/=1000 #convert nm to um

	tck    = interpolate.splrep(w,s, k=2, s=0)
	snew   = interpolate.splev(x,tck,der=0,ext=1)

	if ploton:
		plt.plot(wblue,specblue,label='blue')
		plt.plot(wred,specred,label='red')
		plt.grid(True)
		plt.xlabel('Wavelength (um)')
		plt.ylabel('Transmission')
		plt.title("HISPEC E2E Except Coupling")
		plt.savefig('e2e.png')

	return snew

def get_sky_bg(x,airmass,pwv=1.5):
	"""
	"""
	diam = 10. * u.m
	area = 76. * u.m * u.m
	delta_lb = 0.001 * u.micron
	wave = np.arange(0.8,2.5,delta_lb.value) 
	fwhm = ((wave * u.micron / diam) * u.radian).to(u.arcsec)
	solidangle = fwhm**2 * 1.13 #corrected for Gaussian beam (factor 1.13)
	
	sky_background_MK = np.zeros([4,3,235000])
	sky_background_MK_wv = np.array([1.,1.6,3.,5.])#water vapor 
	sky_background_MK_airmass = np.array([1.,1.5,2.])#airmass
	path = '../../../_DATA/'
	#
	sky_background_MK_tmp = np.genfromtxt(path+'sky/mk_skybg_zm_'+str(pwv)+'_'+str(airmass)+'_ph.dat', skip_header=0)
	sky_background_MK = sky_background_MK_tmp[:,1]
	sky_background_MK_wave = sky_background_MK_tmp[:,0] #* u.nm
	if np.max(wave) < 10: wave *= 1e+3
	sky_background_interp=np.interp(wave, sky_background_MK_wave, sky_background_MK) * u.photon/(u.s*u.arcsec**2*u.nm*u.m**2) * area * solidangle * 2000/100000/3 * u.nm 

	sky_background_interp2=np.interp(x, wave, sky_background_interp) 

	return sky_background_interp2

def inst_bg():
	diam = 10. * u.m
	area = 76. * u.m * u.m
	delta_lb = 0.001 * u.micron
	wave = np.arange(0.8,2.5,delta_lb.value) 
	fwhm = ((wave * u.micron / diam) * u.radian).to(u.arcsec)
	solidangle = fwhm**2 * 1.13 #corrected for Gaussian beam (factor 1.13)
	
	T_tel = 276
	tel_bb = blackbody_lambda(wave * u.micron, T_tel).to(u.erg/(u.micron * u.s * u.cm**2 * u.arcsec**2)) * area.to(u.cm**2) * solidangle
	tel_thermal = tel_em * tel_bb.to(u.photon/u.s/u.micron, equivalencies=u.spectral_density(wave * u.micron)) * 2./100000/3 * u.micron

	datapath = './data/throughput/hispec_subsystems_11032022/'
	#plt.figure()
	for spec in ['red','blue']:
		if spec=='red':
			include = ['tel', 'ao', 'feicom', 'feired','fibred','rspec']#,'coupling']
			temps    = [276,276,276,276,276,77]
		if spec=='blue':
			include = ['tel', 'ao', 'feicom', 'feiblue','fibblue','bspec']#,'coupling']
			temps    = [276,276,276,276,276,77]

		for i in include:
			if i==include[0]:
				w,s = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
				#bb = (blackbody_lambda(w* u.micron, ?).to(u.erg/(u.micron * u.s * u.cm**2 * u.arcsec**2)) * area.to(u.cm**2) * solidangle)
				thermal = fiber_em * fiber_bb.to(u.photon/u.s/u.micron, equivalencies=u.spectral_density(wave* u.micron)) * 2./100000/3 * u.micron	
			else:
				wtemp, stemp = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
				# interpolate onto s
				f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
				s*=f(w)
				#plt.plot(w,s,label=i)

		if spec=='red':
			isub = np.where(w > 1.4) 
			wred = w[isub]
			specred = s[isub]
		if spec=='blue':
			isub = np.where(w<1.4)
			specblue = s[isub]
			wblue = w[isub]
	
	w = np.concatenate([wblue,wred])
	s = np.concatenate([specblue,specred])

	# reinterpolate 
	if np.min(x) > 10:
		x/=1000 #convert nm to um

	tck    = interpolate.splrep(w,s, k=2, s=0)
	snew   = interpolate.splev(x,tck,der=0,ext=1)

	if ploton:
		plt.plot(wblue,specblue,label='blue')
		plt.plot(wred,specred,label='red')
		plt.grid(True)
		plt.xlabel('Wavelength (um)')
		plt.ylabel('Transmission')
		plt.title("HISPEC E2E Except Coupling")
		plt.savefig('e2e.png')

	return snew

def pick_coupling(waves,dynwfe,ttStatic,ttDynamic,LO=30,PLon=0,piaa_boost=1.3):
	"""
	select correct coupling file
	to do:implement interpolation of coupling files instead of rounding variables
	"""
	if np.min(waves) > 10:
		waves/=1000 # convert nm to um

	# check range of each variable
	if ttStatic > 10 or ttStatic < 0:
		raise ValueError('ttStatic is out of range, 0-10')
	if ttDynamic > 10 or ttDynamic < 0:
		raise ValueError('ttDynamic is out of range, 0-10')
	if LO > 100 or LO < 0:
		raise ValueError('LO is out of range,0-100')
	if PLon >1:
		raise ValueError('PL is out of range')

	if PLon:
		points, values_1,values_2,values_3 = grid_interp_coupling(PLon) # move this outside this function ,do one time!
		point = (LO,ttStatic,ttDynamic,waves)
		mode1 = interpn(points, values_1, point,bounds_error=False,fill_value=0) # see example https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html#scipy.interpolate.interpn
		mode2 = interpn(points, values_2, point,bounds_error=False,fill_value=0) 
		mode3 = interpn(points, values_3, point,bounds_error=False,fill_value=0) 

		PLwav,PLdat = load_photonic_lantern() #transfer matrices input mode--> each SMF
		mat = PLdat[10] # use middle one for now
		test1 = mode1 * mat[0,0]  + mode2*mat[1,0] + mode3*mat[2,0]
		test2 = mode1 * mat[0,1]  + mode2*mat[1,1] + mode3*mat[2,1]
		test3 = mode1 * mat[2,2]  + mode2*mat[1,2] + mode3*mat[2,2]
		# apply only to YJ or make matrix diagonal for HK..map onto same wavelength grid somehow
		# get coupling
		raw_coupling = mode1+mode2+mode3 # do dumb things for now
	else:
		points, values_1 = grid_interp_coupling(PLon)
		point = (LO,ttStatic,ttDynamic,waves)
		raw_coupling = interpn(points, values_1, point,bounds_error=False,fill_value=0)

	if np.max(waves) < 10:
		waves*=1000 # nm to match dynwfe

	ho_strehl =  np.exp(-(2*np.pi*dynwfe/waves)**2) # computed per wavelength as grid
	coupling  = raw_coupling * piaa_boost * ho_strehl
	
	return coupling,ho_strehl

def plot_wfe():
	"""
	"""
	modes = np.array(['K','SH','80J','80H','80JK','LGS'])# corresponding modes to match assumptions of text files 
	
	f_tt = np.loadtxt('./data/WFE/HAKA/Kstar_tiptilt.txt').T
	vmags = f_tt[0]
	f_wfe = np.loadtxt('./data/WFE/HAKA/Kstar_HOwfe.txt').T
	
	fig, ax = plt.subplots(2,figsize=(6,6),sharex=True)
	for mode in modes:
		imode = np.where(modes==mode)[0][0]

		wfes      = f_wfe[1:][imode]
		tip_tilts = f_tt[1:][imode] # switch to take best mode for observing case

		ax[0].plot(vmags,tip_tilts,label=mode)
		ax[1].plot(vmags,wfes,label=mode)

	ax[0].set_xlim(3,20)
	ax[0].set_ylim(0,20)
	ax[0].legend(loc=2,fontsize=10)
	ax[0].grid(True)
	ax[0].set_ylabel('Tip Tilt Resid. (mas)')
	ax[1].set_xlabel('V Mag')

	ax[1].set_ylim(100,500)
	ax[1].set_ylabel('HO WFE (nm)')
	ax[1].grid(True)
	plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15)
	ax[0].set_title('K Star HAKA WFE Estimate')

def plot_noise_components():
	"""
	plot spectra and transmission so know what we're dealing with
	"""
	plt.figure()
	plt.plot(so.stel.v,so.hispec.ytransmit)


if __name__=='__main__':	
	#load inputs
	configfile = 'hispec.cfg'
	so    = load_object(configfile)
	cload = fill_data(so)

	# scale spectra
	sat        = 1e5/2#half of full well
	hispec_pixel_column = 3 # pixels in a column
	nramps     = 30 # number of up the ramp frames recorded for read noise reduction
	readnoise  = 12/np.sqrt(nramps)# e-, check this**
	darknoise  = 0.02#0.1 #e-/s/pix/read
	PLon       = 1 # if photonic lantern or not, 1=on
	base_exp_time = 900 # s , assert 15min exposure time longest can go

	# get WFE - (1) ttStatic - tip tilt (2) ttDynamic - tip/tilt (3) dynwfe - dyanamic wfe (4) LO low order -focus
	mode='LGS' # ['K','SH','80J','80H','80JK','LGS']
	guide_star_mag_diff = 10
	ttStatic = 4.2 # mas - fiber decenter - use result from centroid requirement from tracking (0.0042)
	LO       = 50  # nm constant focus term, set to Gary's assumption which was????
	#defocus  = 25  # **nm, need to implement!***

	Vmag    = get_band_mag(so,'Johnson','V',so.stel.factor_0) # get magnitude in r band
	Vmag    -=guide_star_mag_diff
	HOwfe   = get_HO_WFE(Vmag,mode)
	ttresid = get_tip_tilt_resid(Vmag,mode)
	#if ttresid >10: ttresid=10

	# Pick coupling curve!
	base_throughput  = get_base_throughput(x=so.stel.v) # everything except coupling
	coupling, strehl = pick_coupling(so.stel.v,HOwfe,ttStatic,ttresid,LO=LO,PLon=PLon) # includes PIAA and HO term
	#mode_throughput  = # consider dichroics if tracking in sci band 
	total_throughput = base_throughput*coupling

	# calc SNR
	v,s      = hispec_sim_spectrum(so,total_throughput)
	nframes  = round(so.var.exp_time/base_exp_time)
	skybg    = get_sky_bg(v,so.tel.airmass,pwv=1.1)

	# calc noise
	noise = np.sqrt(s + hispec_pixel_column * (nframes*readnoise**2 + (skybg.value + darknoise)*so.var.exp_time)) #noise in reduced pixel column
	snr   = s/noise
	
	# check if one frame hits saturation
	if np.max(s) > sat: print("WARNING-hit SATURATION limit, lower exp time")

	plotsnr=True
	plt.figure(-10,figsize=(7,4))
	plt.plot(v,snr,label='PL: %s'%bool(PLon))
	#plt.plot(v,snr,alpha=0.8,label='Mode: %s'%mode)
	plt.ylabel('SNR')
	plt.xlabel('Wavelength (nm)')
	plt.title('%s=%s, t=4hr, Teff=%s, dark=%se-/s/pix'%(so.filt.band,int(so.var.mag),int(so.var.teff),darknoise))
	plt.axhline(y=30,color='k',ls='--')
	figname = 'snr_%s_%smag_%s_Teff_%s_texp_%ss_dark_%s.png' %(mode,so.filt.band,so.var.mag,so.var.teff,int(base_exp_time*nframes),darknoise)
	plt.legend()
	#plt.savefig('./output/snrplots/' + figname)

	plt.figure(figsize=(7,4))
	plt.plot(so.stel.v,coupling,label='Coupling Only')
	plt.plot(so.stel.v,base_throughput,label='All But Coupling')	
	plt.plot(so.stel.v,coupling*base_throughput,'k',label='Total Throughput')	
	plt.ylabel('Transmission')
	plt.xlabel('Wavelength (nm)')
	plt.title('%s=%s, Vmag=%s, Teff=%s'%(so.filt.band,int(so.var.mag),round(Vmag,1),int(so.var.teff)))
	plt.subplots_adjust(bottom=0.15)
	plt.axhline(y=0.05,color='m',ls='--',label='5%')
	plt.legend()
	plt.grid()
	figname = 'throughput_%s_%smag_%s_Teff_%s_texp_%ss.png' %(mode,so.filt.band,so.var.mag,so.var.teff,int(base_exp_time*nframes))
	#plt.savefig('./output/snrplots/' + figname)

