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
from scipy import signal, interpolate
import pandas as pd
from scipy.interpolate import interpn

font = {'size'   : 14}
matplotlib.rc('font', **font)

sys.path.append('./utils/')
from objects import load_object
from load_inputs import fill_data, load_filter,load_phoenix
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
	vraw,sraw = load_phoenix(so.stel.phoenix_file,wav_start=np.min(x), wav_end=np.max(x)) #phot/m2/s/nm
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

def get_tip_tilt_resid(Rmag):
	"""
	fill in when get data
	"""
	return 6

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

def get_dyn_wfe_pywfs(Jmag):
	"""
	if using pywfs, give it jmag and see if it's better
	"""
	pass

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
	"""
	datapath = './data/throughput/hispec_subsystems_11032022/'
	for spec in ['red','blue']:
		if spec=='red':
			include = ['tel', 'ao', 'feicom', 'feired','fibred','rspec']#,'coupling']
		if spec=='blue':
			include = ['tel', 'ao', 'feicom', 'feiblue','fibblue','bspec']#,'coupling']

		for i in include:
			if i==include[0]:
				w,s = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
			else:
				wtemp, stemp = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
				# interpolate onto s
				f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
				s*=f(w)

		if spec=='red':
			isub = np.where(w > 1.4) 
		if spec=='blue':
			isub = np.where(w<1.4)
		if ploton:
			plt.figure('e2e')
			plt.plot(w[isub],s[isub])

	# reinterpolate 
	if np.min(x) > 10:
		x/=1000 #convert nm to um
	tck    = interpolate.splrep(w,s, k=2, s=0)
	snew   = interpolate.splev(x,tck,der=0,ext=1)

	if ploton:
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
		mode1 = interpn(points, values_1, point) # see example https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html#scipy.interpolate.interpn
		mode2 = interpn(points, values_2, point) 
		mode3 = interpn(points, values_3, point) 

		PLwav,PLdat = load_photonic_lantern() #transfer matrices input mode--> each SMF
		
		# apply only to YJ or make matrix diagonal for HK..map onto same wavelength grid somehow
		# get coupling
	else:
		points, values_1 = grid_interp_coupling(PLon)
		point = (LO,ttStatic,ttDynamic,waves)
		raw_coupling = interpn(points, values_1, point,bounds_error=False,fill_value=0)

	if np.max(waves) < 10:
		waves*=1000 # nm to match dynwfe

	ho_strehl =  np.exp(-(2*np.pi*dynwfe/waves)**2) # computed per wavelength as grid
	coupling  = raw_coupling * piaa_boost * ho_strehl
	
	return coupling,ho_strehl


if __name__=='__main__':	
	#load inputs
	configfile = 'hispec.cfg'
	so    = load_object(configfile)
	cload = fill_data(so)

	# scale spectra
	sat     = 1e5/2#half of full well
	hispec_pixel_column = 3 # pixels in a column
	readnoise  = 12 # e-, check this**
	darknoise  = 0.8 #e-/s/pix
	PLon       = 0 # if photonic lantern or not, 1=on

	# get WFE - (1) ttStatic - tip tilt (2) ttDynamic - tip/tilt (3) dynwfe - dyanamic wfe (4) LO low order -focus
	mode='SH' # mode: shack hartmann (SH) or pywfs
	ttStatic = 0 # fiber decenter - assume 0
	ttDynamic=3  # is this ttresid?
	LO=0         # constant focus term, set to Gary's assumption which was????
	if mode=='SH': 
		Rmag    = get_band_mag(so,'Johnson','R',so.stel.factor_0) # get magnitude in r band
		dynwfe  = get_dyn_wfe(Rmag)
		ttresid = get_tip_tilt_resid(Rmag)
	if mode=='pywfs':
		Jmag    = get_band_mag(so,'Johnson','J',so.stel.factor_0) # get magnitude in J band
		dynwfe  = get_dyn_wfe_pywfs(Jmag)
		ttresid = get_tip_tilt_resid(Jmag)

	# Pick coupling curve!
	base_throughput  = get_base_throughput(x=so.stel.v) # everything except coupling
	coupling, strehl = pick_coupling(so.stel.v,dynwfe,ttStatic,ttDynamic,LO=LO,PLon=PLon) # includes PIAA and HO term
	total_throughput = base_throughput*coupling

	# calc SNR
	v,s  = hispec_sim_spectrum(so,total_throughput)

	# calc noise
	noise = np.sqrt(s + hispec_pixel_column * (readnoise**2 + darknoise*so.var.exp_time)) #noise in reduced pixel column
	snr = s/noise
	
	nframes  =  4*3600/so.var.exp_time
	# check if one frame hits saturation
	if np.max(s) > sat: print("WARNING-hit SATURATION limit, lower exp time")

	plotsnr=True
	plt.figure(figsize=(7,4))
	plt.plot(v,snr * np.sqrt(nframes),'g',label='HISPEC Spectrum')
	plt.ylabel('SNR')
	plt.xlabel('Wavelength (nm)')
	plt.title('%s=%s, t=4hr, Teff=%s'%(so.filt.band,int(so.var.mag),int(so.var.teff)))
	plt.axhline(y=30,color='k',ls='--')
	figname = 'snr_%s_%smag_%s_Teff_%s_texp_%ss.png' %(mode,so.filt.band,so.var.mag,so.var.teff,int(so.var.exp_time*nframes))
	plt.savefig('./output/snrplots/' + figname)

	plt.figure(figsize=(7,4))
	plt.plot(so.stel.v,coupling,label='Coupling Only')
	plt.plot(so.stel.v,base_throughput,label='All But Coupling')	
	plt.plot(so.stel.v,coupling*base_throughput,'k',label='Total Throughput')	
	plt.ylabel('Transmission')
	plt.xlabel('Wavelength (nm)')
	plt.title('%s=%s, Rmag=%s, Teff=%s'%(so.filt.band,int(so.var.mag),round(Rmag,1),int(so.var.teff)))
	plt.subplots_adjust(bottom=0.15)
	plt.axhline(y=0.05,color='m',ls='--',label='5%')
	plt.legend()
	plt.grid()
	figname = 'throughput_%s_%smag_%s_Teff_%s_texp_%ss.png' %(mode,so.filt.band,so.var.mag,so.var.teff,int(so.var.exp_time*nframes))
	plt.savefig('./output/snrplots/' + figname)

def plot_noise_components():
	"""
	plot spectra and transmission so know what we're dealing with
	"""
	plt.figure()
	plt.plot(so.stel.v,so.hispec.ytransmit)

