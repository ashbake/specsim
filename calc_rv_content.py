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

font = {'size'   : 14}
matplotlib.rc('font', **font)

sys.path.append('./utils/')
from objects import load_object
from load_inputs import fill_data, load_filter,load_phoenix
from functions import *
#from kpf_etc.etc import kpf_photon_noise_estimate, kpf_etc_rv, kpf_etc_snr

plt.ion()

def hispec_sim_spectrum(so,airmass,coupling):
	"""
	scale output from kpf scaling to Ca H&K throughput 
	"""
	s_ccd_hires = so.stel.s * so.var.exp_time * so.const.tel_area * (so.hispec.ytransmit * coupling) #* np.abs(so.tel.s)**airmass

	# convolve to lower res
	s_ccd_lores = degrade_spec(so.stel.v, s_ccd_hires, so.const.res_hispec)

	# resample onto res element grid
	lam0=1500#nm take as lam to calc dlambda
	sig = lam0/so.const.res_hispec/3 # lambda/res = dlambda
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

	return wfe


def get_dyn_wfe_pywfs(Jmag):
	"""
	if using pywfs, give it jmag and see if it's better
	"""
	pass


def plot_components():
	"""
	plot spectra and transmission so know what we're dealing with
	"""
	plt.figure()
	plt.plot(so.stel.v,so.hispec.ytransmit)

def grid_interp_coupling(PLon):
	"""
	interpolate coupling files over their various parameters
	"""
	from scipy.interpolate import interpn
	LOs = np.arange(0,125,25)
	ttStatics = np.arange(11)
	ttDynamics = np.arange(0,10,0.5)
	
	path_to_files     = './data/throughput/hispec_subsystems_11032022/coupling/couplingEff_wPL_202212014/'
	filename_skeleton = 'couplingEff_atm0_adc0_PL%s_defoc0nmRMS_LO%snmRMS_ttStatic%smas_ttDynamic%smasRMS.csv'

	# to dfine values, must open up each file. not sure if can deal w/ wavelength
	values_1 = np.zeros((len(LOs),len(ttStatics),len(ttDynamics),100))
	values_2 = np.zeros((len(LOs),len(ttStatics),len(ttDynamics),100))
	values_3 = np.zeros((len(LOs),len(ttStatics),len(ttDynamics),100))	
	for i,LO in enumerate(LOs):
		for j,ttStatic in enumerate(ttStatics):
			for k,ttDynamic in enumerate(ttDynamics):
				if round(ttDynamic)==ttDynamic: ttDynamic=round(ttDynamic)
				f = pd.read_csv(path_to_files+filename_skeleton%(PLon,LO,ttStatic,ttDynamic))
				values_1[i,j,k,:]=f['coupling_eff_mode1'] #what to fill here?
				if PLon:
					values_2[i,j,k,:]=f['coupling_eff_mode2'] #what to fill here?
					values_3[i,j,k,:]=f['coupling_eff_mode3'] #what to fill here?
				#values_hk[i,j,k]=f['coupling_eff_mode1'][50] #what to fill here?
	
	points = (LOs, ttStatics, ttDynamics,f['wavelength_um'].values)

	if PLon:
		return points,values_1,values_2,values_3
	else:
		return points,values_1



if __name__=='__main__':	
	#load inputs
	configfile = 'hispec.cfg'
	so    = load_object(configfile)
	cload = fill_data(so)



