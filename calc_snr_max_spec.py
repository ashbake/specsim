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
from noise_tools import get_sky_bg, get_inst_bg
from throughput_tools import pick_coupling, get_band_mag, get_base_throughput
from wfe_tools import get_tip_tilt_resid, get_HO_WFE

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


def run_one_spec():
	"""
	"""
	pass

if __name__=='__main__':	
	#load inputs
	configfile = 'hispec.cfg'
	so    = load_object(configfile)
	cload = fill_data(so) # put coupling files in load and wfe stuff too

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
	guide_star_mag_diff = 0
	ttStatic = 0# 4.2 # mas - fiber decenter - use result from centroid requirement from tracking (0.0042)
	LO       = 50  # nm constant focus term, set to Gary's assumption which was????
	#defocus  = 25  # **nm, need to implement!***
	
	Vmag    = get_band_mag(so,'Johnson','V',so.stel.factor_0) # get magnitude in r band
	Vmag    -=guide_star_mag_diff
	HOwfe   = get_HO_WFE(Vmag,mode)
	ttDynamic = 0#get_tip_tilt_resid(Vmag,mode)
	#if ttresid >10: ttresid=10

	# Pick coupling curve!
	base_throughput  = get_base_throughput(x=so.stel.v) # everything except coupling
	coupling, strehl = pick_coupling(so.stel.v,HOwfe,ttStatic,ttDynamic,LO=LO,PLon=PLon) # includes PIAA and HO term
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

