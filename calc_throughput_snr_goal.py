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
	lam0=1500#nm take as lam to calc dlambda
	sig = lam0/so.const.res_hispec/3 # lambda/res = dlambda
	v_resamp, s_resamp = resample(so.stel.v,s_ccd_lores,sig=sig, dx=0, eta=1,mode='fast')

	return v_resamp, s_resamp


if __name__=='__main__':	

	configfile = 'hispec.cfg'
	so    = load_object(configfile)
	cload = fill_data(so)
	
	hispec_pixel_column = 3 # pixels in a column

	nframes  =  4*3600/so.var.exp_time
	dark = 0.8 # e-/pix/s
	rn      = 12 # e-, check this**

	throughputs = np.arange(20)/100
	snrmean     = np.zeros_like(throughputs)
	snrmedian   = np.zeros_like(throughputs)
	snrmax      = np.zeros_like(throughputs)
	snr30frac   = np.zeros_like(throughputs)
	for i,throughput in enumerate(throughputs):
		v,s  = hispec_sim_spectrum(so,throughput)
		noise = np.sqrt(s + hispec_pixel_column * (rn**2 + so.var.exp_time*dark)) #noise in reduced pixel column
		snr = s/noise
		snr*= np.sqrt(nframes)
		snrmean[i] = np.mean(snr)
		snrmedian[i] = np.median(snr)
		snrmax[i] = np.max(snr)
		n30 = len(np.where(snr > 30)[0])
		snr30frac[i] = n30/len(snr)

	# plot of results
	plt.figure(figsize=(7,4))
	plt.plot(throughputs,snrmax,label='Max SNR')
	plt.plot(throughputs,snrmean,label='Mean SNR')
	#plt.plot(throughputs,snrmedian,label='Median SNR')
	plt.xlabel('Throughput')
	plt.ylabel('SNR')
	plt.title('%s=%s, t_total=4hr (%ss/frame), Teff=%s'%(so.filt.band,int(so.var.mag),int(so.var.exp_time),int(so.var.teff)))
	plt.grid()
	plt.axhline(y=30,color='k',ls='--')
	plt.legend()
	plt.subplots_adjust(bottom=0.15)
	figname = 'throughputgoal_%smag_%s_Teff_%s_texp_%ss.png' %(so.filt.band,so.var.mag,so.var.teff,int(so.var.exp_time*nframes))
	plt.savefig('./output/snrplots/' + figname)

	plt.figure(figsize=(7,4))
	plt.plot(throughputs,snr30frac)
	plt.xlabel('Throughput')
	plt.ylabel('Fraction of Spectrum above SNR 30')
	plt.title('%s=%s, t_total=4hr (%ss/frame), Teff=%s'%(so.filt.band,int(so.var.mag),int(so.var.exp_time),int(so.var.teff)))
	plt.grid()
	plt.subplots_adjust(bottom=0.15)
	plt.axhline(y=0.5,color='k',ls='--')
	figname = 'snr30frac_%smag_%s_Teff_%s_texp_%ss.png' %(so.filt.band,so.var.mag,so.var.teff,int(so.var.exp_time*nframes))
	plt.savefig('./output/snrplots/' + figname)

	# example SNR plot
	throughput=0.075
	dark1 = 0.8
	dark2 = 1.6
	v,s  = hispec_sim_spectrum(so,throughput)
	noise1 = np.sqrt(s + hispec_pixel_column * (rn**2 + so.var.exp_time*dark1)) #noise in reduced pixel column
	noise2 = np.sqrt(s + hispec_pixel_column * (rn**2 + so.var.exp_time*dark2)) #noise in reduced pixel column
	snr1 = np.sqrt(nframes) * s/noise1
	snr2 = np.sqrt(nframes) * s/noise2	

	plotsnr=True
	plt.figure(figsize=(7,4))
	plt.plot(v,snr1 ,'g',alpha=0.7,label='Dark=%se-/pix/s'%dark1)
	plt.plot(v,snr2 ,'orange',alpha=0.7,label='Dark=%se-/pix/s'%dark2)
	plt.ylabel('SNR')
	plt.xlabel('Wavelength (nm)')
	plt.title('%s=%s, t_total=4hr (%ss/frame), Teff=%s,\nThroughput=%s'%(so.filt.band,int(so.var.mag),int(so.var.exp_time),int(so.var.teff),throughput))
	plt.axhline(y=30,color='k',ls='--')
	plt.subplots_adjust(bottom=0.15,top=0.85)
	plt.legend(fontsize=14)
	figname = 'dark_noise_effect_flat_throughput_%s_%smag_%s_Teff_%s_texp_%ss.png' %(throughput,so.filt.band,so.var.mag,so.var.teff,int(so.var.exp_time*nframes))
	plt.savefig('./output/snrplots/' + figname)

	throughput=0.075
	dark=0.8
	rn1 = 12
	rn2 = 24
	v,s  = hispec_sim_spectrum(so,throughput)
	noise1 = np.sqrt(s + hispec_pixel_column * (rn1**2 + so.var.exp_time*dark)) #noise in reduced pixel column
	noise2 = np.sqrt(s + hispec_pixel_column * (rn2**2 + so.var.exp_time*dark)) #noise in reduced pixel column
	snr1 = np.sqrt(nframes) * s/noise1
	snr2 = np.sqrt(nframes) * s/noise2	

	plotsnr=True
	plt.figure(figsize=(7,4))
	plt.plot(v,snr1 ,'g',alpha=0.7,label='Read Noise=%se-'%rn1)
	plt.plot(v,snr2 ,'orange',alpha=0.7,label='Read Noise=%se-'%rn2)
	plt.ylabel('SNR')
	plt.xlabel('Wavelength (nm)')
	plt.title('%s=%s, t_total=4hr (%ss/frame), Teff=%s,\nThroughput=%s'%(so.filt.band,int(so.var.mag),int(so.var.exp_time),int(so.var.teff),throughput))
	plt.axhline(y=30,color='k',ls='--')
	plt.subplots_adjust(bottom=0.15,top=0.85)
	plt.legend(fontsize=14)
	figname = 'read_noise_effect_flat_throughput_%s_%smag_%s_Teff_%s_texp_%ss.png' %(throughput,so.filt.band,so.var.mag,so.var.teff,int(so.var.exp_time*nframes))
	plt.savefig('./output/snrplots/' + figname)



