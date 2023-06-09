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
from load_inputs import fill_data, load_filter
from functions import *
import noise_tools
#from kpf_etc.etc import kpf_photon_noise_estimate, kpf_etc_rv, kpf_etc_snr

plt.ion()

def hispec_sim_spectrum(so,throughput):
	"""
	scale output from kpf scaling to Ca H&K throughput 
	"""
	s_ccd_hires = so.stel.s * so.obs.texp_frame * so.inst.tel_area * throughput * np.abs(so.tel.s)* so.inst.extraction_frac#**so.tel.airmass

	# convolve to lower res
	s_ccd_lores = degrade_spec(so.stel.v, s_ccd_hires, so.inst.res)

	# resample onto res element grid
	v_resamp, s_resamp = resample(so.stel.v,s_ccd_lores,sig=so.inst.sig, dx=0, eta=1,mode='variable')
	s_resamp_nframes = s_resamp * so.obs.nframes
	
	noise_frame  = noise_tools.sum_total_noise(s_resamp,so.obs.texp_frame, so.obs.nsamp,so.obs.inst_bg_ph, so.obs.sky_bg_ph, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert)
	noise_frame[np.where(np.isnan(noise_frame))] = np.inf
	noise_frame[np.where(noise_frame==0)] = np.inf
	noise_nframes = noise_frame * np.sqrt(so.obs.nframes)
	snr =  s_resamp_nframes/noise_nframes
	
	v_reselement, snr_reselement = resample(v_resamp,snr,sig=3, dx=0, eta=1/np.sqrt(so.inst.res_samp),mode='pixels')

	return v_reselement, snr_reselement

def find_min_throughput(so,snrgoal,nframes,dark=0.8,rn=12,hispec_pixel_column=3,method='max',ploton=False):
	"""
	inputs:
	-------


	output:
	-------
	minimum throughput needed to reach snrgoal

	"""
	#snrgoal = 30 # as stated by requirement

	throughputs = np.arange(20)/100
	snrmean     = np.zeros_like(throughputs)
	snrmedian   = np.zeros_like(throughputs)
	snrmax      = np.zeros_like(throughputs)
	snr30frac   = np.zeros_like(throughputs)
	for i,throughput in enumerate(throughputs):
		v,s  = hispec_sim_spectrum(so,throughput)
		noise = np.sqrt(s + hispec_pixel_column * (rn**2 + so.obs.texp*dark)) #noise in reduced pixel column
		snr = s/noise
		snr*= np.sqrt(nframes)
		snrmean[i] = np.mean(snr)
		snrmedian[i] = np.median(snr)
		snrmax[i] = np.max(snr)
		n30 = len(np.where(snr > snrgoal)[0])
		snr30frac[i] = n30/len(snr)

	# interpolate throughput vs snr30frac curve to more accurately pull out where snr goes above goal
	throughputs_fine = np.arange(np.min(throughputs),np.max(throughputs),0.001)
	if method=='max':
		interpsnr = interp1d(throughputs,snrmax)
		isub      = np.where(interpsnr(throughputs_fine) > snrgoal)

	elif method=='mean':
		interpsnr = interp1d(throughputs,snrmean)
		isub      = np.where(interpsnr(throughputs_fine) > snrgoal)

	if ploton:
		# plot of results
		plt.figure(figsize=(7,4))
		plt.plot(throughputs,snrmax,label='Max SNR')
		plt.plot(throughputs,snrmean,label='Mean SNR')
		#plt.plot(throughputs,snrmedian,label='Median SNR')
		plt.xlabel('Throughput')
		plt.ylabel('SNR')
		plt.title('%s=%s, t_total=4hr (%ss/frame), Teff=%s'%(so.filt.band,int(so.stel.mag),int(so.stel.exp_time),int(so.stel.teff)))
		plt.grid()
		plt.axhline(y=30,color='k',ls='--')
		plt.legend()
		plt.subplots_adjust(bottom=0.15)
		figname = 'throughputgoal_%smag_%s_Teff_%s_texp_%ss.png' %(so.filt.band,so.stel.mag,so.stel.teff,int(so.obs.texp*nframes))
		plt.savefig('./output/snrplots/' + figname)

		plt.figure(figsize=(7,4))
		plt.plot(throughputs,snr30frac)
		plt.xlabel('Throughput')
		plt.ylabel('Fraction of Spectrum above SNR 30')
		plt.title('%s=%s, t_total=4hr (%ss/frame), Teff=%s'%(so.filt.band,int(so.stel.mag),int(so.obs.texp),int(so.stel.teff)))
		plt.grid()
		plt.subplots_adjust(bottom=0.15)
		plt.axhline(y=0.5,color='k',ls='--')
		figname = 'snr30frac_%smag_%s_Teff_%s_texp_%ss.png' %(so.filt.band,so.stel.mag,so.stel.teff,int(so.obs.texp*nframes))
		plt.savefig('./output/snrplots/' + figname)

	return throughputs_fine[isub[0][0]]


def plot_snr(throughput,dark1,dark2,rn1,rn2):
	# example SNR plot
	#throughput=0.075
	#dark1 = 0.8
	#dark2 = 1.6
	v,s  = hispec_sim_spectrum(so,throughput)
	noise1 = np.sqrt(s + hispec_pixel_column * (rn1**2 + so.obs.texp*dark1)) #noise in reduced pixel column
	noise2 = np.sqrt(s + hispec_pixel_column * (rn2**2 + so.obs.texp*dark2)) #noise in reduced pixel column
	snr1 = np.sqrt(nframes) * s/noise1
	snr2 = np.sqrt(nframes) * s/noise2	

	plotsnr=True
	plt.figure(figsize=(7,4))
	plt.plot(v,snr1 ,'g',alpha=0.7,label='Dark=%se-/pix/s, RN=%s'%(dark1,rn1))
	plt.plot(v,snr2 ,'orange',alpha=0.7,label='Dark=%se-/pix/s, RN=%s'%(dark2,rn2))
	plt.ylabel('SNR')
	plt.xlabel('Wavelength (nm)')
	plt.title('%s=%s, t_total=4hr (%ss/frame), Teff=%s,\nThroughput=%s'%(so.filt.band,int(so.stel.mag),int(so.obs.texp),int(so.stel.teff),throughput))
	plt.axhline(y=30,color='k',ls='--')
	plt.subplots_adjust(bottom=0.15,top=0.85)
	plt.legend(fontsize=14)
	figname = 'example_flat_throughput_%s_%smag_%s_Teff_%s_texp_%ss.png' %(throughput,so.filt.band,so.stel.mag,so.stel.teff,int(so.obs.texp*nframes))
	plt.savefig('./output/snrplots/' + figname)

def get_order_snrs(so,v,snr):
	"""
	given array, return max and mean of snr per order
	"""
	order_peaks      = signal.find_peaks(so.inst.base_throughput,height=0.055,distance=2e4,prominence=0.01)
	order_cen_lam    = so.stel.v[order_peaks[0]]
	blaze_angle      =  76
	snr_peaks = []
	snr_means = []
	for i,lam_cen in enumerate(order_cen_lam):
		line_spacing = 0.02 if lam_cen < 1475 else 0.01
		m = np.sin(blaze_angle*np.pi/180) * 2 * (1/line_spacing)/(lam_cen/1000)
		fsr  = lam_cen/m
		isub_test= np.where((so.stel.v> (lam_cen - fsr/2)) & (so.stel.v < (lam_cen+fsr/2))) #FINISH THIS
		#plt.plot(so.stel.v[isub_test],total_throughput[isub_test],'k--')
		sub_snr = snr[np.where((v > (lam_cen - 1.3*fsr/2)) & (v < (lam_cen+1.3*fsr/2)))[0]] #FINISH THIS]
		snr_peaks.append(np.nanmax(sub_snr))
		snr_means.append(np.nanmean(sub_snr))

	return np.array(order_cen_lam), np.array(snr_peaks), np.array(snr_means)



if __name__=='__main__':	
	configfile = 'hispec_snr.cfg'
	so    = load_object(configfile)
	cload = fill_data(so)
	
	snrgoal  = 30 

	#min_throughput = find_min_throughput(so,snrgoal,so.obs.nframes,dark=dark,rn=rn,hispec_pixel_column=hispec_pixel_column,method='mean')

	throughput = 0.019
	v,snr = hispec_sim_spectrum(so,throughput)
	vpeak,speak,_ =  get_order_snrs(so,v,snr)
	plt.figure()
	plt.plot(vpeak,speak)
	plt.plot([500,2500],[30,30],'k--')
	#plot_snr(0.075,dark1=dark,dark2=dark*2,rn1=rn,rn2=rn)

	goal_throughput = throughput * (snrgoal/speak) #(30**2/speak**2)
	# interp onto grid
	f = interpolate.interp1d(vpeak, goal_throughput,bounds_error=False,fill_value=0)
	v,snr = hispec_sim_spectrum(so,f(so.stel.v))
	vpeak,speak,_ =  get_order_snrs(so,v,snr)
	plot_final_snr(vpeak,speak)
	plot_final_thr(vpeak, goal_throughput)


def plot_final_snr(vspeak,speak):
	fig, ax = plt.subplots(1,1, figsize=(6,4))	
	ax.plot(vpeak,speak)
	ax.plot([500,2500],[30,30],'k--')
	peak = np.max(speak)+2
	trough = np.min(speak)-2
	ax.set_ylabel('SNR')
	ax.set_xlabel('Wavelength (nm)')
	ax.axhline(y=30,color='k',ls='--')
	figname = 'base_throughput.png' 
	# duplicate axis to plot filter response
	# plot band
	ax.fill_between(so.inst.y,trough,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(20+np.min(so.inst.y),2+trough, 'y')
	ax.fill_between(so.inst.J,trough,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(so.inst.J),2+trough, 'J')
	ax.fill_between(so.inst.H,trough,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(so.inst.H),2+trough, 'H')
	ax.fill_between(so.inst.K,trough,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(so.inst.K),2+trough, 'K')
	ax.set_xlim(970,2500)
	ax.set_ylim(trough,peak)
	plt.grid()


def plot_final_thr(vpeak, goal_throughput):
	fig, ax = plt.subplots(1,1, figsize=(6,4))	
	ax.plot(vpeak, goal_throughput)
	ax.plot([500,2500],[30,30],'k--')
	peak = np.max(goal_throughput)+0.1
	trough = np.min(goal_throughput)-0.1
	ax.set_ylabel('SNR')
	ax.set_xlabel('Wavelength (nm)')
	ax.axhline(y=30,color='k',ls='--')
	figname = 'base_throughput.png' 
	# duplicate axis to plot filter response
	# plot band
	ax.fill_between(so.inst.y,trough,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(20+np.min(so.inst.y),2+trough, 'y')
	ax.fill_between(so.inst.J,trough,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(so.inst.J),2+trough, 'J')
	ax.fill_between(so.inst.H,trough,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(so.inst.H),2+trough, 'H')
	ax.fill_between(so.inst.K,trough,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(so.inst.K),2+trough, 'K')
	ax.set_xlim(970,2500)
	ax.set_ylim(trough,peak)
	plt.grid()
	plt.title(so.filt.band)
	# do a set of stellar temperatures to calc min throughput for


