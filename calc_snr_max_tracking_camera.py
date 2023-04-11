# calc signal to noise
# max for the calcium H&K
import sys, os
import matplotlib
import numpy as np
import matplotlib.pylab as plt
from scipy import interpolate
import pandas as pd
from datetime import date

font = {'size'   : 14}
matplotlib.rc('font', **font)

sys.path.append('./utils/')
from objects import load_object
from load_inputs import fill_data, load_filter
from functions import tophat,degrade_spec
from noise_tools import get_sky_bg_tracking, get_inst_bg_tracking, sum_total_noise
from throughput_tools import pick_coupling, get_band_mag, get_base_throughput
import wfe_tools
from obs_tools import get_tracking_band, get_tracking_cam, calc_plate_scale
plt.ion()

from matplotlib.ticker import (AutoMinorLocator)

plt.ion()
font = {'size'   : 14}
matplotlib.rc('font', **font)
plt.rcParams['font.size'] = '14'
plt.rcParams['font.family'] = 'sans'
plt.rcParams['axes.linewidth'] = '1.3'
fontname = 'DIN Condensed'

# old
def plot_results_byfield(teff,ao_mode,filt_band='H'):
	"""
	must edit savename and link to folder in centroid/data/ 

	because i move files around to organized
	data in _run_20230221_PDR_offaxis_comparison
	"""
	exptimes = np.array([0.001,0.01,0.1,1,10])
	magarr   = np.arange(1,20,2) # in band of interest
	tracking_requirement_pixel  = get_track_req(so)

	field_x=[0.,0.,3.]
	field_y = [0.,2.,3.]
	camera='h2rg'
	linestyles=['-','--','-.']
	colors2 = ['orange','steelblue','green']

	track_bands = ['z','y','JHgap','J','H','K']
	colors = ['steelblue','orange','green','red','purple','brown']
	syms = ['o','s','d','.','x','^']
	
	ao_mode = 'LGS_STRAP_45'
	teff=3600

	fig, ax = plt.subplots(1,1,figsize=(6,5),sharex=True,sharey=True)
	for jj,x in enumerate(field_x):
		framerates =np.zeros(len(track_bands))
		for ii,track_band in enumerate(track_bands):
			savename = 'cendata_%s_ao_%s_band_%s_teff_%s_%smag_fieldx_%sarcsec_fieldy_%sarcsec.npy' %(camera,ao_mode,track_band,teff,'H',x,field_y[jj])
			try: centroid = np.load('./output/centroid_data/centroid_%s'%savename)
			except: continue
			try:	mag_req = get_mag_limit_per_exptime(exptimes,magarr,centroid,tracking_requirement_pixel)
			except: mag_req = np.zeros_like(exptimes)
			fwhm = np.load('./output/centroid_data/fwhm_%s'%savename)
			snr  = np.load('./output/centroid_data/snr_%s'%savename)
			#if track_band=='y' and ao_mode.startswith('LGS_100'): mag_req-=0.69# 20%y band flux, star is 2.5 mags fainter
			#if track_band=='z' and ao_mode.startswith('LGS_100'): mag_req-=1.6# 3% z band flux, star is 2.5 mags fainter
			#if track_band=='JHgap' and ao_mode.startswith('LGS_100'): mag_req-=.82 # 3% z band flux, star is 2.5 mags fainter
			exptimes_dense = np.arange(np.min(exptimes),np.max(exptimes),0.00001)
			ftemp = interpolate.interp1d(exptimes,mag_req)
			mag_req_dense = ftemp(exptimes_dense)
			try:
				print(track_band, 1/exptimes_dense[np.where(mag_req_dense>14.9)][0])
				framerates[ii] =  1/exptimes_dense[np.where(mag_req_dense>14.9)][0]
			except:
				print(track_band)
				#plt.plot(exptimes,mag_req,syms[ii],c=p[0].get_color())
		for ii in np.arange(len(track_bands)):
			p = ax.scatter(track_bands[ii],framerates[ii],c=colors[ii],marker=syms[ii])
		ax.semilogy(track_bands,framerates,c=colors2[jj],linestyle=linestyles[jj],label='(%s", %s")'%(x,field_y[jj]),lw=0.8,alpha=1)

	ax.set_title('%s'%(ao_mode))
	plt.suptitle('T=%sK'%teff)

	ax.legend()
	ax.set_xlabel('Tracking Band')
	plt.subplots_adjust(bottom=0.15,left=0.15,top=0.85,right=0.9)
	ax.set_ylabel('Max Frame Rate (Hz)')
	ax.grid(color='gray',linestyle='--',alpha=0.5,dash_joinstyle='miter')
	yticks = [1,10,100,1000]
	ax.set_yticks(yticks)
	ax.set_yticklabels(yticks,fontsize=14)

	plt.savefig('./output/trackingcamera/plots/mag_req/frame_rate_fieldpositions_ao_mode_%s_Teff_%sK.png'%(ao_mode,teff))

############## FINAL
def plot_results_var_final(var,teff,datapath,camera,track_bands=['y','z','JHgap','J','H','K']):
	"""
	var: ['signal','noise','snr','fwhm']
	"""
	exptimes = np.load(datapath + 'exptimes.npy')
	magarr   = np.load(datapath + 'magarr.npy')
	tracking_requirement_pixel  = get_track_req(so)
	modes, modes2, linestyles, colors, widths = wfe_tools.get_AO_plot_scheme()
	#teff     = 4200
	filt_band='H' # matches the run
	fig, axs = plt.subplots(3,2,figsize=(7,7),sharex=True,sharey=True)

	for itexp in np.arange(len(exptimes)):
		for ii,track_band in enumerate(track_bands):
			for jj,ao_mode in enumerate(modes):
				savename = 'cendata_%s_ao_%s_band_%s_teff_%s_%smag_fieldr_%sarcsec.npy' %(camera,ao_mode,track_band,teff,'H',0.0)
				try: centroid = np.load(datapath + 'centroid_%s'%(savename))
				except: continue
				data = np.load(datapath + '%s_%s'%(var,savename))
				ifit = np.where((~np.isnan(centroid[:,itexp])) &(centroid[:,itexp]< 3))[0]
				if var=='fwhm':axs[ii%3,ii//3].plot(magarr[ifit],data[ifit],c=colors[jj],linestyle=linestyles[jj],label=ao_mode)
				else: axs[ii%3,ii//3].plot(magarr[ifit],data[:,itexp][ifit],c=colors[jj],linestyle=linestyles[jj],label=ao_mode)
			axs[ii%3,ii//3].set_title('%s'%(track_band))
		plt.suptitle('T=%sK  '%teff +  '$t_{exp}$=%ss'%exptimes[itexp])

	axs[2,0].set_xlabel('%s Magnitude'%filt_band)
	axs[2,1].set_xlabel('%s Magnitude'%filt_band)
	#axs[1].legend(bbox_to_anchor=(1,1))
	plt.subplots_adjust(bottom=0.15,top=0.9,right=0.65)
	for ii,track_band in enumerate(track_bands):
		axs[ii%3,ii//3].axvline(15,c='k',linestyle='--')
		if ii//3==0: axs[ii%3,ii//3].set_ylabel(var)
		axs[ii%3,ii//3].grid(color='gray',linestyle='--',dash_joinstyle='miter')

	savename = '%s_results_ao_mode_%s_Teff_%sK.png'%(var,ao_mode,teff)
	savepath = os.path.join(*datapath.split('/')[0:-1]) + '/_plots/'
	if not os.path.isdir(savepath): os.makedirs(savepath)

	#plt.savefig(savepath + savename) # plots suck too much to save

def plot_results_mag_req_single(track_band='JHgap',filt_band='H'):
	"""
	"""
	datapath = './output/trackingcamera/centroid_arrs/_run_20230313/'
	exptimes = np.load(datapath + 'exptimes.npy')
	magarr   = np.load(datapath + 'magarr.npy')
	tracking_requirement_pixel  = get_track_req(so)
	modes, modes2, linestyles, colors, widths = wfe_tools.get_AO_plot_scheme()
	#teff     = 4200
	#filt_band='H' # matches the run
	track_band='JHgap'

	fig, axs = plt.subplots(1,1,figsize=(7,7),sharex=True,sharey=True)
	modes =['LGS_STRAP_45', 'LGS_100J_130']
	for ii,teff in enumerate([1500,2300,3600,5800]):
		for jj,ao_mode in enumerate(modes):
			savename = 'cendata_%s_ao_%s_band_%s_teff_%s_%smag_fieldr_%sarcsec.npy' %('h2rg',ao_mode,track_band,teff,'H',0.0)
			try: centroid = np.load(datapath + 'centroid_%s'%(savename))
			except: continue
			try:	mag_req = get_mag_limit_per_exptime(exptimes,magarr,centroid,tracking_requirement_pixel)
			except: mag_req = np.zeros_like(exptimes)
			if (teff ==2300) & (ao_mode == modes[0]): continue
			axs.semilogx(exptimes,mag_req,label=str(teff) + 'K')
	
	axs.set_title(track_band + ' Tracking Performance')

	axs.set_xlabel('Exposure Time [s]')
	#plt.legend(bbox_to_anchor=(1,1))
	plt.subplots_adjust(bottom=0.15,top=0.85,right=0.9)
	axs.axhline(15,c='k',linestyle='--')
	axs.set_ylabel('%s Limit' %filt_band)
	axs.grid(color='gray',linestyle='-',alpha=0.2)
	axs.set_ylim(11,20)
	axs.fill_between(exptimes,15,y2=22,facecolor='green',alpha=0.2)
	axs.set_xticks(exptimes)
	axs.set_xticklabels(exptimes,fontsize=12)
	yticks = [10,11,12,13,14,15,16,17,18]
	axs.set_yticks(yticks)
	axs.set_yticklabels(yticks,fontsize=12)
	plt.legend()
	plt.savefig('./output/trackingcamera/plots/mag_req/mag_req_single_trackband_%sK_band_%s.png'%(track_band,filt_band))

def plot_results_mag_req_final(teff,camera,filt_band='H',track_bands=['y','z','JHgap','J','H','K'],datapath='./output/trackingcamera/centroid_arrs/_run_20230313/'):
	"""
	filt_band: the bandpass used to define the magnitude system
	"""
	exptimes = np.load(datapath + 'exptimes.npy')
	magarr   = np.load(datapath + 'magarr.npy')
	tracking_requirement_pixel  = get_track_req(so)
	modes, modes2, linestyles, colors, widths = wfe_tools.get_AO_plot_scheme()
	fig, axs = plt.subplots(3,2,figsize=(7,7),sharex=True,sharey=True)

	for ii,track_band in enumerate(track_bands):
		for jj,ao_mode in enumerate(modes):
			savename = 'cendata_%s_ao_%s_band_%s_teff_%s_%smag_fieldr_%sarcsec.npy' %(camera,ao_mode,track_band,teff,'H',0.0)
			try: centroid = np.load(datapath + 'centroid_%s'%(savename))
			except: continue
			try:	mag_req = get_mag_limit_per_exptime(exptimes,magarr,centroid,tracking_requirement_pixel)
			except: mag_req = np.zeros_like(exptimes)
			axs[ii%3,ii//3].semilogx(exptimes,mag_req,c=colors[jj],linestyle=linestyles[jj],linewidth=widths[jj],label=modes2[jj]+ ' ' +modes[jj])
		axs[ii%3,ii//3].set_title('%s'%(track_band))
	plt.suptitle('T=%sK'%teff)

	axs[2,0].set_xlabel('Exposure Time [s]')
	axs[2,1].set_xlabel('Exposure Time [s]')
	#plt.legend(bbox_to_anchor=(1,1))
	plt.subplots_adjust(bottom=0.15,top=0.85,right=0.9)
	for ii,track_band in enumerate(track_bands):
		axs[ii%3,ii//3].axhline(15,c='k',linestyle='--')
		if ii//3==0: axs[ii%3,ii//3].set_ylabel('%s Limit' %filt_band)
		axs[ii%3,ii//3].grid(color='gray',linestyle='-',alpha=0.2)
		axs[ii%3,ii//3].set_ylim(2,20)
		axs[ii%3,ii//3].fill_between(exptimes,15,y2=22,facecolor='green',alpha=0.2)
		axs[ii%3,ii//3].set_xticks(exptimes)
		axs[ii%3,ii//3].set_xticklabels(exptimes,fontsize=12)
		yticks = [5,10,15,20]
		axs[ii%3,ii//3].set_yticks(yticks)
		axs[ii%3,ii//3].set_yticklabels(yticks,fontsize=12)
	plt.suptitle('T$_{eff}$ = %sK' %teff)
	plt.savefig('./output/trackingcamera/plots/mag_req/mag_req_Teff_%sK_band_%s.png'%(teff,filt_band))

def plot_frame_rates(teff,ao_mode,camera,maglimit=14.9,filt_band='H',track_bands=['JHgap','J','H','K'],datapath = './output/trackingcamera/centroid_arrs/_run_20230313/'):
	"""
	must edit savename and link to folder in centroid/data/ 

	because i move files around to organized
	data in _run_20230221_PDR_offaxis_comparison
	"""
	tracking_requirement_pixel  = get_track_req(so)
	exptimes = np.load(datapath + 'exptimes.npy')
	magarr   = np.load(datapath + 'magarr.npy')

	#colors = ['steelblue','orange','green','purple']
	syms = ['o','s','d','.']
	fig, ax = plt.subplots(1,1,figsize=(6,5),sharex=True,sharey=True)
	framerates =np.zeros(len(track_bands))
	for ii,track_band in enumerate(track_bands):
		if '%s' in ao_mode:
			if track_band=='H': ao_mode_filled = ao_mode%track_band
			else: ao_mode_filled = ao_mode%'J'
		savename = 'cendata_%s_ao_%s_band_%s_teff_%s_%smag_fieldr_%sarcsec.npy' %(camera,ao_mode_filled,track_band,teff,'H',0.0)
		try: centroid = np.load(datapath + 'centroid_%s'%(savename))
		except: continue
		try:	mag_req = get_mag_limit_per_exptime(exptimes,magarr,centroid,tracking_requirement_pixel)
		except: mag_req = np.zeros_like(exptimes)
		fwhm = np.load(datapath + 'fwhm_%s'%savename)
		snr  = np.load(datapath + 'snr_%s'%savename)
		exptimes_dense = np.arange(np.min(exptimes),np.max(exptimes),0.00001)
		ftemp = interpolate.interp1d(exptimes,mag_req)
		mag_req_dense = ftemp(exptimes_dense)
		try:
			print(track_band, 1/exptimes_dense[np.where(mag_req_dense>maglimit)][0])
			framerates[ii] =  1/exptimes_dense[np.where(mag_req_dense>maglimit)][0]
		except:
			print(track_band)
			#plt.plot(exptimes,mag_req,syms[ii],c=p[0].get_color())
	for ii in np.arange(len(track_bands)):
		p = ax.scatter(track_bands[ii],framerates[ii],marker=syms[0])
	p = ax.plot(track_bands,framerates,'k--',lw=0.6)
	
	ax.set_title('T=%sK'%teff)
	#plt.suptitle(ao_mode)
	plt.suptitle('HISPEC Tracking Camera')

	#ax.legend()
	ax.set_xlabel('Tracking Band')
	plt.subplots_adjust(bottom=0.15,left=0.15,top=0.85,right=0.9)
	ax.set_ylabel('Max Frame Rate (Hz)')
	ax.grid(color='gray',linestyle='--',alpha=0.5,dash_joinstyle='miter')
	yticks = [100,150,200]
	ax.set_yticks(yticks)
	ax.set_yticklabels(yticks,fontsize=14)

	savename = 'frame_rate_ao_mode_%s_Teff_%sK.png'%(ao_mode,teff)
	plt.savefig('./output/trackingcamera/plots/mag_req/frame_rate_ao_mode_%s_Teff_%sK.png'%(ao_mode,teff))

def plot_frame_rates_magnitude(teffs,ao_mode,camera,track_bands=['J','Jplus','H','JHgap'],maglimits=[11.9,12.9,13.9,14.9,15.9],filt_band='H',datapath = './output/trackingcamera/centroid_arrs/_run_20230313/'):
	"""
	must edit savename and link to folder in centroid/data/ 

	because i move files around to organized
	data in _run_20230221_PDR_offaxis_comparison
	"""
	tracking_requirement_pixel  = get_track_req(so)
	exptimes = np.load(datapath + 'exptimes.npy')
	magarr   = np.load(datapath + 'magarr.npy')

	#colors = ['steelblue','orange','green','purple']
	syms = ['o','s','d','.']
	fig, ax = plt.subplots(1,1,figsize=(6,5),sharex=True,sharey=True)
	framerates =np.zeros(len(maglimits))
	for track_band in track_bands:
		for ii,maglimit in enumerate(maglimits):
			if '%s' in ao_mode:
				if track_band=='H': ao_mode_filled = ao_mode%track_band
				else: ao_mode_filled = ao_mode%'J'
			else: ao_mode_filled=ao_mode
			savename = 'cendata_%s_ao_%s_band_%s_teff_%s_%smag_fieldr_%sarcsec.npy' %(camera,ao_mode_filled,track_band,teff,'H',0.0)
			try: centroid = np.load(datapath + 'centroid_%s'%(savename))
			except: continue
			try:	mag_req = get_mag_limit_per_exptime(exptimes,magarr,centroid,tracking_requirement_pixel)
			except: mag_req = np.zeros_like(exptimes)
			fwhm = np.load(datapath + 'fwhm_%s'%savename)
			snr  = np.load(datapath + 'snr_%s'%savename)
			exptimes_dense = np.arange(np.min(exptimes),np.max(exptimes),0.00001)
			ftemp = interpolate.interp1d(exptimes,mag_req)
			mag_req_dense = ftemp(exptimes_dense)
			try:
				print(track_band, 1/exptimes_dense[np.where(mag_req_dense>maglimit)][0])
				framerates[ii] =  1/exptimes_dense[np.where(mag_req_dense>maglimit)][0]
			except:
				print(track_band)
				#plt.plot(exptimes,mag_req,syms[ii],c=p[0].get_color())
		p = ax.scatter(maglimits,framerates,marker=syms[0],label=track_band)
		p = ax.plot(maglimits,framerates,'k--',lw=0.8)
	
	ax.set_title('T=%sK'%(teff))
	#plt.suptitle(ao_mode)
	plt.suptitle('Cam: %s, AO Mode: LGS'%(camera))
	ax.legend(fontsize=10)

	ax.set_xlabel('H Mag')
	plt.subplots_adjust(bottom=0.15,left=0.15,top=0.85,right=0.9)
	ax.set_ylabel('Max Frame Rate (Hz)')
	ax.grid(color='gray',linestyle='--',alpha=0.5,dash_joinstyle='miter')
	ax.set_yscale('log')
	yticks = [10,27,100,500,1000]
	ax.set_yticks(yticks)
	ax.set_yticklabels(yticks,fontsize=14)
	ax.set_ylim(1,1200)
	if teff <= 3000:
		ax.fill_between([11.7,16.3],0,27,alpha=0.3,color='r',zorder=-100)
	ax.plot([15,15],[1,1200],'k',lw=2)

	savename = 'frame_rate_ao_mode_%s_Teff_%sK.png'%(ao_mode,teff)
	savepath = os.path.join(*datapath.split('/')[0:-1]) + '/_plots/'
	if not os.path.isdir(savepath): os.makedirs(savepath)

	plt.savefig(savepath + savename)

def plot_saturation(teff, ao_mode):
	"""
	made for run 20230313 

		plot_saturation(teff, 'LGS_100J_130')

	"""
	datapath = './output/trackingcamera/centroid_arrs/_run_20230313/'
	exptimes = np.load(datapath + 'exptimes.npy')
	magarr   = np.load(datapath + 'magarr.npy')
	tracking_requirement_pixel  = get_track_req(so)
	modes, modes2, linestyles, colors, widths = wfe_tools.get_AO_plot_scheme()
	teff     = 3000
	saturation = 80000
	filt_band='H' # matches the run	
	ODs = [0,2,3,4]
	fig, axs = plt.subplots(1,1,figsize=(7,6),sharex=True,sharey=True)

	saturated = np.zeros((len(exptimes)))
	for OD in ODs:#'LGS_STRAP_45','LGS_100J_130']:
		ao_mode='SH'
		track_band = ['y','z','JHgap','J','H','K'][0]
		savename = 'cendata_%s_ao_%s_band_%s_teff_%s_%smag_fieldr_%sarcsec.npy' %('h2rg',ao_mode,track_band,teff,'H',0.0)
		try: centroid = np.load(datapath + 'centroid_%s'%(savename))
		except: continue
		for itexp in np.arange(len(exptimes)):
			fwhm    = np.load(datapath + 'fwhm_%s'%(savename))
			signal  = np.load(datapath + 'signal_%s'%(savename))
			ifit = np.where(~np.isinf(fwhm))[0]
			sig_per_pix = 10**(-1 * OD) * 0.7 * signal[ifit,itexp]/fwhm[ifit] 
			# interpolate
			f = interpolate.interp1d(magarr[ifit],sig_per_pix,bounds_error=False,fill_value=0)
			mag_fine = np.arange(1,20,0.01)
			magarr_sub = mag_fine[np.where(f(mag_fine) > saturation)[0]]
			try: 
				saturated[itexp] = np.max(magarr_sub)
			except ValueError:
				saturated[itexp] = 0

		p = axs.semilogx(exptimes,saturated,label=ao_mode + ' ' + 'x OD%s'%OD)
		axs.fill_between(exptimes,0,saturated,alpha=0.3,color=p[0].get_color())

	plt.legend()
	plt.xlabel('Exposure Time (s) ')
	plt.ylabel('Magnitude')
	plt.ylim(15,0)
	plt.title(str(teff) + 'K')

	plt.savefig('./output/trackingcamera/centroid_arrs/saturation_run_20230313_teff_%s.png'%teff)

############## RUN FUNCTIONS
def get_track_req(so):
	# compute requirement. requirement is 0.2lambda/D in y band
	yband_wavelength       = 1020 # nm, center of y band
	tracking_requirement_arcsec = 206265 * 0.2 * yband_wavelength / (so.inst.tel_diam*10**9) 
	tracking_requirement_pixel  = tracking_requirement_arcsec/so.track.platescale

	return tracking_requirement_pixel

def get_mag_limit_per_exptime(exptimes,magarr,centroid,tracking_requirement_pixel):
	"""
	"""
	mag_requirement = np.zeros((len(exptimes)))
	for j, exptime in enumerate(exptimes):
		# interpolate curve on higher density grid
		ifit = np.where((~np.isnan(centroid[:,j])) & (centroid[:,j] !=0)& (centroid[:,j] <2))[0]
		if len(ifit) < 3: mag_requirement[j]=np.nan; continue
		magdense = np.arange(np.min(magarr[ifit]),np.max(magarr[ifit]),0.01)
		interp   = interpolate.interp1d(magarr[ifit],centroid[:,j][ifit],kind='quadratic',bounds_error=False,fill_value='extrapolate')
		cendense = interp(magdense)
		try:
			ireq1 = np.where(cendense > tracking_requirement_pixel)[0][0]
			ireq2 = np.where(cendense < tracking_requirement_pixel)[0][-1]
			ireq=ireq1 if np.abs(ireq1-ireq2) <2 else ireq2
		except:
			ireq = 0
		mag_requirement[j] = magdense[ireq]
		if ireq==0 and np.max(cendense) > tracking_requirement_pixel : mag_requirement[j]=np.nan
		if ireq==0 and np.max(cendense) < tracking_requirement_pixel : mag_requirement[j]=np.max(magdense)

		#plt.figure()
		#plt.plot(magarr,centroid[:,j])
		#plt.plot(magarr[ifit],centroid[:,j][ifit])
		#plt.plot(magdense,cendense)
		#plt.title(str(exptime) + '  ' + str(mag_requirement[j]))

	return mag_requirement

def run_mags_exptimes(so,magarr,exptimes,output_path = './output/trackingcamera/centroid_arrs/'):
	"""
	"""
	centroid  = np.zeros((len(magarr),len(exptimes)))
	fwhms     = np.zeros((len(magarr)))
	signal    = np.zeros((len(magarr),len(exptimes)))
	noise     = np.zeros((len(magarr),len(exptimes)))
	strehl    = np.zeros((len(magarr),len(exptimes)))
	for i,mag in enumerate(magarr): # this is the magnitude in filter band
		cload.set_filter_band_mag(so,so.filt.band,so.filt.family,mag,trackonly=True)
		fwhms[i] = so.track.fwhm
		if np.isnan(so.track.noise) or np.isinf(so.track.fwhm): continue
		for j, texp in enumerate(exptimes):
			cload.set_tracking_band_texp(so,so.track.band,texp)
			centroid[i,j] = float(so.track.centroid_err)
			signal[i,j]   = float(so.track.nphot)
			noise[i,j]    = float(so.track.noise)
			strehl[i,j]   = float(so.track.strehl)

	tracking_requirement_pixel  = get_track_req(so)

	# save centroid array for stellar temp, magntiude, ao mode, camera
	savename = 'cendata_%s_ao_%s_band_%s_teff_%s_%smag_fieldr_%sarcsec' %(so.track.camera,so.ao.mode,so.track.band,so.stel.teff,so.filt.band,so.track.field_r)
	np.save(output_path + 'centroid_%s'%savename,centroid)
	np.save(output_path + 'fwhm_%s'%savename,fwhms)
	np.save(output_path + 'snr_%s'%savename,signal/noise)
	np.save(output_path + 'signal_%s'%savename,signal)
	np.save(output_path + 'noise_%s'%savename,noise)
	np.save(output_path + 'magarr',magarr)
	np.save(output_path + 'exptimes',exptimes)

	plt.close('all')

def run_all(so,path,track_bands):
	"""
	step through all params and run tracking cam for each
	"""
	# make folder if doesnt exist
	if not os.path.isdir(path):
		os.makedirs(path)
	else:
		text = ''
		while text!='Y' and text!='N':
			text = input ("WARNING, folder %s already exists. Want to still proceed? y/n: ")
			if text=='y': pass
			if text=='n': pass
			if (text!='y') and (text!='n'): print('input must be y/n')

	#plot_bg_noise(so)
	exptimes = np.array([0.001,0.01,0.1,1,10])
	magarr   = np.arange(1,20,1) # in band of interest
	
	#############
	# Guide Camera
	#############
	# each temp gets own ao mode
	run_dic = {}
	run_dic[1000] = ['LGS_100%s_130','SH']
	run_dic[1500] = ['LGS_100%s_130','SH']
	run_dic[2300] = ['LGS_100%s_130', 'LGS_STRAP_45','SH']
	run_dic[3000] = ['LGS_STRAP_130','LGS_100%s_130','SH']
	run_dic[3600] = ['LGS_STRAP_130','SH']
	run_dic[4200] = ['LGS_STRAP_130','SH']
	teffs = run_dic.keys()

	for ii,teff in enumerate(teffs):
		so.stel.teff=teff
		cload      = fill_data(so)
		#plot_tracking_bands(so)
		#plt.close()
		runK=True
		for hh,ao_mode in enumerate(run_dic[teff]):
			for jj, track_band in enumerate(track_bands):
				if '%s' in ao_mode: # fill in with J or H, or dont do it
					if track_band == 'J' or track_band =='H': so.ao.mode = ao_mode%track_band
					elif track_band=='JHgap': so.ao.mode = ao_mode%'J'
					elif track_band=='K' and runK: so.ao.mode = 'LGS_100H_130';runK=True
					else: so.ao.mode=ao_mode%'J'
				else:
					so.ao.mode = ao_mode
				# reset tracking band
				cload.set_tracking_band_texp(so,track_band,1)
				run_mags_exptimes(so,magarr,exptimes,output_path=path)

if __name__=='__main__':
	#load inputs
	configfile = './configs/hispec_tracking_camera.cfg'
	so         = load_object(configfile)
	cload      = fill_data(so)

	# run
	#datapath = './output/trackingcamera/centroid_arrs/_run_20230313/'
	datestr  = date.today().strftime("%Y%m%d")#'20230406' #date.today().strftime("%Y%m%d")
	datapath = './output/trackingcamera/centroid_arrs/_run_%s_%s/'%(datestr,so.track.camera)
	track_bands = ['y','J','Jplus','JHgap','H']
	run_all(so,datapath,track_bands=track_bands)
	os.system('cp %s %s' %(configfile, datapath)) # copy config file used to data path for record

	#########
	# plot
	datapath   = './output/trackingcamera/centroid_arrs/_run_20230410_h2rg/'
	configfile = datapath + 'hispec_tracking_camera.cfg'
	so         = load_object(configfile)
	cload      = fill_data(so)
	teff       = 1500
	ao_mode = 'LGS_STRAP_130' if teff>3000 else 'LGS_100%s_130'#'LGS_STRAP_130'#
	plot_results_var_final('fwhm',teff,datapath,so.track.camera,track_bands=track_bands)
	#plot_results_mag_req_final(teff,so.track.camera,filt_band='H',track_bands=track_bands,datapath=datapath)
	plot_frame_rates_magnitude(teff,ao_mode,maglimits=[11.9,12.9,13.9,14.9,15.9],filt_band='H',camera=so.track.camera,track_bands=track_bands,datapath=datapath)


