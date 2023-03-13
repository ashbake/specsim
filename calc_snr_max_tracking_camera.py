# calc signal to noise
# max for the calcium H&K
import sys
import matplotlib
import numpy as np
import matplotlib.pylab as plt
from scipy import interpolate
import pandas as pd

font = {'size'   : 14}
matplotlib.rc('font', **font)

sys.path.append('./utils/')
from objects import load_object
from load_inputs import fill_data, load_filter
from functions import tophat,degrade_spec
from noise_tools import get_sky_bg_tracking, get_inst_bg_tracking, sum_total_noise
from throughput_tools import pick_coupling, get_band_mag, get_base_throughput
from wfe_tools import get_tip_tilt_resid, get_HO_WFE
from obs_tools import get_tracking_band, get_tracking_cam, calc_plate_scale
plt.ion()

from matplotlib.ticker import (AutoMinorLocator)

plt.ion()
font = {'size'   : 14}
matplotlib.rc('font', **font)
plt.rcParams['font.size'] = '14'
plt.rcParams['font.family'] = 'sans'
plt.rcParams['axes.linewidth'] = '1.3'
fontname = 'Arial Narrow'

def plot_tracking_bands(so):
	"""
	"""
	spectrum = so.track.ytransmit * so.tel.s * so.stel.s/np.max(so.stel.s) / 1.5
	spec_lores = degrade_spec(so.stel.v[::10], spectrum[::10], 2000)
	star_lores = degrade_spec(so.stel.v[::10], so.stel.s[::10]/np.max(so.stel.s), 2000)

	plt.figure(figsize=(8,5))
	for band in ['z','y','J','JHgap','H','K']:
		print(band)
		bandpass, center_wavelength = get_tracking_band(so.stel.v,band)
		p = plt.plot(so.stel.v[::100],bandpass[::100]*2,linewidth=1)
		plt.fill_between(so.stel.v,-1,bandpass*2,alpha=0.1,facecolor=p[0].get_color(),edgecolor=p[0].get_color())
		if band!='JHgap': plt.text(center_wavelength-10, 1,band,c=p[0].get_color())
		if band=='JHgap': plt.text(center_wavelength-50, 0.95,' JH\nGap',c=p[0].get_color())
	
	# get J band
	Jbandpass, center_wavelength = get_tracking_band(so.stel.v,'J')
	sumflux_J = np.trapz(spectrum[np.where(Jbandpass>0.1)],so.stel.v[np.where(Jbandpass>0.1)])
	for i,band in enumerate(['z','y','J','JHgap','H','K']):
		bandpass, center_wavelength = get_tracking_band(so.stel.v,band)
		sumflux = np.trapz(spectrum[np.where(bandpass>0.1)],so.stel.v[np.where(bandpass>0.1)])
		if i%2==0: plt.text(center_wavelength-50, 0.8,str(round(100*sumflux/sumflux_J,1))+'%',fontsize=10)
		if i%2==1: plt.text(center_wavelength-50, 0.85,str(round(100*sumflux/sumflux_J,1))+'%',fontsize=10)
	
	plt.fill_between([500,970],0,2,alpha=0.1,facecolor='m')
	plt.text(550, 0.95,'Visible\nWFS',c='m')

	plt.plot(so.stel.v[::10],star_lores,'k',zorder=-100,label='T=%sK'%so.stel.teff)
	plt.plot(so.stel.v[::10],spec_lores,'gray',alpha=0.8,zorder=-101,label='Throughput x \n Normalized Flux')
	plt.ylim(0,1.15)
	plt.title('Tracking Camera Filter Profiles')
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Relative Transmission')
	plt.legend(fontsize=10,loc=7)

	plt.savefig('output/trackingcamera/plots/filters/tracking_camera_filter_assumptions_%sK.png'%so.stel.teff,dpi=500)

def plot_tracking_throughput(so):
	"""
	"""
	plt.figure(figsize=(8,5))
	plt.plot(so.track.xtransmit,so.track.ytransmit,'k',zorder=110)

	for band in ['z','y','J','JHgap','H','K']:
		print(band)
		bandpass, center_wavelength = get_tracking_band(so.track.xtransmit,band)
		p = plt.plot(so.track.xtransmit,bandpass*2,linewidth=1)
		plt.fill_between(so.track.xtransmit,-1,bandpass*2,alpha=0.1,facecolor=p[0].get_color(),edgecolor=p[0].get_color())
		if band!='JHgap': plt.text(center_wavelength-10, 1,band,c=p[0].get_color())
		if band=='JHgap': plt.text(center_wavelength-50, 0.95,' JH\nGap',c=p[0].get_color())
		avg_th = np.mean(so.track.ytransmit[np.where(bandpass>.1)[0]])
		plt.text(center_wavelength-50, 0.8,str(int(100*avg_th))+'%',c=p[0].get_color())
		
	plt.fill_between([500,970],0,2,alpha=0.1,facecolor='m')
	plt.text(550, 0.95,'Visible\nWFS',c='m')

	plt.ylim(0,1.15)
	plt.title('Tracking Camera Filter Profiles')
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Relative Transmission')

	plt.savefig('output/trackingcamera/plots/tracking_camera_throughput_%s.png'%so.track.camera,dpi=200)

def plot_stellar_colors():
	f = pd.read_csv('./data/WFE/HAKA/color_curves.csv',delimiter='\t')
	
	plt.figure()
	bands = f['Temp'].values
	for key in f.keys():
		if key =='Temp':continue
		if key=='2500':continue
		if key=='3800':continue
		p = plt.plot(bands,f[key]- f[key][6],label=key)
		plt.text(0, f[key][0]- f[key][6] ,key+'K', c=p[0].get_color())
	
	#plt.legend(fontsize=12)
	plt.xlim(-1,len(bands))
	plt.xlabel('Band')
	plt.ylabel('Band - H')
	plt.ylim(18,-2)
	plt.subplots_adjust(bottom=0.15,left=0.15)
	plt.grid()
	plt.savefig('./output/stellar_colors_H.png')

def plot_tracking_cam_spot_rms(camera='h2rg'):
	"""
	"""
	#f = np.loadtxt('./data/WFE/trackingcamera_optics/OAP1_HISPEC_FEI_RMS_SpotRvsField.txt')
	f = np.loadtxt('./data/WFE/trackingcamera_optics/HISPEC_ParaxialTel_OAP_TrackCamParax_SpotSizevsField.txt')
	field, rmstot, rms900,rms1000,rms1200,rms1400,rms1600,rms2200  = f.T #field [deg], rms [um]
	_,pixel_pitch,_,_ = select_tracking_cam(camera=camera)
	plt.figure()
	# multiply rms by sqrt (2) to get a diagonal cut, multiple by 2 to get diameter
	plt.plot(field*3600,np.sqrt(2) * 2*rmstot/pixel_pitch,label='total') 
	plt.plot(field*3600,np.sqrt(2) * 2*rms900/pixel_pitch,label='900nm')
	plt.plot(field*3600,np.sqrt(2) * 2*rms2200/pixel_pitch,label='2200nm')
	plt.xlabel('Field [arcsec]')
	plt.ylabel('RMS Diameter [pix]')
	plt.title('Tracking Camera Spot RMS')
	plt.legend()

def plot_centroiderr_vmag(so,magarr,exptimes,centroid,tracking_requirement_pixel=0.4):
	plt.figure(figsize=(8,6))
	ax = plt.axes()
	for j, exptime in enumerate(exptimes):
		plt.semilogy(magarr,centroid[:,j],label='$t_{exp}$='+ str(exptime) + 's')

	plt.plot(magarr,np.ones_like(magarr)*tracking_requirement_pixel,'k--',label='Requirement')
	plt.fill_between(magarr,np.ones_like(magarr)*tracking_requirement_pixel,facecolor='green',alpha=0.1)

	plt.legend(loc=6)
	plt.title('Cam: %s Band: '%so.track.camera + so.track.band + ' Temp: %sK\nAO: %s'%(int(so.stel.teff),so.ao.mode))
	plt.xlabel('%s Magnitude' %so.filt.band)
	plt.ylabel('Centroid Error [pixels]')
	plt.subplots_adjust(bottom=0.15,top=0.85)
	plt.grid(color='gray',linestyle='--',dash_joinstyle='miter')
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	#ax.yaxis.set_minor_locator(AutoMinorLocator())
	ax.tick_params(which='minor',length=4,color='gray',direction="in")
	plt.xlim(np.min(magarr),np.max(magarr))
	plt.ylim(1e-5,2)
	plt.plot([15,15],[1e-5,2],'k--')
	savename = 'centroid_err_mag_summary_cam_%s_ao_%s_band_%s_texp_%s_teff_%s.png' %(so.track.camera,so.ao.mode,so.track.band,exptimes[j],so.stel.teff)
	plt.savefig('output/trackingcamera/plots/centroid_error/' + savename)
	return ax

def plot_bg_noise(so):
	"""
	"""
	col_table = plt.get_cmap('Spectral_r')
	fig, axs = plt.subplots(3,figsize=(7,9),sharex=True)
	plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.85,top=0.85)

	axs[0].plot([950,2400],[0.5,0.5],'k--',lw=0.7)
	axs[0].fill_between([1450,2400],0,1000,facecolor='gray',alpha=0.2)
	axs[0].fill_between([980,1330],0,1000,facecolor='gray',alpha=0.2)
	axs[0].grid('True')
	axs[0].set_xlim(950,2400)
	axs[0].set_ylabel('Sky Background \n(phot/nm/s)')
	axs[1].set_ylabel('Instrument Background \n(phot/nm/s)')
	axs[2].set_ylabel('Source Photon Noise \n (phot/nm/s)')
	axs[2].set_xlabel('Wavelength [nm]')
	axs[0].set_ylim(-0,1000)
	axs[1].set_ylim(-0,40)
	axs[0].plot(so.stel.v,so.track.sky_bg_spec,'b',alpha=0.5,zorder=100,label='Sky Background')
	axs[1].plot(so.stel.v,so.track.inst_bg_spec,'m',lw=2,alpha=0.5,zorder=100,label='Instrument Background')
	axs[2].plot(so.stel.v,np.sqrt(so.track.signal_spec/so.track.texp),'g',alpha=0.5,zorder=100,label='Instrument Background')

	#ax2.fill_between(so.filt.v,so.filt.s,facecolor='gray',edgecolor='black',alpha=0.2)
	#ax2.set_ylabel('Filter Response')
	# plot band
	for ax in axs:
		ax2 = ax.twinx()
		ax2.fill_between(so.inst.y,0,1,facecolor='k',edgecolor='black',alpha=0.1)
		ax2.text(20+np.min(so.inst.y),0.9, 'y')
		ax2.fill_between(so.inst.J,0,1,facecolor='k',edgecolor='black',alpha=0.1)
		ax2.text(50+np.min(so.inst.J),0.9, 'J')
		ax2.fill_between(so.inst.H,0,1,facecolor='k',edgecolor='black',alpha=0.1)
		ax2.text(50+np.min(so.inst.H),0.9, 'H')
		ax2.fill_between(so.inst.K,0,1,facecolor='k',edgecolor='black',alpha=0.1)
		ax2.text(50+np.min(so.inst.K),0.9, 'K')
		ax2.set_ylim(0,1)

	axs[0].set_title('Tracking Camera Noise \n  %s mag = %s, Teff=%sK '%(so.filt.band,so.stel.mag,int(so.stel.teff)))
	plt.savefig('./output/trackingcamera/noise_flux_%sK_%s_%smag.pdf'%(so.stel.teff,so.filt.band,so.stel.mag))
	plt.savefig('./output/trackingcamera/noise_flux_%sK_%s_%smag.png'%(so.stel.teff,so.filt.band,so.stel.mag))

def plot_signal_noise_fwhm(so,magarr,exptimes,signal,noise,fwhms,strehl,centroid):
	"""
	"""
	j=2
	# set up plot for mag and texp

	fig, axs = plt.subplots(3,figsize=(6,7),sharex=True)

	axs[0].semilogy(magarr,signal[:,j],'k',lw=4,label='signal')
	#axs[0].semilogy(magarr,signal[:,j]*strehl[:,j],'k-',lw=2,label='signal*strehl')
	axs[0].semilogy(magarr,noise[:,j],'gray',lw=4,label='noise')
	axs[0].semilogy(magarr,np.sqrt(signal[:,j]),'m-.',c='orange',label='shot noise')
	axs[0].grid('True')
	axs[0].legend(fontsize=10)
	axs[0].set_ylabel('Counts (e-)')
	axs[0].set_title('Cam: %s Band: %s AO: %s'%(so.track.camera,so.track.band,so.ao.mode) \
			+ '\nTemp: %sK ($t_{exp}$=%ss)'%(int(so.stel.teff),exptimes[j]))

	axs[1].plot(magarr,fwhms,lw=2)
	diffraction_spot_arcsec = 206265 * so.track.center_wavelength/ (so.inst.tel_diam * 10**9) # arcsec
	diffraction_spot_pix = diffraction_spot_arcsec / so.track.platescale
	axs[1].grid('True')
	axs[1].set_ylabel('FWHM [pixels]')
	axs[1].plot(magarr,np.ones_like(magarr)*diffraction_spot_pix/(strehl[:,j]**(1/4)),'k--',label='Diffraction Size/Strehl^(1/4)')
	axs[1].legend()

	axs[2].plot(magarr,centroid[:,j],'g',lw=2)
	axs[2].set_ylabel('Centroid Error (Pix)')
	axs[2].set_ylim(0,4)
	axs[2].grid('True')
	axs[2].set_xlabel('%s Magnitude'%so.filt.band)
	tracking_requirement_pixel = get_track_req(so)
	axs[2].plot(magarr,np.ones_like(magarr)*tracking_requirement_pixel,'k--',label='Requirement')
	axs[2].legend()

	plt.subplots_adjust(hspace=0.1)

	savename = 'snr_fwhm_summary_cam_%s_ao_%s_band_%s_texp_%s_teff_%s.png' %(so.track.camera,so.ao.mode,so.track.band,exptimes[j],so.stel.teff)
	plt.savefig('output/trackingcamera/plots/snr/' + savename)

def plot_mag_limit():
	"""
	OLD
	mag limit vs exp time
	"""
	iplot = np.where(mag_requirement!=0) # dont plot 0s where strehl cant be calculated
	ax.semilogx(exptimes[iplot],mag_requirement[iplot],'--',label='%s Tracking'%track_band)
	ax.semilogx(exptimes[iplot],mag_requirement[iplot],'o',c='k')
	#ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
	#ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
	my_xticks = ['0.001','0.01','0.1','1','10']
	plt.xticks(exptimes, my_xticks)

	#plt.fill_between(exptimes,mag_requirement,y2=15 * np.ones_like(exptimes),facecolor='green',alpha=0.2)
	# label magnitudes
	#for j, exptime in enumerate(exptimes):
	#	plt.text(exptime*0.9,mag_requirement[j]*.93,str(round(mag_requirement[j],1)))

	plt.axhline(15,c='k',linestyle='--')

	#plt.title('Cam: %s Band: '%camera + track_band + ' Temp: %sK'%(int(so.var.teff)))
	plt.title('Cam: %s, Temp: %sK'%(camera,int(so.stel.teff)))
	plt.xlabel('Exposure Time [s]')
	plt.ylabel('%s Magnitude Limit' %so.filt.band)
	plt.subplots_adjust(bottom=0.15,top=0.85)
	plt.grid(color='gray',linestyle='--',dash_joinstyle='miter')
	#ax.xaxis.set_minor_locator(AutoMinorLocator())
	#ax.yaxis.set_minor_locator(AutoMinorLocator())
	#ax.tick_params(which='minor',length=4,color='gray',direction="in")
	plt.legend()
	plt.ylim(4,18)
	plt.fill_between(exptimes,15,y2=18,facecolor='green',alpha=0.2)
	plt.savefig('./output/trackingcamera/limiting_mag_T%sK.png'%so.var.teff,dpi=1000)

def plot_results_mag_req(teff,filt_band='H'):
	"""
	"""
	exptimes = np.array([0.001,0.01,0.1,1,10])
	magarr   = np.arange(1,20,2) # in band of interest
	tracking_requirement_pixel  = get_track_req(so)
	modes    = ['100JH','80J','80H','100K','SH','LGS_100H_130','LGS_100J_130','LGS_100J_45','LGS_STRAP_130','LGS_STRAP_45']
	modes2   = ['NGS', 'NGS', 'NGS','NGS','NGS','','','','','']
	linestyles = ['-','-','-','-','-','--','--','--','-.','-.']
	colors   = ['m','b','orange','gray','g','r','c','b','gray','black']
	widths   = [1, 1.5,  1,    1, 1.5,2, 1.5,1.5,  1, 1]
	#filt_band = 'H'
	#teff = 2300
	
	fig, axs = plt.subplots(3,2,figsize=(6,7),sharex=True,sharey=True)
	
	for ii,track_band in enumerate(['y','z','JHgap','J','H','K']):
		for jj,ao_mode in enumerate(modes):
			savename = 'cendata_%s_ao_%s_band_%s_teff_%s_%smag_fieldx_%sarcsec_fieldy_%sarcsec.npy' %('h2rg',ao_mode,track_band,teff,'H',0.0,0.0)
			try: centroid = np.load('./output/centroid_data/centroid_%s'%savename)
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
	for ii,track_band in enumerate(['y','z','JHgap','J','H','K']):
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

def plot_results(teff,itexps=[1,2,3],filt_band='H'):
	"""
	"""
	exptimes = np.array([0.001,0.01,0.1,1,10])
	magarr   = np.arange(1,20,2) # in band of interest
	tracking_requirement_pixel  = get_track_req(so)
	#teff = 1500
	modes    = ['100JH','80J','80H','100K','SH','LGS_100H_130','LGS_100J_130','LGS_100J_45','LGS_STRAP_130','LGS_STRAP_45']
	modes2   = ['NGS', 'NGS', 'NGS','NGS','NGS','','','','','']
	linestyles = ['-','-','-','-','-','--','--','--','-.','-.']
	colors   = ['m','b','orange','gray','g','r','c','b','gray','black']
	widths   = [1, 1.5,  1,    1, 1.5,2, 1.5,1.5,  1, 1]
	#itexp=4

	fig, axs = plt.subplots(3,2,figsize=(6,7),sharex=True,sharey=True)
	for itexp in itexps:
		for ii,track_band in enumerate(['y','z','JHgap','J','H','K']):
			for jj,ao_mode in enumerate(modes):
				savename = 'cendata_%s_ao_%s_band_%s_teff_%s_%smag.npy' %('h2rg',ao_mode,track_band,teff,'H')
				try: centroid = np.load('./output/centroid_data/centroid_%s'%savename)
				except: continue
				fwhm = np.load('./output/centroid_data/fwhm_%s'%savename)
				snr  = np.load('./output/centroid_data/snr_%s'%savename)
				ifit = np.where((~np.isnan(centroid[:,itexp])) &(centroid[:,itexp]< 3) & (centroid[:,itexp]!=0))[0]
				axs[ii%3,ii//3].semilogy(magarr[ifit],centroid[:,itexp][ifit],c=colors[jj],linestyle=linestyles[jj],label=ao_mode)
			axs[ii%3,ii//3].set_title('%s'%(track_band))
		plt.suptitle('T=%sK  '%teff +  '$t_{exp}$=%ss'%exptimes[itexps])

	axs[2,0].set_xlabel('%s Magnitude'%filt_band)
	axs[2,1].set_xlabel('%s Magnitude'%filt_band)
	#axs[1].legend(bbox_to_anchor=(1,1))
	plt.subplots_adjust(bottom=0.15,left=0.15,top=0.9,right=0.9)
	for ii,track_band in enumerate(['y','z','JHgap','J','H','K']):
		axs[ii%3,ii//3].axhline(0.39,c='k',linestyle='--')
		axs[ii%3,ii//3].axvline(15,c='k',linestyle='--')
		if ii//3==0: axs[ii%3,ii//3].set_ylabel('$\sigma_{cen}$ [pix]')
		axs[ii%3,ii//3].grid(color='gray',linestyle='--',dash_joinstyle='miter')
		axs[ii%3,ii//3].set_ylim(0.0005,1)
		axs[ii%3,ii//3].fill_between(magarr,0,y2=0.39,facecolor='green',alpha=0.2)
		axs[ii%3,ii//3].fill_between(magarr,0.39,y2=1,facecolor='red',alpha=0.2)
		axs[ii%3,ii//3].set_xlim(0,20)
		xticks = [0,5,10,15,20]
		axs[ii%3,ii//3].set_xticks(xticks)
		axs[ii%3,ii//3].set_xticklabels(xticks,fontsize=12)

	plt.savefig('./output/trackingcamera/plots/raw/raw_Teff_%sK_band_%s.png'%(teff,filt_band))

def plot_results_bybandpass(teff,ao_mode,filt_band='H'):
	"""
	"""
	exptimes = np.array([0.001,0.01,0.1,1,10])
	magarr   = np.arange(1,20,2) # in band of interest
	tracking_requirement_pixel  = get_track_req(so)

	camera='h2rg'
	syms = ['o','s','d','.','x','^']
	ao_mode = 'LGS_STRAP_45';linestyle='-'
	teff=4200
	#ao_mode = 'LGS_100%s_130';linestyle='--'

	fig, ax = plt.subplots(1,1,figsize=(6,5),sharex=True,sharey=True)
	for ii,track_band in enumerate(['z','y','JHgap','J','H','K']):
		if '%s' in ao_mode:
			if track_band == 'J' or track_band =='H': fill=track_band
			#elif track_band=='JHgap': fill = 'J'
			#elif track_band=='z' or track_band=='y': fill='I'
			elif track_band=='K': fill='H'
			else: fill='J'
			savename = 'cendata_%s_ao_%s_band_%s_teff_%s_%smag.npy' %(camera,ao_mode%fill,track_band,teff,'H')
		else:
			savename = 'cendata_%s_ao_%s_band_%s_teff_%s_%smag.npy' %(camera,ao_mode,track_band,teff,'H')
		try: centroid = np.load('./output/centroid_data/centroid_%s'%savename)
		except: continue
		try:	mag_req = get_mag_limit_per_exptime(exptimes,magarr,centroid,tracking_requirement_pixel)
		except: mag_req = np.zeros_like(exptimes)
		fwhm = np.load('./output/centroid_data/fwhm_%s'%savename)
		snr  = np.load('./output/centroid_data/snr_%s'%savename)
		#if track_band=='y' and ao_mode.startswith('LGS_100'): mag_req-=0.69# 20%y band flux, star is 2.5 mags fainter
		#if track_band=='z' and ao_mode.startswith('LGS_100'): mag_req-=1.6# 3% z band flux, star is 2.5 mags fainter
		#if track_band=='JHgap' and ao_mode.startswith('LGS_100'): mag_req-=.82 # 3% z band flux, star is 2.5 mags fainter
		p = ax.semilogx(exptimes,mag_req,linestyle=linestyle,marker=syms[ii],label=track_band)
		exptimes_dense = np.arange(np.min(exptimes),np.max(exptimes),0.00001)
		ftemp = interpolate.interp1d(exptimes,mag_req)
		mag_req_dense = ftemp(exptimes_dense)
		try: 
			print(track_band, 1/exptimes_dense[np.where(mag_req_dense>14.9)][0])
		except:
			print(track_band)
			#plt.plot(exptimes,mag_req,syms[ii],c=p[0].get_color())

	ax.set_title('%s'%(ao_mode))
	plt.suptitle('T=%sK Cam:%s '%(teff,camera))

	ax.legend()
	ax.set_xlabel('Exposure Time [s]')
	plt.subplots_adjust(bottom=0.15,left=0.15,top=0.85,right=0.9)
	#ax.axhline(15,c='k',linestyle='--')
	ax.set_ylabel('H Mag Limit')
	ax.grid(color='gray',linestyle='--',dash_joinstyle='miter')
	ax.fill_between(exptimes,15,y2=20,facecolor='green',alpha=0.2)
	yticks = [0,5,10,15,20]
	ax.set_yticks(yticks)
	ax.set_yticklabels(yticks,fontsize=12)
	ax.set_ylim(0,20)

	plt.savefig('./output/trackingcamera/plots/mag_req/per_filter_ao_mode_%s_Teff_%sK_cam_%s.png'%(ao_mode,teff,camera))

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

def plot_results_fwhm(teff):
	"""
	plot fwhm for each mode
	"""
	exptimes = np.array([0.001,0.01,0.1,1,10])
	magarr   = np.arange(1,20,2) # in band of interest
	tracking_requirement_pixel  = get_track_req(so)
	#teff = 2300
	modes    = ['100JH','80J','80H','100K','SH','LGS_100H_130','LGS_100J_130','LGS_100J_45','LGS_STRAP_130','LGS_STRAP_45']
	modes2   = ['NGS', 'NGS', 'NGS','NGS','NGS','','','','','']
	linestyles = ['-','-','-','-','-','--','--','--','-.','-.']
	colors   = ['m','b','orange','gray','g','r','c','b','gray','black']
	widths   = [1, 1.5,  1,    1, 1.5,2, 1.5,1.5,  1, 1]
	itexp=4

	fig, axs = plt.subplots(3,2,figsize=(6,7),sharex=True,sharey=True)

	for ii,track_band in enumerate(['y','z','JHgap','J','H','K']):
		for jj,ao_mode in enumerate(modes):
			savename = 'cendata_%s_ao_%s_band_%s_teff_%s_%smag_fieldx_%sarcsec_fieldy_%sarcsec.npy' %('h2rg',ao_mode,track_band,teff,'H',0.0,0.0)
			try: centroid = np.load('./output/centroid_data/centroid_%s'%savename)
			except: continue
			fwhm = np.load('./output/centroid_data/fwhm_%s'%(savename))
			snr  = np.load('./output/centroid_data/snr_%s'%(savename))
			ifit = np.where((~np.isnan(centroid[:,itexp])))[0]
			axs[ii%3,ii//3].plot(magarr[ifit],fwhm[ifit],c=colors[jj],linestyle=linestyles[jj],label=ao_mode)
			#axtmp = axs[ii%3,ii//3].twinx()
			#axtmp.semilogy(magarr[ifit],snr[:,itexp][ifit],c=colors[jj],linestyle=linestyles[jj],label=ao_mode)
		axs[ii%3,ii//3].set_title('%s'%(track_band))
	plt.suptitle('T=%sK  '%teff )

	diffraction_limits = [1.7, 2, 2.2, 2.4, 3.2, 4.2] # to match the bandpass

	axs[2,0].set_xlabel('%s Magnitude'%so.filt.band)
	axs[2,1].set_xlabel('%s Magnitude'%so.filt.band)
	#axs[1].legend(bbox_to_anchor=(1,1))
	plt.subplots_adjust(bottom=0.15,top=0.9,right=0.9)
	for ii,track_band in enumerate(['y','z','JHgap','J','H','K']):
		axs[ii%3,ii//3].axhline(diffraction_limits[ii],c='k',linestyle='-')
		axs[ii%3,ii//3].axvline(15,c='k',linestyle='--')
		if ii//3==0: axs[ii%3,ii//3].set_ylabel('FWHM (pix)')
		axs[ii%3,ii//3].grid(color='gray',linestyle='--',dash_joinstyle='miter',alpha=0.3)
		axs[ii%3,ii//3].set_ylim(1,8)
	
	plt.savefig('./output/trackingcamera/plots/fwhm/fwhm_Teff_%sK_band_%s.png'%(teff,filt_band))

def plot_results_snr():
	"""
	"""
	exptimes = np.array([0.001,0.01,0.1,1,10])
	magarr   = np.arange(5,17,1) # in band of interest
	tracking_requirement_pixel  = 0.39#get_track_req(so)
	teff = 2300
	ao_mode = 'LGS_100H_45'
	modes    = ['100JH','80J','80H','100K','SH','LGS_100H_130','LGS_100J_130','LGS_100J_45','LGS_STRAP_130','LGS_STRAP_45']
	modes2   = ['NGS', 'NGS', 'NGS','NGS','NGS','','','','','']
	linestyles = ['-','-','-','-','-','--','--','--','-.','-.']
	colors   = ['m','b','orange','gray','g','r','c','b','gray','black']
	widths   = [1, 1.5,  1,    1, 1.5,2, 1.5,1.5,  1, 1]
	itexp=4
	filt_band='H' # matches the run
	run = '_run2_20230209'

	fig, axs = plt.subplots(3,2,figsize=(7,7),sharex=True,sharey=True)

	for ii,track_band in enumerate(['y','z','JHgap','J','H','K']):
		for jj,ao_mode in enumerate(modes):
			savename = 'cendata_%s_ao_%s_band_%s_teff_%s_%smag.npy' %('h2rg',ao_mode,track_band,teff,'H')
			try: centroid = np.load('./output/centroid_data/%s/centroid_%s'%(run,savename))
			except: continue
			fwhm = np.load('./output/centroid_data/%s/fwhm_%s'%(run,savename))
			snr  = np.load('./output/centroid_data/%s/snr_%s'%(run,savename))
			ifit = np.where((~np.isnan(centroid[:,itexp])) &(centroid[:,itexp]< 3))[0]
			axs[ii%3,ii//3].plot(magarr[ifit],snr[:,itexp][ifit],c=colors[jj],linestyle=linestyles[jj],label=ao_mode)
		axs[ii%3,ii//3].set_title('%s'%(track_band))
	plt.suptitle('T=%sK  '%teff +  '$t_{exp}$=%ss'%exptimes[itexp])


	axs[2,0].set_xlabel('%s Magnitude'%filt_band)
	axs[2,1].set_xlabel('%s Magnitude'%filt_band)
	#axs[1].legend(bbox_to_anchor=(1,1))
	plt.subplots_adjust(bottom=0.15,top=0.9,right=0.65)
	for ii,track_band in enumerate(['y','z','JHgap','J','H','K']):
		axs[ii%3,ii//3].axhline(0.39,c='k',linestyle='--')
		axs[ii%3,ii//3].axvline(15,c='k',linestyle='--')
		if ii//3==0: axs[ii%3,ii//3].set_ylabel('Centroid Error (pix)')
		axs[ii%3,ii//3].grid(color='gray',linestyle='--',dash_joinstyle='miter')
		axs[ii%3,ii//3].set_ylim(0.001,1)
		axs[ii%3,ii//3].fill_between(magarr,0,y2=0.39,facecolor='green',alpha=0.2)
		axs[ii%3,ii//3].fill_between(magarr,0.39,y2=1,facecolor='red',alpha=0.2)

##############
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

def run_mags_exptimes(so,magarr,exptimes):
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

	#PLOT			f = pd.read_csv(so.ao.ho_wfe_set,header=[0,1])

	tracking_requirement_pixel  = get_track_req(so)
	ax = plot_centroiderr_vmag(so,magarr,exptimes,centroid,tracking_requirement_pixel=tracking_requirement_pixel)
	ax = plot_signal_noise_fwhm(so,magarr,exptimes,signal,noise,fwhms,strehl,centroid)
	#mag_requirement = fit_mag_limit_per_exptime(exptimes,magarr,centroid,tracking_requirement_pixel)

	# save centroid array for stellar temp, magntiude, ao mode, camera
	savename = 'cendata_%s_ao_%s_band_%s_teff_%s_%smag_fieldx_%sarcsec_fieldy_%sarcsec' %(so.track.camera,so.ao.mode,so.track.band,so.stel.teff,so.filt.band,so.track.field_x,so.track.field_y)
	np.save('./output/centroid_data/centroid_%s'%savename,centroid)
	#np.save('./output/centroid_data/mag_req_%s'%savename,mag_requirement)
	np.save('./output/centroid_data/fwhm_%s'%savename,fwhms)
	np.save('./output/centroid_data/snr_%s'%savename,signal/noise)

	plt.close('all')



if __name__=='__main__':
	#load inputs
	configfile = 'hispec_tracking_camera.cfg'
	so         = load_object(configfile)
	cload      = fill_data(so)

	# note that i changed the TT residual=FWHM assumption to be FWHM= 1/.44 *tt 
	# which will make things slightly worse and need to rerun post PDR
	# will only change slightly bc where things blow up is where the cutoff is
	# not huge changes by factor of 2ish in TT since it's small portion of FWHM

	#plot_bg_noise(so)
	exptimes = np.array([0.001,0.01,0.1,1,10])
	magarr   = np.arange(1,20,2) # in band of interest
	#cameras   = ['h2rg','cred-2'] # only do cred2 for one mode and teff
	
	#############
	# Guide Camera
	#############
	# each track band gets own mag to check
	track_bands = ['z','y','J','JHgap','H','K']
	if so.track.camera=='cred2': track_bands=['y','J','JHgap','H']
	
	# each temp gets own ao mode
	run_dic = {}
	run_dic[1000] = ['LGS_100%s_130','80%s','SH']
	run_dic[1500] = ['LGS_100%s_130','80%s','SH']
	run_dic[2300] = ['LGS_100%s_130', 'LGS_STRAP_45','80%s','SH']
	run_dic[3000] = ['LGS_STRAP_45','LGS_100%s_130','80%s','SH']
	run_dic[3600] = ['LGS_STRAP_45','LGS_100%s_130','80%s','SH']
	run_dic[4200] = ['LGS_STRAP_45','80%s','SH']
	run_dic[5800] = ['LGS_STRAP_45','SH']
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
					elif track_band=='z' or track_band=='y' and so.ao.mode.startswith('LGS'): so.ao.mode = ao_mode%'J'
					#elif track_band=='K' and not runK: so.ao.mode = '100K';runK=True # hack what to run for K band
					elif track_band=='K' and runK: so.ao.mode = 'LGS_100H_130';runK=True
					else: so.ao.mode=ao_mode%'J'
				else:
					so.ao.mode = ao_mode
				# reset tracking band
				cload.set_tracking_band_texp(so,track_band,1)
				run_mags_exptimes(so,magarr,exptimes)


