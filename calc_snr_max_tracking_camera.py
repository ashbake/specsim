# calc signal to noise
# max for the calcium H&K
import sys
import matplotlib
import numpy as np
import matplotlib.pylab as plt
from scipy import interpolate

font = {'size'   : 14}
matplotlib.rc('font', **font)

sys.path.append('./utils/')
from objects import load_object
from load_inputs import fill_data, load_filter
from functions import tophat
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


def run_one(so,mag,texp,band='JHgap',camera='h2rg'):
	"""
	"""
	nphot = get_signal(so,mag,texp,band,camera)

def get_signal(so,mag,texp,band,camera):
	"""
	get tracking camera centroid for one frame
	"""
	# SIGNAL
	_, _, qe_mod,_ = get_tracking_cam(camera=camera)
	bandpass,_     = get_tracking_band(so,band)

	s_ccd_hires = so.stel.s*10**(-0.4*(mag - so.stel.mag))* texp *\
		 			so.inst.tel_area * so.track.ytransmit*qe_mod*\
		 			np.abs(so.tel.s)**so.tel.airmass
	
	total_photons = np.trapz(s_ccd_hires * bandpass,so.stel.v)

	return total_photons



def get_mag_limit_per_exptime(exptimes,magarr,centroid,tracking_requirement_pixel):
	"""
	"""
	magdense = np.arange(np.min(magarr),np.max(magarr)-1,0.01)
	mag_requirement = np.zeros((len(exptimes)))
	for j, exptime in enumerate(exptimes):
		# interpolate curve on higher density grid
		interp   = interp1d(magarr,centroid[:,j])
		cendense = interp(magdense)
		try:
			ireq = np.where(cendense > tracking_requirement_pixel)[0][0]
		except:
			ireq = 0
		mag_requirement[j] = magdense[ireq]

	return mag_requirement


def plot_tracking_bands():
	"""
	"""
	plt.figure(figsize=(8,5))
	for band in ['z','y','J','JH gap','H','K']:
		print(band)
		if band=='z':
			l0,lf = 800,950
			x = np.arange(l0-5,lf+5)
			throughput = 1
			bandpass = tophat(x,l0,lf,throughput) #make up fake band

		if band=='y':
			l0,lf = 980,1100
			x = np.arange(l0-5,lf+5)
			center_wavelength = 1050
			throughput = 1
			bandpass = tophat(x,l0,lf,throughput) #make up fake band

		if band=='JH gap':
			l0,lf= 1335,1485
			x = np.arange(l0-5,lf+5)
			throughput = 1
			bandpass = tophat(x,l0,lf,throughput)

		if band=='J' or band=='H' or band=='K':
			# take 20%
			x,bandpass = load_filter(so,'Johnson',band)
			bandpass[np.where(bandpass>0.2)] = 0.2
			band  = '20% ' + band
		#plt.fill_between(so.stel.v,bandpass,alpha=0.8,label=band,edgecolor='k')
		plt.plot(x,bandpass,linewidth=3,label=band)


	#for band in ['R','I','J','H','K']:
	#	x,y = load_filter(so,'Johnson',band)
	#	plt.plot(x,y,alpha=0.5,c='k')
	#	if band=='R': 
	#		plt.fill_between(x,y,alpha=0.1,facecolor='k',label='Generic \nJohnson \nRIJHK')
	#	else:
	#		plt.fill_between(x,y,alpha=0.1,facecolor='k')
	x,y = load_filter(so,'Johnson','V')
	plt.plot(x,y,alpha=0.5,c='k')
	plt.fill_between(x,y,alpha=0.1,facecolor='m',label='Johnson V')
	
	#plt.plot(so.stel.v,so.tel.s * so.stel.s/np.max(so.stel.s),'gray',alpha=0.5,zorder=-100)
	spectrum = so.hispec.ytransmit * so.tel.s * so.stel.s/np.max(so.stel.s)
	spec_lores = degrade_spec(so.stel.v[::10], spectrum[::10], 2000)
	star_lores = degrade_spec(so.stel.v[::10], so.stel.s[::10]/np.max(so.stel.s), 2000)
	plt.plot(so.stel.v[::10],star_lores,'k',zorder=-100,label='T=%sK'%so.var.teff)
	plt.plot(so.stel.v[::10],spec_lores,'gray',alpha=0.8,zorder=-101,label='Throughput x \n Normalized Flux')
	plt.title('Tracking Camera Filter Profiles')
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Transmission')
	plt.legend(fontsize=10,loc=1)

	plt.savefig('output/trackingcamera/tracking_camera_filter_assumptions_%sK.png'%so.var.teff,dpi=500)

def plot_psf_size_vs_mag():
	"""
	"""
	#compute platescale 
	platescale_arcsec_um = 206265 / fratio / (D * 10**6) #arc/um
	platescale_arcsec_pix = platescale_arcsec_um * pitch

	sizes = []
	magarr=np.arange(0,20)
	wavelengths = 1e3*np.array([0.9,1.02,1.23,1.63,2.19])
	for wv in wavelengths:
		sizes = []
		for mag in magarr:
			D   = 10 #meters
			wfe = get_wfe(Rmag) #
			strehl = np.exp(-(2*np.pi*wfe/wv)**2)

			diffraction_spot_arcsec = 206265 * wv/ (D * 10**9) # arcsec
			diffraction_spot_pix = diffraction_spot_arcsec / platescale_arcsec_pix
			sizes.append(diffraction_spot_pix / strehl**(1/4)) # 1/strehl**.25 from dimitri, to account for broadening deviation from diffraction limit
		
		plt.figure('mag plot')
		plt.plot(magarr,sizes,label=wv)
	plt.xlabel('Magnitude')
	plt.ylabel('PSF FWHM [pixels]')
	plt.legend()
	plt.title('Tracking Camera Spot Sizes')
	plt.xlim(0,18)
	plt.ylim(0,15)
	plt.savefig('./output/psf_vs_mag.png')

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

def plot_snr_and_fwhm():
	# plot noise breakdown related
	j = -1  # corresponds to one second
	npix = np.pi * fwhm**2
	rn, pitch, qe, dark = select_tracking_cam(camera=camera)
	shot = np.sqrt(signal)
	readnoise = np.sqrt(npix * rn**2)
	shotdark = np.sqrt(npix* exptime*dark)

	#plt.figure(figsize=(7,4))
	fig, axs = plt.subplots(2,figsize=(6,8),sharex=True)

	axs[0].semilogy(magarr,signal[:,j],'k',lw=4,label='signal')
	axs[0].semilogy(magarr,signal[:,j]*strehl[:,j],'k-',lw=2,label='signal*strehl')
	axs[0].semilogy(magarr,noise[:,j],'gray',lw=4,label='noise')
	axs[0].semilogy(magarr,shot[:,j],'m-.',c='orange',label='shot noise')
	axs[0].semilogy(magarr,readnoise[:,j],'--',c='m',label='read noise')
	axs[0].semilogy(magarr,shotdark[:,j],'g-',label='dark')
	axs[0].grid('True')
	axs[0].legend(fontsize=10)
	axs[0].set_ylabel('Counts (e-)')
	axs[0].set_title('Cam: %s Band: '%camera + track_band + ' Temp: %sK ($t_{exp}$=1s)'%(int(so.var.teff)))

	axs[1].semilogy(magarr,fwhm[:,j],label=exptime)
	axs[1].grid('True')
	axs[1].set_ylabel('FWHM [pixels]')
	axs[1].set_xlabel('%s Magnitude'%so.filt.band)
	axs[1].yaxis.set_minor_formatter(mticker.ScalarFormatter())
	axs[1].yaxis.set_major_formatter(mticker.ScalarFormatter())

	plt.subplots_adjust(bottom=0.15,hspace=0,left=0.15)

def plot_wfe():
	wfes = []
	for Rmag in np.arange(20):
		wfes.append(get_wfe(Rmag))

	wfes = np.array(wfes)
	wavelength = 1630 # nm, H band
	strehl = np.exp(-(2*np.pi*wfes/wavelength)**2)

	plt.figure()
	plt.plot(np.arange(20),strehl)
	plt.show()
	plt.xlabel('Rmag')
	plt.ylabel('H-band SR')

def plot_centroiderr_vmag():
	plt.figure(figsize=(8,6))
	ax = plt.axes()
	for j, exptime in enumerate(exptimes):
		plt.semilogy(magarr,centroid[:,j],label='$t_{exp}$='+ str(exptime) + 's')

	plt.plot(magarr,np.ones_like(magarr)*tracking_requirement_pixel,'k--',label='Requirement')
	plt.fill_between(magarr,np.ones_like(magarr)*tracking_requirement_pixel,facecolor='green',alpha=0.1)

	plt.legend(loc=6)
	plt.title('Cam: %s Band: '%camera + track_band + ' Temp: %sK'%(int(so.var.teff)))
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

def compute_band_photon_counts():
	"""
	"""
	newmags = []
	all_bands = []
	Johnson_bands = ['U','B','V','R','I','J','H','K']
	for i,band in enumerate(Johnson_bands):
		newmags.append(get_band_mag(so,'Johnson',band,so.stel.factor_0))
		all_bands.append(band)

	#newmags.append(get_band_mag(so,'Sloan','uprime_filter',so.stel.factor_0))
	#all_bands.append('uprime_filter')

	get_band_mag(so,'SLOAN','uprime_filter',so.stel.factor_0)


if __name__=='__main__':
	#load inputs
	configfile = 'hispec_tracking_camera.cfg'
	so         = load_object(configfile)
	cload      = fill_data(so)

	#############
	# Guide Camera
	#############
	camera     = 'h2rg'
	track_band = 'JHgap'  # this is the tracking band
	track_bands = ['z','y','J','JHgap','H','K']
	exptimes = np.array([0.001,0.01,0.1,1,10])
	magarr   = np.arange(0,20,1)

	plt.figure(0,figsize=(7,6))
	ax = plt.axes()	
	for track_band in track_bands:
		snr_arr   = np.zeros((len(magarr),len(exptimes)))
		signal    = np.zeros((len(magarr),len(exptimes)))
		noise     = np.zeros((len(magarr),len(exptimes)))
		fwhm      = np.zeros((len(magarr),len(exptimes)))
		centroid  = np.zeros((len(magarr),len(exptimes)))
		strehl    = np.zeros((len(magarr),len(exptimes)))
		Vmags     = np.zeros((len(magarr)))
		for i,mag in enumerate(magarr): # this is the magnitude in filter band
			factor_0       = so.stel.factor_0 * 10**(-0.4 *(mag-so.var.mag)) # instead could use factor_0 = scale_stellar(so, mag)
			Vmags[i]       = get_band_mag(so,'Johnson','V',factor_0) 
			wfe      = get_HO_WFE(Vmags[i],mode)
			tt_resid = get_tip_tilt_resid(Vmags[i],mode)
			#test_mag      = get_band_mag(so,'Johnson',band,factor_0);plt.plot(mag,test_mag) # if want to test mags match, they do now..
			for j, exptime in enumerate(exptimes):
				signal[i,j]             = calc_trackingcamera_photons(so,exptime,mag,band=track_band)
				fwhm[i,j],strehl[i,j]   = get_fwhm(mag,track_band,tt_resid,camera=camera,offaxis=1)
				# calculate bkg here
				noise[i,j]              = calc_trackingcam_noise(signal[i,j],bkg,exptime,fwhm[i,j],camera=camera)
				snr_arr[i,j]            = signal[i,j]/noise[i,j]
				centroid[i,j]           = (1/np.pi) * fwhm[i,j]/(strehl[i,j]* snr_arr[i,j]) 
				#snr_to_centroidaccuracy(snr_arr[i,j],fwhm[i,j],wfe)

		# compute requirement. requirement is 0.2lambda/D in y band
		platescale_arcsec_pix  = calc_plate_scale(camera)
		yband_wavelength       = 1020 # nm
		tracking_requirement_arcsec = 206265 * 0.2 * yband_wavelength / (so.const.tel_diam*10**9) 
		tracking_requirement_pixel  = tracking_requirement_arcsec/platescale_arcsec_pix

		# get intersection of tracking requiremet for each exposure time
		mag_requirement = get_mag_limit_per_exptime(exptimes,magarr,centroid,tracking_requirement_pixel)

		# save to text file for certain temperature
		f  = open('./output/trackingcamera/%s_maglimit_%s_%smag.txt' %(camera,track_band,so.filt.band),'a+')
		writeme = str(so.var.teff) + '\n' + track_band + '\n' + str(so.var.exp_time) + '\n' 
		for exptime in exptimes: writeme+=str(exptime) + ' '
		writeme+='\n'
		for mag in mag_requirement: writeme+=str(mag) + ' '
		writeme+='\n'
		f.write(writeme)
		f.close()

		
		# PLOT !!!
		#plot_centroiderr_vmag()


		# mag limit vs exp time

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
		plt.title('Cam: %s, Temp: %sK'%(camera,int(so.var.teff)))
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


	#plot_tracking_bands()

