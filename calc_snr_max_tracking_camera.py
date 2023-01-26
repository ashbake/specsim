# calc signal to noise
# max for the calcium H&K
import sys,matplotlib, os

import numpy as np
import matplotlib.pylab as plt
from scipy import interpolate
import matplotlib.ticker as mticker
from astropy import units as u
from astropy import constants as c 
from astropy.io import fits
from astropy.table import Table

font = {'size'   : 14}
matplotlib.rc('font', **font)
plt.rcParams['font.size'] = '14'
plt.rcParams['font.family'] = 'sans'
plt.rcParams['axes.linewidth'] = '1.3'
fontname = 'Arial Narrow'

sys.path.append('./utils/')
from objects import load_object
from load_inputs import fill_data, load_filter,load_phoenix,load_sonora
from functions import *
from noise_tools import get_sky_bg, get_inst_bg
#from kpf_etc.etc import kpf_photon_noise_estimate, kpf_etc_rv, kpf_etc_snr

from matplotlib.ticker import (AutoMinorLocator)

plt.ion()

def tophat(x,l0,lf,throughput):
	ion = np.where((x > l0) & (x<lf))[0]
	bandpass = np.zeros_like(x)
	bandpass[ion] = throughput
	return bandpass

def select_tracking_cam(camera='h2rg'):
	"""
	gary assumes 0.9 for the QE of the H2RG, so modify throughput accordingly
	"""
	if camera=='h2rg':
		rn = 12 #e-
		pixel_pitch = 18 #um
		qe_mod = 1 # relative to what is assumed in the throughput model
		dark=0.8 #e-/s

	if camera=='alladin':
		rn = 43 #e- ??????
		pixel_pitch = 18 #um ?????
		qe_mod = 0.8/0.9 # ???? not sure if we know this
		dark = 0.2 # e-/s

	if camera=='cred2':
		rn = 40 #e-  what is claimed online
		pixel_pitch = 15 #um https://www.axiomoptics.com/products/c-red-2/
		qe_mod = 0.8/.9 # 
		dark=600 #e-/s liquid cooling mode -40

	if camera=='cred2_xswir':
		#extended NIR
		rn = 50 #e-  what is claimed online
		pixel_pitch = 15 #um https://www.axiomoptics.com/products/c-red-2/
		qe_mod = 0.8 /0.9 # 
		dark=10000 #e-/s liquid cooling mode -40

	return rn, pixel_pitch, qe_mod, dark

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

def get_wfe(Rmag):
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

	f_ngs = interpolate.interp1d(mag_arr,aowfe_ngs, bounds_error=False,fill_value=100000)
	f_lgs = interpolate.interp1d(mag_arr,aowfe_lgs, bounds_error=False,fill_value=100000)

	wfe = np.min([f_ngs(Rmag), f_lgs(Rmag)])

	return wfe

def calc_strehl(wfe,wavelength):
	strehl = np.exp(-(2*np.pi*wfe/wavelength)**2)

	return strehl

def snr_to_centroidaccuracy(snr,fwhm,wfe):
	"""
	translate snr to accuracy of centroiding for guiding
	"""
	# get theoretical
	sigx_theory = (1/np.pi) * fwhm/snr

	# get err vs RMS WFE
	wfe_rms = [0, 22.65, 37.25, 51.78, 66.39, 80.87, 95.53, 110.15, 124.41, 139.16, 153.92, 168.37, 182.68, 197.41, 212.18, 226.78];

	com_wferr_10k = [0.0071, 0.0156, 0.0243, 0.0342, 0.0455, 0.058, 0.0728, 0.088, 0.1058, 0.1265, 0.1486, 0.1722, 0.1993, 0.2335, 0.2643, 0.3045];

	qdr_full_wferr_10k = [0.033, 0.0333, 0.0339, 0.0353, 0.0386, 0.0433, 0.0507, 0.06, 0.072, 0.0857, 0.1024, 0.1194, 0.1402, 0.1646, 0.1887, 0.2192];

	jw1dg_wferr_10k = [0.0041, 0.0093, 0.015, 0.0216, 0.0296, 0.0392, 0.0505, 0.0633, 0.078, 0.0953, 0.1144, 0.1346, 0.1589, 0.1891, 0.2169, 0.2542]
	fsigx2      = interpolate.interp1d(wfe_rms,jw1dg_wferr_10k, bounds_error=False,fill_value="extrapolate")
	sigx_wfe    = fsigx2(wfe)

	return np.sqrt(sigx_theory**2)#+ sigx_wfe**2)

def pick_tracking_band(so,band):
	"""
	pick tracking band and get some stats on it
	"""
	if band=='z':
		l0,lf = 800,950
		center_wavelength = 875
		throughput = 1 # 20% was applied in total throughput of new dichroic
		bandpass = tophat(so.stel.v,l0,lf,throughput) #make up fake band

	if band=='y':
		l0,lf = 980,1100
		center_wavelength = 1050
		throughput = 1
		bandpass = tophat(so.stel.v,l0,lf,throughput) #make up fake band

	if band=='JHgap':
		l0,lf= 1335,1485
		center_wavelength = 1400
		throughput = 1
		bandpass = tophat(so.stel.v,l0,lf,throughput)

	if band=='J' or band=='H' or band=='K':
		# take 20%
		x,y = load_filter(so,'Johnson',band)
		y[np.where(y>0.2)] = 0.2
		f = interpolate.interp1d(x,y, bounds_error=False,fill_value=0)
		bandpass = f(so.stel.v)
		center_wavelength = np.mean(x)

	return bandpass, center_wavelength

def calc_plate_scale(camera, D=10, fratio=40):
	"""
	D: diameter in meters
	fratio: 40 default

	return :
	-------
	platescale_arcsec_pix
	"""
	platescale_arcsec_um = 206265 / fratio / (D * 10**6) #arc/um
	_, pitch, _, _ = select_tracking_cam(camera=camera)
	platescale_arcsec_pix = platescale_arcsec_um * pitch
	return platescale_arcsec_pix

def calc_trackingcamera_photons(so,exptime,mag,band='z',camera='h2rg'):
	"""
	get total photons through tracking band

	bands: z, JHgap, J, H, K
	Each band is made using a tophat function to mimic some future dichroic
	"""
	
	# load qe mod factor to modify QE of detector in next step
	_, _, qe_mod,_ = select_tracking_cam(camera=camera)

	# scale to diff magnitudes from one that was loaded
	s_ccd_hires = so.stel.s*10**(-0.4*(mag - so.var.mag))* exptime *\
		 so.const.tel_area * so.hispec.ytransmit*qe_mod * np.abs(so.tel.s)**so.tel.airmass

	# load bandpass profile to integrate over
	bandpass,_= pick_tracking_band(so,band)

	# integrate over passband * photons through tracking camera
	total_photons = np.trapz(s_ccd_hires * bandpass,so.stel.v)

	return total_photons

def get_fwhm(mag,band,tt_resid,camera='h2rg',offaxis=0):
	"""
	combine DL by strehlt and tip/tilt error and off axis

	to do:
	update fwhm_tt and fwhm_offaxis
	"""
	platescale_arcsec_pix = calc_plate_scale(camera, D=10, fratio=40)
	
	# get WFE
	_,wavelength = pick_tracking_band(so,band)
	wfe    = get_wfe(mag) #pulls from Rich's 
	strehl = np.exp(-(2*np.pi*wfe/wavelength)**2)

	D=10 #m
	diffraction_spot_arcsec = 206265 * wavelength/ (D * 10**9) # arcsec
	diffraction_spot_pix = diffraction_spot_arcsec / platescale_arcsec_pix
	fwhm_ho = diffraction_spot_pix / strehl**(1/4) # 1/strehl**.25 from dimitri, to account for broadening deviation from diffraction limit

	fwhm_tt = tt_resid*10**-3/platescale_arcsec_pix # assume 10 [arcsec] for now

	fwhm_offaxis=offaxis *4 # [pix] if off axis input is 1, assume at edge of field where RMS is 4 pix
	# ^ need to use mm instead of 4pix, and load camera pixel pitch
	# ^ also should just load the curve from Mitsuko of RMS diameter vs field angle
	
	fwhm = np.sqrt(fwhm_tt**2 + fwhm_ho**2 + fwhm_offaxis**2)

	return fwhm,strehl

def calc_trackingcam_noise(nphot,bkg,exptime,fwhm,camera='h2rg'):
	"""
	bkg in counts per second per pixel
	"""
	# load tracking camera stats
	rn, pitch, qe, dark = select_tracking_cam(camera=camera)

	npix = np.pi * fwhm**2

	noise =  np.sqrt(nphot + npix * (rn**2  + exptime*(dark+bkg))) #have to add sky #noise in reduced pixel column

	return noise

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

def get_sky_bg(x,airmass,pwv=1.5):
	"""
	"""
	diam = 10. * u.m
	area = 76. * u.m * u.m
	delta_lb = 0.001 * u.micron
	wave = np.arange(0.8,2.5,delta_lb.value) 
	fwhm = ((wave * u.micron / diam) * u.radian).to(u.arcsec)
	solidangle = fwhm**2 * 1.13 #corrected for Gaussian beam (factor 1.13)
	npix = 3
	lam0=2000.
	R=100000

	sky_background_MK = np.zeros([4,3,235000])
	sky_background_MK_wv = np.array([1.,1.6,3.,5.])#water vapor 
	sky_background_MK_airmass = np.array([1.,1.5,2.])#airmass
	path = '../../../_DATA/'
	#
	sky_background_MK_tmp = np.genfromtxt(path+'sky/mk_skybg_zm_'+str(pwv)+'_'+str(airmass)+'_ph.dat', skip_header=0)
	sky_background_MK = sky_background_MK_tmp[:,1]
	sky_background_MK_wave = sky_background_MK_tmp[:,0] #* u.nm
	if np.max(wave) < 10: wave *= 1e+3
	pix_nm  = lam0/R/npix
	sky_background_interp=np.interp(wave, sky_background_MK_wave, sky_background_MK) * u.photon/(u.s*u.arcsec**2*u.nm*u.m**2) * area * solidangle * 2000/100000/3 * u.nm 

	sky_background_interp2=np.interp(x, wave, sky_background_interp) 

	return sky_background_interp2

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

def plot_joshs_sims():
	"""
	show that can RSS centroid errors from theory and his sims
	"""
	#RMS WFE (Generated by Dan's WF RMS Computer)
	wfe_rms = [0,22.68,45.35,67.92,90.79,113.40,135.92,158.50,181.13,203.76,226.83,249.31,271.06,294.80,316.32,340.16,362.77,385.12,406.71,430.43,452.76];

	#SNR: 218 (Hmag 5.33)
	jw1dg_wferr_10k_SNR218 = [0.0073, 0.0113, 0.0198, 0.0315, 0.0474, 0.0678, 0.0915, 0.1218, 0.1576, 0.1993, 0.2555, 0.3318, 0.4291, 0.6297, 0.896, 1.3435, 1.894, 2.4346, 3.0503, 3.683, 4.256];

	com_wferr_10k = [0.0119, 0.0183, 0.0312, 0.0476, 0.0674, 0.0916, 0.1184, 0.1519, 0.191, 0.2333, 0.2914, 0.365, 0.4562, 0.6456, 0.9061, 1.3437, 1.8962, 2.4408, 3.0617, 3.7012, 4.2817];

	qdr_full_wferr_10k = [0.0338, 0.034, 0.0353, 0.0397, 0.049, 0.0639, 0.0834, 0.1086, 0.1386, 0.1743, 0.2203, 0.2838, 0.3705, 0.5567, 0.8186, 1.2651, 1.8417, 2.4003, 3.0471, 3.7017, 4.304];

	#SNR: 438 (Hmag 4.0)
	jw1dg_wferr_10k_SNR438 = [0.0036, 0.0092, 0.0187, 0.0309, 0.0465, 0.0664, 0.0918, 0.1212, 0.1568, 0.1995, 0.254, 0.3254, 0.4314, 0.6057, 0.928, 1.3275, 1.8898, 2.4559, 3.114, 3.7535, 4.3606];

	com_wferr_10k = [0.0081, 0.0162, 0.0301, 0.047, 0.0664, 0.0902, 0.1185, 0.1509, 0.1898, 0.2341, 0.2897, 0.3589, 0.4579, 0.6235, 0.9345, 1.3288, 1.8889, 2.4609, 3.1259, 3.7735, 4.383];

	qdr_full_wferr_10k = [0.0327, 0.0331, 0.0347, 0.0392, 0.0477, 0.0624, 0.0829, 0.1074, 0.1383, 0.1741, 0.2197, 0.2786, 0.3724, 0.5318, 0.8481, 1.251, 1.8294, 2.4212, 3.1077, 3.7727, 4.3967];

	#SNR: 72 (Hmag 6.67)
	jw1dg_wferr_10k_SNR72 = [0.0179, 0.02, 0.0256, 0.0359, 0.0504, 0.07, 0.0945, 0.1259, 0.1609, 0.2048, 0.2566, 0.3277, 0.4353, 0.6537, 0.948, 1.3271, 1.87, 2.5134, 3.0553, 3.7183, 4.3578];

	com_wferr_10k = [0.0263, 0.0302, 0.0392, 0.0535, 0.0721, 0.0949, 0.1227, 0.1566, 0.1954, 0.2405, 0.293, 0.3627, 0.4655, 0.6793, 0.9658, 1.341, 1.8882, 2.5417, 3.0875, 3.7569, 4.4077];

	qdr_full_wferr_10k = [0.0396, 0.0399, 0.0413, 0.0459, 0.055, 0.0691, 0.0889, 0.1144, 0.1437, 0.1802, 0.2238, 0.2854, 0.3783, 0.5852, 0.8825, 1.2697, 1.8278, 2.5262, 3.0923, 3.7738, 4.4539];
	
	jw1dg_newfreq_SNR218 = [0.0073, 0.0117, 0.0217, 0.0374, 0.0642, 0.1043, 0.1666, 0.2519, 0.38, 0.6171, 1.4476, 3.3702, 6.0105, 8.2094, 9.9621, 11.284, 12.2018, 13.1987, 13.8982, 14.3467, 14.9818];

	qdr_newfreq_SNR218 = [0.0334, 0.0339, 0.0358, 0.0421, 0.0608, 0.0941, 0.1518, 0.233, 0.3529, 0.5788, 1.4198, 3.3773, 6.0769, 8.33, 10.1048, 11.4713, 12.3742, 13.4173, 14.1734, 14.6812, 15.3916];

	com_newfreq_SNR218 = [0.0117, 0.0191, 0.0339, 0.0545, 0.086, 0.129, 0.1928, 0.2782, 0.4042, 0.6321, 1.4616, 3.3961, 6.0591, 8.2858, 10.0506, 11.4179, 12.3475, 13.3846, 14.1629, 14.6636, 15.3953];

	plt.figure()
	plt.plot(wfe_rms,jw1dg_wferr_10k_SNR72,label='SNR72')
	plt.plot(wfe_rms,jw1dg_wferr_10k_SNR218,label='SNR218')
	plt.plot(wfe_rms,jw1dg_wferr_10k_SNR438,label='SNR438')
	fwhm = 4.28
	sigx_theory = (1/np.pi) * fwhm/snr
	plt.plot(wfe_rms,np.sqrt(sigx_theory**2 + np.array(jw1dg_wferr_10k_SNR438)**2))

	plt.figure()
	plt.plot(wfe_rms,jw1dg_wferr_10k_SNR218,label='SNR218')
	plt.plot(wfe_rms,jw1dg_newfreq_SNR218,label='SNR218,New Freq')

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
	# questions
	# how to split up wfe to get right centroid error

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
	mode     = 'LGS' # 	modes = np.array(['K','SH','80J','80H','80JK','LGS'])# corresponding modes to match assumptions of text files 
	exptimes = np.array([0.001,0.01,0.1,1,10])
	magarr   = np.arange(0,20,1)
	bkg      = 0#skybkg : #e-/sec/arcsec2/nm/m2 from file, incorporate this and instrument bkg

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
			#wfe           = get_wfe(Rmags[i])
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

