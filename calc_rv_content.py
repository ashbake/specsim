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
from load_inputs import fill_data, load_filter,load_phoenix,load_sonora
from functions import *
#from kpf_etc.etc import kpf_photon_noise_estimate, kpf_etc_rv, kpf_etc_snr

plt.ion()

def hispec_sim_spectrum(so,throughput):
	"""
	scale output from kpf scaling to Ca H&K throughput 
	"""
	s_ccd_hires = so.stel.s * so.var.exp_time * so.const.tel_area * throughput #* np.abs(so.tel.s)**so.tel.airmass

	# convolve to lower res
	s_ccd_lores = degrade_spec(so.stel.v, s_ccd_hires, so.const.res_hispec)

	# resample onto res element grid
	lam0=1500#nm take as lam to calc dlambda, perhaps fix this to have diff res per order..
	sig = lam0/so.const.res_hispec/so.const.res_sampling # lambda/res = dlambda
	v_resamp, s_resamp = resample(so.stel.v,s_ccd_lores,sig=sig, dx=0, eta=1,mode='fast')

	return v_resamp, s_resamp

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

def get_tip_tilt_resid(Rmag):
	"""
	fill in when get data
	"""
	return 6


if __name__=='__main__':	
	#load inputs
	configfile = 'hispec.cfg'
	so         = load_object(configfile)
	cload      = fill_data(so)
	
	airmass = 1.5

	# scale spectra
	sat        = 1e5/2#half of full well
	hispec_pixel_column = 3 # pixels in a column
	readnoise  = 12 # e-, check this**
	darknoise  = 0.012 #e-/s/pix
	PLon       = 0 # if photonic lantern or not, 1=on

	# get WFE - (1) ttStatic - tip tilt (2) ttDynamic - tip/tilt (3) dynwfe - dyanamic wfe (4) LO low order -focus
	mode='SH'    # mode: shack hartmann (SH) or pywfs
	ttStatic = 0 # fiber decenter - assume 0
	ttDynamic=3  # is this ttresid?
	LO=0         # constant focus term, set to Gary's assumption which was????
	if mode=='SH': 
		Rmag    = get_band_mag(so,'Johnson','R',so.stel.factor_0) # get magnitude in r band
		Jmag    = get_band_mag(so,'Johnson','J',so.stel.factor_0) # get magnitude in r band
		dynwfe  = get_dyn_wfe(Rmag)
		ttresid = get_tip_tilt_resid(Rmag)
	if mode=='pywfs':
		Jmag    = get_band_mag(so,'Johnson','J',so.stel.factor_0) # get magnitude in J band
		dynwfe  = get_dyn_wfe_pywfs(Jmag)
		ttresid = get_tip_tilt_resid(Jmag)

	# Pick coupling curve!
	base_throughput  = get_base_throughput(x=so.stel.v) # everything except coupling
	coupling, strehl = pick_coupling(so.stel.v,dynwfe,ttStatic,ttDynamic,LO=LO,PLon=PLon) # includes PIAA and HO term
	total_throughput = np.abs(base_throughput*coupling)
	
	# use throughput to get order bounds, also define echelle orders
	order_peaks      = signal.find_peaks(base_throughput,height=0.055,distance=2e4,prominence=0.01)
	order_cen_lam    = so.stel.v[order_peaks[0]]
	#ms               = np.concatenate((np.arange(59,97),np.arange(111,149))) # for getting fsr approximately
	#ms = ms[::-1]
	blaze_angle      =  76

	# calculate spectrum
	v,s  = hispec_sim_spectrum(so,total_throughput)

	# make corresponding telluric spectrum with same sampling to make a mask
	telluric_spec = np.abs(so.tel.s)**so.tel.airmass
	sig = 1500/so.const.res_hispec/so.const.res_sampling # lambda/res = dlambda
	v_tel, s_tel = resample(so.stel.v,telluric_spec,sig=sig, dx=0, eta=1,mode='fast')
	s_tel/=np.max(s_tel)  # do a rough normalization
	telluric_cutoff = 0.01 # reject lines greater than 1% depth
	telluric_mask = np.ones_like(s_tel)
	telluric_mask[np.where(s_tel < (1-telluric_cutoff))[0]] = 0


	#############
	# start RV business, stealing from Sam's code

	# define order bounds
	flux_interp = interpolate.InterpolatedUnivariateSpline(v,s, k=1)
	dflux = flux_interp.derivative()
	spec_deriv = dflux(v)
	sigma_ord = np.abs(s) ** 0.5
	sigma_ord[np.where(sigma_ord ==0)] = 1e10
	all_w = (v ** 2.) * (spec_deriv ** 2.) / sigma_ord ** 2. # include read noise and dark here!!
	SPEEDOFLIGHT = 2.998e8 # m/s

	# loop through orders
	dv_vals = []
	for i,lam_cen in enumerate(order_cen_lam):
		line_spacing = 0.02 if lam_cen < 1475 else 0.01
		m = np.sin(blaze_angle*np.pi/180) * 2 * (1/line_spacing)/(lam_cen/1000)
		fsr  = lam_cen/m
		isub_test= np.where((so.stel.v> (lam_cen - fsr/2)) & (so.stel.v < (lam_cen+fsr/2))) #FINISH THIS
		#plt.plot(so.stel.v[isub_test],total_throughput[isub_test],'k--')
		isub_order = np.where((v > (lam_cen - 1.3*fsr/2)) & (v < (lam_cen+1.3*fsr/2))) #FINISH THIS
		w_ord = all_w[isub_order] * telluric_mask[isub_order]
		dv_order  = SPEEDOFLIGHT / (np.nansum(w_ord[1:-1])**0.5) # m/s
		dv_photon = 1. / (np.nansum(1./dv_order**2.))**0.5
		dv_vals.append(dv_order)

	dv_vals = np.array(dv_vals)
	#wvl_norm = (np.nanmean(wvl_sampled_ord) - 4200.) / (7200. - 4200.)
 	
	dv_photon = 1. / (np.nansum(1./dv_vals**2.))**0.5

	##################
	#
	# PLOTTT
	col_table = plt.get_cmap('Spectral_r')
	fig, axs = plt.subplots(2,figsize=(7,6),sharex=True)
	plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.85)

	axs[1].plot([950,2400],[0.5,0.5],'k--',lw=0.7)
	axs[1].fill_between([1450,2400],0,1000,facecolor='gray',alpha=0.2)
	axs[1].fill_between([980,1330],0,1000,facecolor='gray',alpha=0.2)
	axs[1].grid('True')
	axs[1].set_ylim(-0,3*np.median(dv_vals))
	axs[1].set_xlim(950,2400)
	axs[1].set_ylabel('$\sigma_{RV}$ [m/s]')
	axs[1].set_xlabel('Wavelength [nm]')

	axs[0].set_ylabel('Counts (e-)')
	axs[0].set_title('M$_%s$=%s, T$_{eff}$=%sK,\n ($t_{exp}$=%ss), vsini=%skm/s'%(so.filt.band,so.var.mag,int(so.var.teff),int(so.var.exp_time),so.stel.vsini))

	axs[0].grid('True')
	ax2 = axs[0].twinx() 
	ax2.plot(v_tel,s_tel,'gray',alpha=0.5,zorder=-100,label='Telluric Absorption')
	ax2.plot(so.stel.v,total_throughput,'k',alpha=0.5,zorder=-100,label='Total Throughput')
	ax2.set_ylabel('Transmission',fontsize=12)
	for i,lam_cen in enumerate(order_cen_lam):
		wvl_norm = (lam_cen - 900.) / (2500. - 900.)
		isub_order = np.where((v > (lam_cen - 1.3*fsr/2)) & (v < (lam_cen+1.3*fsr/2))) #FINISH THIS
		axs[0].plot(v[isub_order],s[isub_order],zorder=200,color=col_table(wvl_norm))
		axs[1].plot(lam_cen,dv_vals[i],'o',zorder=100,color=col_table(wvl_norm),markeredgecolor='k')
	

	sub_yj = dv_vals[np.where((dv_vals!=np.inf) & (order_cen_lam < 1400))[0]]
	sub_hk = dv_vals[np.where((dv_vals!=np.inf) & (order_cen_lam > 1400))[0]]
	rvmed_yj = np.sqrt(np.sum(dv_vals[np.where((dv_vals!=np.inf) & (order_cen_lam < 1400))[0]]**2))/np.sum(sub_yj)
	rvmed_hk = np.median(dv_vals[np.where((dv_vals!=np.inf) & (order_cen_lam > 1400))[0]])
	dv_yj = 1. / (np.nansum(1./sub_yj**2.))**0.5	# 
	dv_hk = 1. / (np.nansum(1./sub_hk**2.))**0.5	# 
	axs[1].text(1050,2*np.median(dv_vals),'$\sigma_{yJ}$=%sm/s'%round(dv_yj,1),fontsize=12,zorder=101)
	axs[1].text(1500,2*np.median(dv_vals),'$\sigma_{HK}$=%sm/s'%round(dv_hk,1),fontsize=12,zorder=101)
	ax2.legend(fontsize=8,loc=1)
	plt.savefig('./output/rv_precision/RV_precision_%sK_%smag%s_%ss_vsini%skms.png'%(so.var.teff,so.filt.band,so.var.mag,so.var.exp_time,so.stel.vsini))

	#####################
	# PLOTTT
	# plot rv content for different stellar types like how sam does it


	# plot RV vs vsini


	####################
	# plot TESS M dwarfs Temp vs magnitude and histogram of vsini
	planets_filename = './data/populations/confirmed_planets_PS_2023.01.12_16.07.07.csv'
	planet_data =  pd.read_csv(planets_filename,delimiter=',',comment='#')

	jmags = planet_data['sy_jmag']
	teffs = planet_data['st_teff']

	fig, axs = plt.subplots(1,2,figsize=(10,4),sharey=True)
	plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.85)
	axs[0].scatter(jmags,teffs,marker='.',c='k')
	axs[1].scatter(jmags,teffs,marker='.',c='k')
	axs[0].set_xlabel('J Mag')
	axs[0].set_ylabel('T$_{eff}$')
	axs[0].set_title('TESS Confirmed Planets')

	axs[0].set_xlim(13,5)
	axs[1].set_xlim(13,5)
	axs[0].set_ylim(2800,7000)
	plt.subplots_adjust(bottom=0.15,hspace=0,left=0.15)

	# add velocity precision contours


	#
	plt.figure()
	isub4000 = np.where(teffs < 4000)[0]
	plt.hist(planet_data['st_vsin'],bins=100,histtype='stepfilled',alpha=0.3,ec='k',label='All')
	plt.hist(planet_data['st_vsin'][isub4000],bins=20,histtype='stepfilled',alpha=0.3,ec='k',label='Teff< 4000K')
	plt.xlim(0,10)
	plt.xlabel('Vsini [km/s]')
	plt.ylabel('Counts')
	plt.title('Vsini of TESS Confirmed Planets')
	plt.legend()





