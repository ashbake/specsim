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

sys.path.append('/Users/ashbake/Documents/Research/ToolBox/hispec_snr/utils/')
from objects import load_object
from load_inputs import fill_data, load_filter,get_band_mag
from functions import *
from noise_tools import get_sky_bg, get_inst_bg, sum_total_noise
from throughput_tools import pick_coupling, get_base_throughput
from wfe_tools import get_tip_tilt_resid, get_HO_WFE
import plot_tools,obs_tools

plt.ion()

# PLOT FROM DATA RUN
def plot_snr_teff(so, temp_arr,snr_arr):
	"""
	"""
	fig, ax = plt.subplots(1,1, figsize=(10,8))	
	for i,temp in enumerate(temp_arr):
		ax.plot(so.obs.v,snr_arr[i],label='T=%sK'%temp)

	ax.set_ylabel('SNR')
	ax.set_xlabel('Wavelength (nm)')
	ax.set_title('AO Mode: %s, %s=%s, t=%shr'%(so.ao.mode,so.filt.band,int(so.stel.mag)))
	ax.axhline(y=30,color='k',ls='--')
	figname = 'snr_%s_%smag_%s_texp_%ss_dark_%s.png' %(so.ao.mode,so.filt.band,so.stel.mag,so.obs.texp,so.inst.darknoise)
	plt.legend()
	# duplicate axis to plot filter response
	ax2 = ax.twinx()
	#ax2.fill_between(so.filt.v,so.filt.s,facecolor='gray',edgecolor='black',alpha=0.2)
	#ax2.set_ylabel('Filter Response')
	# plot band
	ax2.fill_between(so.inst.y,0,1,facecolor='k',edgecolor='black',alpha=0.1)
	ax2.text(20+np.min(so.inst.y),0.9, 'y')
	ax2.fill_between(so.inst.J,0,1,facecolor='k',edgecolor='black',alpha=0.1)
	ax2.text(50+np.min(so.inst.J),0.9, 'J')
	ax2.fill_between(so.inst.H,0,1,facecolor='k',edgecolor='black',alpha=0.1)
	ax2.text(50+np.min(so.inst.H),0.9, 'H')
	ax2.fill_between(so.inst.K,0,1,facecolor='k',edgecolor='black',alpha=0.1)
	ax2.text(50+np.min(so.inst.K),0.9, 'K')
	ax2.set_ylim(0,1)
	ax.set_xlim(970,2500)
	plt.savefig('./output/snrplots/' + figname)

def plot_snr_mag_peaks(so, mag_arr,v,snr_arr,mode='max'):
	"""
	"""
	fig, ax = plt.subplots(1,1, figsize=(10,8))	
	for i,mag in enumerate(mag_arr):
		cen_lam, snr_peaks,snr_means = obs_tools.get_order_value(so,v,snr_arr[i])
		if mode=='max':
			ax.semilogy(cen_lam,snr_peaks,'-o',label='m=%s'%mag)
		elif mode=='mean':
			ax.semilogy(cen_lam,snr_means,'-o',label='m=%s'%mag)

	ax.set_ylabel('SNR per Resolution Element')
	ax.set_xlabel('Wavelength (nm)')
	ax.set_title('AO Mode: %s, %sband, Teff:%s, t=4hr'%(so.ao.mode,so.filt.band,so.stel.teff))
	ax.axhline(y=30,color='k',ls='--')
	figname = 'snr_%s_band_%s_teff_%s_texp_%ss.png' %(so.ao.mode,so.filt.band,so.stel.teff,so.obs.texp)
	plt.legend()
	# duplicate axis to plot filter response
	ax2 = ax.twinx()
	#ax2.fill_between(so.filt.v,so.filt.s,facecolor='gray',edgecolor='black',alpha=0.2)
	#ax2.set_ylabel('Filter Response')
	# plot band
	ax2.fill_between(so.inst.y,0,1,facecolor='k',edgecolor='black',alpha=0.1)
	ax2.text(20+np.min(so.inst.y),0.9, 'y')
	ax2.fill_between(so.inst.J,0,1,facecolor='k',edgecolor='black',alpha=0.1)
	ax2.text(50+np.min(so.inst.J),0.9, 'J')
	ax2.fill_between(so.inst.H,0,1,facecolor='k',edgecolor='black',alpha=0.1)
	ax2.text(50+np.min(so.inst.H),0.9, 'H')
	ax2.fill_between(so.inst.K,0,1,facecolor='k',edgecolor='black',alpha=0.1)
	ax2.text(50+np.min(so.inst.K),0.9, 'K')
	ax2.set_ylim(0,1)
	ax.set_xlim(970,2500)
	plt.savefig('./output/snrplots/' + figname)

def plot_snr_mag_peaks_2d(so, mag_arr,v,snr_arr,xextent=[980,2460],mode='max'):
	"""
	"""
	def fmt(x):
		return str(int(x))

	snr_arr_order= []
	for i,mag in enumerate(mag_arr):
		cen_lam, snr_peaks,snr_means = obs_tools.get_order_value(so,v,snr_arr[i])
		if mode=='max':
			#ax.plot(cen_lam,snr_peaks,label='m=%s'%mag)
			snr_arr_order.append(snr_peaks)
		elif mode=='mean':
			#ax.plot(cen_lam,snr_means,label='m=%s'%mag)
			snr_arr_order.append(snr_means)
	
	snr_arr_order = np.array(snr_arr_order)

	# resample onto regular grid for imshow
	#xs = np.arange(np.min(cen_lam),np.max(cen_lam))
	xs = np.arange(np.min(xextent), np.max(xextent))
	snr_arr_order_regular = np.zeros((len(mag_arr),len(xs)))
	for i,mag in enumerate(mag_arr):
		tmp = interpolate.interp1d(cen_lam, snr_arr_order[i],kind='linear',bounds_error=False,fill_value=0)
		snr_arr_order_regular[i,:] = tmp(xs)

	extent = (np.min(xs),np.max(xs),np.min(mag_arr),np.max(mag_arr))

	fig, ax = plt.subplots(1,1, figsize=(9,8))	
	ax.imshow(snr_arr_order_regular,aspect='auto',origin='lower',\
				interpolation='quadric',cmap='nipy_spectral',\
				extent=extent,vmax=1000,vmin=0)
	cs = ax.contour(snr_arr_order_regular, levels=[30,50,100,200,500,1000] ,\
				colors=['r','k','k','k','k','k'],origin='lower',\
				extent=extent)
	ax.invert_yaxis()
	ax.clabel(cs, cs.levels, inline=True,fmt=fmt,fontsize=10,\
		colors=['r','r','r','r'],zorder=101)
	#plt.xticks(list(cen_lam))
	blackout = [1327,1490]
	plt.fill_between(blackout,np.min(mag_arr),np.max(mag_arr),facecolor='k',\
			hatch='X',edgecolor='gray',zorder=100)
	try:
		c30   = cs.collections[0].get_paths()[1].vertices # extract 30 snr curve
		c30_2 = cs.collections[0].get_paths()[0].vertices # extract 30 snr curve
		plt.fill_between(c30[:,0],np.max(mag_arr),c30[:,1],hatch='/',fc='k',edgecolor='r')
		plt.fill_between(c30_2[:,0],np.max(mag_arr),c30_2[:,1],hatch='/',fc='k',edgecolor='r')
	except IndexError:
		pass

	ax.set_ylabel('%s Magnitude'%so.filt.band)
	ax.set_xlabel('Wavelength (nm)')
	ax.set_xlim(xextent)
	ax.set_title('AO Mode: %s, %sband, Teff:%s, t=4hr'%(so.ao.mode,so.filt.band,so.stel.teff))
	figname = 'snr2d_%s_band_%s_teff_%s_texp_%ss.png' %(so.ao.mode,so.filt.band,so.stel.teff,so.obs.texp)
	# duplicate axis to plot filter response
	plt.savefig('./output/snrplots/' + figname)

def plot_snr_teff_peaks(so, temp_arr,v,snr_arr,xextent=[980,2460],yextent=[0,1000],mode='max'):
	"""
	take peaks of orders
	"""
	fig, ax = plt.subplots(1,1, figsize=(10,8))	
	for i,temp in enumerate(temp_arr):
		cen_lam, snr_peaks,snr_means = obs_tools.get_order_value(so,v,snr_arr[i])
		if mode=='max':
			ax.plot(cen_lam,snr_peaks,label='T=%sK'%temp)
		elif mode=='mean':
			ax.plot(cen_lam,snr_means,label='T=%sK'%temp)

	ax.set_ylabel('SNR')
	ax.set_xlabel('Wavelength (nm)')
	ax.set_title('%s=%s, t=%shr'%(so.filt.band,int(so.stel.mag),int(so.obs.texp/3600)))
	ax.axhline(y=30,color='k',ls='--')
	figname = 'snr_teff_%s_%smag_%s_texp_%ss.png' %(so.ao.mode,so.filt.band,so.stel.mag,so.obs.texp)
	plt.legend()
	# duplicate axis to plot filter response
	ax2 = ax.twinx()
	#ax2.fill_between(so.filt.v,so.filt.s,facecolor='gray',edgecolor='black',alpha=0.2)
	#ax2.set_ylabel('Filter Response')
	# plot band
	ax2.fill_between(so.inst.y,0,1,facecolor='k',edgecolor='black',alpha=0.1)
	ax2.text(20+np.min(so.inst.y),0.9, 'y')
	ax2.fill_between(so.inst.J,0,1,facecolor='k',edgecolor='black',alpha=0.1)
	ax2.text(50+np.min(so.inst.J),0.9, 'J')
	ax2.fill_between(so.inst.H,0,1,facecolor='k',edgecolor='black',alpha=0.1)
	ax2.text(50+np.min(so.inst.H),0.9, 'H')
	ax2.fill_between(so.inst.K,0,1,facecolor='k',edgecolor='black',alpha=0.1)
	ax2.text(50+np.min(so.inst.K),0.9, 'K')
	ax2.set_ylim(0,1)
	ax.set_ylim(yextent)
	ax.set_xlim(xextent)
	plt.savefig('./output/snrplots/' + figname)

def plot_noise_teff(v,n_arr,temp_arr):
	"""
	"""
	fig, ax = plt.subplots(1,1, figsize=(10,8))	
	for i,temp in enumerate(temp_arr):
		ax.plot(v,s_arr[i],label='T=%sK'%temp)

	fig, ax = plt.subplots(1,1, figsize=(10,8))	
	for i,temp in enumerate(temp_arr):
		ax.plot(v,n_arr[i],label='T=%sK'%temp)

	fig, ax = plt.subplots(1,1, figsize=(10,8))	
	for i,temp in enumerate(temp_arr):
		ax.plot(so.stel.v,c_arr[i],label='T=%sK'%temp)

	plt.plot(so.stel.v,coupling,label='Coupling Only')
	plt.plot(so.stel.v,base_throughput,label='All But Coupling')	
	plt.plot(so.stel.v,coupling*base_throughput,'k',label='Total Throughput')	
	plt.ylabel('Transmission')
	plt.xlabel('Wavelength (nm)')
	plt.title('%s=%s, Vmag=%s, Teff=%s'%(so.filt.band,int(so.stel.mag),round(so.ao.v_mag,1),int(so.stel.teff)))
	plt.subplots_adjust(bottom=0.15)
	plt.axhline(y=0.05,color='m',ls='--',label='5%')
	plt.legend()
	plt.grid()
	figname = 'throughput_%s_%smag_%s_Teff_%s_texp_%ss.png' %(mode,so.filt.band,so.stel.mag,so.stel.teff,int(so.obs.texp_frame*nframes))
	#plt.savefig('./output/snrplots/' + figname)

def plot_median_bin_snr(so):
	"""
	need so to make this work bc need so.obs.v
	"""
	#temps were 3600 for this
	mag_arr = np.arange(5,16,1)
	bands = ['y','J','H','K']
	xextents= [so.inst.y,so.inst.J,so.inst.H,so.inst.K]

	plt.figure()
	for i,band in enumerate(bands):
		f = np.load('./output/snr/snr_arr_mag_teff_%s_band_%s.npy'%(3600,band))
		new_f = []
		for j,mag in enumerate(mag_arr): 
			new_v, temp_f, _ = obs_tools.get_order_value(so,so.obs.v,f[j])
			new_f.append(temp_f)
		new_f = np.array(new_f)
		iband = np.where((new_v > xextents[i][0]) & (new_v <xextents[i][1]))[0]
		plt.semilogy(mag_arr,np.sqrt(3)*np.median(new_f[:,iband],axis=1),label=band)
	plt.plot(mag_arr,mag_arr*0 + 30,'k--')
	plt.legend()
	plt.xlabel('Magnitude')
	plt.ylabel('SNR per Resolution Element')


	plt.figure()
	for i,band in enumerate(bands):
		f = np.load('./output/snr/snr_arr_mag_teff_%s_band_%s.npy'%(3600,band))
		iband = np.where((so.obs.v > xextents[i][0]) & (so.obs.v <xextents[i][1]))[0]
		plt.semilogy(mag_arr,np.sqrt(3)*np.median(f[:,iband],axis=1),label=band)
	plt.plot(mag_arr,mag_arr*0 + 30,'k--')
	plt.legend()
	plt.xlabel('Magnitude')
	plt.ylabel('Median bin SNR')

	my_xticks = mag_arr
	plt.xticks(mag_arr, mag_arr)

	my_yticks = [10,100,1000]
	plt.yticks(my_yticks,my_yticks)
	plt.subplots_adjust(left=0.15,bottom=0.15)
	plt.grid()
	plt.text(6,33,'SNR=30')
	plt.savefig('./output/snr/median_bin_snr_per_band.png')


###############
def run_snr_v_mag(teff=3600):
	"""
	"""
	temp_arr   = [1000,1500,2300,3000,3600,4200,5800]
	ao_modes   = ['LGS_100J_45','LGS_100J_45','LGS_100J_45','LGS_100J_45','LGS_100J_45','LGS_STRAP_45','LGS_STRAP_45','LGS_STRAP_45']
	itemp = np.where(np.array(temp_arr) ==teff)[0][0]
	ao_mode = np.array(ao_modes)[itemp]

	so.ao.mode = ao_mode

	cload.set_teff_mag(so,3600,so.stel.mag,star_only=False)
	mag_arr= np.arange(5,16)
	snr_arr = []
	s_arr   = []
	n_arr   = []
	c_arr   = []
	snr_reselement= []

	for mag in mag_arr:
		cload.set_mag(so,mag) 
		snr_arr.append(so.obs.snr)
		s_arr.append(so.obs.s_frame)
		snr_reselement.append(so.obs.snr_reselement) # plot snr per res element
		n_arr.append(so.obs.noise_frame)
		c_arr.append(so.inst.coupling)
	
	#plot_snr_mag_peaks(so, mag_arr,so.obs.v_resamp,snr_reselement,mode='max')
	#plot_snr_mag_peaks_2d(so, mag_arr,so.obs.v_resamp,snr_reselement,mode='max')

	return mag_arr,snr_arr,s_arr,n_arr,c_arr,snr_reselement

def run_snr_v_teff(mag=15):
	"""
	"""
	temp_arr   = [1000,1500,2300,3000,3600,4200,5800]
	ao_mode   = ['LGS_100J_45','LGS_100J_45','LGS_100J_45','LGS_100J_45','LGS_100J_45','LGS_STRAP_45','LGS_STRAP_45','LGS_STRAP_45']
	snr_arr = []
	s_arr   = []
	n_arr   = []
	c_arr   = []
	snr_reselement= []

	for i,temp in enumerate(temp_arr):
		so.ao.mode=ao_mode[i]
		cload.set_teff_mag(so,temp,mag,star_only=False) 
		snr_arr.append(so.obs.snr) # plot snr per res element
		snr_reselement.append(so.obs.snr_reselement) # plot snr per res element
		s_arr.append(so.obs.s_frame)
		n_arr.append(so.obs.noise_frame)
		c_arr.append(so.inst.coupling)
	
	#plot_snr_teff_peaks(so, temp_arr,so.obs.v_resamp,snr_reselement,mode='max')
	
	#plot_snr(so,temp_arr,v,snr_arr)
	#plot_throughput(v,coupling,base_throughput,so)
	#plot_noise(so.obs.v,n_arr,temp_arr)
	return temp_arr,snr_arr,s_arr,n_arr,c_arr,snr_reselement

def check_AO_counts_match(so):
	"""
	lianqi assumes some flux for his AO models

	make sure these magnitudes and flux values are consistent
	with my magnitudes
	"""
	# lianqi values
	texp =1/800.
	ytransmit = 0.437
	Rmag = [8,9,10,11,12,13,14,15,16]
	pdes = [921.854, 366.997, 146.104, 58.1651, 23.1559, 9.21854, 3.66997, 1.46104, 0.581651] 
	detector_size = 0.5 * 0.5 #m2 subaperture

	l0,lf = 600,750 #nfiraos pyramid wave bandpass
	center_wavelength = (l0+lf)/2
	bandpass = tophat(so.stel.v,l0,lf,ytransmit) #make up fake band

	total = detector_size * texp * np.trapz(so.stel.s * bandpass,x=so.stel.v) # phot to pyramid
	print(total)

	flux_top_atm = np.array(pdes) / detector_size / texp / ytransmit



if __name__=='__main__':
	#load inputs
	configfile = './modhis_snr.cfg'
	so    = load_object(configfile)
	cload = fill_data(so) # put coupling files in load and wfe stuff too

	plot_tools.plot_snr_orders(so,snrtype=1,mode='peak',height=0.08,savepath='./output/snrplots/')

	# mag_arr,snr_arr,s_arr,n_arr,c_arr,snr_reselement= run_snr_v_mag(teff=3600)
	# np.save('./output/snr/snr_arr_mag_teff_%s_band_%s'%(so.stel.teff,so.filt.band),snr_arr)
	

	#xextent = so.inst.y

	#plot_snr_mag_peaks_2d(so, mag_arr,so.obs.v_resamp,snr_reselement,xextent=xextent,mode='max')
	#temp_arr,snr_arr,s_arr,n_arr,c_arr,snr_reselement2 = run_snr_v_teff(mag=15)
	#plot_snr_teff_peaks(so, temp_arr,so.obs.v_resamp,snr_reselement2,xextent=xextent,yextent=[0,150],mode='max')

	#plot_snr_mag_peaks(so, mag_arr,so.obs.v_resamp,snr_reselement,mode='max')


