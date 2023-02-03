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
from functions import *
from noise_tools import get_sky_bg, get_inst_bg, sum_total_noise
from throughput_tools import pick_coupling, get_band_mag, get_base_throughput
from wfe_tools import get_tip_tilt_resid, get_HO_WFE

plt.ion()

def sim_spectrum(so):
	"""
	scale output from kpf scaling to Ca H&K throughput 
	"""
	s_ccd_hires = so.stel.s * so.obs.texp_frame * so.inst.tel_area * so.inst.ytransmit * np.abs(so.tel.s)**so.tel.airmass

	# convolve to lower res
	s_ccd_lores = degrade_spec(so.stel.v, s_ccd_hires, so.inst.res)

	# resample onto res element grid
	lam0=1500#nm take as lam to calc dlambda, perhaps fix this to have diff res per order..
	sig = lam0/so.inst.res/so.inst.res_samp # lambda/res = dlambda
	v_resamp, s_resamp = resample(so.stel.v,s_ccd_lores,sig=sig, dx=0, eta=1,mode='fast')

	return v_resamp, s_resamp

def run_one_spec(so):
	"""
	"""
	# get WFE - (1) ttStatic - tip tilt (2) ttDynamic - tip/tilt (3) dynwfe - dyanamic wfe (4) LO low order -focus
	#so.ao.mode='LGS' # ['K','SH','80J','80H','80JK','LGS']

	# calc SNR
	v,s_frame = sim_spectrum(so)
	# do some texp calc here if default requested
	skybg    = get_sky_bg(v,so.tel.airmass,pwv=1.1,skypath=so.tel.skypath)
	instbg   = get_inst_bg(v,npix=so.inst.pix_vert,lam0=2000,R=so.inst.res,diam=so.inst.tel_diam,area=so.inst.tel_area)

	# calc noise
	noise_frame  = sum_total_noise(s_frame,so.obs.texp_frame, so.obs.nramp,instbg, skybg, so.inst.darknoise,so.inst.readnoise,so.inst.pix_vert)
	noise_frame[np.where(np.isnan(noise_frame))] = np.inf
	noise_frame[np.where(noise_frame==0)] = np.inf
	snr_frame    = s_frame/noise_frame
	snr          = np.sqrt(so.obs.nframes) * snr_frame
	
	# calc snr per res element
	v_res_elem , snr_res_elem= rebin(v,snr,nbin=int(so.inst.res_samp), eta=(1/np.sqrt(so.inst.res_samp)))
	
	if np.max(s_frame) > so.inst.saturation: print("WARNING-hit SATURATION limit, lower exp time")

	return v_res_elem , snr_res_elem, noise_frame, s_frame

def save_one_spec(path='./output/snr_data/'):
	"""
	"""
	pass

def plot_snr_teff(so, temp_arr,v,snr_arr):
	"""
	"""
	fig, ax = plt.subplots(1,1, figsize=(10,8))	
	for i,temp in enumerate(temp_arr):
		ax.plot(v,snr_arr[i],label='T=%sK'%temp)

	ax.set_ylabel('SNR')
	ax.set_xlabel('Wavelength (nm)')
	ax.set_title('AO Mode: %s, %s=%s, t=4hr'%(so.ao.mode,so.filt.band,int(so.stel.mag)))
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
		cen_lam, snr_peaks,snr_means = get_order_snrs(so,v,snr_arr[i])
		if mode=='max':
			ax.plot(cen_lam,snr_peaks,label='m=%s'%mag)
		elif mode=='mean':
			ax.plot(cen_lam,snr_means,label='m=%s'%mag)

	ax.set_ylabel('SNR')
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

def plot_snr_mag_peaks_2d(so, mag_arr,v,snr_arr,mode='max'):
	"""
	"""
	def fmt(x):
		return str(int(x))

	snr_arr_order= []
	for i,mag in enumerate(mag_arr):
		cen_lam, snr_peaks,snr_means = get_order_snrs(so,v,snr_arr[i])
		if mode=='max':
			#ax.plot(cen_lam,snr_peaks,label='m=%s'%mag)
			snr_arr_order.append(snr_peaks)
		elif mode=='mean':
			#ax.plot(cen_lam,snr_means,label='m=%s'%mag)
			snr_arr_order.append(snr_means)
	
	snr_arr_order = np.array(snr_arr_order)

	# resample onto regular grid for imshow
	xs = np.arange(np.min(cen_lam),np.max(cen_lam))
	snr_arr_order_regular = np.zeros((len(mag_arr),len(xs)))
	for i,mag in enumerate(mag_arr):
		tmp = interpolate.interp1d(cen_lam, snr_arr_order[i],kind='linear',bounds_error=False,fill_value=0)
		snr_arr_order_regular[i,:] = tmp(xs)

	extent = (np.min(xs),np.max(xs),np.min(mag_arr),np.max(mag_arr))

	fig, ax = plt.subplots(1,1, figsize=(12,8))	
	ax.imshow(snr_arr_order_regular,aspect='auto',origin='lower',\
				interpolation='quadric',cmap='nipy_spectral',\
				extent=extent)
	cs = ax.contour(snr_arr_order_regular, levels=[30,50,100,500,1000] ,\
				colors=['r','k','k','k','k'],origin='lower',\
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

	ax.set_title('AO Mode: %s, %sband, Teff:%s, t=4hr'%(so.ao.mode,so.filt.band,so.stel.teff))
	figname = 'snr2d_%s_band_%s_teff_%s_texp_%ss.png' %(so.ao.mode,so.filt.band,so.stel.teff,so.obs.texp)
	# duplicate axis to plot filter response
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

	return order_cen_lam, snr_peaks, snr_means

def plot_snr_teff_peaks(so, temp_arr,v,snr_arr,mode='max'):
	"""
	take peaks of orders
	"""
	fig, ax = plt.subplots(1,1, figsize=(10,8))	
	for i,temp in enumerate(temp_arr):
		cen_lam, snr_peaks,snr_means = get_order_snrs(so,v,snr_arr[i])
		if mode=='max':
			ax.plot(cen_lam,snr_peaks,label='T=%sK'%temp)
		elif mode=='mean':
			ax.plot(cen_lam,snr_means,label='T=%sK'%temp)

	ax.set_ylabel('SNR')
	ax.set_xlabel('Wavelength (nm)')
	ax.set_title('AO Mode: %s, %s=%s, t=4hr'%(so.ao.mode,so.filt.band,int(so.stel.mag)))
	ax.axhline(y=30,color='k',ls='--')
	figname = 'snr_%s_%smag_%s_texp_%ss.png' %(so.ao.mode,so.filt.band,so.stel.mag,so.obs.texp)
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

def plot_bg(v,instbg,skybg):
	fig, ax = plt.subplots(1,1, figsize=(8,5))	
	ax.plot(v,instbg+skybg)
	ax.set_xlim(900,2500)
	ax.set_ylim(0,0.5)
	ax.set_xlabel('Wavelength (nm)')
	ax.set_ylabel('Sky + Inst Bg (e-/s/pix)')
	ax2 = ax.twinx()
	#ax2.fill_between(so.filt.v,so.filt.s,facecolor='gray',edgecolor='black',alpha=0.2)
	#ax2.set_ylabel('Filter Response')
	# plot band
	ax2.fill_between(so.inst.y,0,1,facecolor='k',edgecolor='black',alpha=0.2)
	ax2.text(20+np.min(so.inst.y),0.9, 'y')
	ax2.fill_between(so.inst.J,0,1,facecolor='k',edgecolor='black',alpha=0.2)
	ax2.text(50+np.min(so.inst.J),0.9, 'J')
	ax2.fill_between(so.inst.H,0,1,facecolor='k',edgecolor='black',alpha=0.2)
	ax2.text(50+np.min(so.inst.H),0.9, 'H')
	ax2.fill_between(so.inst.K,0,1,facecolor='k',edgecolor='black',alpha=0.2)
	ax2.text(50+np.min(so.inst.K),0.9, 'K')
	ax2.set_ylim(0,1)

	fig, ax = plt.subplots(1,1, figsize=(8,5))	
	ax.plot(v,instbg)
	ax.set_xlim(900,2500)
	ax.set_ylim(0,0.5)
	ax.set_xlabel('Wavelength (nm)')
	ax.set_ylabel('Inst Bg (e-/s/pix)')
	ax2 = ax.twinx()
	#ax2.fill_between(so.filt.v,so.filt.s,facecolor='gray',edgecolor='black',alpha=0.2)
	#ax2.set_ylabel('Filter Response')
	# plot band
	ax2.fill_between(so.inst.y,0,1,facecolor='k',edgecolor='black',alpha=0.2)
	ax2.text(20+np.min(so.inst.y),0.9, 'y')
	ax2.fill_between(so.inst.J,0,1,facecolor='k',edgecolor='black',alpha=0.2)
	ax2.text(50+np.min(so.inst.J),0.9, 'J')
	ax2.fill_between(so.inst.H,0,1,facecolor='k',edgecolor='black',alpha=0.2)
	ax2.text(50+np.min(so.inst.H),0.9, 'H')
	ax2.fill_between(so.inst.K,0,1,facecolor='k',edgecolor='black',alpha=0.2)
	ax2.text(50+np.min(so.inst.K),0.9, 'K')
	ax2.set_ylim(0,1)


	fig, ax = plt.subplots(1,1, figsize=(8,5))	
	ax.plot(v,skybg)
	ax.set_xlim(900,2500)
	ax.set_ylim(0,0.5)
	ax.set_xlabel('Wavelength (nm)')
	ax.set_ylabel('Sky Bg (e-/s/pix)')
	ax2 = ax.twinx()
	#ax2.fill_between(so.filt.v,so.filt.s,facecolor='gray',edgecolor='black',alpha=0.2)
	#ax2.set_ylabel('Filter Response')
	# plot band
	ax2.fill_between(so.inst.y,0,1,facecolor='k',edgecolor='black',alpha=0.2)
	ax2.text(20+np.min(so.inst.y),0.9, 'y')
	ax2.fill_between(so.inst.J,0,1,facecolor='k',edgecolor='black',alpha=0.2)
	ax2.text(50+np.min(so.inst.J),0.9, 'J')
	ax2.fill_between(so.inst.H,0,1,facecolor='k',edgecolor='black',alpha=0.2)
	ax2.text(50+np.min(so.inst.H),0.9, 'H')
	ax2.fill_between(so.inst.K,0,1,facecolor='k',edgecolor='black',alpha=0.2)
	ax2.text(50+np.min(so.inst.K),0.9, 'K')
	ax2.set_ylim(0,1)

def plot_noise(v,n_arr):
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

if __name__=='__main__':
	#load inputs
	configfile = 'hispec.cfg'
	so    = load_object(configfile)
	cload = fill_data(so) # put coupling files in load and wfe stuff too

	cload.set_teff(so,3600) 
	mag_arr= np.arange(8,16)
	snr_arr = []
	s_arr   = []
	n_arr   = []
	c_arr   = []

	for mag in mag_arr:
		cload.set_mag(so,mag) 
		v,snr, n_frame, s_frame = run_one_spec(so)
		snr_arr.append(snr)
		s_arr.append(s_frame)
		n_arr.append(n_frame)
		c_arr.append(so.inst.coupling)
	
	plot_snr_mag_peaks(so, mag_arr,v,snr_arr,mode='max')
	plot_snr_mag_peaks_2d(so, mag_arr,v,snr_arr,mode='max')


	temp_arr   = [1000,1500,2300,3000,3600,4200,5800]
	snr_arr = []
	s_arr   = []
	n_arr   = []
	c_arr   = []

	for temp in temp_arr:
		cload.set_teff(so,temp) 
		v,snr, n_frame, s_frame = run_one_spec(so)
		snr_arr.append(snr)
		s_arr.append(s_frame)
		n_arr.append(n_frame)
		c_arr.append(so.inst.coupling)

	#plot_snr(so,temp_arr,v,snr_arr)
	# plot_throughput(v,coupling,base_throughput,so)
	#plot_snr_teff_peaks(so, temp_arr,v,snr_arr,mode='max')

