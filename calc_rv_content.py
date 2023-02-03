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
SPEEDOFLIGHT = 2.998e8 # m/s

plt.ion()


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

def plot_rv_err(so, lam_cen, dv_vals,savefig=True):
	"""
	"""
	col_table = plt.get_cmap('Spectral_r')
	fig, axs = plt.subplots(2,figsize=(7,7),sharex=True)
	plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.85,top=0.85)

	axs[1].plot([950,2400],[0.5,0.5],'k--',lw=0.7)
	axs[1].fill_between([1450,2400],0,1000,facecolor='gray',alpha=0.2)
	axs[1].fill_between([980,1330],0,1000,facecolor='gray',alpha=0.2)
	axs[1].grid('True')
	axs[1].set_ylim(-0,3*np.median(dv_vals))
	axs[1].set_xlim(950,2400)
	axs[1].set_ylabel('$\sigma_{RV}$ [m/s]')
	axs[1].set_xlabel('Wavelength [nm]')

	axs[0].set_ylabel('SNR')
	axs[0].set_title('M$_%s$=%s, T$_{eff}$=%sK,\n ($t_{exp}$=%ss), vsini=%skm/s'%(so.filt.band,so.stel.mag,int(so.stel.teff),int(so.obs.texp),so.stel.vsini))

	axs[0].grid('True')
	ax2 = axs[0].twinx() 
	ax2.plot(so.tel.v,so.tel.s,'gray',alpha=0.5,zorder=-100,label='Telluric Absorption')
	ax2.plot(so.stel.v,so.inst.ytransmit,'k',alpha=0.5,zorder=-100,label='Total Throughput')
	ax2.set_ylabel('Transmission',fontsize=12)
	for i,lam_cen in enumerate(order_cens):
		wvl_norm = (lam_cen - 900.) / (2500. - 900.)
		axs[0].plot(so.obs.v[order_inds[i]],so.obs.s[order_inds[i]]/so.obs.noise[order_inds[i]],zorder=200,color=col_table(wvl_norm))
		axs[1].plot(lam_cen,dv_vals[i],'o',zorder=100,color=col_table(wvl_norm),markeredgecolor='k')
	
	sub_yj = dv_vals[np.where((dv_vals!=np.inf) & (order_cens < 1400))[0]]
	sub_hk = dv_vals[np.where((dv_vals!=np.inf) & (order_cens > 1400))[0]]
	rvmed_yj = np.sqrt(np.sum(dv_vals[np.where((dv_vals!=np.inf) & (order_cens < 1400))[0]]**2))/np.sum(sub_yj)
	rvmed_hk = np.median(dv_vals[np.where((dv_vals!=np.inf) & (order_cens > 1400))[0]])
	dv_yj = 1. / (np.nansum(1./sub_yj**2.))**0.5	# 
	dv_hk = 1. / (np.nansum(1./sub_hk**2.))**0.5	# 
	axs[1].text(1050,2*np.median(dv_vals),'$\sigma_{yJ}$=%sm/s'%round(dv_yj,1),fontsize=12,zorder=101)
	axs[1].text(1500,2*np.median(dv_vals),'$\sigma_{HK}$=%sm/s'%round(dv_hk,1),fontsize=12,zorder=101)
	ax2.legend(fontsize=8,loc=1)
	if savefig:
		plt.savefig('./output/rv_precision/RV_precision_%sK_%smag%s_%ss_vsini%skms.png'%(so.stel.teff,so.filt.band,so.stel.mag,so.obs.texp,so.stel.vsini))

	return fig,axs

def plot_tess_data():
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


if __name__=='__main__':
	#load inputs
	configfile = 'hispec_rv_err.cfg'
	so    = load_object(configfile)
	cload = fill_data(so) # put coupling files in load and wfe stuff too

	def get_order_bounds(so):
		"""
		given array, return max and mean of snr per order
		"""
		order_peaks      = signal.find_peaks(so.inst.base_throughput,height=0.055,distance=2e4,prominence=0.01)
		order_cen_lam    = so.stel.v[order_peaks[0]]
		blaze_angle      =  76
		order_indices    =[]
		for i,lam_cen in enumerate(order_cen_lam):
			line_spacing = 0.02 if lam_cen < 1475 else 0.01
			m = np.sin(blaze_angle*np.pi/180) * 2 * (1/line_spacing)/(lam_cen/1000)
			fsr  = lam_cen/m
			isub_test= np.where((so.stel.v> (lam_cen - fsr/2)) & (so.stel.v < (lam_cen+fsr/2))) #FINISH THIS
			#plt.plot(so.stel.v[isub_test],total_throughput[isub_test],'k--')
			order_indices.append(np.where((so.obs.v > (lam_cen - 1.3*fsr/2)) & (so.obs.v  < (lam_cen+1.3*fsr/2)))[0])

		return order_cen_lam,order_indices

	def make_telluric_mask(so,cutoff=0.01):
		telluric_spec = np.abs(so.tel.s/so.tel.rayleigh)**so.tel.airmass
		telluric_spec[np.where(np.isnan(telluric_spec))] = 0
		telluric_spec_lores = degrade_spec(so.stel.v, telluric_spec, so.inst.res)
		# resample onto v array
		filt_interp     = interpolate.interp1d(so.stel.v, telluric_spec_lores, bounds_error=False,fill_value=0)
		s_tel           = filt_interp(so.obs.v)/np.max(filt_interp(so.obs.v))    # filter profile resampled to phoenix times phoenix flux density

		cutoff = 0.01 # reject lines greater than 1% depth
		telluric_mask = np.ones_like(s_tel)
		telluric_mask[np.where(s_tel < (1-cutoff))[0]] = 0

		return telluric_mask

	def get_rv_content(v,s,n):
		"""
		"""
		flux_interp = interpolate.InterpolatedUnivariateSpline(v,s, k=1)
		dflux = flux_interp.derivative()
		spec_deriv = dflux(v)
		sigma_ord = np.abs(s) ** 0.5 # np.abs(n)
		sigma_ord[np.where(sigma_ord ==0)] = 1e10
		all_w = (v ** 2.) * (spec_deriv ** 2.) / sigma_ord ** 2. # include read noise and dark here!!
		
		return all_w

	def get_rv_precision(all_w,order_cens,order_inds,noise_floor=0.5,mask=None):
		dv_vals = np.zeros_like(order_cens)
		for i,lam_cen in enumerate(order_cens):
			w_ord = all_w[order_inds[i]] * telluric_mask[order_inds[i]]
			dv_order  = SPEEDOFLIGHT / (np.nansum(w_ord[1:-1])**0.5) # m/s
			dv_vals[i]  = dv_order
		
		dv_tot  = np.sqrt(dv_vals**2 + noise_floor**2)
		dv_spec  = 1. / (np.nansum(1./dv_vals**2.))**0.5

		return dv_tot,dv_spec

	order_cens, order_inds  = get_order_bounds(so)
	telluric_mask           = make_telluric_mask(so,cutoff=0.01)
	all_w                   = get_rv_content(so.obs.v,so.obs.s,so.obs.noise)
	dv_vals,dv_spec         = get_rv_precision(all_w,order_cens,order_inds,noise_floor=0.5,mask=telluric_mask)

	fig,axs=plot_rv_err(so, lam_cen, dv_vals)





