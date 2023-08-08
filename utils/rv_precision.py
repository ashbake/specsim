# calc signal to noise
import sys,os
import matplotlib
import numpy as np
import matplotlib.pylab as plt
from scipy import interpolate
import pandas as pd
from matplotlib.colors import LogNorm
from datetime import date

font = {'size'   : 14}
matplotlib.rc('font', **font)

sys.path.append('/Users/huihaoz/specsim/utils/')
from objects import load_object
from load_inputs import fill_data, load_filter,get_band_mag
from functions import *
from noise_tools import get_sky_bg, get_inst_bg, sum_total_noise
from throughput_tools import pick_coupling, get_base_throughput
from wfe_tools import get_tip_tilt_resid, get_HO_WFE
import plot_tools,obs_tools

water_only=False;line_spacing=None; peak_spacing=2e4;height=0.055

def get_order_bounds(so,line_spacing=0.02,peak_spacing=1e3,height=0.055):
	"""
	given array, return max and mean of snr per order
	"""
	order_peaks	  = signal.find_peaks(so.inst.base_throughput,height=height,distance=peak_spacing,prominence=0.01)
	order_cen_lam	= so.stel.v[order_peaks[0]]
	blaze_angle	  =  76
	order_indices	=[]
	for i,lam_cen in enumerate(order_cen_lam):
		if line_spacing == None: line_spacing_now = 0.02 if lam_cen < 1475 else 0.01
		else: line_spacing_now=line_spacing
		m = np.sin(blaze_angle*np.pi/180) * 2 * (1/line_spacing_now)/(lam_cen/1000)
		fsr  = lam_cen/m
		isub_test= np.where((so.stel.v> (lam_cen - fsr/2)) & (so.stel.v < (lam_cen+fsr/2))) #FINISH THIS
		#plt.plot(so.stel.v[isub_test],so.inst.ytransmit[isub_test],'k--')
		order_indices.append(np.where((so.obs.v > (lam_cen - 0.9*fsr/2)) & (so.obs.v  < (lam_cen+0.9*fsr/2)))[0])

	return order_cen_lam,order_indices


def make_telluric_mask(so,cutoff=0.01,velocity_cutoff=5,water_only=False):
	"""
	"""
	telluric_spec = np.abs(so.tel.s/so.tel.rayleigh)**so.tel.airmass
	if water_only: telluric_spec = np.abs(so.tel.h2o)**so.tel.airmass #h2o only
	telluric_spec[np.where(np.isnan(telluric_spec))] = 0
	telluric_spec_lores = degrade_spec(so.stel.v, telluric_spec, so.inst.res)
	# resample onto v array
	filt_interp	 = interpolate.interp1d(so.stel.v, telluric_spec_lores, bounds_error=False,fill_value=0)
	s_tel		 = filt_interp(so.obs.v)/np.max(filt_interp(so.obs.v))	# filter profile resampled to phoenix times phoenix flux density

	#cutoff = 0.01 # reject lines greater than 1% depth
	telluric_mask = np.ones_like(s_tel)
	telluric_mask[np.where(s_tel < (1-cutoff))[0]] = 0
	# avoid +/-5km/s  (5pix) around telluric
	for iroll in range(velocity_cutoff):
		telluric_mask[np.where(np.roll(s_tel,iroll) < (1-cutoff))[0]] = 0
		telluric_mask[np.where(np.roll(s_tel,-1*iroll) < (1-cutoff))[0]] = 0

	return telluric_mask,s_tel

def get_rv_content(v,s,n):
	"""
	"""
	flux_interp = interpolate.InterpolatedUnivariateSpline(v,s, k=1)
	dflux = flux_interp.derivative()
	spec_deriv = dflux(v)
	sigma_ord = np.abs(n) #np.abs(s) ** 0.5 # np.abs(n)
	sigma_ord[np.where(sigma_ord ==0)] = 1e10
	all_w = (v ** 2.) * (spec_deriv ** 2.) / sigma_ord ** 2. # include read noise and dark here!!
	
	return all_w

def get_rv_precision(all_w,order_cens,order_inds,noise_floor=0.5,mask=None):
	if np.any(mask==None):
		mask = np.ones_like(all_w)
	dv_vals = np.zeros_like(order_cens)
	for i,lam_cen in enumerate(order_cens):
		w_ord = all_w[order_inds[i]] * mask[order_inds[i]]
		dv_order  = SPEEDOFLIGHT / (np.nansum(w_ord[1:-1])**0.5) # m/s
		dv_vals[i]  = dv_order
	
	dv_tot  = np.sqrt(dv_vals**2 + noise_floor**2)
	dv_spec  = 1. / (np.nansum(1./dv_vals**2.))**0.5
	dv_spec_floor  = 1. / (np.nansum(1./dv_tot**2.))**0.5

	return dv_tot,dv_spec,dv_vals


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
	dv_yj_tot = (so.inst.rv_floor**2 +dv_yj**2.)**0.5	# 
	dv_hk_tot = (so.inst.rv_floor**2 +dv_hk**2.)**0.5	# # 

	axs[1].text(1050,2*np.median(dv_vals),'$\sigma_{yJ}$=%sm/s'%round(dv_yj_tot,1),fontsize=12,zorder=101)
	axs[1].text(1500,2*np.median(dv_vals),'$\sigma_{HK}$=%sm/s'%round(dv_hk_tot,1),fontsize=12,zorder=101)
	ax2.legend(fontsize=8,loc=1)
	if savefig:
		plt.savefig('./output/rv_precision/RV_precision_%sK_%smag%s_%ss_vsini%skms.png'%(so.stel.teff,so.filt.band,so.stel.mag,so.obs.texp,so.stel.vsini))

	return fig,axs


if __name__=='__main__':
	SPEEDOFLIGHT = 2.998e8 # m/s
	GAUSSCONST = (2. * (2. * np.log(2))**0.5)

	#load inputs
	configfile = './hispec_snr.cfg'
	so    = load_object(configfile)
	cload = fill_data(so) # put coupling files in load and wfe stuff too

	plot_tools.plot_snr_orders(so,snrtype=1,mode='peak',savepath='./output/')

	# RV
	water_only=False;line_spacing=None; peak_spacing=2e4;height=0.055
	order_cens, order_inds  = get_order_bounds(so,line_spacing=line_spacing,peak_spacing=peak_spacing,height=height) # None and 2e4 for hispec
	telluric_mask,s_tel     = make_telluric_mask(so,cutoff=0.01,velocity_cutoff=10,water_only=water_only)
	all_w				    = get_rv_content(so.obs.v,so.obs.s,so.obs.noise)
	dv_tot,dv_spec,dv_vals	= get_rv_precision(all_w,order_cens,order_inds,noise_floor=so.inst.rv_floor,mask=telluric_mask)

	#fig,axs = plot_rv_err_agnostic(so, order_cens, dv_vals, savefig=False)
	fig,axs = plot_rv_err(so, order_cens, dv_vals, savefig=False)