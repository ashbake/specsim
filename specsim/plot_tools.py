##############################################################
# General functions for calc_snr_max
###############################################################

import numpy as np
from scipy import interpolate
from scipy import signal
import matplotlib.pylab as plt
from astropy.io import fits
import pandas as pd

import os,sys
sys.path.append('./')
os.chdir('./')

from specsim import ccf_tools, noise_tools, wfe_tools, obs_tools, throughput_tools
from specsim.objects import load_object
from specsim.load_inputs import fill_data
from specsim.functions import *



all = {}

DATAPATH = '../data/'
SAVEPATH  = '../output/'


# instrument project plots
def plot_doppler_spectrographs(so,cload):
	"""
	make figures to add to sam's cool rv landscape plot
	"""	
	configfile = './configs/hispec_snr.cfg'
	so         = load_object(configfile)
	so.inst.l0 = 370
	so.inst.lf = 2650
	cload = fill_data(so) # put coupling files in load and wfe stuff too

	# plot m star and g star
	cload.set_teff_mag(so,2500,10,star_only=True)
	mstar = so.stel.s
	cload.set_teff_mag(so,5800,10,star_only=True)
	gstar = so.stel.s
	
	fig, ax = plt.subplots(1,1, figsize=(15,4),facecolor='black')	
	ax.set_facecolor('black')
	ax.plot(so.stel.v,mstar*2,'firebrick',lw=0.5)
	ax.plot(so.stel.v,gstar,'royalblue',lw=0.5)	
	ax.set_ylim(-10,5000)
	ax.set_xlim(380,2650)

	#tellurics
	tel_nir = so.tel.s.copy()
	data = fits.getdata('./data/telluric/psg_out_2015.06.17_l0_380nm_l1_900nm_res_0.002nm_lon_204.53_lat_19.82_pres_0.5826.fits')
	_,ind     = np.unique(data['Wave/freq'],return_index=True)
	tck_tel   = interpolate.splrep(data['Wave/freq'][ind],data['Total'][ind], k=2, s=0)
	tel_vis = interpolate.splev(cload.x,tck_tel,der=0,ext=1)

	fig, ax = plt.subplots(1,1, figsize=(15,4),facecolor='black')	
	ax.set_facecolor('black')
	ax.plot(so.stel.v[1000001:],tel_nir[1000001:],'gray',lw=0.5)
	ax.plot(so.stel.v[0:1000001],tel_vis[0:1000001],'gray',lw=0.5)
	ax.set_ylim(-2,2)
	ax.set_xlim(380,2650)

	# plot filters
	families=['Johnson','cfht','2mass','2mass','2mass']
	for i,band in enumerate(['R','y','J','H','K']):
		so.filt.family=families[i]
		so.filt.band=band
		cload.filter(so)
		ax.plot(so.filt.xraw,so.filt.yraw*.5,'darkgray',lw=0.8)

def plot_rv_err(so,savefig=True,savepath=SAVEPATH):
	"""
	plots RV precision and SNR spectrum 
	
	inputs
	------
	so
	savefig  - bool
	savepath - defaults SAVEPATH defined here 

	"""
	col_table = plt.get_cmap('Spectral_r')
	fig, axs = plt.subplots(2,figsize=(7,7),sharex=True)
	plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.85,top=0.85)

	axs[1].plot([950,2400],[so.inst.rv_floor,so.inst.rv_floor],'k--',lw=0.7)
	axs[1].fill_between([1450,2400],0,1e10,facecolor='gray',alpha=0.2)
	axs[1].fill_between([980,1330],0,1e10,facecolor='gray',alpha=0.2)
	axs[1].grid('True')
	axs[1].set_ylim(-0,3*np.median(so.obs.rv_order))
	axs[1].set_xlim(950,2400)
	axs[1].set_ylabel('$\sigma_{RV}$ [m/s]')
	axs[1].set_xlabel('Wavelength [nm]')

	axs[0].set_ylabel('SNR/pixel')
	axs[0].set_title('M$_%s$=%s, T$_{eff}$=%sK,\n $t_{exp}$=%ss, vsini=%skm/s'%(so.filt.band,so.stel.mag,int(so.stel.teff),int(so.obs.texp),so.stel.vsini))

	axs[0].grid('True')
	ax2 = axs[0].twinx() 
	ax2.plot(so.tel.v,so.tel.s,'gray',alpha=0.5,zorder=-100,label='Telluric Absorption')
	ax2.plot(so.stel.v,so.inst.ytransmit,'k',alpha=0.5,zorder=-100,label='Total Throughput')
	ax2.set_ylabel('Transmission',fontsize=12)
	for i,lam_cen in enumerate(so.inst.order_cens):
		wvl_norm = (lam_cen - 900.) / (2500. - 900.)
		axs[0].plot(so.obs.v_res_element[so.obs.order_inds[i]],so.obs.snr_res_element[so.obs.order_inds[i]]/np.sqrt(so.inst.res_samp),zorder=200,color=col_table(wvl_norm))
		axs[1].plot(lam_cen,so.obs.rv_order[i],'o',zorder=100,color=col_table(wvl_norm),markeredgecolor='k')
	
	sub_yj = so.obs.rv_order[np.where((so.obs.rv_order!=np.inf) & (so.inst.order_cens < 1400))[0]]
	sub_hk = so.obs.rv_order[np.where((so.obs.rv_order!=np.inf) & (so.inst.order_cens > 1400))[0]]
	rvmed_yj = np.sqrt(np.sum(so.obs.rv_order[np.where((so.obs.rv_order!=np.inf) & (so.inst.order_cens < 1400))[0]]**2))/np.sum(sub_yj)
	rvmed_hk = np.median(so.obs.rv_order[np.where((so.obs.rv_order!=np.inf) & (so.inst.order_cens > 1400))[0]])
	dv_yj = 1. / (np.nansum(1./sub_yj**2.))**0.5	# 
	dv_hk = 1. / (np.nansum(1./sub_hk**2.))**0.5	# 
	dv_yj_tot = (so.inst.rv_floor**2 +dv_yj**2.)**0.5	# 
	dv_hk_tot = (so.inst.rv_floor**2 +dv_hk**2.)**0.5	# # 
	# 2*np.median(so.obs.rv_order)
	axs[1].text(1050,.5,'$\sigma_{yJ}$=%sm/s'%round(dv_yj_tot,1),fontsize=12,zorder=101)
	axs[1].text(1500,.5,'$\sigma_{HK}$=%sm/s'%round(dv_hk_tot,1),fontsize=12,zorder=101)
	ax2.legend(fontsize=8,loc=1)
	if savefig:
		plt.savefig(savepath + './RV_precision_%s_%sK_%smag%s_%ss_vsini%skms.png'%(so.run.tag,so.stel.teff,so.filt.band,so.stel.mag,so.obs.texp,so.stel.vsini))

	return fig,axs


def plot_rv_err_gen(v,s,order_cens,rv_order,rv_floor=0.3,savefig=True,tag='test',annotate=False):
	"""

	"""
	col_table = plt.get_cmap('Spectral_r')
	fig, axs = plt.subplots(2,figsize=(7,7),sharex=True)
	plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.85,top=0.85)

	axs[1].plot([950,2400],[rv_floor,rv_floor],'k--',lw=0.7)
	axs[1].fill_between([1450,2400],0,1e10,facecolor='gray',alpha=0.2)
	axs[1].fill_between([980,1330],0,1e10,facecolor='gray',alpha=0.2)
	axs[1].grid('True')
	axs[1].set_ylim(-0,3*np.median(rv_order))
	axs[1].set_xlim(950,2400)
	axs[1].set_ylabel('$\sigma_{RV}$ [m/s]')
	axs[1].set_xlabel('Wavelength [nm]')

	axs[0].set_ylabel('SNR/pixel')
	axs[0].set_title(tag)

	axs[0].grid('True')
	for i,lam_cen in enumerate(order_cens):
		wvl_norm = (lam_cen - 900.) / (2500. - 900.)
		axs[0].plot(v,s,zorder=200,color=col_table(wvl_norm))
		axs[1].plot(lam_cen,rv_order[i],'o',zorder=100,color=col_table(wvl_norm),markeredgecolor='k')
	
	if annotate:
		sub_yj = rv_order[np.where((rv_order!=np.inf) & (order_cens < 1400))[0]]
		sub_hk = rv_order[np.where((rv_order!=np.inf) & (order_cens > 1400))[0]]
		dv_yj = 1. / (np.nansum(1./sub_yj**2.))**0.5	# 
		dv_hk = 1. / (np.nansum(1./sub_hk**2.))**0.5	# 
		dv_yj_tot = (rv_floor**2 +dv_yj**2.)**0.5	# 
		dv_hk_tot = (rv_floor**2 +dv_hk**2.)**0.5	# # 
		# 2*np.median(so.obs.rv_order)
		axs[1].text(1050,0.5,'$\sigma_{yJ}$=%sm/s'%round(dv_yj_tot,1),fontsize=12,zorder=101)
		axs[1].text(1500,0.5,'$\sigma_{HK}$=%sm/s'%round(dv_hk_tot,1),fontsize=12,zorder=101)
	if savefig:
		plt.savefig('./RV_precision_%s.png'%tag)

	return fig,axs



def load_brown_dwarfs_AB():
    """
    load adam burgasser's file

    returns hmag, teff
    """
    xl = pd.ExcelFile('/Users/ashbake/Documents/Research/Projects/HISPEC/Tests/TrackingCamera/data/ucd_sheet_teff.xlsx')
    #xl.sheet_names
    first_sheet = xl.sheet_names[0]
    df = xl.parse(first_sheet)
    #df.head()
    teff = df['teff']
    hmag = df['H_2MASS']

    return  hmag.values,teff.values



# STAND ALONE PLOTTING FUNCTIONS
def setup_plot_style():
	"""
	"""
	import matplotlib
	font = {'size'   : 14}
	matplotlib.rc('font', **font)
	plt.rcParams['font.size'] = '14'
	plt.rcParams['font.family'] = 'sans'
	plt.rcParams['axes.linewidth'] = '1.3'
	fontname = 'DIN Condensed'#'Arial Narrow'

def plot_stellar_colors():
	"""
	open colors of stars relative to H band and plot
	"""
	f = pd.read_csv(DATAPATH + 'WFE/HAKA/color_curves.csv',delimiter='\t')
	
	plt.figure()
	bands = f['Temp'].values
	for key in f.keys():
		if key=='Temp':continue
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
	plt.savefig(SAVEPATH + 'stellar_colors_H.png')

def plot_tracking_cam_spot_rms(camera='h2rg'):
	"""
	"""
	#f = np.loadtxt('./data/WFE/trackingcamera_optics/OAP1_HISPEC_FEI_RMS_SpotRvsField.txt')
	#f = np.loadtxt(DATAPATH + 'WFE/trackingcamera_optics/HISPEC_ParaxialTel_OAP_TrackCamParax_SpotSizevsField.txt')
	f = np.loadtxt('/Users/ashbake/Documents/Research/Projects/HISPEC/SNR_calcs/data/WFE/trackingcamera_optics/HISPEC_ParaxialTel_OAP_TrackCamParax_SpotSizevsField.txt')
	field, rmstot, rms900,rms1000,rms1200,rms1400,rms1600,rms2200  = f.T #field [deg], rms [um]
	#_,pixel_pitch,_,_ = select_tracking_cam(camera=camera)
	pixel_pitch=18
	plt.figure()
	# multiply rms by sqrt (2) to get a diagonal cut, multiple by 2 to get diameter
	plt.plot(field*3600,np.sqrt(2) * 2*rmstot/pixel_pitch,label='total') 
	plt.plot(field*3600,np.sqrt(2) * 2*rms900/pixel_pitch,label='900nm')
	plt.plot(field*3600,np.sqrt(2) * 2*rms2200/pixel_pitch,label='2200nm')
	plt.xlabel('Field Radius [arcsec]')
	plt.ylabel('RMS Diameter [pix]')
	plt.title('Tracking Camera Spot RMS')
	plt.legend()

	plt.savefig(SAVEPATH + 'tracking_cam_spot_RMS.png')

def plot_cool_stars():
	planets_filename = './data/populations/rv_less2earthrad_less360Teq_less4000Teff_planets_.csv'
	planet_data =  pd.read_csv(planets_filename,delimiter=',',comment='#')

	hmags = planet_data['sy_hmag']
	teffs = planet_data['st_teff']
	mass = planet_data['pl_bmassj']
	teq  = planet_data['pl_eqt']
	names = planet_data['pl_name']
	hostnames = planet_data['hostname']
	rvamps = planet_data['pl_rvamp']

	plt.figure()
	plt.scatter(teffs,hmags)

def plot_brown_dwarfs():
	"""
	"""
	bd_filename = './data/populations/UltracoolSheetMain.csv'
	bd_data =  pd.read_csv(bd_filename,delimiter=',',comment='#')

	#sp_type = bd_data['spt_opt']
	sp_type = bd_data['spt_ir']
	hmags = bd_data['H_MKO']
	#jmags = bd_data['J_2MASS']
	jmags = bd_data['J_MKO']
	W1    = bd_data['W1']
	W2    = bd_data['W2']
	sp  =[]
	for x in sp_type.values:
		if type(x)==str: 
			sp.append(x[0:2])
		else:
			sp.append(0)

	teffs = np.zeros_like(sp_type,dtype=float)
	teffs[np.where(sp_type.values=='M6')[0]] = 2600
	teffs[np.where(sp_type.values=='M7')[0]] = 2400
	teffs[np.where(sp_type.values=='M8')[0]] = 2200
	teffs[np.where(sp_type.values=='M9')[0]] = 2100
	teffs[np.where(sp_type.values=='L0')[0]] = 2000
	teffs[np.where(sp_type.values=='L1')[0]] = 1950
	teffs[np.where(sp_type.values=='L2')[0]] = 1900
	teffs[np.where(sp_type.values=='L3')[0]] = 1850
	teffs[np.where(sp_type.values=='L4')[0]] = 1800
	teffs[np.where(sp_type.values=='L5')[0]] = 1750
	teffs[np.where(sp_type.values=='L6')[0]] = 1700
	teffs[np.where(sp_type.values=='L7')[0]] = 1600
	teffs[np.where(sp_type.values=='L8')[0]] = 1500
	teffs[np.where(sp_type.values=='T1')[0]] = 1300
	teffs[np.where(sp_type.values=='T2')[0]] = 1200
	teffs[np.where(sp_type.values=='T3')[0]] = 1100
	teffs[np.where(sp_type.values=='T4')[0]] = 1000
	teffs[np.where(sp_type.values=='T5')[0]] = 900
	teffs[np.where(sp_type.values=='T6')[0]] = 800

	plt.figure()
	plt.hist(hmags,bins=100,alpha=0.7)

	plt.figure()
	plt.plot(hmags,teffs,'o',alpha=0.8)
	plt.xlabel('H Mag')
	plt.ylabel('T$_{eff}$ (K)')
	plt.subplots_adjust(left=0.15)

def plot_throughput_nice(telluric_file,datapath='./data/throughput/hispec_subsystems_11032022/',outputdir='../output/'):
    """
    plot throughput plot for MRI proposal
    """
    # plot red only
    telluric_file = './data/telluric/psg_out_2020.08.02_l0_800nm_l1_2700nm_res_0.001nm_lon_204.53_lat_19.82_pres_0.5826.fits'
    outputdir     = './output/'
    w, ngs_bspec = np.loadtxt(outputdir + 'ngs_throughput_bspec.txt').T
    w, lgs_bspec = np.loadtxt(outputdir + 'lgs_throughput_bspec.txt').T
    w, ngs_rspec = np.loadtxt(outputdir + 'ngs_throughput_HK.txt').T
    w, lgs_rspec = np.loadtxt(outputdir + 'lgs_throughput_HK.txt').T

    plt.figure(figsize=(10,5))
    ib = np.where(w < 1.333)[0]
    ir = np.where(w > 1.333)[0]
    #cen_lam, snr_peaks,snr_means = obs_tools.get_order_value(so,w*1000,ngs_bspec)
    plt.plot(w[ib],ngs_bspec[ib],c='seagreen',linewidth=1)
    plt.plot(w[ir],ngs_bspec[ir],c='seagreen',linewidth=1)
    
    plt.plot(w[ib],lgs_bspec[ib],c='gray',linewidth=1)
    plt.plot(w[ir],lgs_rspec[ir],c='gray',linewidth=1)
    
    plt.xlabel('Wavelength (microns)',color='k')
    plt.ylabel('End-to-End Throughput',color='k')
    plt.ylim(0.005,0.075)
    #plt.xlim(0.980, 1.490)
    plt.axhline(np.max(ngs),c='k',linestyle='--',linewidth=2)
    plt.fill_between([1.33, 1.49],0.00,y2=1,facecolor='w',zorder=110)
    plt.fill_between([0.98, 1.07],0.0,y2=1,facecolor='gray',alpha=0.2,zorder=-110)
    plt.fill_between([1.170,1.327],0.0,y2=1,facecolor='gray',alpha=0.2,zorder=-110)
    plt.fill_between([1.490,1.780],0.0,y2=1,facecolor='gray',alpha=0.2,zorder=110)
    plt.fill_between([1.990,2.460],0.0,y2=1,facecolor='gray',alpha=0.2,zorder=110)

    #plt.title("HISPEC E2E Except Coupling")
    # y lines
    yticks = [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.8]
    #yticks = [0.01, 0.03, 0.09, 0.27, 0.81]
    xticks = np.round((np.arange(0.98, 1.49,0.04)),2)
    plt.yticks(ticks=yticks,labels=yticks,color='k',fontsize=12)
    plt.xticks(rotation=90,ticks=xticks,labels=xticks,color='k',fontsize=12)
    plt.grid(axis='y',alpha=0.4)
    plt.subplots_adjust(bottom=0.17)
    plt.title('HISPEC yJ Throughput')
    #plt.savefig(outputdir + 'e2e_mri_plot_yJ.png')
    #plt.savefig(outputdir + 'e2e_mri_plot_yJ.pdf')


# REQUIRES SO INSTANCE
def plot_snr(so,snrtype='pixel',savepath='./'):
	"""
	Plots SNR calculated in so instance
	
	inputs
	------
	so : object
		instance of specsim storage object

	snrtype: 'pixel' or 'res_element'
		'pixel' selects per pixel SNR
		'res_element' selects per resolution element SNR

	savepath: str
		path to save the plot to
	"""
	fig, ax = plt.subplots(1,1, figsize=(10,8))	
	if snrtype =='pixel':  ax.plot(so.obs.v,so.obs.snr)
	elif snrtype=='res_element': ax.plot(so.obs.v_res_element,so.obs.snr_res_element)
	else: print('Choose pixel or res_element for snrtype'); return
	ax.set_ylabel('SNR')
	ax.set_xlabel('Wavelength (nm)')
	ax.set_title('AO Mode: %s, %s=%s, t=%shr, Teff=%sK'%(so.ao.mode_chosen,\
			so.filt.band,round(so.stel.mag,1),np.round(so.obs.texp/3600,2),\
			int(so.stel.teff)))
	ax.axhline(y=30,color='k',ls='--')
	#plt.legend()
	# duplicate axis to plot filter response
	ax2 = ax.twinx()
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
	figname = 'snr_%s_%smag_%s_texp_%ss_dark_%s.png' %(so.ao.mode,so.filt.band,so.stel.mag,so.obs.texp,so.inst.darknoise)
	plt.savefig(savepath + figname)

def plot_snr_orders(so,snrtype='res_element',mode='mean',height=0.055,savepath=SAVEPATH):
	"""
	inputs:
	-------
	so : object
		instance of specsim storage object

	snrtype: 'pixel' or 'res_element'
		'pixel' selects per pixel SNR
		'res_element' selects per resolution element SNR
	
	mode: 'mean' or 'peak'
		plots SNR as either the average ('mean') or the peak ('peak') of each order

	"""
	if snrtype=='pixel': cen_lam, snr_peaks,snr_means = obs_tools.get_order_value(so.obs.v,so.obs.snr,so.inst.order_bounds_file)
	if snrtype=='res_element':cen_lam, snr_peaks,snr_means = obs_tools.get_order_value(so.obs.v_res_element,so.obs.snr_res_element,so.inst.order_bounds_file)

	fig, ax = plt.subplots(1,1, figsize=(8,6))	
	if mode=='peak': ax.plot(cen_lam, snr_peaks,lw=2)
	elif mode=='mean': ax.plot(cen_lam, snr_means,lw=2)
	ax.set_ylabel('SNR')
	ax.set_xlabel('Wavelength (nm)')
	ax.set_title('AO Mode: %s, T$_{eff}$=%sK, %s=%s, t=%shr'%(so.ao.mode,so.stel.teff,so.filt.band,so.stel.mag,round(so.obs.texp/3600,2)))
	#ax.axhline(y=30,color='k',ls='--')
	figname = 'snr_%s_%smag_%s_texp_%ss_dark_%s.png' %(so.ao.mode,so.filt.band,so.stel.mag,so.obs.texp,so.inst.darknoise)

	# duplicate axis to plot filter response
	# plot band
	ax.fill_between(so.inst.y,0,np.max(snr_peaks)+10,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(20+np.min(so.inst.y),0.09, 'y')
	ax.fill_between(so.inst.J,0,np.max(snr_peaks)+10,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(so.inst.J),0.09, 'J')
	ax.fill_between(so.inst.H,0,np.max(snr_peaks)+10,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(so.inst.H),0.09, 'H')
	ax.fill_between(so.inst.K,0,np.max(snr_peaks)+10,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(so.inst.K),0.09, 'K')
	ax.set_xlim(970,2500)
	ax.set_ylim(0,np.max(snr_peaks)+10)
	ax.fill_between([1333,1500],0,np.max(snr_peaks)+10,facecolor='white',zorder=100)

	plt.savefig(savepath + figname)

	return cen_lam, snr_peaks,snr_means
	
def plot_base_throughput(so,savepath=SAVEPATH):
	"""
	"""
	fig, ax = plt.subplots(1,1, figsize=(10,8))	
	ax.plot(so.inst.xtransmit,so.inst.base_throughput)
	peak = np.max(so.inst.base_throughput)

	ax.set_ylabel('Base Throughput')
	ax.set_xlabel('Wavelength (nm)')
	ax.axhline(y=30,color='k',ls='--')
	figname = 'base_throughput.png' 
	# duplicate axis to plot filter response
	# plot band
	ax.fill_between(so.inst.y,0,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(20+np.min(so.inst.y),0.09, 'y')
	ax.fill_between(so.inst.J,0,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(so.inst.J),0.09, 'J')
	ax.fill_between(so.inst.H,0,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(so.inst.H),0.09, 'H')
	ax.fill_between(so.inst.K,0,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(so.inst.K),0.09, 'K')
	ax.set_xlim(970,2500)
	plt.grid()
	ax.set_ylim(0,peak)
	plt.savefig(savepath + figname)

def plot_coupling(so,savepath=SAVEPATH):
	"""
	"""
	fig, ax = plt.subplots(1,1, figsize=(10,8))	
	#ax.plot(so.inst.xtransmit,so.inst.ytransmit)
	ax.plot(so.inst.xtransmit,so.inst.coupling)
	peak = np.max(so.inst.coupling)

	ax.set_ylabel('Coupling')
	ax.set_xlabel('Wavelength (nm)')
	ax.axhline(y=30,color='k',ls='--')
	figname = 'coupling_throughput.png' 
	# duplicate axis to plot filter response
	# plot band
	ax.fill_between(so.inst.y,0,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(20+np.min(so.inst.y),0.09, 'y')
	ax.fill_between(so.inst.J,0,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(so.inst.J),0.09, 'J')
	ax.fill_between(so.inst.H,0,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(so.inst.H),0.09, 'H')
	ax.fill_between(so.inst.K,0,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(so.inst.K),0.09, 'K')
	ax.set_xlim(970,2500)
	plt.grid()
	ax.set_ylim(0,peak)
	plt.savefig(savepath + figname)

def plot_track_background(so,savepath=SAVEPATH):
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
	axs[0].set_ylabel('Sky Bkg \n(phot/nm/s)')
	axs[1].set_ylabel('Instrument Bkg \n(phot/nm/s)')
	axs[2].set_ylabel('Source Photons \n (phot/nm/s)')
	axs[2].set_xlabel('Wavelength [nm]')
	axs[0].set_ylim(-0,1000)
	axs[1].set_ylim(-0,20)
	axs[0].plot(so.stel.v,so.track.sky_bg_spec,'b',alpha=0.5,zorder=100,label='Sky Background')
	axs[1].plot(so.stel.v,so.track.inst_bg_spec,'m',lw=2,alpha=0.5,zorder=100,label='Instrument Background')
	axs[2].plot(so.stel.v,so.track.signal_spec/so.track.texp,'g',alpha=0.5,zorder=100,label='Source Photons')

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
	#plt.savefig('./output/trackingcamera/noise_flux_%sK_%s_%smag.pdf'%(so.stel.teff,so.filt.band,so.stel.mag))
	plt.savefig(savepath + 'noise_flux_%sK_%s_%smag.png'%(so.stel.teff,so.filt.band,so.stel.mag))

def plot_spec_background(so,savepath=SAVEPATH):
	"""
	
	"""
	col_table = plt.get_cmap('Spectral_r')
	fig, ax = plt.subplots(1,figsize=(7,4),sharex=True)
	axs = [ax,]
	plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.85,top=0.85)

	axs[0].plot([950,2400],[0.5,0.5],'k--',lw=0.7)
	#axs[0].fill_between([1450,2400],0,1000,facecolor='gray',alpha=0.2)
	#axs[0].fill_between([980,1330],0,1000,facecolor='gray',alpha=0.2)
	axs[0].grid('True')
	axs[0].set_xlim(950,2450)
	#axs[1].set_ylabel('Sky Bkg \n(phot/nm/s)')
	axs[0].set_ylim(-0.001,0.6)
	#axs[1].set_ylim(-0,20)
	#axs[1].plot(so.stel.v,so.obs.sky_bg_ph,'m',alpha=0.5,zorder=100,label='Sky Background')
	#axs[0].plot(so.obs.v,so.obs.inst_bg_ph,'b',lw=2,alpha=0.5,zorder=100,label='Instrument Background')
	axs[0].plot(so.obs.v,so.obs.sky_bg_ph,'b',lw=2,alpha=0.5,zorder=100,label='Instrument Background')
	axs[0].set_xlabel('Wavelength [nm]')
	axs[0].set_ylabel('Instrument + Sky Bkg \n(phot/s)')

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
		ax2.set_ylim(0.1,1)

	plt.savefig(savepath + 'noise_flux_%sK_%s_%smag.png'%(so.stel.teff,so.filt.band,so.stel.mag))
	#np.savetxt('inst_background.txt',np.vstack((so.obs.v,so.obs.inst_bg_ph)).T,header='wave[nm],bkg[ph/s]')
	#np.savetxt('sky_background.txt',np.vstack((so.obs.v,so.obs.sky_bg_ph)).T,header='wave[nm],bkg[ph/s]')

def plot_tracking_bands(so,trackbands=['J','JHgap','H'],savepath=SAVEPATH):
	"""
	"""
	#trackbands=['y','Jplus','H','K'] #['J','JHgap','H'] #'Hplus50','Jplus']#['y',

	spectrum = so.track.ytransmit * so.tel.s * so.stel.s/np.max(so.stel.s) / 1.5
	spec_lores = degrade_spec(so.stel.v[::10], spectrum[::10], 2000)
	star_lores = degrade_spec(so.stel.v[::10], so.stel.s[::10]/np.max(so.stel.s), 2000)

	plt.figure(figsize=(8,5))
	for band in trackbands:
		print(band)
		bandpass, center_wavelength = obs_tools.get_tracking_band(so.stel.v,band)
		p = plt.plot(so.stel.v[::100],bandpass[::100],linewidth=1)
		plt.fill_between(so.stel.v,-1,bandpass,alpha=0.1,facecolor=p[0].get_color(),edgecolor=p[0].get_color())
		if band!='JHgap': plt.text(center_wavelength-10, np.max(bandpass),band,c=p[0].get_color())
		if band=='JHgap': plt.text(center_wavelength-50, 0.95,' JH\nGap',c=p[0].get_color())
	
	# get J band
	Jbandpass, center_wavelength = obs_tools.get_tracking_band(so.stel.v,'J')
	sumflux_J = np.trapz(spectrum[np.where(Jbandpass>0.1)],so.stel.v[np.where(Jbandpass>0.1)])
	for i,band in enumerate(trackbands):
		bandpass, center_wavelength = obs_tools.get_tracking_band(so.stel.v,band)
		sumflux = np.trapz(spectrum[np.where(bandpass>0.1)],so.stel.v[np.where(bandpass>0.1)])
		if i%2==0: plt.text(center_wavelength-50, 0.8*np.max(bandpass),str(round(100*sumflux/sumflux_J,1))+'%',fontsize=10)
		if i%2==1: plt.text(center_wavelength-50, 0.9*np.max(bandpass),str(round(100*sumflux/sumflux_J,1))+'%',fontsize=10)
	
	plt.fill_between([500,970],0,2,alpha=0.1,facecolor='m')
	plt.text(550, 0.95,'Visible\nWFS',c='m')

	plt.plot(so.stel.v[::10],star_lores,'k',zorder=-100,label='T=%sK'%so.stel.teff)
	plt.plot(so.stel.v[::10],spec_lores,'gray',alpha=0.8,zorder=-101,label='Throughput x \n Normalized Flux')
	plt.ylim(0,1.15)
	plt.title('Tracking Camera Filter Profiles')
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Relative Transmission')
	plt.legend(fontsize=10,loc=7)

	if plot_telluric: 
		ytel = degrade_spec(so.tel.v,so.tel.s,4000)
		plt.plot(so.tel.v[::100],ytel[::100],'gray',alpha=0.3)

	plt.savefig(savepath + 'tracking_camera_filter_assumptions_%sK.png'%so.stel.teff,dpi=500)

def plot_photonic_lantern_boost(so,cload):
	"""
	"""
	pass


if __name__=='__main__':
	print('Loaded Plot Tools')
