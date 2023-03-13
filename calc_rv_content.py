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
GAUSSCONST = (2. * (2. * np.log(2))**0.5)
import pandas as pd
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
	dv_yj_tot = (0.5**2 +dv_yj**2.)**0.5	# 
	dv_hk_tot = (0.5**2 +dv_hk**2.)**0.5	# # 

	axs[1].text(1050,2*np.median(dv_vals),'$\sigma_{yJ}$=%sm/s'%round(dv_yj_tot,1),fontsize=12,zorder=101)
	axs[1].text(1500,2*np.median(dv_vals),'$\sigma_{HK}$=%sm/s'%round(dv_hk_tot,1),fontsize=12,zorder=101)
	ax2.legend(fontsize=8,loc=1)
	if savefig:
		plt.savefig('./output/rv_precision/RV_precision_%sK_%smag%s_%ss_vsini%skms.png'%(so.stel.teff,so.filt.band,so.stel.mag,so.obs.texp,so.stel.vsini))

	return fig,axs

def plot_rv_err_HKonly(so, lam_cen, dv_vals,savefig=True):
	"""
	"""
	col_table = plt.get_cmap('Spectral_r')
	fig, axs = plt.subplots(2,figsize=(7,6),sharex=True)

	plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.85,top=0.85)

	axs[1].plot([950,2460],[0.5,0.5],'k--',lw=0.7)
	axs[1].fill_between([1490,1780],0,1000,facecolor='gray',alpha=0.2)
	axs[1].fill_between([1950,2460],0,1000,facecolor='gray',alpha=0.2)
	axs[1].grid('True')
	axs[1].set_ylim(-0,3*np.median(dv_vals))
	axs[1].set_xlim(1490,2460)
	axs[1].set_ylabel('$\sigma_{RV}$ [m/s]')
	axs[1].set_xlabel('Wavelength [nm]')

	axs[0].set_ylabel('SNR')
	axs[0].set_title('M$_%s$=%s, T$_{eff}$=%sK,\n $t_{exp}$=%ss, vsini=%skm/s'%(so.filt.band,so.stel.mag,int(so.stel.teff),int(so.obs.texp),so.stel.vsini))

	axs[0].grid('True')
	ax2 = axs[0].twinx() 
	ax2.plot(so.tel.v,so.tel.s,'gray',alpha=0.5,zorder=-100,label='Telluric Absorption')
	ax2.plot(so.stel.v,so.inst.ytransmit,'k',alpha=0.5,zorder=-100,label='Total Throughput')
	ax2.set_ylabel('Transmission',fontsize=12)
	for i,lam_cen in enumerate(order_cens):
		wvl_norm = (lam_cen - 900.) / (2500. - 900.)
		axs[0].plot(so.obs.v[order_inds[i]],so.obs.s[order_inds[i]]/so.obs.noise[order_inds[i]],zorder=200,color=col_table(wvl_norm))
		axs[1].plot(lam_cen,dv_vals[i],'o',zorder=100,color=col_table(wvl_norm),markeredgecolor='k')
	
	wavesplit = 1820
	sub_1 = np.where((dv_vals!=np.inf) & (order_cens < wavesplit))[0]
	sub_2 = np.where((dv_vals!=np.inf) & (order_cens > wavesplit))[0]
	dv_1 = 1. / (np.nansum(1./dv_vals[sub_1]**2.))**0.5	# 
	dv_2 = 1. / (np.nansum(1./dv_vals[sub_2]**2.))**0.5	# 
	dv_1_tot = (0.5**2 +dv_1**2.)**0.5	# 
	dv_2_tot = (0.5**2 +dv_2**2.)**0.5	# # 

	axs[1].text(np.mean(order_cens[sub_1])+150,12*dv_2,'$\sigma_{H}$=%sm/s'%round(dv_1_tot,2),fontsize=12,zorder=101)
	axs[1].text(np.mean(order_cens[sub_2])-200,12*dv_2,'$\sigma_{K}$=%sm/s'%round(dv_2_tot,2),fontsize=12,zorder=101)
	ax2.legend(fontsize=8,loc=1)

	if savefig:
		plt.savefig('./output/rv_precision/RV_precision_HKonly_%sK_%smag%s_%ss_vsini%skms.png'%(so.stel.teff,so.filt.band,so.stel.mag,so.obs.texp,so.stel.vsini))

	return fig,axs

def plot_tess_data():
	"""
	"""
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

	# add velocity precision contour

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

def plot_TOI_data():
	planets_filename = './data/populations/TOI_2023.02.08_13.32.46.csv'
	planet_data =  pd.read_csv(planets_filename,delimiter=',',comment='#')

	tessmags = planet_data['st_tmag']
	teffs = planet_data['st_teff']

	fig, ax = plt.subplots(1,1,figsize=(6,5),sharey=True)
	plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.85)
	ax.scatter(tessmags,teffs,marker='.',c='k')
	ax.set_xlabel('TESS Mag')
	ax.set_ylabel('T$_{eff}$')
	ax.set_title('TESS Objects of Interest')

	ax.set_xlim(13,5)
	ax.set_ylim(2800,7000)
	plt.subplots_adjust(bottom=0.15,hspace=0,left=0.15)

def plot_confirmed_planets():
	"""
	plot landscape of confirmed planets
	"""
	planets_filename = './data/populations/PS_2023.02.08_13.41.45.csv'
	planet_data =  pd.read_csv(planets_filename,delimiter=',',comment='#')

	hmags = planet_data['sy_hmag']
	teffs = planet_data['st_teff']
	mass = planet_data['pl_bmassj']
	method = planet_data['discoverymethod']
	iRV = np.where(method=='Radial Velocity')[0]
	iim =np.where(method=='Imaging')[0]
	nomass = np.where(np.isnan(mass))[0]

	fig, ax = plt.subplots(1,1,figsize=(6,5),sharey=True)
	plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.85)
	ax.scatter(hmags,teffs,marker='.',c='k',alpha=0.5,label='Other')
	ax.scatter(hmags[iRV],teffs[iRV],marker='^',c='purple',alpha=0.5,label='RV')
	ax.scatter(hmags[iim],teffs[iim],marker='s',c='b',alpha=1,label='Imaging')
	ax.set_xlabel('H Mag')
	ax.set_ylabel('T$_{eff}$')
	ax.set_title('Confirmed Planets')

	ax.set_xlim(15,1)
	ax.set_ylim(2300,7000)
	ax.legend(fontsize=12)
	plt.subplots_adjust(bottom=0.15,hspace=0,left=0.15)


	fig, ax = plt.subplots(1,1,figsize=(6,5),sharey=True)
	plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.85)
	ax.scatter(hmags,teffs,marker='.',c='k')
	ax.scatter(hmags[nomass],teffs[nomass],marker='.',c='red',label='No Mass Msmt.')
	ax.set_xlabel('H Mag')
	ax.set_ylabel('T$_{eff}$')
	ax.set_title('Confirmed Planets')

	ax.set_xlim(15,1)
	ax.set_ylim(2300,7000)
	ax.legend(fontsize=12)
	plt.subplots_adjust(bottom=0.15,hspace=0,left=0.15)

	# add velocity precision contours

def plot_rv_err_lfc(snr_arr,rv_arr,ax=None,label=''):
	"""
	"""
	if ax == None:
		fig, ax = plt.subplots(1,figsize=(7,5))
	plt.subplots_adjust(bottom=0.15,left=0.15,right=0.85,top=0.85)

	ax.loglog(snr_arr,rv_arr,label=label)
	ax.set_ylabel('$\sigma_{RV}$ [m/s]')
	ax.set_xlabel('SNR')

	ax.grid(True)
	ax.set_title('RV Error vs Cal. SNR')
	plt.legend()

	return ax

def plot_rv_grid_2d():
	"""
	"""
	def fmt(x):
		return str(float(x)) + 'm/s'

	teffs = [2300,2500, 3000, 3600,4200,5800,6600,8000,9600]
	magarr = np.arange(4,17,1)
	ao_modes = ['SH','LGS_100J_130']# assuming 100J mean JHgap
	texp=900.0

	savename_1 = './output/rv_precision/rv_grid_data/rv_error_texp_%ss_ao_%s_teff_%s_%smag.npy' %(texp,ao_modes[0],teffs,magarr)
	savename_2 = './output/rv_precision/rv_grid_data/rv_error_texp_%ss_ao_%s_teff_%s_%smag.npy' %(texp,ao_modes[1],teffs,magarr)
	rv_grid_1 = np.load(savename_1)
	rv_grid_2 = np.load(savename_2)

	# pick max
	rv_err_grid  = np.zeros((len(teffs), len(magarr))) #best of all ao modes
	ao_mode_grid  = np.zeros((len(teffs), len(magarr))) #best of all ao modes
	for ii,teff in enumerate(teffs):
		for jj,mag in enumerate(magarr): # this is the magnitude in filter band
			rv_err_grid[ii,jj] = np.min((rv_grid_1[ii,jj],rv_grid_2[ii,jj]))
			ao_mode_grid[ii,jj] = np.argmin((rv_grid_1[ii,jj],rv_grid_2[ii,jj]))

	# resample onto regular grid for imshow
	rv_err_grid= rv_err_grid.T
	ao_mode_grid = ao_mode_grid.T
	rv_err_grid[0,-1] = 0.86
	extent = (np.min(teffs),np.max(teffs),np.min(magarr),np.max(magarr))
	rv_err_grid = np.sqrt(rv_err_grid**2 - .25)

	################
	fig, ax = plt.subplots(1,1, figsize=(11,8))	
	ax.imshow(rv_err_grid,aspect='auto',origin='lower',\
				interpolation='quadric',cmap='nipy_spectral',\
				extent=extent,norm=matplotlib.colors.LogNorm(vmin=0.5,vmax=300))
	cs = ax.contour(rv_err_grid, levels=[0.5,1,3,5,10,50,100,200] ,\
				colors=['w','w','w','w','w','w','w','w'],origin='lower',\
				extent=extent)
	ax.invert_yaxis()
	ax.clabel(cs, cs.levels, inline=True,fmt=fmt,fontsize=10,\
		colors=['w','w','w','w','w','w','w','w'],zorder=101)

	c3_1   = cs.collections[2].get_paths()[0].vertices # extract rv 3 curve
	c3_2   = cs.collections[2].get_paths()[1].vertices # extract rv 3 curve
	c3 = np.concatenate((c3_1,c3_2))

	ax.set_ylim(4,16)
	ax.set_ylabel('%s Magnitude'%so.filt.band)
	ax.set_xlabel('Temperature (K)')

	ax.set_title('RV Precision in t=%ss'%(int(texp)))
	figname = 'snr2d_band_%s_texp_%ss.png' %(so.filt.band,texp)
	# duplicate axis to plot filter response

	# now plot ao_mode_grid
	def fmt2(x):
		return 'NGS - LGS Boundary'

	#plt.imshow(ao_mode_grid)
	cs = ax.contour(ao_mode_grid, levels=[0.5],origin='lower',colors=['r'],\
				extent=extent)
	ax.clabel(cs, cs.levels, inline=True,fmt=fmt2,fontsize=10,\
		colors=['r'],zorder=101)
	if plot_planets:
		pl_hmags, pl_teffs,method,rvamps  = load_planets()
		ax.plot(pl_teffs,pl_hmags,'o',alpha=0.4,c='gray')

		ax.set_xlim(np.min(teffs),np.max(teffs))
		plt.savefig('./output/rv_precision/plots/planets_ms_' + figname)
	else:
		ax.set_xlim(np.min(teffs),np.max(teffs))
		plt.savefig('./output/rv_precision/plots/' + figname)

	return c3

def load_planets():
    planets_filename = './data/populations/confirmed_uncontroversial_planets_2023.03.08_14.19.56.csv'
    planets_filename = './data/populations/rv_less2earthrad_less380Teq_less4000Teff_planets_.csv'
    planet_data =  pd.read_csv(planets_filename,delimiter=',',comment='#')
    # add brown dwarfs!
    hmags = planet_data['sy_hmag']
    teffs = planet_data['st_teff']
    rvamps = planet_data['pl_rvamp']
    method = planet_data['discoverymethod']
    return hmags,teffs,method,rvamps


def load_confirmed_planets():
	planets_filename = './data/populations/confirmed_planets_PS_2023.01.12_16.07.07.csv'
	planet_data =  pd.read_csv(planets_filename,delimiter=',',comment='#')
	# add brown dwarfs!
	hmags = planet_data['sy_hmag']
	teffs = planet_data['st_teff']
	return hmags,teffs

def run_rv_error_grids(rv_floor=0.5):
	"""
	run RV grids!

	step through temperature and J mag
	"""
	teffs = [2300,2500, 3000, 3600,4200,5800,6600,8000,9600]
	vsini = [1,2,2,2,1,1,2,20,30] # https://aa.oma.be/stellar_rotation but take low end distribution
	telluric_mask	 = make_telluric_mask(so,cutoff=0.01,velocity_cutoff=10)
	order_cens, order_inds  = get_order_bounds(so)
	magarr = np.arange(4,17,1)

	rv_err_grid  = np.zeros((len(teffs), len(magarr)))
	for ii,teff in enumerate(teffs):
		so.stel.teff=teff
		so.stel.vsin = vsini[ii]
		cload      = fill_data(so)
		for jj,mag in enumerate(magarr): # this is the magnitude in filter band
			cload.set_filter_band_mag(so,so.filt.band,so.filt.family,mag,trackonly=False)
			so.obs.noise[np.where(telluric_mask==0)] = np.inf
			all_w				    = get_rv_content(so.obs.v,so.obs.s,so.obs.noise)
			dv_tot,dv_spec,dv_vals	= get_rv_precision(all_w,order_cens,order_inds,noise_floor=1,mask=telluric_mask)
			rv_err_grid[ii,jj] = np.sqrt(dv_spec**2 + rv_floor**2)

	savename = 'rv_error_texp_%ss_ao_%s_teff_%s_%smag' %(so.obs.texp,so.ao.mode,teffs,magarr)
	np.save('./output/rv_precision/rv_grid_data/%s'%savename,rv_err_grid)

	return rv_err_grid

def load_kpf_3ms_line():
	"""
	the contour method doesnt work too well - will need to redo
	"""
	from kpf_etc.etc import kpf_photon_noise_estimate, kpf_etc_rv, kpf_etc_snr
	
	teffs = [2701, 2800, 2900, 3000, 3600,4200,5800]
	colors = [7.1,6.8,5.6,5.2,3.8,2.2,1.5,1,.05,-0.09]
	magarr = np.arange(-8,17,1)

	rv_err_grid_kpf  = np.zeros((len(teffs), len(magarr)))
	for ii,teff in enumerate(teffs):
		for jj,mag in enumerate(magarr): # this is the magnitude in filter band
			try: 
				sigma_rv_val, wvl_arr, snr_ord, dv_ord = kpf_photon_noise_estimate(teff,mag+colors[ii],900)  
				rv_err_grid_kpf[ii,jj] = sigma_rv_val
			except:
				continue

	rv_err_grid= rv_err_grid_kpf.T
	extent = (np.min(teffs),np.max(teffs),np.min(magarr),np.max(magarr))
	rv_err_grid = np.sqrt(rv_err_grid**2 - .25)

	fig, ax = plt.subplots(1,1, figsize=(8,6))	
	ax.imshow(rv_err_grid,aspect='auto',origin='lower',\
				interpolation='quadric',cmap='nipy_spectral',\
				extent=extent,vmax=100,vmin=.01)
	cs = ax.contour(rv_err_grid, levels=[1,3,5,10,50] ,\
				colors=['r'],origin='lower',\
				extent=extent)
	ax.invert_yaxis()
	ax.clabel(cs, cs.levels, inline=True,fmt=fmt,fontsize=10,\
		colors=['r','r','r','r'],zorder=101)

	c3_1   = cs.collections[1].get_paths()[0].vertices # extract rv 3 curve
	c3_2   = cs.collections[1].get_paths()[1].vertices # extract rv 3 curve
	c3_kpf = np.concatenate((c3_1,c3_2))

	return c3_kpf

def plot_temperate_planets_MRI(c3):
	"""
	inputs:
	c3 - contour of rv=3m/s 

	add spirou, ilocater (assume 2 8.4m and combine rvs)
	"""
	planets_filename = './data/populations/rv_less2earthrad_less380Teq_less4000Teff_planets_.csv'
	planet_data =  pd.read_csv(planets_filename,delimiter=',',comment='#')

	hmags = planet_data['sy_hmag']
	teffs = planet_data['st_teff']
	mass = planet_data['pl_bmassj']
	mass_earth = planet_data['pl_bmasse']
	period = planet_data['pl_orbper']
	starmass = planet_data['st_mass']
	teq  = planet_data['pl_eqt']
	names = planet_data['pl_name']
	hostnames = planet_data['hostname']
	rvamps = planet_data['pl_rvamp']
	radii = planet_data['pl_rade']
	method=planet_data['discoverymethod']
	incl = planet_data['pl_orbincl']
	itransit = np.where((incl>89))[0]
	

	# get masses through mass - rad relation
	mass_est = 0.00314558 * radii**1.81 # 0.003 to jupiter mass , https://www.aanda.org/articles/aa/pdf/2017/08/aa29922-16.pdf  1.81= 1/.55
	rv_est = 203 * period**(-1/3) * mass_est/(starmass + 9.548*10**-4 * mass_est)**(2/3) #https://exoplanetarchive.ipac.caltech.edu/docs/poet_calculations.html

	fig, ax = plt.subplots()
	s = ax.scatter(teffs[itransit],hmags[itransit],c=teq[itransit],s=30*radii[itransit]**2,cmap='RdYlBu_r',ec='k')
	fig.colorbar(s, ax=ax)
	hostnames_plotted = []
	host_dic = {}
	for i in teffs[itransit].keys():
		if np.isnan(rv_est[i]): continue
		if hostnames[i] not in hostnames_plotted:
			host_dic[hostnames[i]] = []
			host_dic[hostnames[i]].append(rv_est[i])
			hostnames_plotted.append(hostnames[i])
			#ax.annotate(str(np.round(rv_est[i],1)), (teffs[i],hmags[i]-0.2),fontsize=8)
		else:
			host_dic[hostnames[i]].append(rv_est[i])

	hostnames_plotted = []
	for i in teffs[itransit].keys():
		if np.isnan(rv_est[i]): continue
		if hostnames[i] not in hostnames_plotted:
			hostnames_plotted.append(hostnames[i])
			rv_max =  np.max(host_dic[hostnames[i]])
			ax.annotate(str(np.round(rv_max,1)), (teffs[i],hmags[i]-0.2),fontsize=8)

	plt.xlabel('T$_{eff}$ (K)')
	plt.ylabel('H (mag)')


	# fill 3m/s section
	# plt.fill_between()
	#c3 = plot_rv_grid_2d()
	#c3 = np.concatenate((np.array([[2300,11.9]]),c3))

	plt.fill_between(c3[:,0],np.min(magarr),c3[:,1],fc='red',alpha=0.3)
	#plt.fill_between([3000,3800],np.min(magarr),[12,10],fc='red',alpha=0.3)
	#plt.fill_between([2500,3000],np.min(magarr),[12,12],fc='red',alpha=0.3)
	# fill kpf

	#c3_kpf = load_kpf_3ms_line()
	#c3_kpf = np.concatenate((np.array([[2700,9.8]]),c3_kpf))
	c3_kpf_x = [2700, 2800,3000,3100,3400,3700,3800]
	c3_kpf_y = [9.7,10.3,10.8,10.83,11.3,11.5,11.65]
	plt.fill_between(c3_kpf_x,np.min(magarr),c3_kpf_y,fc='steelblue',alpha=0.3)

	ax.set_ylim(7.8,14)
	ax.set_xlim(2500,4000)

def get_order_bounds(so):
	"""
	given array, return max and mean of snr per order
	"""
	order_peaks	  = signal.find_peaks(so.inst.base_throughput,height=0.055,distance=2e4,prominence=0.01)
	order_cen_lam	= so.stel.v[order_peaks[0]]
	blaze_angle	  =  76
	order_indices	=[]
	for i,lam_cen in enumerate(order_cen_lam):
		line_spacing = 0.02 if lam_cen < 1475 else 0.01
		m = np.sin(blaze_angle*np.pi/180) * 2 * (1/line_spacing)/(lam_cen/1000)
		fsr  = lam_cen/m
		isub_test= np.where((so.stel.v> (lam_cen - fsr/2)) & (so.stel.v < (lam_cen+fsr/2))) #FINISH THIS
		#plt.plot(so.stel.v[isub_test],total_throughput[isub_test],'k--')
		order_indices.append(np.where((so.obs.v > (lam_cen - fsr/2)) & (so.obs.v  < (lam_cen+fsr/2)))[0])

	return order_cen_lam,order_indices

def make_telluric_mask(so,cutoff=0.01,velocity_cutoff=5):
	telluric_spec = np.abs(so.tel.s/so.tel.rayleigh)**so.tel.airmass
	telluric_spec[np.where(np.isnan(telluric_spec))] = 0
	telluric_spec_lores = degrade_spec(so.stel.v, telluric_spec, so.inst.res)
	# resample onto v array
	filt_interp	 = interpolate.interp1d(so.stel.v, telluric_spec_lores, bounds_error=False,fill_value=0)
	s_tel		   = filt_interp(so.obs.v)/np.max(filt_interp(so.obs.v))	# filter profile resampled to phoenix times phoenix flux density

	#cutoff = 0.01 # reject lines greater than 1% depth
	telluric_mask = np.ones_like(s_tel)
	telluric_mask[np.where(s_tel < (1-cutoff))[0]] = 0
	# avoid +/-5km/s  (5pix) around telluric
	for iroll in range(velocity_cutoff):
		telluric_mask[np.where(np.roll(s_tel,iroll) < (1-cutoff))[0]] = 0
		telluric_mask[np.where(np.roll(s_tel,-1*iroll) < (1-cutoff))[0]] = 0

	return telluric_mask

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

	return dv_tot,dv_spec,dv_tot


if __name__=='__main__':
	#load inputs
	configfile = 'hispec_snr_hd189733.cfg'
	so	= load_object(configfile)
	cload = fill_data(so) # put coupling files in load and wfe stuff too

	# change to use spec_rv_noise_calc in ccf_tools.py
	order_cens, order_inds  = get_order_bounds(so)
	telluric_mask		    = make_telluric_mask(so,cutoff=0.01,velocity_cutoff=10)
	all_w				    = get_rv_content(so.obs.v,so.obs.s,so.obs.noise)
	dv_tot,dv_spec,dv_vals	= get_rv_precision(all_w,order_cens,order_inds,noise_floor=0.5,mask=telluric_mask)

	fig,axs = plot_rv_err(so, order_cens, dv_vals)
	#plot_rv_err_HKonly(so, order_cens, dv_vals,savefig=True)

	#run_rv_error_grids(rv_floor=0.5)





	##########
	# read noise dependence
	dv_rn = []
	rn_arr = np.arange(0,20)
	for rn in rn_arr:
		noise = np.sqrt(so.obs.s + rn**2 * so.inst.pix_vert)
		all_w				    = get_rv_content(so.obs.v,so.obs.s,noise)
		dv_tot,dv_spec,dv_vals	= get_rv_precision(all_w,order_cens,order_inds,noise_floor=1,mask=telluric_mask)
		dv_rn.append(dv_spec)






