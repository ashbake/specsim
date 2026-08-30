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

from dataclasses import replace

from specsim import throughput_tools
from specsim.instrument import get_order_bounds, get_tracking_band
from specsim.bandpass import Bandpass, YJHK
from specsim.star import Star, StarParams
from specsim.functions import *



all = {}

# Repo-root-relative, matching the ./data/ paths in configs/instruments/*.yaml.
# The examples chdir to the repo root before calling into here.
DATAPATH = './data/'
SAVEPATH = './output/'


def get_AO_plot_scheme():
	"""
	Define a fixed set of AO mode names plus matching plot styling
	(guide-star-type label, linestyle, color, linewidth) for consistent
	plotting across figures

	output
	------
	modes : list of str
		AO mode names
	modes2 : list of str
		guide star type label per mode ('NGS' or '' for LGS modes)
	linestyles : list of str
		matplotlib linestyle per mode
	colors : list of str
		matplotlib color per mode
	widths : list of float
		matplotlib linewidth per mode
	"""
	modes    = ['100JH','80J','80H','100K','SH','LGS_100H_130','LGS_100J_130','LGS_100J_45','LGS_STRAP_130','LGS_STRAP_45']
	modes2   = ['NGS', 'NGS', 'NGS','NGS','NGS','','','','','']
	linestyles = ['-','-','-','-','-','--','--','--','-.','-.']
	colors   = ['m','b','orange','gray','g','r','c','b','gray','black']
	widths   = [1, 1.5,  1,    1, 1.5,2, 1.5,1.5,  1, 1]

	return modes, modes2, linestyles, colors, widths


# instrument project plots
def plot_doppler_spectrographs(sim):
	"""
	Make figures to add to Sam's "cool RV landscape" plot.

	Note: this used to reload './configs/hispec_snr.cfg' internally and
	widen the wavelength range to 370-2650nm regardless of what was passed
	in. Now it uses the already-built Simulate passed in as-is -- to
	reproduce the original 370-2650nm range, build `sim` from a
	Spectrograph(l0=370, l1=2650, ...) before calling this. Produces two
	black-background figures:
	1. M star (Teff=2500K, scaled x2) and G star (Teff=5800K) spectra
	   vs. wavelength [nm], both at H=10 mag, with y limited to (-10,5000).
	2. Telluric transmission vs. wavelength [nm] (visible portion from a
	   PSG telluric FITS file spliced with the NIR portion from
	   sim.atmosphere.s), with y limited to (-2,2), overplotted with
	   Johnson R, CFHT y, and 2MASS J/H/K filter transmission curves (each
	   scaled by 0.5).

	Parameters
	----------
	sim : specsim.simulate.Simulate
		an already-built scene; uses sim.x, sim.star (for its
		StarParams -- teff/mag are overridden to build the M/G
		comparison stars), sim.filt, sim.atmosphere, sim.filter_path,
		sim.zp_file

	Returns
	-------
	None
		Does not return anything and does not call plt.savefig; the two
		figures are left open on the current pyplot state for the caller
		to save or display.
	"""
	# plot m star and g star (re-loaded at fixed teff/mag off the same model grid/filter as sim.star)
	mstar_params = replace(sim.star.params, teff=2500, mag=10)
	mstar = Star(mstar_params).load(sim.x, sim.filt).s
	gstar_params = replace(sim.star.params, teff=5800, mag=10)
	gstar = Star(gstar_params).load(sim.x, sim.filt).s

	fig, ax = plt.subplots(1,1, figsize=(15,4),facecolor='black')
	ax.set_facecolor('black')
	ax.plot(sim.x,mstar*2,'firebrick',lw=0.5)
	ax.plot(sim.x,gstar,'royalblue',lw=0.5)
	ax.set_ylim(-10,5000)
	ax.set_xlim(380,2650)

	#tellurics
	tel_nir = sim.atmosphere.s.copy()
	data = fits.getdata('./data/telluric/psg_out_2015.06.17_l0_380nm_l1_900nm_res_0.002nm_lon_204.53_lat_19.82_pres_0.5826.fits')
	_,ind     = np.unique(data['Wave/freq'],return_index=True)
	tck_tel   = interpolate.splrep(data['Wave/freq'][ind],data['Total'][ind], k=2, s=0)
	tel_vis = interpolate.splev(sim.x,tck_tel,der=0,ext=1)

	fig, ax = plt.subplots(1,1, figsize=(15,4),facecolor='black')
	ax.set_facecolor('black')
	ax.plot(sim.x[1000001:],tel_nir[1000001:],'gray',lw=0.5)
	ax.plot(sim.x[0:1000001],tel_vis[0:1000001],'gray',lw=0.5)
	ax.set_ylim(-2,2)
	ax.set_xlim(380,2650)

	# plot filters
	for band in ['R','y','J','H','K']:
		filt_i = Bandpass.load(sim.filter_path, sim.zp_file, band)
		ax.plot(filt_i.xraw,filt_i.yraw*.5,'darkgray',lw=0.8)

def plot_rv_err(observation,spectrograph,rv_result,atmosphere,star,filt,savefig=True,savepath=SAVEPATH,text_pos=0,tag='test'):
	"""
	Plot per-order SNR/pixel and per-order RV precision vs. wavelength,
	side by side in two vertically stacked panels, to compare achievable
	RV precision and SNR across the spectrograph's wavelength range for
	a given star/exposure setup.

	Top panel: SNR per pixel (observation.snr_res_element/sqrt(res_samp))
	vs. wavelength [nm] for each order, colored by order center wavelength
	(Spectral_r colormap), with telluric absorption (atmosphere.s) and
	total spectrograph throughput (spectrograph.ytransmit) overplotted on a
	secondary y-axis (Transmission).
	Bottom panel: RV precision per order (rv_result.rv_order) [m/s] vs.
	order center wavelength (spectrograph.order_cens), with a dashed line at
	the spectrograph RV floor (spectrograph.rv_floor), gray shading over the
	1450-2400nm and 980-1330nm bands, and text annotations giving the
	combined yJ-band and HK-band RV precision (quadrature sum with the
	RV floor).

	Parameters
	----------
	observation : specsim.observation.Observation
		already .run(); uses .v_res_element, .snr_res_element, .order_inds, .texp
	spectrograph : specsim.instrument.Spectrograph
		already .load(); uses .rv_floor, .order_cens, .res_samp, .ytransmit
	rv_result : specsim.analyze.RVPrecisionResult
		from Analyze.rv_precision(); uses .rv_order
	atmosphere : specsim.atmosphere.Atmosphere
		already .load(); uses .v, .s
	star : specsim.star.Star
		the on-axis star; uses .params.mag/.teff/.vsini and .v (for the
		throughput overplot's x-axis)
	filt : specsim.bandpass.Bandpass
		uses .band
	savefig : bool
		if True, save the figure to disk (default True)
	savepath : str
		directory to save the figure into if savefig is True; defaults
		to the module-level SAVEPATH
	text_pos : float
		y-axis position (in RV precision units, m/s) at which the yJ/HK
		RV precision text annotations are placed in the bottom panel
		(default 0)
	tag : str
		label used in the saved filename in place of the old so.run.tag
		(default 'test')

	Returns
	-------
	fig, axs : matplotlib Figure and array of Axes
		the created figure and its two subplots. If savefig is True, the
		figure is also written to
		'<savepath>/./RV_precision_<tag>_<star.params.teff>K_<filt.band>mag<star.params.mag>_<observation.texp>s_vsini<star.params.vsini>kms.png'
	"""
	col_table = plt.get_cmap('Spectral_r')
	fig, axs = plt.subplots(2,figsize=(7,7),sharex=True)
	plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.85,top=0.85)

	axs[1].plot([950,2400],[spectrograph.rv_floor,spectrograph.rv_floor],'k--',lw=0.7)
	axs[1].fill_between([1450,2400],0,1e10,facecolor='gray',alpha=0.2)
	axs[1].fill_between([980,1330],0,1e10,facecolor='gray',alpha=0.2)
	axs[1].grid('True')
	dv_vals = rv_result.rv_order
	max_y_lim = 3*np.median(dv_vals[np.where(~np.isinf(dv_vals))])
	if np.isnan(max_y_lim): max_y_lim = 1
	axs[1].set_ylim(0,max_y_lim)
	axs[1].set_xlim(950,2400)
	axs[1].set_ylabel(r'$\sigma_{RV}$ [m/s]')
	axs[1].set_xlabel('Wavelength [nm]')

	axs[0].set_ylabel('SNR/pixel')
	axs[0].set_title('M$_%s$=%s, T$_{eff}$=%sK,\n $t_{exp}$=%ss, vsini=%skm/s'%(filt.band,star.params.mag,int(star.params.teff),int(observation.texp),star.params.vsini))

	axs[0].grid('True')
	ax2 = axs[0].twinx()
	ax2.plot(atmosphere.v,atmosphere.s,'gray',alpha=0.5,zorder=-100,label='Telluric Absorption')
	ax2.plot(star.v,spectrograph.ytransmit,'k',alpha=0.5,zorder=-100,label='Total Throughput')
	ax2.set_ylabel('Transmission',fontsize=12)
	for i,lam_cen in enumerate(spectrograph.order_cens):
		wvl_norm = (lam_cen - 900.) / (2500. - 900.)
		axs[0].plot(observation.v_res_element[observation.order_inds[i]],observation.snr_res_element[observation.order_inds[i]]/np.sqrt(spectrograph.res_samp),zorder=200,color=col_table(wvl_norm))
		axs[1].plot(lam_cen,rv_result.rv_order[i],'o',zorder=100,color=col_table(wvl_norm),markeredgecolor='k')

	sub_yj = rv_result.rv_order[np.where((rv_result.rv_order!=np.inf) & (spectrograph.order_cens < 1400))[0]]
	sub_hk = rv_result.rv_order[np.where((rv_result.rv_order!=np.inf) & (spectrograph.order_cens > 1400))[0]]
	rvmed_yj = np.sqrt(np.sum(rv_result.rv_order[np.where((rv_result.rv_order!=np.inf) & (spectrograph.order_cens < 1400))[0]]**2))/np.sum(sub_yj)
	rvmed_hk = np.median(rv_result.rv_order[np.where((rv_result.rv_order!=np.inf) & (spectrograph.order_cens > 1400))[0]])
	dv_yj = 1. / (np.nansum(1./sub_yj**2.))**0.5	#
	dv_hk = 1. / (np.nansum(1./sub_hk**2.))**0.5	#
	dv_yj_tot = (spectrograph.rv_floor**2 +dv_yj**2.)**0.5	#
	dv_hk_tot = (spectrograph.rv_floor**2 +dv_hk**2.)**0.5	# #
	# 2*np.median(rv_result.rv_order)
	axs[1].text(1050,text_pos,r'$\sigma_{yJ}$=%sm/s'%round(dv_yj_tot,1),fontsize=12,zorder=101)
	axs[1].text(1500,text_pos,r'$\sigma_{HK}$=%sm/s'%round(dv_hk_tot,1),fontsize=12,zorder=101)
	ax2.legend(fontsize=8,loc=1)
	if savefig:
		plt.savefig(savepath + './RV_precision_%s_%sK_%smag%s_%ss_vsini%skms.png'%(tag,star.params.teff,filt.band,star.params.mag,observation.texp,star.params.vsini))

	return fig,axs


def plot_telluric_mask(observation,rv_result,atmosphere):
	"""
	Not tested.

	Plot the telluric mask alongside the telluric and stellar spectra
	over a narrow wavelength window (2192-2198nm), to visually check
	which wavelengths get masked out due to telluric absorption.

	Plots, all vs. wavelength [nm]:
	- inverted telluric mask (1 - rv_result.telluric_mask) as a filled
	  black region labeled 'Masked'
	- telluric spectrum labeled 'Telluric' (atmosphere.s interpolated
	  onto observation.v; the exact intermediate the old so.obs.s_tel
	  held -- a degraded/resampled, continuum-normalized telluric-only
	  spectrum -- isn't retained by Analyze.rv_precision, so
	  this is an approximation of the same shape for the visual check)
	- normalized stellar spectrum (observation.s/max(observation.s))
	  labeled 'Stellar'

	Parameters
	----------
	observation : specsim.observation.Observation
		already .run(); uses .v, .s
	rv_result : specsim.analyze.RVPrecisionResult
		from Analyze.rv_precision(); uses .telluric_mask
	atmosphere : specsim.atmosphere.Atmosphere
		already .load(); uses .v, .s

	Returns
	-------
	None
		Does not return anything and does not call plt.savefig; draws
		on a new figure left open for the caller to save or display.
	"""
	s_tel = np.interp(observation.v,atmosphere.v,atmosphere.s)
	plt.figure()
	plt.fill_between(observation.v,1-rv_result.telluric_mask,facecolor='k',alpha=0.8,label='Masked')
	plt.plot(observation.v,s_tel,label='Telluric')
	#plt.plot(observation.v,20*all_w/np.max(all_w),label='IC')
	plt.plot(observation.v,observation.s/np.max(observation.s),'k',label='Stellar')
	plt.xlim(2192,2198)
	plt.legend()
	plt.xlabel('Wavelength (nm)')
	plt.subplots_adjust(bottom=0.15)
	plt.ylabel('Flux (arb. units)')


def plot_rv_err_gen(v,s,order_cens,rv_order,rv_floor=0.3,savefig=True,tag='test',annotate=False):
	"""
	Generic (non-`so`-based) version of plot_rv_err: plot SNR/pixel and
	per-order RV precision vs. wavelength from raw arrays, useful for
	comparing RV precision across spectrograph designs without needing
	a full storage_object instance.

	Top panel: spectrum s vs. wavelength v, redrawn once per order
	center in order_cens, each colored by its normalized wavelength
	position (Spectral_r colormap).
	Bottom panel: rv_order [m/s] vs. order_cens [nm], with a dashed
	line at rv_floor, gray shading over the 1450-2400nm and 980-1330nm
	bands, and (if annotate=True) text annotations giving the combined
	yJ-band and HK-band RV precision (quadrature sum of the per-order
	values with rv_floor).

	Parameters
	----------
	v : array
		wavelength array [nm] plotted in the top (SNR) panel
	s : array
		spectrum/SNR array plotted against v in the top panel
	order_cens : array
		center wavelength [nm] of each spectrograph order; used as the
		x-axis in the bottom (RV precision) panel and to color the top
		panel traces
	rv_order : array
		RV precision [m/s] per order, plotted against order_cens in the
		bottom panel
	rv_floor : float
		systematic RV noise floor [m/s] plotted as a dashed reference
		line and combined in quadrature with rv_order when annotate is
		True (default 0.3)
	savefig : bool
		if True, save the figure to disk (default True)
	tag : str
		label used as the top panel title and in the saved filename
		(default 'test')
	annotate : bool
		if True, add text annotations of the combined yJ-band and
		HK-band RV precision to the bottom panel (default False)

	Returns
	-------
	fig, axs : matplotlib Figure and array of Axes
		the created figure and its two subplots. If savefig is True, the
		figure is also written to './RV_precision_<tag>.png'
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
	axs[1].set_ylabel(r'$\sigma_{RV}$ [m/s]')
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
		axs[1].text(1050,0.5,r'$\sigma_{yJ}$=%sm/s'%round(dv_yj_tot,1),fontsize=12,zorder=101)
		axs[1].text(1500,0.5,r'$\sigma_{HK}$=%sm/s'%round(dv_hk_tot,1),fontsize=12,zorder=101)
	if savefig:
		plt.savefig('./RV_precision_%s.png'%tag)

	return fig,axs



def load_brown_dwarfs_AB(sheet_file=DATAPATH + 'populations/ucd_sheet_teff.xlsx'):
    """
    Load Adam Burgasser's brown dwarf compilation spreadsheet
    ('ucd_sheet_teff.xlsx') and pull out H magnitude and effective
    temperature columns for use as a brown dwarf population sample
    (e.g. for overplotting on H mag vs. Teff figures).

    Not a plotting function; it only reads the spreadsheet. The file is not
    shipped with the repo -- pass sheet_file to point at your own copy.

    Parameters
    ----------
    sheet_file : str
        path to the .xlsx compilation (default
        DATAPATH+'populations/ucd_sheet_teff.xlsx')

    Returns
    -------
    hmag : ndarray
        2MASS H-band magnitudes ('H_2MASS' column)
    teff : ndarray
        effective temperatures ('teff' column) [K]
    """
    xl = pd.ExcelFile(sheet_file)
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
	Set global matplotlib rcParams used by the plotting functions in
	this module (font size 14, sans-serif font family, axes line width
	1.3). Intended to be called once before making plots to give them a
	consistent style.

	Parameters
	----------
	None

	Returns
	-------
	None
		Does not return anything and does not save or draw a figure;
		only mutates global matplotlib.rcParams / plt.rcParams state.
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
	Plot stellar color-vs-band curves relative to H band for a set of
	stellar temperatures, to compare how much brighter/fainter each
	photometric band is relative to H across spectral types.

	Reads DATAPATH+'WFE/HAKA/color_curves.csv' (tab-delimited, one
	column per stellar temperature plus a 'Temp' column of band names)
	and, for each temperature column (excluding 'Temp', '2500', and
	'3800'), plots (band - H) vs. band index, with a text label of the
	temperature (in K) placed at the first band.

	Parameters
	----------
	None

	Returns
	-------
	None
		Does not return anything. Saves the figure to
		SAVEPATH+'stellar_colors_H.png'.
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

def plot_tracking_cam_spot_rms(camera='h2rg',aberrations_file=DATAPATH + 'track/HISPEC_ParaxialTel_OAP_TrackCamParax_SpotSizevsField.txt'):
	"""
	Plot tracking camera optical spot size (RMS diameter, in pixels) vs.
	field radius, to check how spot blur grows toward the edge of the
	tracking camera field of view at different wavelengths.

	Reads a spot-size vs. field text file (default
	DATAPATH+'track/HISPEC_ParaxialTel_OAP_TrackCamParax_SpotSizevsField.txt',
	the same file the tracking camera's aberrations_file points at) containing
	field angle [deg] and RMS spot size [um] at several wavelengths, and
	plots RMS spot diameter (converted to pixels via a fixed 18um pixel
	pitch and a sqrt(2) diagonal-cut factor) vs. field radius [arcsec]
	for the total RMS, 900nm, and 2200nm cases.

	Parameters
	----------
	camera : str
		name of the tracking camera; accepted but currently unused in
		the function body (the pixel pitch is hardcoded to 18um instead
		of being looked up for this camera) (default 'h2rg')
	aberrations_file : str
		path to the ZEMAX spot-size vs. field text file (default
		DATAPATH+'track/HISPEC_ParaxialTel_OAP_TrackCamParax_SpotSizevsField.txt')

	Returns
	-------
	None
		Does not return anything. Saves the figure to
		SAVEPATH+'tracking_cam_spot_RMS.png'.
	"""
	f = np.loadtxt(aberrations_file)
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
	"""
	Scatter plot of host star effective temperature vs. H magnitude
	for confirmed planets around cool (Teff < 4000K), small (Rp < 2 Rearth),
	short period (Teq < 360K) stars
	"""
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
	Plot a brown dwarf population sample from the UltracoolSheet
	compilation, to look at the H-magnitude distribution and the
	relationship between H magnitude and estimated effective
	temperature for late M through T dwarfs.

	Reads './data/populations/UltracoolSheetMain.csv' and derives an
	approximate Teff [K] for each object from its infrared spectral
	type (spt_ir, M6-T6) via a fixed spectral-type-to-Teff lookup table;
	objects outside M6-T6 (or with non-string spectral type) get Teff=0.
	Produces two separate figures:
	1. Histogram of H magnitude (H_MKO column, 100 bins).
	2. Scatter plot of H magnitude vs. derived Teff [K].

	Parameters
	----------
	None

	Returns
	-------
	None
		Does not return anything and does not call plt.savefig; the two
		figures are left open on the current pyplot state for the caller
		to save or display.
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
    Plot the HISPEC yJ-band end-to-end throughput for the MRI proposal,
    comparing NGS (natural guide star) vs. LGS (laser guide star) AO
    mode throughput curves across the blue and red spectrograph arms.

    Note: telluric_file and outputdir are both overwritten internally
    with hardcoded values, so the values passed in are not used.

    Loads precomputed throughput text files ('ngs_throughput_bspec.txt',
    'lgs_throughput_bspec.txt', 'ngs_throughput_HK.txt',
    'lgs_throughput_HK.txt') from outputdir and plots throughput vs.
    wavelength [microns] for NGS (seagreen) and LGS (gray) mode, split
    at 1.333 microns into blue-arm and red-arm segments. Y-axis is
    'End-to-End Throughput' (limited to 0.005-0.075), with the
    1.33-1.49 micron gap whited out and the yJ order-gap/edge bands
    shaded gray. Custom log-spaced y-ticks and rotated x-ticks are set.

    Parameters
    ----------
    telluric_file : str
        path to a telluric transmission FITS file; accepted but unused,
        as the function immediately reassigns it to a hardcoded path
    datapath : str
        directory containing per-subsystem throughput files; accepted
        but not referenced in the function body (default
        './data/throughput/hispec_subsystems_11032022/')
    outputdir : str
        directory to read the precomputed *_throughput_*.txt files
        from; accepted but immediately overwritten to './output/'
        inside the function (default '../output/')

    Returns
    -------
    None
        Does not return anything. The plt.savefig calls that would
        write to outputdir+'e2e_mri_plot_yJ.png'/'.pdf' are currently
        commented out, so the figure is left open rather than saved.
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
    #cen_lam, snr_peaks,snr_means = get_order_value(so,w*1000,ngs_bspec)
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


# REQUIRES OBSERVATION/AO/FILT/STAR/INSTRUMENT INSTANCES
def plot_snr(observation,ao_system,filt,star,spectrograph,snrtype='pixel',savepath='./',bands=YJHK):
	"""
	Plot SNR vs. wavelength for the whole spectrum computed in the
	observation, to check the overall SNR level and its shape across the
	spectrograph bandpass for a given AO mode/exposure/star setup.

	Plots SNR (either observation.snr per pixel or
	observation.snr_res_element per resolution element) vs. wavelength
	[nm], with a dashed horizontal reference line at SNR=30, x-axis
	limited to 970-2500nm, a title giving the AO mode, filter
	band/magnitude, exposure time [hr], and Teff, and a secondary y-axis
	showing the y/J/H/K filter bands as shaded regions with text labels.

	Parameters
	----------
	observation : specsim.observation.Observation
		already .run(); uses .v, .snr, .v_res_element, .snr_res_element, .texp
	ao_system : specsim.aosystem.AOSystem
		already .select(); uses .mode_chosen (title) and .mode (filename)
	filt : specsim.bandpass.Bandpass
		uses .band
	star : specsim.star.Star
		uses .params.mag/.teff
	spectrograph : specsim.instrument.Spectrograph
		uses .darknoise

	snrtype : str
		'pixel' selects per-pixel SNR (observation.snr vs observation.v);
		'res_element' selects per-resolution-element SNR
		(observation.snr_res_element vs observation.v_res_element); any
		other value prints an error message and returns without plotting
		(default 'pixel')

	savepath : str
		directory to save the figure into (default './')

	bands : dict
		band name -> [lo,hi] wavelength [nm] edges used to shade the
		y/J/H/K bands (default specsim.bandpass.YJHK)

	Returns
	-------
	None
		Does not return anything. Saves the figure to
		'<savepath>/snr_<ao_system.mode>_<filt.band>mag_<star.params.mag>_texp_<observation.texp>s_dark_<spectrograph.darknoise>.png',
		unless snrtype is invalid, in which case nothing is saved.
	"""
	fig, ax = plt.subplots(1,1, figsize=(10,8))
	if snrtype =='pixel':  ax.plot(observation.v,observation.snr)
	elif snrtype=='res_element': ax.plot(observation.v_res_element,observation.snr_res_element)
	else: print('Choose pixel or res_element for snrtype'); return
	ax.set_ylabel('SNR')
	ax.set_xlabel('Wavelength (nm)')
	ax.set_title('AO Mode: %s, %s=%s, t=%shr, Teff=%sK'%(ao_system.mode_chosen,\
			filt.band,round(star.params.mag,1),np.round(observation.texp/3600,2),\
			int(star.params.teff)))
	ax.axhline(y=30,color='k',ls='--')
	#plt.legend()
	# duplicate axis to plot filter response
	ax2 = ax.twinx()
	# plot band
	ax2.fill_between(bands['y'],0,1,facecolor='k',edgecolor='black',alpha=0.1)
	ax2.text(20+np.min(bands['y']),0.9, 'y')
	ax2.fill_between(bands['J'],0,1,facecolor='k',edgecolor='black',alpha=0.1)
	ax2.text(50+np.min(bands['J']),0.9, 'J')
	ax2.fill_between(bands['H'],0,1,facecolor='k',edgecolor='black',alpha=0.1)
	ax2.text(50+np.min(bands['H']),0.9, 'H')
	ax2.fill_between(bands['K'],0,1,facecolor='k',edgecolor='black',alpha=0.1)
	ax2.text(50+np.min(bands['K']),0.9, 'K')
	ax2.set_ylim(0,1)
	ax.set_xlim(970,2500)
	figname = 'snr_%s_%smag_%s_texp_%ss_dark_%s.png' %(ao_system.mode,filt.band,star.params.mag,observation.texp,spectrograph.darknoise)
	plt.savefig(savepath + figname)

def plot_snr_orders(observation,spectrograph,ao_system,filt,star,snrtype='res_element',mode='mean',height=0.055,savepath=SAVEPATH,bands=YJHK):
	"""
	Plot per-order SNR (mean or peak) vs. order center wavelength, to
	compare SNR across the spectrograph's echelle orders in a single
	summary curve rather than the full per-pixel spectrum.

	Uses get_order_value to collapse the SNR spectrum
	(observation.snr or observation.snr_res_element, depending on
	snrtype) into a peak and mean SNR value per order, using
	spectrograph.order_bounds_file to define order boundaries. Plots the
	chosen statistic (snr_peaks or snr_means) vs. order center wavelength
	cen_lam [nm], with a title giving AO mode, Teff, filter
	band/magnitude, and exposure time [hr], x-axis limited to
	970-2500nm, y-axis limited to (0, max+10), the y/J/H/K filter bands
	shaded on the same axis, and the 1333-1500nm gap whited out.

	Parameters
	----------
	observation : specsim.observation.Observation
		already .run(); uses .v, .snr, .v_res_element, .snr_res_element, .texp
	spectrograph : specsim.instrument.Spectrograph
		uses .order_bounds_file, .darknoise
	ao_system : specsim.aosystem.AOSystem
		uses .mode
	filt : specsim.bandpass.Bandpass
		uses .band
	star : specsim.star.Star
		uses .params.teff/.mag

	snrtype : str
		'pixel' selects per-pixel SNR (observation.snr, observation.v)
		as the input spectrum; 'res_element' selects
		per-resolution-element SNR (observation.snr_res_element,
		observation.v_res_element) (default 'res_element')

	mode : str
		plots SNR as either the average ('mean') or the peak ('peak')
		of each order (default 'mean')

	height : float
		accepted as a parameter but not referenced anywhere in the
		function body (default 0.055)

	savepath : str
		directory to save the figure into (default SAVEPATH)

	bands : dict
		band name -> [lo,hi] wavelength [nm] edges used to shade the
		y/J/H/K bands (default specsim.bandpass.YJHK)

	Returns
	-------
	cen_lam, snr_peaks, snr_means : arrays
		order center wavelengths [nm] and the per-order peak/mean SNR
		values computed by get_order_value. The figure is also
		saved to
		'<savepath>/snr_<ao_system.mode>_<filt.band>mag_<star.params.mag>_texp_<observation.texp>s_dark_<spectrograph.darknoise>.png'
	"""
	if snrtype=='pixel': cen_lam, snr_peaks,snr_means = get_order_value(observation.v,observation.snr,spectrograph.order_bounds_file)
	if snrtype=='res_element':cen_lam, snr_peaks,snr_means = get_order_value(observation.v_res_element,observation.snr_res_element,spectrograph.order_bounds_file)

	fig, ax = plt.subplots(1,1, figsize=(8,6))
	if mode=='peak': ax.plot(cen_lam, snr_peaks,lw=2)
	elif mode=='mean': ax.plot(cen_lam, snr_means,lw=2)
	ax.set_ylabel('SNR')
	ax.set_xlabel('Wavelength (nm)')
	ax.set_title('AO Mode: %s, T$_{eff}$=%sK, %s=%s, t=%shr'%(ao_system.mode,star.params.teff,filt.band,star.params.mag,round(observation.texp/3600,2)))
	#ax.axhline(y=30,color='k',ls='--')
	figname = 'snr_%s_%smag_%s_texp_%ss_dark_%s.png' %(ao_system.mode,filt.band,star.params.mag,observation.texp,spectrograph.darknoise)

	# duplicate axis to plot filter response
	# plot band
	ax.fill_between(bands['y'],0,np.max(snr_peaks)+10,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(20+np.min(bands['y']),0.09, 'y')
	ax.fill_between(bands['J'],0,np.max(snr_peaks)+10,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(bands['J']),0.09, 'J')
	ax.fill_between(bands['H'],0,np.max(snr_peaks)+10,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(bands['H']),0.09, 'H')
	ax.fill_between(bands['K'],0,np.max(snr_peaks)+10,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(bands['K']),0.09, 'K')
	ax.set_xlim(970,2500)
	ax.set_ylim(0,np.max(snr_peaks)+10)
	ax.fill_between([1333,1500],0,np.max(snr_peaks)+10,facecolor='white',zorder=100)

	plt.savefig(savepath + figname)

	return cen_lam, snr_peaks,snr_means
	
def plot_base_throughput(spectrograph,savepath=SAVEPATH,bands=YJHK):
	"""
	Plot the spectrograph base throughput (excluding fiber coupling) vs.
	wavelength, to inspect the shape of the non-coupling throughput
	budget across the spectrograph bandpass.

	Plots spectrograph.base_throughput vs. spectrograph.xtransmit [nm], with
	a dashed horizontal line at y=30 (note: this is likely a leftover
	from an SNR plot, since base_throughput is a fraction typically
	<=1, so the line falls outside the visible y-range), x-axis limited
	to 970-2500nm, y-axis limited to (0, peak throughput), a grid, and
	the y/J/H/K filter bands shaded on the same axis with text labels.

	Parameters
	----------
	spectrograph : specsim.instrument.Spectrograph
		already .load(); uses .xtransmit, .base_throughput

	savepath : str
		directory to save the figure into (default SAVEPATH)

	bands : dict
		band name -> [lo,hi] wavelength [nm] edges used to shade the
		y/J/H/K bands (default specsim.bandpass.YJHK)

	Returns
	-------
	None
		Does not return anything. Saves the figure to
		'<savepath>/base_throughput.png'.
	"""
	fig, ax = plt.subplots(1,1, figsize=(10,8))
	ax.plot(spectrograph.xtransmit,spectrograph.base_throughput)
	peak = np.max(spectrograph.base_throughput)

	ax.set_ylabel('Base Throughput')
	ax.set_xlabel('Wavelength (nm)')
	ax.axhline(y=30,color='k',ls='--')
	figname = 'base_throughput.png'
	# duplicate axis to plot filter response
	# plot band
	ax.fill_between(bands['y'],0,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(20+np.min(bands['y']),0.09, 'y')
	ax.fill_between(bands['J'],0,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(bands['J']),0.09, 'J')
	ax.fill_between(bands['H'],0,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(bands['H']),0.09, 'H')
	ax.fill_between(bands['K'],0,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(bands['K']),0.09, 'K')
	ax.set_xlim(970,2500)
	plt.grid()
	ax.set_ylim(0,peak)
	plt.savefig(savepath + figname)

def plot_coupling(spectrograph,savepath=SAVEPATH,bands=YJHK):
	"""
	Plot the fiber coupling efficiency vs. wavelength, to inspect how
	coupling efficiency varies across the spectrograph bandpass.

	Plots spectrograph.coupling vs. spectrograph.xtransmit [nm], with a
	dashed horizontal line at y=30 (note: this is likely a leftover from
	an SNR plot, since coupling is a fraction typically <=1, so the line
	falls outside the visible y-range), x-axis limited to 970-2500nm,
	y-axis limited to (0, peak coupling), a grid, and the y/J/H/K
	filter bands shaded on the same axis with text labels.

	Parameters
	----------
	spectrograph : specsim.instrument.Spectrograph
		already .load(); uses .xtransmit, .coupling

	savepath : str
		directory to save the figure into (default SAVEPATH)

	bands : dict
		band name -> [lo,hi] wavelength [nm] edges used to shade the
		y/J/H/K bands (default specsim.bandpass.YJHK)

	Returns
	-------
	None
		Does not return anything. Saves the figure to
		'<savepath>/coupling_throughput.png'.
	"""
	fig, ax = plt.subplots(1,1, figsize=(10,8))
	#ax.plot(spectrograph.xtransmit,spectrograph.ytransmit)
	ax.plot(spectrograph.xtransmit,spectrograph.coupling)
	peak = np.max(spectrograph.coupling)

	ax.set_ylabel('Coupling')
	ax.set_xlabel('Wavelength (nm)')
	ax.axhline(y=30,color='k',ls='--')
	figname = 'coupling_throughput.png'
	# duplicate axis to plot filter response
	# plot band
	ax.fill_between(bands['y'],0,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(20+np.min(bands['y']),0.09, 'y')
	ax.fill_between(bands['J'],0,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(bands['J']),0.09, 'J')
	ax.fill_between(bands['H'],0,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(bands['H']),0.09, 'H')
	ax.fill_between(bands['K'],0,peak,facecolor='k',edgecolor='black',alpha=0.1)
	ax.text(50+np.min(bands['K']),0.09, 'K')
	ax.set_xlim(970,2500)
	plt.grid()
	ax.set_ylim(0,peak)
	plt.savefig(savepath + figname)

def plot_track_background(star,tracking_camera,filt,savepath=SAVEPATH,bands=YJHK):
	"""
	Plot tracking-camera sky background, instrument background, and
	source signal vs. wavelength in three stacked panels, to compare
	the relative contribution of each noise/signal source across the
	tracking camera's wavelength range.

	Top panel: sky background (tracking_camera.sky_bg_spec) [phot/nm/s]
	vs. wavelength [nm], with a dashed reference line at y=0.5 and gray
	shading over the 1450-2400nm and 980-1330nm bands, y-axis limited
	to (0,1000).
	Middle panel: instrument background (tracking_camera.inst_bg_spec)
	[phot/nm/s], y-axis limited to (0,20).
	Bottom panel: source photon rate
	(tracking_camera.signal_spec/tracking_camera.texp) [phot/nm/s].
	Each panel has a secondary y-axis showing the y/J/H/K filter bands
	as shaded regions with text labels. Title gives filter band,
	magnitude, and Teff.

	Parameters
	----------
	star : specsim.star.Star
		uses .v (wavelength grid) and .params.mag/.teff
	tracking_camera : specsim.tracking_camera.TrackingCamera
		already .observe(); uses .sky_bg_spec, .inst_bg_spec, .signal_spec, .texp
	filt : specsim.bandpass.Bandpass
		uses .band

	savepath : str
		directory to save the figure into (default SAVEPATH)

	bands : dict
		band name -> [lo,hi] wavelength [nm] edges used to shade the
		y/J/H/K bands (default specsim.bandpass.YJHK)

	Returns
	-------
	None
		Does not return anything. Saves the figure to
		'<savepath>/noise_flux_<star.params.teff>K_<filt.band>_<star.params.mag>mag.png'.
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
	axs[0].plot(star.v,tracking_camera.sky_bg_spec,'b',alpha=0.5,zorder=100,label='Sky Background')
	axs[1].plot(star.v,tracking_camera.inst_bg_spec,'m',lw=2,alpha=0.5,zorder=100,label='Instrument Background')
	axs[2].plot(star.v,tracking_camera.signal_spec/tracking_camera.texp,'g',alpha=0.5,zorder=100,label='Source Photons')

	#ax2.fill_between(filt.x,filt.y,facecolor='gray',edgecolor='black',alpha=0.2)
	#ax2.set_ylabel('Filter Response')
	# plot band
	for ax in axs:
		ax2 = ax.twinx()
		ax2.fill_between(bands['y'],0,1,facecolor='k',edgecolor='black',alpha=0.1)
		ax2.text(20+np.min(bands['y']),0.9, 'y')
		ax2.fill_between(bands['J'],0,1,facecolor='k',edgecolor='black',alpha=0.1)
		ax2.text(50+np.min(bands['J']),0.9, 'J')
		ax2.fill_between(bands['H'],0,1,facecolor='k',edgecolor='black',alpha=0.1)
		ax2.text(50+np.min(bands['H']),0.9, 'H')
		ax2.fill_between(bands['K'],0,1,facecolor='k',edgecolor='black',alpha=0.1)
		ax2.text(50+np.min(bands['K']),0.9, 'K')
		ax2.set_ylim(0,1)

	axs[0].set_title('Tracking Camera Noise \n  %s mag = %s, Teff=%sK '%(filt.band,star.params.mag,int(star.params.teff)))
	#plt.savefig('./output/trackingcamera/noise_flux_%sK_%s_%smag.pdf'%(star.params.teff,filt.band,star.params.mag))
	plt.savefig(savepath + 'noise_flux_%sK_%s_%smag.png'%(star.params.teff,filt.band,star.params.mag))

def plot_spec_background(observation,star,filt,savepath=SAVEPATH,bands=YJHK):
	"""
	Plot the spectrograph's combined spectrograph+sky background vs.
	wavelength, to inspect the background photon rate the science
	spectrum sits on top of across the spectrograph bandpass.

	Plots observation.sky_bg_ph [phot/s] vs. observation.v [nm], with a
	dashed reference line at y=0.5, x-axis limited to 950-2450nm, y-axis
	limited to (-0.001,0.6), and a secondary y-axis showing the y/J/H/K
	filter bands as shaded regions with text labels.

	Parameters
	----------
	observation : specsim.observation.Observation
		already .run(); uses .v, .sky_bg_ph
	star : specsim.star.Star
		uses .params.teff/.mag
	filt : specsim.bandpass.Bandpass
		uses .band

	savepath : str
		directory to save the figure into (default SAVEPATH)

	bands : dict
		band name -> [lo,hi] wavelength [nm] edges used to shade the
		y/J/H/K bands (default specsim.bandpass.YJHK)

	Returns
	-------
	None
		Does not return anything. Saves the figure to
		'<savepath>/noise_flux_<star.params.teff>K_<filt.band>_<star.params.mag>mag.png'
		(same filename pattern as plot_track_background, so calling
		both with the same savepath/star parameters will overwrite one
		another).
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
	#axs[1].plot(star.v,observation.sky_bg_ph,'m',alpha=0.5,zorder=100,label='Sky Background')
	#axs[0].plot(observation.v,observation.inst_bg_ph,'b',lw=2,alpha=0.5,zorder=100,label='Instrument Background')
	axs[0].plot(observation.v,observation.sky_bg_ph,'b',lw=2,alpha=0.5,zorder=100,label='Instrument Background')
	axs[0].set_xlabel('Wavelength [nm]')
	axs[0].set_ylabel('Spectrograph + Sky Bkg \n(phot/s)')

	# plot band
	for ax in axs:
		ax2 = ax.twinx()
		ax2.fill_between(bands['y'],0,1,facecolor='k',edgecolor='black',alpha=0.1)
		ax2.text(20+np.min(bands['y']),0.9, 'y')
		ax2.fill_between(bands['J'],0,1,facecolor='k',edgecolor='black',alpha=0.1)
		ax2.text(50+np.min(bands['J']),0.9, 'J')
		ax2.fill_between(bands['H'],0,1,facecolor='k',edgecolor='black',alpha=0.1)
		ax2.text(50+np.min(bands['H']),0.9, 'H')
		ax2.fill_between(bands['K'],0,1,facecolor='k',edgecolor='black',alpha=0.1)
		ax2.text(50+np.min(bands['K']),0.9, 'K')
		ax2.set_ylim(0.1,1)

	plt.savefig(savepath + 'noise_flux_%sK_%s_%smag.png'%(star.params.teff,filt.band,star.params.mag))
	#np.savetxt('inst_background.txt',np.vstack((observation.v,observation.inst_bg_ph)).T,header='wave[nm],bkg[ph/s]')
	#np.savetxt('sky_background.txt',np.vstack((observation.v,observation.sky_bg_ph)).T,header='wave[nm],bkg[ph/s]')

def plot_tracking_bands(star,atmosphere,ao_system,trackbands=['J','JHgap','H'],plot_telluric=True,savepath=SAVEPATH):
	"""
	Plot the tracking camera's filter bandpass profiles together with
	the (degraded-resolution) stellar spectrum and throughput-weighted
	flux, to compare where each tracking band sits relative to the
	star's flux and to show what fraction of J-band flux each band
	captures.

	For each band in trackbands, plots the band's transmission profile
	(from instrument.get_tracking_band) vs. wavelength [nm] as a filled
	curve, labeled with the band name (or 'JH Gap' for 'JHgap') and
	annotated with the percentage of J-band-normalized in-band flux it
	captures. Also overplots the degraded-resolution (R~2000) normalized
	stellar spectrum and the throughput x flux product, shades a
	500-970nm 'Visible WFS' region, and (if plot_telluric) overplots a
	degraded telluric transmission curve.

	Note: this doesn't need a live TrackingCamera -- the old
	so.track.ytransmit stood in for a rough throughput scaling in the
	"Throughput x Normalized Flux" curve, which this replaces with
	ao_system.pywfs_dichroic (the AO dichroic shape), since building a
	full TrackingCamera just for that illustrative curve would add an
	unnecessary dependency. The old `plot_telluric` name was previously
	an undefined module-level global (a bug -- calling this function
	always raised NameError there); it's now a real parameter.

	Parameters
	----------
	star : specsim.star.Star
		uses .v, .s, .params.teff
	atmosphere : specsim.atmosphere.Atmosphere
		already .load(); uses .v, .s
	ao_system : specsim.aosystem.AOSystem
		already .select(); uses .pywfs_dichroic

	trackbands : list of str
		names of tracking camera bands to plot, passed to
		instrument.get_tracking_band (default ['J','JHgap','H'])

	plot_telluric : bool
		if True, overplot a degraded telluric transmission curve
		(default True)

	savepath : str
		directory to save the figure into (default SAVEPATH)

	Returns
	-------
	None
		Does not return anything. Saves the figure to
		'<savepath>/tracking_camera_filter_assumptions_<star.params.teff>K.png'
		at dpi=500.
	"""
	#trackbands=['y','Jplus','H','K'] #['J','JHgap','H'] #'Hplus50','Jplus']#['y',

	spectrum = ao_system.pywfs_dichroic * atmosphere.s * star.s/np.max(star.s) / 1.5
	spec_lores = degrade_spec(star.v[::10], spectrum[::10], 2000)
	star_lores = degrade_spec(star.v[::10], star.s[::10]/np.max(star.s), 2000)

	plt.figure(figsize=(8,5))
	for band in trackbands:
		print(band)
		bandpass, center_wavelength = get_tracking_band(star.v,band)
		p = plt.plot(star.v[::100],bandpass[::100],linewidth=1)
		plt.fill_between(star.v,-1,bandpass,alpha=0.1,facecolor=p[0].get_color(),edgecolor=p[0].get_color())
		if band!='JHgap': plt.text(center_wavelength-10, np.max(bandpass),band,c=p[0].get_color())
		if band=='JHgap': plt.text(center_wavelength-50, 0.95,' JH\nGap',c=p[0].get_color())

	# get J band
	Jbandpass, center_wavelength = get_tracking_band(star.v,'J')
	sumflux_J = np.trapz(spectrum[np.where(Jbandpass>0.1)],star.v[np.where(Jbandpass>0.1)])
	for i,band in enumerate(trackbands):
		bandpass, center_wavelength = get_tracking_band(star.v,band)
		sumflux = np.trapz(spectrum[np.where(bandpass>0.1)],star.v[np.where(bandpass>0.1)])
		if i%2==0: plt.text(center_wavelength-50, 0.8*np.max(bandpass),str(round(100*sumflux/sumflux_J,1))+'%',fontsize=10)
		if i%2==1: plt.text(center_wavelength-50, 0.9*np.max(bandpass),str(round(100*sumflux/sumflux_J,1))+'%',fontsize=10)

	plt.fill_between([500,970],0,2,alpha=0.1,facecolor='m')
	plt.text(550, 0.95,'Visible\nWFS',c='m')

	plt.plot(star.v[::10],star_lores,'k',zorder=-100,label='T=%sK'%star.params.teff)
	plt.plot(star.v[::10],spec_lores,'gray',alpha=0.8,zorder=-101,label='Throughput x \n Normalized Flux')
	plt.ylim(0,1.15)
	plt.title('Tracking Camera Filter Profiles')
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Relative Transmission')
	plt.legend(fontsize=10,loc=7)

	if plot_telluric:
		ytel = degrade_spec(atmosphere.v,atmosphere.s,4000)
		plt.plot(atmosphere.v[::100],ytel[::100],'gray',alpha=0.3)

	plt.savefig(savepath + 'tracking_camera_filter_assumptions_%sK.png'%star.params.teff,dpi=500)

def plot_photonic_lantern_boost(sim):
	"""
	Not implemented. Placeholder for a planned plot of the SNR/coupling
	boost from using a photonic lantern (spectrograph.pl_on), presumably
	comparing performance with and without the photonic lantern in the
	blue channel.

	Parameters
	----------
	sim : specsim.simulate.Simulate
		an already-built scene (unused; function body is only `pass`).
		Would presumably use sim.spectrograph.pl_on and re-run
		sim.snr()/sim.rv_precision() with pl_on toggled once implemented.

	Returns
	-------
	None
		Function body is currently just `pass`; it does no computation,
		draws nothing, and saves nothing.
	"""
	pass


if __name__=='__main__':
	print('Loaded Plot Tools')


########## THROUGHPUT PLOTS (moved here from throughput_tools.py) 
def plot_throughput(spectrograph, star, filt, ao_system, observation, savepath=SAVEPATH):
    """
    Plot fiber coupling efficiency, base throughput (everything but
    coupling), and total throughput vs wavelength for a single simulation
    run, and save the figure to disk. Draws a horizontal 5% reference line
    for quick readout of where the total throughput crosses that threshold.

    inputs
    ------
    spectrograph : specsim.instrument.Spectrograph
        already .load(); uses .coupling, .base_throughput, .ytransmit
    star : specsim.star.Star
        the on-axis star; uses .v (wavelength grid) and .params.mag/.teff
    filt : specsim.bandpass.Bandpass
        uses .band
    ao_system : specsim.instrument.AOSystem
        uses .mode (the requested mode, e.g. 'auto' -- not .mode_chosen)
    observation : specsim.observation.Observation
        already .run(); uses .texp (total exposure time)
    savepath : str
        directory to save the figure into (default SAVEPATH)

    outputs
    -------
    None
        Draws the figure on a new matplotlib figure/axes and saves it to
        '<savepath>/throughput_<ao_system.mode>_<filt.band>mag_<star.params.mag>_Teff_<star.params.teff>_texp_<observation.texp>s.png'
    """
    plt.figure(figsize=(7,4))
    plt.plot(star.v,spectrograph.coupling,label='Coupling Only')
    plt.plot(star.v,spectrograph.base_throughput,label='All But Coupling')
    plt.plot(star.v,spectrograph.ytransmit,'k',label='Total Throughput')
    plt.ylabel('Transmission')
    plt.xlabel('Wavelength (nm)')
    plt.title('%s=%s, Teff=%s, AO mode: %s'%(filt.band,int(star.params.mag),int(star.params.teff),ao_system.mode))
    plt.subplots_adjust(bottom=0.15)
    plt.axhline(y=0.05,color='m',ls='--',label='5%')
    plt.legend()
    plt.grid()
    figname = 'throughput_%s_%smag_%s_Teff_%s_texp_%ss.png' %(ao_system.mode,filt.band,star.params.mag,star.params.teff,int(observation.texp))
    plt.savefig(savepath + figname)


def plot_throughput_components_HK(telluric_file=DATAPATH + 'telluric/psg_out_2020.08.02_l0_800nm_l1_2700nm_res_0.001nm_lon_204.53_lat_19.82_pres_0.5826.fits',
                                    transmission_path = DATAPATH + 'instrument/hispec/throughput/',
                                    outputdir=SAVEPATH,
                                    ngs_wfe=[130,3],
                                    lgs_wfe=[220,9.4],
                                    atm=1,adc=1):
    """
    Build and plot the cumulative (red-path, H+K band) system throughput
    for the MRI proposal figure: telluric atmosphere, telescope, Keck AO,
    front-end injection (FEI) optics, fiber coupling (NGS and LGS cases via
    pick_coupling_rounded), and spectrograph, each curve layered
    cumulatively on a log-scale plot. Saves the figure as PNG and PDF; does
    not return anything.

    inputs
    ------
    telluric_file : string, optional
        path to a FITS file (PSG telluric transmission model) with
        'Wave/freq' and 'Total' columns, used for the atmospheric
        transmission curve
    transmission_path : string, optional
        path to the directory of per-surface '<surface>_throughput.csv'
        files (and the 'coupling/' grid used by pick_coupling_rounded)
    outputdir : string, optional
        directory to save the output 'e2e_plot_HK.png'/'.pdf' figures.
        Default SAVEPATH
    ngs_wfe : list of 2 floats, optional [nm, mas]
        [high-order WFE (ho_wfe), dynamic tip-tilt (tt_dynamic)] for the
        natural-guide-star AO case, passed to pick_coupling_rounded.
        Default [130,3]
    lgs_wfe : list of 2 floats, optional [nm, mas]
        [high-order WFE (ho_wfe), dynamic tip-tilt (tt_dynamic)] for the
        laser-guide-star AO case, passed to pick_coupling_rounded.
        Default [220,9.4]
    atm : int, optional
        0 or 1, whether atmosphere was included in the coupling simulation
        grid selected by pick_coupling_rounded. Default 1
    adc : int, optional
        0 or 1, whether the ADC was included in the coupling simulation grid
        selected by pick_coupling_rounded. Default 1

    outputs
    -------
    None
        Draws a new matplotlib figure and saves it to
        '<outputdir>/e2e_plot_HK.png' and '<outputdir>/e2e_plot_HK.pdf'
    """
    data={}
    data['red'] = {}

    colors = ['b','orange','gray','yellow','lightblue','green','k']
    labels = ['Atmosphere','Telescope','Keck AO','FEI','Fiber \nCoupling','Fiber\nPropogation',]
    for spec in ['red']:
        if spec=='red':
            include = ['tel', 'ao', 'feicom', 'feired','fibred','rspec']#,'coupling']
        if spec=='blue':
            include = ['tel', 'ao', 'feicom', 'feiblue','fibblue','bspec']#,'coupling']

        for i in include:
            if i==include[0]:
                w,s = np.loadtxt(transmission_path + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                data[spec][i] = s
            else:
                wtemp, stemp = np.loadtxt(transmission_path + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                # interpolate onto s
                f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
                data[spec][i] = f(w)
                #plt.plot(w,s,label=i)

    #load atmosphere and degrade to lower res, resample onto w
    teldata      = fits.getdata(telluric_file)
    _,ind  = np.unique(teldata['Wave/freq'],return_index=True)
    tck_tel   = interpolate.splrep(teldata['Wave/freq'][ind],teldata['Total'][ind], k=2, s=0)
    telluric  = interpolate.splev(1000*w,tck_tel,der=0,ext=1)
    telluric_spec  = degrade_spec(w,telluric,1000)
    data['atm'] = telluric_spec
 
    #load coupling for two options
    # inputs : waves,dynwfe,ttStatic,ttDynamic
    data['coupling_NGS'],strehl  = throughput_tools.pick_coupling_rounded(transmission_path,w,ngs_wfe[0], ngs_wfe[1])
    data['coupling_LGS'],strehl2 = throughput_tools.pick_coupling_rounded(transmission_path,w,lgs_wfe[0], lgs_wfe[1])


    if np.max(w)>1000: w/=1000
    # plot red only
    spec = 'red'
    plt.figure(figsize=(7,6))
    plt.semilogy(w,data['atm'],c='royalblue',linewidth=1)
    plt.plot(w,data['atm']*data[spec]['tel'],c='darkorange',linewidth=1)
    plt.plot(w,data['atm']* data[spec]['tel'] * data[spec]['ao'],c='silver',linewidth=1)
    plt.plot(w, data['atm']*data[spec]['tel'] * data[spec]['ao']* \
                    data[spec]['feicom']*data[spec]['feired'],c='gold',linewidth=1)
    
    plt.plot(w,data['atm']*data[spec]['tel'] * data[spec]['ao']* \
                    data[spec]['feicom']*data[spec]['feired']*data[spec]['fibred']*\
                    data['coupling_NGS'],c='steelblue',alpha=0.8,linewidth=1)
    
    ngs =  data[spec]['tel'] * data[spec]['ao']* \
                    data[spec]['feicom']*data[spec]['feired']*data[spec]['fibred']*\
                    data[spec]['rspec']*data['coupling_NGS']
    lgs =  data[spec]['tel'] * data[spec]['ao']* \
                    data[spec]['feicom']*data[spec]['feired']*data[spec]['fibred']*\
                    data[spec]['rspec']*data['coupling_LGS']
    #np.savetxt(outputdir + 'ngs_throughput_HK.txt',np.vstack((w,ngs)).T)
    #np.savetxt(outputdir + 'lgs_throughput_HK.txt',np.vstack((w,lgs)).T)

    ngs[np.where(ngs<0.015)[0]] = np.nan
    lgs[np.where((ngs<0.015) & (w > 1.9))[0]] = np.nan
    lgs[np.where(lgs<0.01)[0]] = np.nan
    lgs[np.where((lgs<0.015) & (w > 1.9))[0]] = np.nan
    plt.plot(w,data['atm']*ngs,c='seagreen',linewidth=1)
    
    plt.plot(w,data['atm']*lgs,c='seagreen',alpha=0.5,linewidth=1)
    
    plt.xlabel('Wavelength (microns)',color='k')
    plt.ylabel('Cumulative Throughput (log)',color='k')
    plt.ylim(0.01,1)
    plt.xlim(1.490, 2.455)
    plt.axhline(np.max(ngs),c='k',linestyle='--',linewidth=2)
    plt.fill_between([1.810, 1.960],0.01,y2=1,facecolor='w',zorder=110)
    plt.fill_between([1.490,1.780],0.01,y2=1,facecolor='gray',alpha=0.2,zorder=110)
    plt.fill_between([1.990,2.460],0.01,y2=1,facecolor='gray',alpha=0.2,zorder=110)

    #plt.title("HISPEC E2E Except Coupling")
    # y lines
    #yticks = [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.8]
    yticks = [0.01, 0.03, 0.09, 0.27, 0.81]
    xticks = np.round((np.arange(1.49, 2.45,0.04)),2)
    plt.yticks(ticks=yticks,labels=yticks,color='k',fontsize=12)
    plt.xticks(rotation=90,ticks=xticks,labels=xticks,color='k',fontsize=12)
    plt.grid(axis='y',alpha=0.4)
    plt.subplots_adjust(bottom=0.17)
    plt.title('HK Throughput')
    plt.savefig(outputdir + 'e2e_plot_HK.png')
    plt.savefig(outputdir + 'e2e_plot_HK.pdf')


def plot_throughput_components_YJ(telluric_file=DATAPATH + 'telluric/psg_out_2020.08.02_l0_800nm_l1_2700nm_res_0.001nm_lon_204.53_lat_19.82_pres_0.5826.fits',
                                    transmission_path = DATAPATH + 'instrument/hispec/throughput/coupling/',
                                    outputdir=SAVEPATH,
                                    ngs_wfe=[130,3],
                                    lgs_wfe=[220,9.4],
                                    atm=1,adc=1):
    """
    Build and plot the cumulative (blue-path, y+J band) system throughput
    for the MRI proposal figure: telescope, Keck AO, front-end injection
    (FEI) optics, fiber coupling (NGS and LGS cases, computed with
    photonic lantern on via grid_interp_coupling/pick_coupling_rounded with
    lo_wfe=50, defocus=30), and spectrograph, layered cumulatively on a
    log-scale plot. Also saves the NGS/LGS cumulative throughput arrays to
    text files. Saves the figure as PNG and PDF; does not return anything.

    inputs
    ------
    telluric_file : string, optional
        path to a FITS file (PSG telluric transmission model) with
        'Wave/freq' and 'Total' columns; loaded and degraded but not
        plotted in this function (see semilogy line, currently commented
        out)
    transmission_path : string, optional
        path to the directory of per-surface '<surface>_throughput.csv'
        files; also used (with a 'coupling/' suffix) as the coupling grid
        path for grid_interp_coupling and pick_coupling_rounded
    outputdir : string, optional
        directory to save the output 'e2e_mri_plot_yJ.png'/'.pdf' figures
        and the 'ngs_throughput_bspec.txt'/'lgs_throughput_bspec.txt' data
        files. Default SAVEPATH
    ngs_wfe : list of 2 floats, optional [nm, mas]
        [high-order WFE (ho_wfe), dynamic tip-tilt (tt_dynamic)] for the
        natural-guide-star AO case, passed to pick_coupling_rounded.
        Default [130,3]
    lgs_wfe : list of 2 floats, optional [nm, mas]
        [high-order WFE (ho_wfe), dynamic tip-tilt (tt_dynamic)] for the
        laser-guide-star AO case, passed to pick_coupling_rounded.
        Default [220,9.4]
    atm : int, optional
        0 or 1, whether atmosphere was included in the coupling simulation
        grid selected by grid_interp_coupling/pick_coupling_rounded.
        Default 1
    adc : int, optional
        0 or 1, whether the ADC was included in the coupling simulation
        grid selected by grid_interp_coupling/pick_coupling_rounded.
        Default 1

    outputs
    -------
    None
        Draws a new matplotlib figure and saves it to
        '<outputdir>/e2e_mri_plot_yJ.png' and
        '<outputdir>/e2e_mri_plot_yJ.pdf'; also writes
        '<outputdir>/ngs_throughput_bspec.txt' and
        '<outputdir>/lgs_throughput_bspec.txt'
    """
    data={}
    data['red'] = {}
    data['blue'] =  {}

    #data['Atmosphere'] = pass

    colors = ['b','orange','gray','yellow','lightblue','green','k']
    labels = ['Atmosphere','Telescope','Keck AO','FEI','Fiber \nCoupling','Fiber\nPropogation',]
    for spec in ['red','blue']:
        if spec=='red':
            include = ['tel', 'ao', 'feicom', 'feired','fibred','rspec']#,'coupling']
        if spec=='blue':
            include = ['tel', 'ao', 'feicom', 'feiblue','fibblue','bspec']#,'coupling']

        for i in include:
            if i==include[0]:
                w,s = np.loadtxt(transmission_path + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                data[spec][i] = s
            else:
                wtemp, stemp = np.loadtxt(transmission_path + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                # interpolate onto s
                f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
                data[spec][i] = f(w)
                #plt.plot(w,s,label=i)

    #load atmosphere and degrade to lower res, resample onto w
    teldata      = fits.getdata(telluric_file)
    _,ind  = np.unique(teldata['Wave/freq'],return_index=True)
    tck_tel   = interpolate.splrep(teldata['Wave/freq'][ind],teldata['Total'][ind], k=2, s=0)
    telluric = interpolate.splev(1000*w,tck_tel,der=0,ext=1)
    telluric_spec  = degrade_spec(w,telluric,100)
    data['atm'] = telluric_spec
 
    #load coupling for two options
    # inputs : waves,dynwfe,ttStatic,ttDynamic
    out = throughput_tools.grid_interp_coupling(1,path=transmission_path  + 'coupling/',atm=atm,adc=adc)
    #data['coupling_NGS'],strehl  = pick_coupling(w,ngs_wfe[0],0,ngs_wfe[1],LO=0,PLon=1,points=out[0],values=out[1:])
    data['coupling_NGS'],strehl = throughput_tools.pick_coupling_rounded(transmission_path,w,ngs_wfe[0], ngs_wfe[1], lo_wfe=50, tt_static=0, defocus=30, atm=1,adc=1,pl_on=1,piaa_boost=1.3)
    out = throughput_tools.grid_interp_coupling(1,path=transmission_path +'coupling/',atm=atm,adc=adc)
    #data['coupling_LGS'],strehl2 = pick_coupling(w,lgs_wfe[0],0,lgs_wfe[1],LO=30,PLon=1,points=out[0],values=out[1:])
    data['coupling_LGS'],strehl2  = throughput_tools.pick_coupling_rounded(transmission_path,w,lgs_wfe[0], lgs_wfe[1], lo_wfe=50, tt_static=0, defocus=30, atm=1,adc=1,pl_on=1,piaa_boost=1.3)

    if np.max(w)>1000: w/=1000
    # plot blue only
    spec = 'blue'
    plt.figure(figsize=(7,6))
    #plt.semilogy(w,data['atm'],c='royalblue',linewidth=1)
    plt.plot(w,data[spec]['tel'],c='darkorange',linewidth=1)
    plt.plot(w, data[spec]['tel'] * data[spec]['ao'],c='silver',linewidth=1)
    plt.plot(w,data[spec]['tel'] * data[spec]['ao']* \
                    data[spec]['feicom']*data[spec]['feiblue'],c='gold',linewidth=1)
    
    plt.plot(w, data[spec]['tel'] * data[spec]['ao']* \
                    data[spec]['feicom']*data[spec]['feiblue']*data[spec]['fibblue']*\
                    data['coupling_NGS'],c='steelblue',alpha=0.8,linewidth=1)
    
    ngs =  data[spec]['tel'] * data[spec]['ao']* \
                    data[spec]['feicom']*data[spec]['feiblue']*data[spec]['fibblue']*\
                    data[spec]['bspec']*data['coupling_NGS']
    lgs = data[spec]['tel'] * data[spec]['ao']* \
                    data[spec]['feicom']*data[spec]['feiblue']*data[spec]['fibblue']*\
                    data[spec]['bspec']*data['coupling_LGS']
    
    np.savetxt(outputdir + 'ngs_throughput_bspec.txt',np.vstack((w,ngs)).T)
    np.savetxt(outputdir+'lgs_throughput_bspec.txt',np.vstack((w,lgs)).T)

    ngs[np.where(ngs<0.015)[0]] = np.nan
    lgs[np.where((ngs<0.015) & (w > 1.9))[0]] = np.nan
    lgs[np.where(lgs<0.005)[0]] = np.nan
    lgs[np.where((lgs<0.005) & (w > 1.9))[0]] = np.nan
    plt.plot(w,ngs,c='seagreen',linewidth=1)
    
    plt.plot(w,lgs,c='seagreen',alpha=0.5,linewidth=1)
    
    plt.xlabel('Wavelength (microns)',color='k')
    plt.ylabel('Cumulative Throughput (log)',color='k')
    plt.ylim(0.005,1)
    plt.xlim(0.980, 1.490)
    plt.axhline(np.max(ngs),c='k',linestyle='--',linewidth=2)
    plt.fill_between([1.33, 1.49],0.00,y2=1,facecolor='w',zorder=110)
    plt.fill_between([0.98, 1.07],0.0,y2=1,facecolor='gray',alpha=0.2,zorder=-110)
    plt.fill_between([1.170,1.327],0.0,y2=1,facecolor='gray',alpha=0.2,zorder=-110)
    
    #plt.title("HISPEC E2E Except Coupling")
    # y lines
    yticks = [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.8]
    #yticks = [0.01, 0.03, 0.09, 0.27, 0.81]
    xticks = np.round((np.arange(0.98, 1.49,0.04)),2)
    #plt.yticks(ticks=yticks,labels=yticks,color='k',fontsize=12)
    plt.xticks(rotation=90,ticks=xticks,labels=xticks,color='k',fontsize=12)
    plt.grid(axis='y',alpha=0.4)
    plt.subplots_adjust(bottom=0.17)
    plt.title('yJ Throughput')
    plt.savefig(outputdir + 'e2e_mri_plot_yJ.png')
    plt.savefig(outputdir + 'e2e_mri_plot_yJ.pdf')



def plot_throughput_components(telluric_file=DATAPATH + 'telluric/psg_out_2020.08.02_l0_800nm_l1_2700nm_res_0.001nm_lon_204.53_lat_19.82_pres_0.5826.fits',
                                    transmission_path = DATAPATH + 'instrument/hispec/throughput/',
                                    outputdir=SAVEPATH,
                                    ngs_wfe=[130,3],
                                    lgs_wfe=[220,9.4],
                                    atm=1,adc=1):
    """
    Build and plot the cumulative system throughput across both the blue
    (y+J) and red (H+K) paths on a single figure: telescope, Keck AO,
    front-end injection (FEI) optics, fiber coupling (NGS and LGS, photonic
    lantern off, via grid_interp_coupling/pick_coupling_rounded with
    lo_wfe=50, defocus=30), fiber propagation, and spectrograph, layered
    cumulatively with labeled annotations. Also saves the per-band NGS/LGS
    cumulative throughput arrays to text files, and saves the figure as PNG
    and PDF.

    inputs
    ------
    telluric_file : string, optional
        path to a FITS file (PSG telluric transmission model) with
        'Wave/freq' and 'Total' columns, loaded/degraded into data['atm']
        (not directly plotted in this function)
    transmission_path : string, optional
        path to the directory of per-surface '<surface>_throughput.csv'
        files; also used (with a 'coupling/' suffix) as the coupling grid
        path for grid_interp_coupling and pick_coupling_rounded
    outputdir : string, optional
        directory to save the output 'e2e_plot_all.png'/'.pdf' figures and
        the 'ngs_throughput_<band>.txt'/'lgs_throughput_<band>.txt' data
        files (band = 'blue' or 'red'). Default SAVEPATH
    ngs_wfe : list of 2 floats, optional [nm, mas]
        [high-order WFE (ho_wfe), dynamic tip-tilt (tt_dynamic)] for the
        natural-guide-star AO case, passed to pick_coupling_rounded.
        Default [130,3]
    lgs_wfe : list of 2 floats, optional [nm, mas]
        [high-order WFE (ho_wfe), dynamic tip-tilt (tt_dynamic)] for the
        laser-guide-star AO case, passed to pick_coupling_rounded.
        Default [220,9.4]
    atm : int, optional
        0 or 1, whether atmosphere was included in the coupling simulation
        grid selected by grid_interp_coupling/pick_coupling_rounded.
        Default 1
    adc : int, optional
        0 or 1, whether the ADC was included in the coupling simulation
        grid selected by grid_interp_coupling/pick_coupling_rounded.
        Default 1

    outputs
    -------
    allw : list of arrays
        wavelength array (microns) used for each band, in the loop order
        ['blue','red']
    allngs : list of arrays
        cumulative NGS throughput (with low values set to NaN for plot
        clarity) for each band, in the loop order ['blue','red']
    alllgs : list of arrays
        cumulative LGS throughput (with low values set to NaN for plot
        clarity) for each band, in the loop order ['blue','red']
    data : dict
        nested dict of intermediate per-surface throughput curves and
        coupling arrays used to build the plot (keys 'red', 'blue', 'atm',
        'coupling_NGS', 'coupling_LGS')

    Side effects
    ------------
    Draws a new matplotlib figure and saves it to
    '<outputdir>/e2e_plot_all.png' and '<outputdir>/e2e_plot_all.pdf'; also
    writes '<outputdir>/ngs_throughput_<band>.txt' and
    '<outputdir>/lgs_throughput_<band>.txt' for band in ['blue','red']
    """
    data={}
    data['red'] = {}
    data['blue'] =  {}

    colors = ['b','orange','gray','yellow','lightblue','green','k']
    labels = ['Atmosphere','Telescope','Keck AO','FEI','Fiber \nCoupling','Fiber\nPropogation',]
    for spec in ['red','blue']:
        if spec=='red':
            include = ['tel', 'ao', 'feicom', 'feired','fibred','rspec']#,'coupling']
        if spec=='blue':
            include = ['tel', 'ao', 'feicom', 'feiblue','fibblue','bspec']#,'coupling']

        for i in include:
            if i==include[0]:
                w,s = np.loadtxt(transmission_path + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                data[spec][i] = s
            else:
                wtemp, stemp = np.loadtxt(transmission_path + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                # interpolate onto s
                f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
                data[spec][i] = f(w)
                #plt.plot(w,s,label=i)

    #load atmosphere and degrade to lower res, resample onto w
    teldata      = fits.getdata(telluric_file)
    _,ind  = np.unique(teldata['Wave/freq'],return_index=True)
    tck_tel   = interpolate.splrep(teldata['Wave/freq'][ind],teldata['Total'][ind], k=2, s=0)
    telluric  = interpolate.splev(1000*w,tck_tel,der=0,ext=1)
    telluric_spec  = degrade_spec(w,telluric,100)
    data['atm']    = telluric_spec
 
    #load coupling for two options
    # inputs : waves,dynwfe,ttStatic,ttDynamic
    out = throughput_tools.grid_interp_coupling(1,path=transmission_path  + 'coupling/',atm=atm,adc=adc)
    data['coupling_NGS'],strehl = throughput_tools.pick_coupling_rounded(transmission_path,w,ngs_wfe[0], ngs_wfe[1], lo_wfe=50, tt_static=0, defocus=30, atm=atm,adc=adc,pl_on=0,piaa_boost=1.3)
 
    out = throughput_tools.grid_interp_coupling(1,path=transmission_path +'coupling/',atm=atm,adc=adc)
    data['coupling_LGS'],strehl2  = throughput_tools.pick_coupling_rounded(transmission_path,w,lgs_wfe[0], lgs_wfe[1], lo_wfe=50, tt_static=0, defocus=30, atm=atm,adc=adc,pl_on=0,piaa_boost=1.3)

    if np.max(w)>1000: w/=1000
    lw=2
    # save ngs,lgs total
    allw,allngs,alllgs = [],[],[]
    # plot red only
    fig, ax = plt.subplots(1,figsize=(9,6))
    colors = ['blue','red','yellow','purple','green','cyan']
    for spec in ['blue','red']:
        if spec=='red':
            include = ['tel', 'ao', 'feicom', 'feired', 'fibred','rspec']#,'coupling']
        if spec=='blue':
            include = ['tel', 'ao', 'feicom', 'feiblue','fibblue','bspec']#,'coupling']

        #plt.semilogy(w,data['atm'],c='royalblue',linewidth=1)
        # telescope
        ax.plot(w,data[spec]['tel'],c='darkorange',linewidth=lw)
        #nfiraos ao
        ax.plot(w, data[spec]['tel'] * data[spec]['ao'],c='gray',linewidth=lw)
        #fei
        ax.plot(w, data[spec]['tel'] * data[spec]['ao']* \
                        data[spec]['feicom']*data[spec][include[3]],c='gold',linewidth=lw)
        #coupling
        ax.plot(w,data[spec]['tel'] * data[spec]['ao']* \
                        data[spec]['feicom']*data[spec][include[3]]*data['coupling_NGS'],\
                        c='steelblue',alpha=0.8,linewidth=lw)
        #fibers
        ax.plot(w,data[spec]['tel'] * data[spec]['ao']* \
                        data[spec]['feicom']*data[spec][include[3]]*data[spec][include[4]]*\
                        data['coupling_NGS'],c='purple',alpha=0.8,linewidth=lw)
        
        ngs =  data[spec]['tel'] * data[spec]['ao']* \
                        data[spec]['feicom']*data[spec][include[3]]*data[spec][include[4]]*\
                        data[spec][include[5]]*data['coupling_NGS']
        lgs =  data[spec]['tel'] * data[spec]['ao']* \
                        data[spec]['feicom']*data[spec][include[3]]*data[spec][include[4]]*\
                        data[spec][include[5]]*data['coupling_LGS']
        np.savetxt(outputdir + 'ngs_throughput_%s.txt'%spec,np.vstack((w,ngs)).T)
        np.savetxt(outputdir + 'lgs_throughput_%s.txt'%spec,np.vstack((w,lgs)).T)

        ngs[np.where(ngs<0.015)[0]] = np.nan
        lgs[np.where((ngs<0.015) & (w > 1.9))[0]] = np.nan
        lgs[np.where(lgs<0.01)[0]] = np.nan
        lgs[np.where((lgs<0.015) & (w > 1.9))[0]] = np.nan
        #spec
        ax.plot(w,ngs,c='seagreen',linewidth=lw)
        allngs.append(ngs)
        allw.append(w)
        alllgs.append(lgs)
        #plt.plot(w,lgs,c='seagreen',alpha=0.5,linewidth=lw)
    
    ax.set_ylim(0.01,1)
    ax.set_xlim(0.985, 2.455)
    ax.axhline(np.max(ngs),c='k',linestyle='--',linewidth=2)
    ax.fill_between([1.327, 1.5],0.01,y2=1,facecolor='w',zorder=110)
    ax.fill_between([1.490,1.780],0.01,y2=1,facecolor='gray',alpha=0.2,zorder=110)
    ax.fill_between([1.990,2.460],0.01,y2=1,facecolor='gray',alpha=0.2,zorder=110)
    ax.fill_between([0.98, 1.07],0.0,y2=1,facecolor='gray',alpha=0.2,zorder=-110)
    ax.fill_between([1.170,1.327],0.0,y2=1,facecolor='gray',alpha=0.2,zorder=-110)

    # add text to plot
    ax.text(1.34,0.89,'Telescope',zorder=200,fontsize=10)
    ax.text(1.34,0.74,'NFIRAOS',zorder=200,fontsize=10)
    ax.text(1.34,0.45,'FEI Optics',zorder=200,fontsize=10)
    ax.text(1.34,0.27,'Coupling',zorder=200,fontsize=10)
    ax.text(1.34,0.23,'Fibers',zorder=200,fontsize=10)
    ax.text(1.34,0.09,'SPEC',zorder=200,fontsize=10)


    #plt.title("HISPEC E2E Except Coupling")
    # y lines
    yticks = [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.8]
    #yticks = [0.01, 0.03, 0.09, 0.27, 0.81]
    xticks = np.round((np.arange(1.49, 2.45,0.04)),2)
    #plt.yticks(ticks=yticks,labels=yticks,color='k',fontsize=12)
    #plt.xticks(rotation=90,ticks=xticks,labels=xticks,color='k',fontsize=12)
    ax.yaxis.grid(True, which='both',alpha=0.5)
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    plt.subplots_adjust(bottom=0.17)
    ax.set_title('Cumulative Throughput')
    ax.set_xlabel('Wavelength (microns)',color='k')
    ax.set_ylabel('Throughput',color='k')
    plt.savefig(outputdir + 'e2e_plot_all.png')
    plt.savefig(outputdir + 'e2e_plot_all.pdf')
    return allw,allngs,alllgs,data


##############################################################
# Helpers moved here from obs_tools.py (only used by plots)
###############################################################

def get_order_value(x,y,order_filename):
    """
    Given the order centers and free spectral ranges tabulated in
    order_filename, return the peak and mean of y (e.g. SNR) within (a
    padded window around) each order.

    inputs:
    -------
    x - array
        wavelength array [nm] corresponding to y

    y - array
        quantity to summarize per order (e.g. SNR), evaluated at x

    order_filename - str
        path to order bounds file (wavelength [nm], order width [nm],
        comma delimited) passed to instrument.get_order_bounds

    returns:
    --------
    order_cen_lam - array
        center wavelength [nm] of each order, from order_filename

    snr_peaks - array
        maximum value of y within +/-1.3*(fsr/2) of each order center

    snr_means - array
        mean value of y within +/-1.3*(fsr/2) of each order center
    """
    order_cen_lam,fsr = get_order_bounds(order_filename)

    snr_peaks = []
    snr_means = []
    for i,lam_cen in enumerate(order_cen_lam):
        sub_snr = y[np.where((x > (lam_cen - 1.3*fsr[i]/2)) & (x < (lam_cen+1.3*fsr[i]/2)))[0]]
        snr_peaks.append(np.nanmax(sub_snr))
        snr_means.append(np.nanmean(sub_snr))

    return np.array(order_cen_lam), np.array(snr_peaks), np.array(snr_means)


def load_confirmed_planets(planets_filename=DATAPATH + 'populations/confirmed_planets_PS_2023.01.12_16.07.07.csv'):
    """
    Load host star H magnitudes and effective temperatures from a
    confirmed planets catalog (NASA Exoplanet Archive format)

    input
    -----
    planets_filename - str
        path to confirmed planets csv file

    output
    ------
    hmags - array
        host star H magnitudes
    teffs - array
        host star effective temperatures [K]
    """
    planet_data =  pd.read_csv(planets_filename,delimiter=',',comment='#')
    hmags = planet_data['sy_hmag']
    teffs = planet_data['st_teff']
    return hmags,teffs
