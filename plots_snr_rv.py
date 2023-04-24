# calc signal to noise
# max for the calcium H&K
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

font = {'size'   : 14}
matplotlib.rc('font', **font)

sys.path.append('./utils/')
from objects import load_object
from load_inputs import fill_data, load_filter
from functions import *

import throughput_tools, wfe_tools

plt.ion()


def get_wfe_landscape():
    """
    figure out which AO mode is best for which temperature and mag star

    save to file to be loaded by plot_ao_landscape
    """
    configfile = 'hispec_tracking_camera.cfg'
    so         = load_object(configfile)
    cload      = fill_data(so)
  
    temp_arr   = [1000,1500,2300,2700,3000,3600,4200,5800]
    magarr     = np.arange(1,15,2)
    best_ao_mode_arr = np.zeros((len(temp_arr),len(magarr)),dtype=int)
    best_ao_coupling = np.zeros((len(temp_arr),len(magarr)))
    for i,temp in enumerate(temp_arr):
        cload.set_teff_mag(so,temp,so.stel.mag,star_only=True)
        for j,mag in enumerate(magarr):
            print('mag=%s'%mag)
            factor_0 = so.stel.factor_0 * 10**(-0.4*(mag - so.stel.mag))
            ao_modes = ['SH','80J','LGS_100J_130']
            #ao_modes = ['SH','80J','LGS_100J_130']
            coupling = np.zeros(len(ao_modes))
            for k,ao_mode in enumerate(ao_modes):
                so.ao.mode=ao_mode
                # get wfe
                f = pd.read_csv(so.ao.ho_wfe_set,header=[0,1])
                so.ao.modes = f.columns
                mags             = f['mag'].values.T[0]
                wfes             = f[so.ao.mode].values.T[0]
                so.ao.ho_wfe_band= f[so.ao.mode].columns[0] # this is the mag band wfe is defined in, must be more readable way..
                so.ao.ho_wfe_mag = throughput_tools.get_band_mag(so,'Johnson',so.ao.ho_wfe_band,factor_0) # get magnitude of star in appropriate band
                f_howfe          = interpolate.interp1d(mags,wfes, bounds_error=False,fill_value=10000)
                so.ao.ho_wfe     = float(f_howfe(so.ao.ho_wfe_mag))
                # get tiptilt
                f = pd.read_csv(so.ao.ttdynamic_set,header=[0,1])
                so.ao.modes_tt = f.columns # should match howfe..
                mags            = f['mag'].values.T[0]
                tts             = f[so.ao.modes_tt].values.T[0]
                so.ao.ttdynamic_band=f[so.ao.mode].columns[0] # this is the mag band wfe is defined in, must be more readable way..
                so.ao.ttdynamic_mag = throughput_tools.get_band_mag(so,'Johnson',so.ao.ttdynamic_band,so.stel.factor_0) # get magnitude of star in appropriate band
                f_ttdynamic     =  interpolate.interp1d(mags,tts, bounds_error=False,fill_value=10000)
                so.ao.tt_dynamic= float(f_ttdynamic(so.ao.ttdynamic_mag))
                # get coupling
                if so.ao.tt_dynamic>20:
                    so.ao.tt_dynamic=20
                so.inst.coupling, so.inst.strehl = throughput_tools.pick_coupling(so.stel.v,so.ao.ho_wfe,so.ao.tt_static,so.ao.tt_dynamic,points=so.inst.grid_points,values=so.inst.grid_values,LO=so.ao.lo_wfe,PLon=so.inst.pl_on,transmission_path=so.inst.transmission_path) # includes PIAA and HO term
                coupling[k] = np.median(so.inst.coupling)
            if np.max(coupling)==0: 
                best_ao_mode_arr[i,j]=-1
                print('all modes suck')
            else:
                best_ao_mode_arr[i,j] = np.argmax(coupling)
                best_ao_coupling[i,j] = coupling[np.argmax(coupling)]
                print('best_mode: %s'%ao_modes[best_ao_mode_arr[i,j]])

    np.save('./output/ao_modes/best_ao_mode_%smag_[1,15,2]_temps_1000,1500,2300,3000,3600,4200,5800'%so.filt.band,best_ao_mode_arr)
    np.save('./output/ao_modes/best_ao_coupling_%smag_[1,15,2]_temps_1000,1500,2300,3000,3600,4200,5800'%so.filt.band,best_ao_coupling)


def plot_planets_ao_modes():
    """
    plot planet populations with regions of AO mode and coupling performance

    show with and without a pyramid mode - have to edit which plotting

    To Do: 
        - rerun coupling curves once gary extends tip/tilt grid
        - rerun with denser mag grid
        - label ao regions and double check correct orientation is shown
        - add brown dwarfs to plot
        - do again with contours as peak SNR in a 15 min exposure
    must be careful to pick right ao_modes array since not recorded in header
    """
    band='H'
    ao_modes  = ['SH','80J','LGS_100H_130']
    best_ao_mode_arr      =  np.load('./output/ao_modes/best_ao_mode_Hmag_[1,15,2]_temps_1000,1500,2300,3000,3600,4200,5800.npy')
    best_ao_coupling_arr  =  np.load('./output/ao_modes/best_ao_coupling_%smag_[1,15,2]_temps_1000,1500,2300,3000,3600,4200,5800.npy'%band)

    #best2_ao_mode_arr      =  np.load('./output/ao_modes/best2_ao_mode_Hmag_[1,15,2]_temps_1000,1500,2300,3000,3600,4200,5800.npy')
    #best2_ao_coupling      =  np.load('./output/ao_modes/best2_ao_coupling_%smag_[1,15,2]_temps_1000,1500,2300,3000,3600,4200,5800.npy'%band)
    #ao_modes2   = ['SH','LGS_100H_130']

    temp_arr   = [1000,1500,2300,2700,3000,3600,4200,5800]
    magarr     = np.arange(1,15,2)
    extent     = (np.min(magarr),np.max(magarr),np.min(temp_arr),np.max(temp_arr))

    # plot contours of it
    planets_filename = './data/populations/PS_2023.02.08_13.41.45.csv'
    planet_data =  pd.read_csv(planets_filename,delimiter=',',comment='#')
    # add brown dwarfs!
    hmags = planet_data['sy_hmag']
    teffs = planet_data['st_teff']
    mass  = planet_data['pl_bmassj']
    method = planet_data['discoverymethod']
    iRV    = np.where(method=='Radial Velocity')[0]
    iim    = np.where(method=='Imaging')[0]
    nomass = np.where(np.isnan(mass))[0]

    def fmt(x):
        "format labels for contour plot"
        return str(np.round(100*x)) + '%'

    fig, ax = plt.subplots(1,1,figsize=(6,5))
    ax.imshow(best_ao_mode_arr,aspect='auto',origin='lower',\
            interpolation='None',cmap='nipy_spectral',\
            extent=extent)
    cs = ax.contour(best_ao_coupling_arr ,\
                colors=['r'],origin='lower',\
                extent=extent)
    ax.clabel(cs, cs.levels, inline=True,fmt=fmt,fontsize=10,\
            colors=['r'],zorder=101)

    plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.85)
    ax.scatter(hmags,teffs,marker='.',c='c',alpha=0.5,label='Planet Hosts')
    #ax.scatter(hmags_bd,teffs_bd,marker='.',c='c',alpha=0.5,label='Brown Dwarfs')
    ax.set_xlabel('H Mag')
    ax.set_ylabel('T$_{eff}$')
    ax.set_title('HISPEC Performance Landscape')
    ax.set_xlim(15,1)
    ax.set_ylim(2300,7000)
    ax.legend(fontsize=12)
    plt.subplots_adjust(bottom=0.15,hspace=0,left=0.15)

    #plt.savefig('./output/')

def load_planets():
    planets_filename = './data/populations/confirmed_uncontroversial_planets_2023.03.08_14.19.56.csv'
    planet_data =  pd.read_csv(planets_filename,delimiter=',',comment='#')
    # add brown dwarfs!
    hmags = planet_data['sy_hmag']
    teffs = planet_data['st_teff']
    rvamps = planet_data['pl_rvamp']
    method = planet_data['discoverymethod']
    return hmags,teffs,method,rvamps


def load_brown_dwarfs(ploton=False):
    """
    """
    bd_filename = './data/populations/UltracoolSheetMain.csv'
    bd_data =  pd.read_csv(bd_filename,delimiter=',',comment='#')

    #sp_type = bd_data['spt_opt']
    sp_type = bd_data['spt_ir']
    hmags = bd_data['H_MKO']
    #jmags = bd_data['J_2MASS']
    hmags = bd_data['H_2MASS']
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

    if ploton:
        plt.figure()
        plt.hist(hmags,bins=100,alpha=0.7)

        plt.figure()
        plt.plot(hmags,teffs,'o',alpha=0.8)
        plt.xlabel('H Mag')
        plt.ylabel('T$_{eff}$ (K)')
        plt.subplots_adjust(left=0.15)

    return hmags, teffs

def get_strehl():
    """
    """
    configfile = 'hispec_tracking_camera.cfg'
    so         = load_object(configfile)
    cload      = fill_data(so)

    temp_arr   = [1000,1500,2300,2700,3000,3600,4200,5800]

    f        = pd.read_csv(so.ao.ho_wfe_set,header=[0,1])
    ftt      = pd.read_csv(so.ao.ttdynamic_set,header=[0,1])
    modes    = ['LGS_100J_130','SH']
    rawmags  = f['mag'].values.T[0]
    best_mode_arr = np.zeros((len(temp_arr),len(rawmags)))
    for i,teff in enumerate(temp_arr):
        so.stel.teff = teff
        cload.set_teff_mag(so,teff,so.stel.mag,star_only=True)
        wfes     = []
        tts      = []
        strehl   = []
        for mode in modes:
            wfes.append(f[mode].values.T[0])
            tts.append(ftt[mode].values.T[0])
            band            = f[mode].columns[0] # this is the mag band wfe is defined in, must be more readable way..
            onemag          = throughput_tools.get_band_mag(so,'Johnson',band,so.stel.factor_0) # get magnitude of star in appropriate band
            color           = so.stel.mag - onemag
            strehl_ho = wfe_tools.calc_strehl(f[mode].values.T[0],so.filt.center_wavelength)
            strehl_tt = wfe_tools.tt_to_strehl(ftt[mode].values.T[0],so.filt.center_wavelength,so.inst.tel_diam)
            strehl_tmp = strehl_tt*strehl_ho
            # reinterpolate onto rawmags
            strehl_interp = interpolate.interp1d(rawmags + color,strehl_tmp,bounds_error=False,fill_value="extrapolate")
            strehl.append(strehl_interp(rawmags))

        best_mode_arr[i,:] = np.argmax(np.array(strehl),axis=0)
    np.save('./output/ao_modes/best_ao_mode_%smag_temps_1000,1500,2300,3000,3600,4200,5800_modes_%s'%(so.filt.band,modes),best_mode_arr)
    np.save('./output/ao_modes/best_ao_mode_%smag_mags'%(so.filt.band),rawmags)

def plot_planets_ao_modes_strehl():
    """
    plot planet populations with regions of AO mode and coupling performance

    show with and without a pyramid mode - have to edit which plotting

    To Do: 
        - rerun coupling curves once gary extends tip/tilt grid
        - rerun with denser mag grid
        - label ao regions and double check correct orientation is shown
        - add brown dwarfs to plot
        - do again with contours as peak SNR in a 15 min exposure
    must be careful to pick right ao_modes array since not recorded in header
    """
    band='H'
    ao_modes  = ['LGS_100J_130','SH']
    best_ao_mode_arr      =  np.load('./output/ao_modes/best_ao_mode_Hmag_temps_1000,1500,2300,3000,3600,4200,5800_modes_%s.npy'%ao_modes)
    magarr = np.load('./output/ao_modes/best_ao_mode_Hmag_mags.npy')
    #ao_modes  = ['LGS_100H_130','SH']
    #best_ao_mode_arr= np.load('./output/ao_modes/best_ao_mode_Hmag_temps_1000,1500,2300,3000,3600,4200,5800_modes_[LGS_100J_130_SH.npy')
    #best_ao_coupling_arr  =  np.load('./output/ao_modes/best_ao_coupling_%smag_[1,15,2]_temps_1000,1500,2300,3000,3600,4200,5800.npy'%band)

    #best2_ao_mode_arr      =  np.load('./output/ao_modes/best2_ao_mode_Hmag_[1,15,2]_temps_1000,1500,2300,3000,3600,4200,5800.npy')
    #best2_ao_coupling      =  np.load('./output/ao_modes/best2_ao_coupling_%smag_[1,15,2]_temps_1000,1500,2300,3000,3600,4200,5800.npy'%band)
    #ao_modes2   = ['SH','LGS_100H_130']

    temp_arr   = [1000,1500,2300,2700,3000,3600,4200,5800]
    extent     = (np.min(temp_arr),np.max(temp_arr),np.min(magarr),np.max(magarr))

    # plot contours of it
    hmags, teffs,method,rvamps = load_planets()
    i_image = np.where(method=='Imaging')[0]
    i_ms= np.where(rvamps<3)[0]
    hmags_bd, teffs_bd = load_brown_dwarfs()

    def fmt(x):
        "format labels for contour plot"
        return str(np.round(100*x)) + '%'

    ##########
    fig, ax = plt.subplots(1,1,figsize=(6,5))
    ax.imshow(best_ao_mode_arr.T,aspect='auto',origin='lower',\
            interpolation='None',cmap='gray',\
            extent=extent,vmin=0,vmax=2)
    #cs = ax.contour(best_ao_coupling ,\
    #            colors=['r'],origin='lower',\
    #            extent=extent)
    #ax.clabel(cs, cs.levels, inline=True,fmt=fmt,fontsize=10,\
    #        colors=['r'],zorder=101)


    ax.scatter(teffs,hmags,marker='.',s=65,c='m',ec='purple',alpha=0.8,label='Confirmed Planet Hosts')
    #ax.scatter(teffs[i_image],hmags[i_image],marker='s',s=40,c='brown',ec='k',alpha=1,label='Imaged Planets')
    ax.scatter(teffs_bd,hmags_bd,marker='d',s=32,c='darkcyan',ec='b',alpha=1,label='Brown Dwarfs')
    #ax.scatter(hmags_bd,teffs_bd,marker='.',c='c',alpha=0.5,label='Brown Dwarfs')
    ax.set_ylabel('H Mag')
    ax.set_xlabel('T$_{eff}$ (K)')
    ax.set_title('HISPEC AO Mode Landscape')
    ax.set_ylim(0,16)
    ax.set_xlim(1000,5800)
    #ax.legend(fontsize=10,loc=1)
    plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.85)

    # add region for cred2 interrupted science region
    if True:
        mag_limit_jhgap_cred2  = [13.5,13.7,14.3,15,15]
        mag_limit_jhgap_h2rg   = [14.3,14.5,15,15,15]
        teff_limit_jhgap = [1000,1500,2300,3000,6000]
        ax.plot(teff_limit_jhgap,mag_limit_jhgap_h2rg,'cyan',lw=2,label='H2RG')
        ax.plot(teff_limit_jhgap,mag_limit_jhgap_cred2,'--',c='red',lw=2,label='C-RED2')
        ax.set_title('Uninterrupted Science Limits')
        ax.set_ylim(5,15)
        ax.legend(fontsize=10)
        ntotal_bd = len(np.where((hmags_bd < 15) & (teffs_bd >1000) & (teffs_bd < 5800))[0])
        ntotal_pl = len(np.where((hmags < 15) & (teffs >1000) & (teffs < 5800))[0])

        plt.savefig('./output/trackingcamera/ao_mode_landscape_brown_dwarfs_trackingcutoff.png',dpi=300)
    else:
        plt.savefig('./output/ao_modes/ao_mode_landscape_brown_dwarfs.png')


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



def plot_coupling_ao_mode():
    configfile = 'hispec_snr.cfg'
    so    = load_object(configfile)
    cload = fill_data(so) # put coupling files in load and wfe stuff too

    so.stel.teff = 3600
    so.ao.mode='LGS_100J_130'
    cload.set_teff_aomode(so,so.stel.teff,so.ao.mode) # put coupling files in load and wfe stuff too
    fig, ax = plt.subplots(1,1,figsize=(7,5))
    fig.subplots_adjust(bottom=0.15,left=0.15)
    fig2, ax2 = plt.subplots(1,1,figsize=(7,5))
    fig2.subplots_adjust(bottom=0.15)
    ax.plot(so.inst.xtransmit,so.inst.ytransmit,'k',zorder=-101,lw=2,label=so.ao.mode)
    cen_lam, snr_peaks,snr_means = get_order_snrs(so,so.obs.v,so.obs.snr)
    ax2.plot(cen_lam, snr_peaks,'k',lw=2,zorder=-101,label=so.ao.mode)

    so.ao.mode='SH'
    cload.set_teff_aomode(so,so.stel.teff,so.ao.mode) # put coupling files in load and wfe stuff too
    ax.plot(so.inst.xtransmit,so.inst.ytransmit,'gray',lw=2,zorder=-101,label=so.ao.mode)
    cen_lam, snr_peaks,snr_means = get_order_snrs(so,so.obs.v,so.obs.snr)
    ax2.plot(cen_lam, snr_peaks,'gray',lw=2,zorder=-101,label=so.ao.mode)

    so.ao.mode='80J'
    cload.set_teff_aomode(so,so.stel.teff,so.ao.mode) # put coupling files in load and wfe stuff too
    ax.plot(so.inst.xtransmit,so.inst.ytransmit,'g',lw=2,zorder=-101,label=so.ao.mode)
    cen_lam, snr_peaks,snr_means = get_order_snrs(so,so.obs.v,so.obs.snr)
    ax2.plot(cen_lam, snr_peaks,'g',lw=2,zorder=-101,label=so.ao.mode)

    ax.set_ylabel('Throughput')
    ax.set_xlabel('Wavelength (nm)')
    ax.legend()
    ax.set_xlim(980,2460)
    ax.set_title('Teff: %sK, %s=%s'%(so.stel.teff,so.filt.band,so.stel.mag))
    ax.fill_between([1330,1494],-1,0.75,facecolor='white',zorder=-100)
    ax.grid()
    ax.set_ylim(0,0.1)
    figname = 'coupling_boost_%sK_%s_%s_%ss.png'%(so.stel.teff,so.filt.band,so.stel.mag,so.obs.texp)

    ax2.set_ylabel('SNR')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.legend()
    ax2.set_xlim(980,2460)
    ax2.fill_between([1330,1494],-10,np.max(snr_peaks)+10,facecolor='white',zorder=-100)
    ax2.grid()
    ax2.set_ylim(-1,np.max(snr_peaks)+400)
    ax2.set_title('Teff: %sK, %s=%s'%(so.stel.teff,so.filt.band,so.stel.mag))
    figname2 = 'snr_boost_%sK_%s_%s_texp_%ss.png'%(so.stel.teff,so.filt.band,so.stel.mag,so.obs.texp)

    fig.savefig('./output/ao_modes/' + figname)
    fig2.savefig('./output/ao_modes/' + figname2)


def plot_coupling_photonic_lantern():
    configfile = 'hispec_snr.cfg'
    so    = load_object(configfile)
    so.inst.pl_on= 1
    so.stel.teff = 3600
    so.stel.mag  = 5
    so.ao.mode='LGS_100J_130'
    cload = fill_data(so) # put coupling files in load and wfe stuff too
    
    fig, ax = plt.subplots(1,1,figsize=(7,5))
    fig.subplots_adjust(bottom=0.15,left=0.15)
    fig2, ax2 = plt.subplots(1,1,figsize=(7,5))
    fig2.subplots_adjust(bottom=0.15)

    ax.plot(so.inst.xtransmit,so.inst.ytransmit,'k',zorder=-101,lw=2,label='with PL')
    cen_lam, snr_peaks,snr_means = get_order_snrs(so,so.obs.v_resamp,so.obs.snr_reselement)
    ax2.plot(cen_lam, snr_peaks,'k',lw=2,zorder=-101,label='with PL')


    so.inst.pl_on=0
    cload = fill_data(so)  # put coupling files in load and wfe stuff too
    ax.plot(so.inst.xtransmit,so.inst.ytransmit,'c',zorder=-101,lw=2,label='without PL')
    cen_lam, snr_peaks,snr_means = get_order_snrs(so,so.obs.v_resamp,so.obs.snr_reselement)
    ax2.plot(cen_lam, snr_peaks,'c',lw=2,zorder=-101,label='without PL')

    ax.set_ylabel('Throughput')
    ax.set_xlabel('Wavelength (nm)')
    ax.legend()
    ax.set_xlim(980,1333)
    ax.set_title('Teff: %sK, %s=%s'%(so.stel.teff,so.filt.band,so.stel.mag))
    ax.grid()
    ax.set_ylim(0,0.05)
    figname = 'PL_coupling_boost_%sK_%s_%s_%ss.png'%(so.stel.teff,so.filt.band,so.stel.mag,so.obs.texp)

    ax2.set_ylabel('SNR')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.legend()
    ax2.set_xlim(980,1333)
    ax2.grid()
    ax2.set_ylim(-1,np.max(snr_peaks)+5)
    ax2.set_title('Teff: %sK, %s=%s'%(so.stel.teff,so.filt.band,so.stel.mag))
    figname2 = 'PL_snr_boost_%sK_%s_%s_texp_%ss.png'%(so.stel.teff,so.filt.band,so.stel.mag,so.obs.texp)

    fig.savefig('./output/throughput/photonic_lantern/' + figname)
    fig2.savefig('./output/throughput/photonic_lantern/' + figname2)



if __name__=='__main__':	
	configfile = 'hispec_snr.cfg'
	so    = load_object(configfile)
	cload = fill_data(so)
	


