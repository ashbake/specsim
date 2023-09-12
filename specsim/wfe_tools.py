##############################################################
# General functions for calc_snr_max
###############################################################

import numpy as np
from scipy.integrate import trapz
from scipy import signal
from scipy import signal, interpolate
import matplotlib.pylab as plt
import pandas as pd 

all = {}



def load_WFE(ho_wfe_file, tt_wfe_file, zenith_angle, seeing):
    """
    load new format WFE files where 
    AO performance is defined per seeing and zenith angle

    """
    data  = {}

    try:
        ho_ao_file = pd.read_csv(ho_wfe_file,header=[0,1,2,3,4])
        tt_ao_file = pd.read_csv(tt_wfe_file,header=[0,1,2,3,4])
    except:
        raise ValueError('Failed to read WFE files. Please check files exist in correct format')

    mags_ho     = ho_ao_file['mag'].values.T[0]
    mags_tt     = tt_ao_file['mag'].values.T[0]

    # require certain zenith angles - later can interp or round
    if zenith_angle not in [0,30,45,60]: raise ValueError('please specify zenith angle as 0, 30, 45, or 60deg')
    if seeing not in ['good','average','bad']: raise ValueError('please specify seeing good, average, or bad')
    
    ho_wfe_per_mode = ho_ao_file['WFE[nm]'][seeing][str(int(zenith_angle))]
    tt_wfe_per_mode = tt_ao_file['WFE[nm]'][seeing][str(int(zenith_angle))]
    
    for item in ho_wfe_per_mode.columns:
        ao_mode    = item[0]
        ho_wfes    = ho_wfe_per_mode[ao_mode].values.T[0]
        wfe_band   = ho_wfe_per_mode[ao_mode].columns[0]
        tt_wfes    = tt_wfe_per_mode[ao_mode].values.T[0]
        if wfe_band != tt_wfe_per_mode[ao_mode].columns[0]:
            raise ValueError('Check AO files! The bands per AO mode does not match')
        data[ao_mode] = {}
        data[ao_mode]['band']    = wfe_band
        data[ao_mode]['ho_wfe']  = ho_wfes
        data[ao_mode]['tt_wfe']  = tt_wfes
        data[ao_mode]['ho_mag']  = mags_ho
        data[ao_mode]['tt_mag']  = mags_tt

    return data


def calc_strehl(wfe,wavelength):
    """
    inputs
    ------
    wfe: nm
    wavelength: nm, grid or single number

    outputs
    -------
    strehl at wavelength
    """
    strehl = np.exp(-(2*np.pi*wfe/wavelength)**2)

    return strehl

def tt_to_strehl(tt,lam,D):
    """
    convert tip tilt residuals in mas to strehl according to Rich's equation
    
    equation 4.60 from Hardy 1998 (adaptive optics for astronomy) matches this

    inputs
    ------
    lam: nm
        wavelength(s)
    D: m
        telescope diameter
    tt: mas
        tip tilt rms

    outputs
    -------
    strehl_tt  
    """
    tt_rad = tt * 1e-3/206265 # convert to radians from mas
    lam_m =lam * 1e-9
    bottom = 1 + np.pi**2/2*((tt_rad)/(lam_m/D))**2
    strehl_tt = 1/bottom

    #sig_D = 0.44* lam_m/D
    #1/(1 + tt_rad**2/sig_D**2) KAON1322 doc eq 5 method matches Richs eqn

    return strehl_tt





def plot_strehl():
    """
    plot strehl for modhis for the different ao modes
    """
    configfile = './modhis_snr.cfg'
    so    = load_object(configfile)
    cload = fill_data(so) # put coupling files in load and wfe stuff too

    data = load_WFE(so.ao.ho_wfe_set, so.ao.ttdynamic_set, 60, 'bad')
    ao_modes =  np.array(list(data.keys()))


    colors   = ['b','orange','limegreen','g','r','c','b','gray','black']
    widths   = [2,2,2, 1.5,2, 1.5,1.5,  1, 1]

    from specsim.load_inputs import get_band_mag

    fig, ax = plt.subplots(1,figsize=(7,5),sharex=True)
    for i,ao_mode in enumerate(ao_modes):
        # interpolate over WFEs and sample HO and TT at correct mag
        strehl_ho = calc_strehl(data[ao_mode]['ho_wfe'],so.filt.center_wavelength)
        strehl_tt = tt_to_strehl(data[ao_mode]['tt_wfe'],so.filt.center_wavelength,so.inst.tel_diam)
        strehl = strehl_ho * strehl_tt
        ax.plot(data[ao_mode]['ho_mag'],strehl,label=(ao_mode),linestyle='--',lw=1,c=colors[i])

    ax.set_xlim(0,21)
    ax.set_ylim(0,1.1)
    ax.legend(loc='best',fontsize=10)#,bbox_to_anchor=(1.2,1))
    ax.grid(True)
    ax.set_ylabel('%s Strehl'%so.filt.band)
    ax.set_xlabel('R Mag' )

    plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.7)
    ax.set_title('NFIRAOS Performance'%int(so.stel.teff))

    plt.savefig('output/strehl_ao_modes.png')



def plot_coupling_modhis():
    """
    plot coupling for different strehl regimes
    """
    configfile = './modhis_snr.cfg'
    so    = load_object(configfile)

    ao_data = {}
    ao_data['lgs_on']  = [2.43, 181.] # avg, 45 deg
    ao_data['ngs']     = [1.36, 150.] # taken from 
    ao_data['lgs_off'] = [2.24, 200.]
    
    colors   = ['b','orange','limegreen','g','r','c','b','gray','black']

    coupling = []
    fig, ax = plt.subplots(1,figsize=(7,5),sharex=True)
    so.inst.pl_on=0
    for i,mode in enumerate(ao_data.keys()):
        so.ao.ttdynamic_set = ao_data[mode][0]
        so.ao.ho_wfe_set    = ao_data[mode][1]
        cload = fill_data(so) # put coupling files in load and wfe stuff too
        coupling.append(so.inst.coupling)

        ax.plot(so.inst.xtransmit,so.inst.coupling,linestyle='-',label=mode,lw=1.5,c=colors[i])
    
    so.inst.pl_on=1
    for i,mode in enumerate(ao_data.keys()):
        so.ao.ttdynamic_set = ao_data[mode][0]
        so.ao.ho_wfe_set    = ao_data[mode][1]
        cload = fill_data(so) # put coupling files in load and wfe stuff too
        coupling.append(so.inst.coupling)
        iyj = np.where(so.inst.xtransmit < 1500)[0]
        ax.plot(so.inst.xtransmit[iyj],so.inst.coupling[iyj],linestyle='--',lw=1,c=colors[i])


    ax.set_xlim(980,2450)
    ax.set_ylim(0,0.8)
    ax.legend(loc='best',fontsize=10)#,bbox_to_anchor=(1.2,1))
    ax.grid(True)
    ax.set_ylabel('Coupling')
    ax.set_xlabel('Wavelength [nm]')

    plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.7)
    ax.set_title('Coupling Performance')

    plt.savefig('output/coupling_ao_modes.png')






#
# old
##
#
#
def plot_wfe():
    """
    """
    configfile = 'hispec_tracking_camera.cfg'
    so         = load_object(configfile)
    cload      = fill_data(so)

    f = pd.read_csv(so.ao.ho_wfe_set,header=[0,1])
    ftt = pd.read_csv(so.ao.ttdynamic_set,header=[0,1])
    #modes    = [''LGS_100H_130','LGS_100J_130']
    modes    = ['80J','80H','100K','SH','LGS_100H_130','LGS_100J_130','LGS_100J_45','LGS_STRAP_130','LGS_STRAP_45']
    modes2   = [ 'NGS', 'NGS','NGS','NGS','','','','','']
    linestyles = ['-','-','-','-','--','--','--','-.','-.']
    colors   = ['b','orange','gray','g','r','c','b','gray','black']
    widths   = [1.5,  1,    1, 1.5,2, 1.5,1.5,  1, 1]
    rawmags  = f['mag'].values.T[0]
    wfes     = []
    tts      = []
    mags     = []
    strehl   = []
    for mode in modes:
        wfes.append(f[mode].values.T[0])
        tts.append(ftt[mode].values.T[0])
        band            = f[mode].columns[0] # this is the mag band wfe is defined in, must be more readable way..
        onemag          = get_band_mag(so,'Johnson',band,so.stel.factor_0) # get magnitude of star in appropriate band
        color           = so.stel.mag - onemag
        mags.append(rawmags + color)

    fig, ax = plt.subplots(2,figsize=(7,6),sharex=True)
    for i,mode in enumerate(modes):
        iplot = np.where(np.array(wfes[i]) < 500)[0]
        ax[0].plot(mags[i],tts[i],label=modes2[i] + ' ' + mode,linestyle=linestyles[i],lw=widths[i],c=colors[i])
        print(modes2[i] + '' + mode)
        ax[1].plot(mags[i][iplot],wfes[i][iplot],label=(modes2[i] + ' ' + mode),linestyle=linestyles[i],lw=widths[i],c=colors[i])


    ax[0].set_xlim(0,17.5)
    ax[0].set_ylim(0,20)
    ax[0].legend(loc='best',fontsize=10,bbox_to_anchor=(1,1))
    ax[0].grid(True)
    ax[0].set_ylabel('Tip Tilt Resid. (mas)')
    ax[1].set_xlabel('%s Mag' %so.filt.band)

    ax[1].set_ylim(0,500)
    ax[1].set_ylabel('HO WFE (nm)')
    ax[1].grid(True)
    plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.75)
    ax[0].set_title('%sK Star HAKA WFE Estimate'%int(so.stel.teff))

    plt.savefig('output/ao_modes/AO_modes_Teff_%s.png'%so.stel.teff)
    #plt.savefig('output/ao_modes/AO_modes_Teff_%s.pdf'%so.stel.teff)


def plot_strehl():
    """
    """
    configfile = 'hispec_tracking_camera.cfg'
    so         = load_object(configfile)
    cload      = fill_data(so)

    f = pd.read_csv(so.ao.ho_wfe_set,header=[0,1])
    ftt = pd.read_csv(so.ao.ttdynamic_set,header=[0,1])
    #modes    = [''LGS_100H_130','LGS_100J_130']
    modes    = ['80J','80H','100K','SH','LGS_100H_130','LGS_100J_130','LGS_100J_45','LGS_STRAP_130','LGS_STRAP_45']
    modes2   = [ 'NGS', 'NGS','NGS','NGS','','','','','']
    linestyles = ['-','-','-','-','--','--','--','-.','-.']
    colors   = ['b','orange','gray','g','r','c','b','gray','black']
    widths   = [1.5,  1,    1, 1.5,2, 1.5,1.5,  1, 1]
    rawmags  = f['mag'].values.T[0]
    wfes     = []
    tts      = []
    mags     = []
    strehl   = []
    for mode in modes:
        wfes.append(f[mode].values.T[0])
        tts.append(ftt[mode].values.T[0])
        band            = f[mode].columns[0] # this is the mag band wfe is defined in, must be more readable way..
        onemag          = get_band_mag(so,'Johnson',band,so.stel.factor_0) # get magnitude of star in appropriate band
        color           = so.stel.mag - onemag
        mags.append(rawmags + color)
        strehl_ho = calc_strehl(f[mode].values.T[0],so.filt.center_wavelength)
        strehl_tt = tt_to_strehl(ftt[mode].values.T[0],so.filt.center_wavelength,so.inst.tel_diam)
        strehl.append(strehl_tt*strehl_ho)

    fig, ax = plt.subplots(1,figsize=(7,5),sharex=True)
    for i,mode in enumerate(modes):
        iplot = np.where(np.array(wfes[i]) < 500)[0]
        ax.plot(mags[i][iplot],strehl[i][iplot],label=(modes2[i] + ' ' + mode),linestyle=linestyles[i],lw=widths[i],c=colors[i])

    ax.set_xlim(0,17.5)
    ax.set_ylim(0,1.1)
    ax.legend(loc='best',fontsize=10,bbox_to_anchor=(1.5,1))
    ax.grid(True)
    ax.set_ylabel('%s Strehl'%so.filt.band)
    ax.set_xlabel('%s Mag' %so.filt.band)

    plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15,right=0.7)
    ax.set_title('%sK Star HAKA Strehl Estimate'%int(so.stel.teff))

    plt.savefig('output/ao_modes/strehl_ao_modes_Teff_%s.png'%so.stel.teff)


def get_wfe_landscape(configfile = '../configs/hispec_tracking_camera.cfg'):
    """
    figure out which AO mode is best for which temperature and mag star
    """
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
                so.ao.ttdynamic_mag = get_band_mag(so,'Johnson',so.ao.ttdynamic_band,so.stel.factor_0) # get magnitude of star in appropriate band
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

    best2_ao_mode_arr      =  np.load('./output/ao_modes/best2_ao_mode_Hmag_[1,15,2]_temps_1000,1500,2300,3000,3600,4200,5800.npy')
    best2_ao_coupling      =  np.load('./output/ao_modes/best2_ao_coupling_%smag_[1,15,2]_temps_1000,1500,2300,3000,3600,4200,5800.npy'%band)
    ao_modes2   = ['SH','LGS_100H_130']

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
    ax.imshow(best2_ao_mode_arr,aspect='auto',origin='lower',\
            interpolation='None',cmap='nipy_spectral',\
            extent=extent)
    cs = ax.contour(best2_ao_coupling ,\
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


def get_AO_plot_scheme():
    """
    define colors and linestyles for AO modes
    """
    modes    = ['100JH','80J','80H','100K','SH','LGS_100H_130','LGS_100J_130','LGS_100J_45','LGS_STRAP_130','LGS_STRAP_45']
    modes2   = ['NGS', 'NGS', 'NGS','NGS','NGS','','','','','']
    linestyles = ['-','-','-','-','-','--','--','--','-.','-.']
    colors   = ['m','b','orange','gray','g','r','c','b','gray','black']
    widths   = [1, 1.5,  1,    1, 1.5,2, 1.5,1.5,  1, 1]

    return modes, modes2, linestyles, colors, widths






# ##################################
# OLD PAST HERE
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


def plot_wfe_old():
    """
    """
    modes = np.array(['K','SH','80J','80H','80JK','LGS'])# corresponding modes to match assumptions of text files 
    
    f_tt = np.loadtxt('./data/WFE/HAKA/Kstar_tiptilt.txt').T
    vmags = f_tt[0]
    f_wfe = np.loadtxt('./data/WFE/HAKA/Kstar_HOwfe.txt').T
    
    fig, ax = plt.subplots(2,figsize=(6,6),sharex=True)
    for mode in modes:
        imode = np.where(modes==mode)[0][0]

        wfes      = f_wfe[1:][imode]
        tip_tilts = f_tt[1:][imode] # switch to take best mode for observing case

        ax[0].plot(vmags,tip_tilts,label=mode)
        ax[1].plot(vmags,wfes,label=mode)

    ax[0].set_xlim(3,20)
    ax[0].set_ylim(0,20)
    ax[0].legend(loc=2,fontsize=10)
    ax[0].grid(True)
    ax[0].set_ylabel('Tip Tilt Resid. (mas)')
    ax[1].set_xlabel('V Mag')

    ax[1].set_ylim(100,500)
    ax[1].set_ylabel('HO WFE (nm)')
    ax[1].grid(True)
    plt.subplots_adjust(bottom=0.15,hspace=0.1,left=0.15)
    ax[0].set_title('K Star HAKA WFE Estimate')



def get_tip_tilt_resid_old(Vmag, mode):
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

def get_HO_WFE_old(Vmag, mode):
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
