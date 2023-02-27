##############################################################
# General functions for calc_snr_max
###############################################################

import numpy as np
from scipy import signal, interpolate
import pandas as pd

from scipy.integrate import trapz
from scipy import signal
import matplotlib.pylab as plt

import load_inputs #load_filter,load_phoenix, load_sonora
from functions import integrate, degrade_spec
from astropy.io import fits

all = {}


def get_band_mag(so,family,band,factor_0):
    """
    factor_0: scaling model to photons
    """
    x,y          = load_inputs.load_filter(so.filt.filter_path,family,band)
    filt_interp  = interpolate.interp1d(x, y, bounds_error=False,fill_value=0)
    dl_l         = np.mean(integrate(x,y)/x) # dlambda/lambda to account for spectral fraction
    
    # load stellar the multiply by scaling factor, factor_0, and filter. integrate
    #if so.stel.model=='phoenix':
    #    vraw,sraw = load_inputs.load_phoenix(so.stel.stel_file,wav_start=np.min(x), wav_end=np.max(x)) #phot/m2/s/nm
    #elif so.stel.model=='sonora':
    #    vraw,sraw = load_inputs.load_sonora(so.stel.stel_file,wav_start=np.min(x), wav_end=np.max(x)) #phot/m2/s/nm
    
    filtered_stel = factor_0 * so.stel.sraw * filt_interp(so.stel.vraw)
    flux = integrate(so.stel.vraw,filtered_stel)    #phot/m2/s

    phot_per_s_m2_per_Jy = 1.51*10**7 # convert to phot/s/m2 from Jansky
    
    flux_Jy = flux/phot_per_s_m2_per_Jy/dl_l
    
    # get zps
    zps                     = np.loadtxt(so.filt.zp_file,dtype=str).T
    izp                     = np.where((zps[0]==family) & (zps[1]==band))[0]
    zp                      = np.float(zps[2][izp])

    mag = -2.5*np.log10(flux_Jy/zp)

    return mag

def pick_coupling(waves,dynwfe,ttStatic,ttDynamic,LO=30,PLon=0,piaa_boost=1.3,points=None,values=None,transmission_path='/Users/ashbake/Documents/Research/Projects/HISPEC/SNR_calcs/data/throughput/hispec_subsystems_11032022/'):
    """
    select correct coupling file
    to do:implement interpolation of coupling files instead of rounding variables
    """
    PLon = int(PLon)

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
        if np.any(points==None): 
            out = grid_interp_coupling(PLon,path=transmission_path) # move this outside this function ,do one time!
            points = out[0]
            values_1,values_2,values_3 = out[1:]
        else:
            values_1,values_2,values_3 = values
        point = (LO,ttStatic,ttDynamic,waves)
        mode1 = interpolate.interpn(points, values_1, point,bounds_error=False,fill_value=0) # see example https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html#scipy.interpolate.interpn
        mode2 = interpolate.interpn(points, values_2, point,bounds_error=False,fill_value=0) 
        mode3 = interpolate.interpn(points, values_3, point,bounds_error=False,fill_value=0) 

        #PLwav,PLdat = load_photonic_lantern() #transfer matrices input mode--> each SMF
        #mat = PLdat[10] # use middle one for now
        #test1 = mode1 * mat[0,0]  + mode2*mat[1,0] + mode3*mat[2,0]
        #test2 = mode1 * mat[0,1]  + mode2*mat[1,1] + mode3*mat[2,1]
        #test3 = mode1 * mat[2,2]  + mode2*mat[1,2] + mode3*mat[2,2]
        # apply only to YJ or make matrix diagonal for HK..map onto same wavelength grid somehow
        # get coupling
        losses = 0.95
        raw_coupling = losses*(mode1+mode2+mode3) # do dumb things for now #0.95 is a recombination loss term 
    else:
        if np.any(points==None): points, values = grid_interp_coupling(PLon,path=transmission_path) # move this outside this function ,do one time!
        values_1= values
        points, values_1 = grid_interp_coupling(PLon)
        point = (LO,ttStatic,ttDynamic,waves)
        raw_coupling = interpolate.interpn(points, values_1, point,bounds_error=False,fill_value=0)

    if np.max(waves) < 10:
        waves*=1000 # nm to match dynwfe

    ho_strehl =  np.exp(-(2*np.pi*dynwfe/waves)**2) # computed per wavelength as grid
    coupling  = raw_coupling * piaa_boost * ho_strehl
    
    return coupling, ho_strehl

def grid_interp_coupling(PLon,path='/Users/ashbake/Documents/Research/Projects/HISPEC/SNR_calcs/data/throughput/hispec_subsystems_11032022/'):
    """
    interpolate coupling files over their various parameters
    """
    LOs = np.arange(0,125,25)
    ttStatics = np.arange(11)
    ttDynamics = np.arange(0,10,0.5)
    
    if PLon: 
        path_to_files     = path + 'coupling/couplingEff_wPL_202212014/'
        filename_skeleton = 'couplingEff_atm0_adc0_PL%s_defoc0nmRMS_LO%snmRMS_ttStatic%smas_ttDynamic%smasRMS.csv'
    else:
        path_to_files     = path + 'coupling/couplingEff_20221005/'
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

def plot_throughput(so):
    """
    """
    plt.figure(figsize=(7,4))
    plt.plot(so.stel.v,so.inst.coupling,label='Coupling Only')
    plt.plot(so.stel.v,so.inst.base_throughput,label='All But Coupling')    
    plt.plot(so.stel.v,so.inst.ytransmit,'k',label='Total Throughput')  
    plt.ylabel('Transmission')
    plt.xlabel('Wavelength (nm)')
    plt.title('%s=%s, Teff=%s, AO mode: %s'%(so.filt.band,int(so.stel.mag),int(so.stel.teff),so.ao.mode))
    plt.subplots_adjust(bottom=0.15)
    plt.axhline(y=0.05,color='m',ls='--',label='5%')
    plt.legend()
    plt.grid()
    figname = 'throughput_%s_%smag_%s_Teff_%s_texp_%ss.png' %(so.ao.mode,so.filt.band,so.stel.mag,so.stel.teff,int(so.obs.texp_frame*nframes))
    plt.savefig('./output/snrplots/' + figname)

def plot_throughput_components(telluric_file,datapath = './data/throughput/hispec_subsystems_11032022/',outputdir='../output/'):
    """
    plot throughput plot for MRI proposal
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
                w,s = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                data[spec][i] = s
            else:
                wtemp, stemp = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                # interpolate onto s
                f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
                data[spec][i] = f(w)
                #plt.plot(w,s,label=i)


    #load atmosphere and degrade to lower res, resample onto w
    teldata      = fits.getdata(telluric_file)
    _,ind  = np.unique(teldata['Wave/freq'],return_index=True)
    tck_tel   = interpolate.splrep(teldata['Wave/freq'][ind],teldata['Total'][ind], k=2, s=0)
    telluric = interpolate.splev(1000*w,tck_tel,der=0,ext=1)
    telluric_spec  = degrade_spec(w,telluric,2000)
    data['atm'] = telluric_spec
 
    #load coupling for two options
    # inputs : waves,dynwfe,ttStatic,ttDynamic
    data['coupling_NGS'],strehl = pick_coupling(w,130,0,3,LO=0,PLon=1)
    data['coupling_LGS'],strehl2 = pick_coupling(w,220,0,9.4,LO=30,PLon=1)

    if np.max(w)>1000: w/=1000
    # plot red only
    spec = 'red'
    plt.figure(figsize=(7,6))
    plt.semilogy(w,data['atm'],c='royalblue',linewidth=1)
    plt.plot(w,data['atm'] * data[spec]['tel'],c='darkorange',linewidth=1)
    plt.plot(w,data['atm'] * data[spec]['tel'] * data[spec]['ao'],c='silver',linewidth=1)
    plt.plot(w,data['atm'] * data[spec]['tel'] * data[spec]['ao']* \
                    data[spec]['feicom']*data[spec]['feired'],c='gold',linewidth=1)
    
    plt.plot(w,data['atm'] * data[spec]['tel'] * data[spec]['ao']* \
                    data[spec]['feicom']*data[spec]['feired']*data[spec]['fibred']*\
                    data['coupling_NGS'],c='steelblue',alpha=0.8,linewidth=1)
    
    ngs = data['atm'] * data[spec]['tel'] * data[spec]['ao']* \
                    data[spec]['feicom']*data[spec]['feired']*data[spec]['fibred']*\
                    data[spec]['rspec']*data['coupling_NGS']
    lgs = data['atm'] * data[spec]['tel'] * data[spec]['ao']* \
                    data[spec]['feicom']*data[spec]['feired']*data[spec]['fibred']*\
                    data[spec]['rspec']*data['coupling_LGS']
    np.savetxt('output/ngs_throughput.txt',np.vstack((w,ngs)).T)
    np.savetxt('output/lgs_throughput.txt',np.vstack((w,lgs)).T)

    ngs[np.where(ngs<0.015)[0]] = np.nan
    lgs[np.where((ngs<0.015) & (w > 1.9))[0]] = np.nan
    lgs[np.where(lgs<0.01)[0]] = np.nan
    lgs[np.where((lgs<0.015) & (w > 1.9))[0]] = np.nan
    plt.plot(w,ngs,c='seagreen',linewidth=1)
    
    plt.plot(w,lgs,c='seagreen',alpha=0.5,linewidth=1)
    
    plt.xlabel('Wavelength (microns)',color='k')
    plt.ylabel('Cumulative Throughput (log)',color='k')
    plt.ylim(0.01,1)
    plt.xlim(1.490, 2.455)
    plt.axhline(np.max(ngs),c='k',linestyle='--',linewidth=2)
    plt.fill_between([1.810, 1.960],0.01,y2=1,facecolor='w',zorder=110)
    
    #plt.title("HISPEC E2E Except Coupling")
    # y lines
    yticks = [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.8]
    yticks = [0.01, 0.03, 0.09, 0.27, 0.81]
    xticks = np.round((np.arange(1.49, 2.45,0.04)),2)
    plt.yticks(ticks=yticks,labels=yticks,color='k',fontsize=12)
    plt.xticks(rotation=90,ticks=xticks,labels=xticks,color='k',fontsize=12)
    plt.grid(axis='y',alpha=0.4)
    plt.subplots_adjust(bottom=0.17)
    plt.savefig(outputdir + 'e2e_mri_plot.png')
    plt.savefig(outputdir + 'e2e_mri_plot.pdf')

def get_emissivity(wave,datapath = './data/throughput/hispec_subsystems_11032022/'):
    """
    get throughput except leave out couplingalso store emissivity
    """
    x = wave.copy()
    if np.min(x) > 10:
        x/=1000 #convert nm to um

    red_include = ['tel', 'ao', 'feicom', 'feired','fibred','rspec']#,'coupling']
    blue_include = ['tel', 'ao', 'feicom', 'feiblue','fibblue','bspec']
    temps = [276,276,276,276,276,77]

    em_red, em_blue = [],[]
    for i in red_include:
        wtemp, stemp = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
        f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
        em_red.append(1-f(x)) # 1 - interp throughput onto x

    for i in blue_include:
        wtemp, stemp = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
        f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
        em_blue.append(1-f(x)) # 1 - interp throughput onto x

    return em_red,em_blue,temps

def get_emissivities(wave,surfaces=['tel'],datapath = './data/throughput/hispec_subsystems_11032022/'):
    """
    get throughput except leave out couplingalso store emissivity
    """
    x = wave.copy()
    if np.min(x) > 10:
        x/=1000 #convert nm to um

    em= []
    for i in surfaces:
        wtemp, stemp = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
        f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
        em.append(1-f(x)) # 1 - interp throughput onto x

    return em

def get_base_throughput(x,ploton=False,datapath = './data/throughput/hispec_subsystems_11032022/'):
    """
    get throughput except leave out coupling

    also store emissivity
    """
    #plt.figure()
    for spec in ['red','blue']:
        if spec=='red':
            include = ['tel', 'ao', 'feicom', 'feired','fibred','rspec']#,'coupling']
        if spec=='blue':
            include = ['tel', 'ao', 'feicom', 'feiblue','fibblue','bspec']#,'coupling']

        for i in include:
            if i==include[0]:
                w,s = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                #plt.plot(w,s,label=i)
            else:
                wtemp, stemp = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                # interpolate onto s
                f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
                s*=f(w)
                #plt.plot(w,s,label=i)

        if spec=='red':
            isub = np.where(w > 1.4) 
            wred = w[isub]
            specred = s[isub]
        if spec=='blue':
            isub = np.where(w<1.4)
            specblue = s[isub]
            wblue = w[isub]
    
    w = np.concatenate([wblue,wred])
    s = np.concatenate([specblue,specred])

    # reinterpolate 
    if np.min(x) > 10:
        x/=1000 #convert nm to um

    tck    = interpolate.splrep(w,s, k=2, s=0)
    snew   = interpolate.splev(x,tck,der=0,ext=1)

    if ploton:
        plt.plot(wblue,specblue,label='blue')
        plt.plot(wred,specred,label='red')
        plt.grid(True)
        plt.xlabel('Wavelength (um)')
        plt.ylabel('Transmission')
        plt.title("HISPEC E2E Except Coupling")
        plt.savefig('e2e.png')

    return snew

def load_photonic_lantern():
    """
    load PL info like unitary matrices
    """
    wavearr = np.linspace(970,1350,20)
    data = np.load('./data/throughput/photonic_lantern/unitary_matrices.npy')
    
    return wavearr,data


