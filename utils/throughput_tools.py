##############################################################
# General functions for calc_snr_max
###############################################################

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy import signal

all = {}


def get_band_mag(so,family,band,factor_0):
    """
    factor_0: scaling model to photons
    """
    x,y          = load_filter(so,family,band)
    filt_interp  =  interpolate.interp1d(x, y, bounds_error=False,fill_value=0)
    dl_l         =   np.mean(integrate(x,y)/x) # dlambda/lambda to account for spectral fraction
    
    # load stellar the multiply by scaling factor, factor_0, and filter. integrate
    if so.stel.model=='phoenix':
        vraw,sraw = load_phoenix(so.stel.stel_file,wav_start=np.min(x), wav_end=np.max(x)) #phot/m2/s/nm
    elif so.stel.model=='sonora':
        vraw,sraw = load_sonora(so.stel.stel_file,wav_start=np.min(x), wav_end=np.max(x)) #phot/m2/s/nm
    
    filtered_stel = factor_0 * sraw * filt_interp(vraw)
    flux = integrate(vraw,filtered_stel)    #phot/m2/s

    phot_per_s_m2_per_Jy = 1.51*10**7 # convert to phot/s/m2 from Jansky
    
    flux_Jy = flux/phot_per_s_m2_per_Jy/dl_l
    
    # get zps
    zps                     = np.loadtxt(so.filt.zp_file,dtype=str).T
    izp                     = np.where((zps[0]==family) & (zps[1]==band))[0]
    zp                      = np.float(zps[2][izp])

    mag = -2.5*np.log10(flux_Jy/zp)

    return mag

def pick_coupling(waves,dynwfe,ttStatic,ttDynamic,LO=30,PLon=0,piaa_boost=1.3):
    """
    select correct coupling file
    to do:implement interpolation of coupling files instead of rounding variables
    """
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
        points, values_1,values_2,values_3 = grid_interp_coupling(PLon) # move this outside this function ,do one time!
        point = (LO,ttStatic,ttDynamic,waves)
        mode1 = interpn(points, values_1, point,bounds_error=False,fill_value=0) # see example https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html#scipy.interpolate.interpn
        mode2 = interpn(points, values_2, point,bounds_error=False,fill_value=0) 
        mode3 = interpn(points, values_3, point,bounds_error=False,fill_value=0) 

        PLwav,PLdat = load_photonic_lantern() #transfer matrices input mode--> each SMF
        mat = PLdat[10] # use middle one for now
        test1 = mode1 * mat[0,0]  + mode2*mat[1,0] + mode3*mat[2,0]
        test2 = mode1 * mat[0,1]  + mode2*mat[1,1] + mode3*mat[2,1]
        test3 = mode1 * mat[2,2]  + mode2*mat[1,2] + mode3*mat[2,2]
        # apply only to YJ or make matrix diagonal for HK..map onto same wavelength grid somehow
        # get coupling
        raw_coupling = mode1+mode2+mode3 # do dumb things for now
    else:
        points, values_1 = grid_interp_coupling(PLon)
        point = (LO,ttStatic,ttDynamic,waves)
        raw_coupling = interpn(points, values_1, point,bounds_error=False,fill_value=0)

    if np.max(waves) < 10:
        waves*=1000 # nm to match dynwfe

    ho_strehl =  np.exp(-(2*np.pi*dynwfe/waves)**2) # computed per wavelength as grid
    coupling  = raw_coupling * piaa_boost * ho_strehl
    
    return coupling,ho_strehl

def grid_interp_coupling(PLon):
    """
    interpolate coupling files over their various parameters
    """
    LOs = np.arange(0,125,25)
    ttStatics = np.arange(11)
    ttDynamics = np.arange(0,10,0.5)
    
    if PLon: 
        path_to_files     = './data/throughput/hispec_subsystems_11032022/coupling/couplingEff_wPL_202212014/'
        filename_skeleton = 'couplingEff_atm0_adc0_PL%s_defoc0nmRMS_LO%snmRMS_ttStatic%smas_ttDynamic%smasRMS.csv'
    else:
        path_to_files     = './data/throughput/hispec_subsystems_11032022/coupling/couplingEff_20221005/'
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

def plot_throughput(v,throughput):
    """
    """
    plt.figure('throughput')
    plt.plot(v,throughput)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission')
    plt.xlim(980,2450)
    plt.ylim(0,0.08)

def get_emissivity(wave):
    """
    get throughput except leave out couplingalso store emissivity
    """
    datapath = './data/throughput/hispec_subsystems_11032022/'
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

def get_base_throughput(x,ploton=False):
    """
    get throughput except leave out coupling

    also store emissivity
    """
    datapath = './data/throughput/hispec_subsystems_11032022/'
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


