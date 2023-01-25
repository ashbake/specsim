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

