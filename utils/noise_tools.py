##############################################################
# General functions for calc_snr_max
###############################################################

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy import signal

from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu
from astropy import units as u
from astropy import constants as c 

from throughput_tools import get_emissivity

all = {}

def get_sky_bg(x,airmass=1.5,pwv=1.5,npix=3,lam0=2000,R=100000,diam=10,area=76,skypath = '../../../../_DATA/sky/'):
    """
    generate sky background per pixel, default to HIPSEC. Source: DMawet jup. notebook

    inputs:
    -------

    outputs:
    --------
    sky background (photons)
    """
    diam *= u.m
    area = area * u.m * u.m
    x*=u.nm

    fwhm = ((x  / diam) * u.radian).to(u.arcsec)
    solidangle = fwhm**2 * 1.13 #corrected for Gaussian beam (factor 1.13)


    sky_background_MK_tmp = np.genfromtxt(path+'mk_skybg_zm_'+str(pwv)+'_'+str(airmass)+'_ph.dat', skip_header=0)
    sky_background_MK = sky_background_MK_tmp[:,1]
    sky_background_MK_wave = sky_background_MK_tmp[:,0] #* u.nm

    pix_width_nm  = (lam0/R/npix) * u.nm 
    sky_background_interp=np.interp(x.value, sky_background_MK_wave, sky_background_MK) * u.photon/(u.s*u.arcsec**2*u.nm*u.m**2) * area * solidangle * pix_width_nm 


    return sky_background_interp



def get_inst_bg(x,npix=3,lam0=2000,R=100000,diam=10,area=76):
    """
    generate sky background per pixel, default to HIPSEC. Source: DMawet jup. notebook

    inputs:
    -------

    outputs:
    --------
    sky background (photons)
    """
    em_red,em_blue, temps = get_emissivity(x)

    # telescope
    diam *= u.m
    area *= u.m * u.m
    lam0*=u.nm
    wave = x*u.nm

    fwhm = ((wave  / diam) * u.radian).to(u.arcsec)
    solidangle = fwhm**2 * 1.13 #corrected for Gaussian beam (factor 1.13)
    
    # step through temperatures and emissivities for red and blue
    # red
    for i,temp in enumerate(temps):
        bbtemp = blackbody_lambda(wave, temp).to(u.erg/(u.micron * u.s * u.cm**2 * u.arcsec**2)) * area.to(u.cm**2) * solidangle
        if i==0:
            tel_thermal_red  = em_red[i] * bbtemp.to(u.photon/u.s/u.micron, equivalencies=u.spectral_density(wave)) * lam0/R/npix 
            tel_thermal_blue = em_blue[i] * bbtemp.to(u.photon/u.s/u.micron, equivalencies=u.spectral_density(wave)) * lam0/R/npix 
        else:
            therm_red_temp   = em_red[i] * bbtemp.to(u.photon/u.s/u.micron, equivalencies=u.spectral_density(wave)) * lam0/R/npix 
            therm_blue_temp  = em_blue[i] * bbtemp.to(u.photon/u.s/u.micron, equivalencies=u.spectral_density(wave)) * lam0/R/npix
            tel_thermal_red+= therm_red_temp
            tel_thermal_blue+= therm_blue_temp

    # interpolate and combine into one thermal spectrum
    isubred = np.where(wave > 1.4*u.um)[0]
    em_red_tot  = tel_thermal_red[isubred].decompose()
    isubblue = np.where(wave <1.4*u.um)[0]
    em_blue_tot  = tel_thermal_blue[isubblue].decompose()

    w = np.concatenate([x[isubblue],x[isubred]])
    s = np.concatenate([em_blue_tot,em_red_tot])

    tck        = interpolate.splrep(w,s.value, k=2, s=0)
    em_total   = interpolate.splev(x,tck,der=0,ext=1)

    return em_total

def combine_noise_sources():
    """
    """
    pass



def plot_noise_components():
    """
    plot spectra and transmission so know what we're dealing with
    """
    plt.figure()
    plt.plot(so.stel.v,so.hispec.ytransmit)


