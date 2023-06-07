##############################################################
# General functions for calc_snr_max
###############################################################

import numpy as np
from scipy import interpolate

from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu
from astropy import units as u
from astropy import constants as c 

from throughput_tools import get_emissivity, get_emissivities

all = {'get_sky_bg','get_inst_bg','sum_total_noise','plot_noise_components'}

def get_sky_bg(x,airmass=1.5,pwv=1.5,npix=3,lam0=2000,R=100000,diam=10,area=76,skypath = '../../../../_DATA/sky/'):
    """
    generate sky background per reduced pixel, default to HIPSEC. Source: DMawet jup. notebook

    inputs:
    -------

    outputs:
    --------
    sky background (ph/s)
    """
    diam *= u.m
    area = area * u.m * u.m
    wave=x*u.nm

    fwhm = ((wave  / diam) * u.radian).to(u.arcsec)
    solidangle = fwhm**2 * 1.13 #corrected for Gaussian beam (factor 1.13)

    sky_background_MK_tmp  = np.genfromtxt(skypath+'mk_skybg_zm_'+str(pwv)+'_'+str(airmass)+'_ph.dat', skip_header=0)
    sky_background_MK      = sky_background_MK_tmp[:,1]
    sky_background_MK_wave = sky_background_MK_tmp[:,0] #* u.nm

    pix_width_nm  = (wave/R/npix) #* u.nm 
    sky_background_interp=np.interp(wave.value, sky_background_MK_wave, sky_background_MK) * u.photon/(u.s*u.arcsec**2*u.nm*u.m**2) * area * solidangle * pix_width_nm 

    return sky_background_interp.value # ph/s

def get_inst_bg(x,npix=3,R=100000,diam=10,area=76,datapath='./data/throughput/hispec_subsystems_11032022/'):
    """
    generate sky background per reduced pixel, default to HIPSEC. Source: DMawet jup. notebook

    inputs:
    -------

    outputs:
    --------
    sky background (photons/s) already considering PSF sampling

    """
    em_red,em_blue, temps = get_emissivity(x,datapath=datapath)

    # telescope
    diam *= u.m
    area *= u.m * u.m
    wave = x*u.nm

    fwhm = ((wave  / diam) * u.radian).to(u.arcsec)
    solidangle = fwhm**2 * 1.13 #corrected for Gaussian beam (factor 1.13)
    pix_width_nm  = (wave/R/npix) #* u.nm 

    # step through temperatures and emissivities for red and blue
    # red
    for i,temp in enumerate(temps):
        bbtemp = blackbody_lambda(wave, temp).to(u.erg/(u.micron * u.s * u.cm**2 * u.arcsec**2)) * area.to(u.cm**2) * solidangle
        if i==0:
            tel_thermal_red  = em_red[i] * bbtemp.to(u.photon/u.s/u.micron, equivalencies=u.spectral_density(wave)) * pix_width_nm
            tel_thermal_blue = em_blue[i] * bbtemp.to(u.photon/u.s/u.micron, equivalencies=u.spectral_density(wave)) * pix_width_nm
        else:
            therm_red_temp   = em_red[i] * bbtemp.to(u.photon/u.s/u.micron, equivalencies=u.spectral_density(wave)) * pix_width_nm
            therm_blue_temp  = em_blue[i] * bbtemp.to(u.photon/u.s/u.micron, equivalencies=u.spectral_density(wave)) * pix_width_nm
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

    return em_total # units of ph/s/pix

def get_sky_bg_tracking(x,fwhm,airmass=1.5,pwv=1.5,area=76,skypath = '../../../../_DATA/sky/'):
    """
    generate sky background per pixel, default to HIPSEC. Source: DMawet jup. notebook

    inputs:
    -------
    fwhm: arcsec

    outputs:
    --------
    sky background (ph/s)
    """
    area = area * u.m * u.m
    wave=x*u.nm

    fwhm *= u.arcsec
    solidangle = fwhm**2 * 1.13 #corrected for Gaussian beam (factor 1.13)

    sky_background_MK_tmp  = np.genfromtxt(skypath+'mk_skybg_zm_'+str(pwv)+'_'+str(airmass)+'_ph.dat', skip_header=0)
    sky_background_MK      = sky_background_MK_tmp[:,1] * u.photon/(u.s*u.arcsec**2*u.nm*u.m**2) 
    sky_background_MK_wave = sky_background_MK_tmp[:,0] * u.nm

    sky_background_interp=np.interp(wave, sky_background_MK_wave, sky_background_MK)
    sky_background_interp*= area * solidangle 
    
    return sky_background_interp.value # ph/s/nm

def get_inst_bg_tracking(x,fwhm,area=76,datapath='./data/throughput/hispec_subsystems_11032022/'):
    """
    generate sky background per pixel, default to HIPSEC. Source: DMawet jup. notebook
    change this to take emissivities and temps as inputs so dont
    have to rely on get_emissivities

    inputs:
    -------

    outputs:
    --------
    sky background (photons/s) already considering PSF sampling

    """
    temps = [276,276,276]
    em = get_emissivities(x,surfaces=['tel','ao','feicom'],datapath=datapath)

    # telescope
    area *= u.m * u.m
    wave = x*u.nm

    fwhm *= u.arcsec
    solidangle = fwhm**2 * 1.13 #corrected for Gaussian beam (factor 1.13)
    
    # step through temperatures and emissivities for red and blue
    # red
    for i,temp in enumerate(temps):
        bbtemp = solidangle * blackbody_lambda(wave, temp).to(u.erg/(u.nm * u.s * u.cm**2 * u.arcsec**2)) * area.to(u.cm**2) 
        if i==0:
            tel_thermal = em[i] * bbtemp.to(u.photon/u.s/u.nm, equivalencies=u.spectral_density(wave)) 
        else:
            therm_temp   = em[i] * bbtemp.to(u.photon/u.s/u.nm, equivalencies=u.spectral_density(wave)) 
            tel_thermal += therm_temp

    return tel_thermal.value # units of ph/nm/s

def sum_total_noise(flux,texp, nsamp, inst_bg, sky_bg,darknoise,readnoise,npix,noisecap=None):
    """
    noise in 1 exposure

    inputs:
    --------
    flux - array [e-] 
        spectrum of star in units of electrons
    texp - float [seconds]
        exposure time, (0s,900s] (for one frame)
    nsamp - int
        number of samples in a ramp which will reduce read noise [1,inf] - 16 max for kpic
    inst_bg - array or float [e-/s]
        instrument background, if array should match sampling of flux
    sky_bg - array or float [e-/s]
        sky background, if array should match sampling of flux
    darknoise - float [e-/s/pix]
        dark noise of detector
    readnoise - float [e-/s]
        read noise of detector
    npix - float [pixels]
        number of pixels in cross dispersion of spectrum being combined into one 1D spectrum
    noisecap - float or None (default: None)
        noise cap to be applied. Defined relative to flux such that 1/noisecap is the max SNR allowed
    
    outputs:
    -------
    noise: array [e-]
        total noise sampled on flux grid
    """
    # shot noise - array w/ wavelength or integrated over band
    sig_flux = np.sqrt(flux)

    # background (instrument and sky) - array w/ wavelength matching flux array sampling or integrated over band
    total_bg = (inst_bg + sky_bg) # per reduced pixel already so dont need to include vertical pixel extent
    sig_bg   = np.sqrt(inst_bg + sky_bg) 

    # read noise  - reduces by number of ramps, limit to 6 at best
    sig_read = np.max((3,(readnoise/np.sqrt(nsamp))))
    
    # dark current - times time and pixels
    sig_dark = np.sqrt(darknoise * npix * texp) #* get dark noise every sample
    
    noise = np.sqrt(sig_flux **2 + sig_bg**2 + npix * sig_read**2 + sig_dark**2)

    # cap the noise if a number is provided
    if noisecap is not None:
        noise[np.where(noise < noisecap)] = noisecap * flux # noisecap is fraction of flux, 1/noisecap gives max SNR

    return noise

def read_noise(rn,npix):
    """
    input:
    ------
    rn: [e-/pix]
        read noise
    npix [pix]
        number of pixels
    """
    return np.sqrt(npix * rn**2)

def dark_noise(darknoise,npix,texp):
    """
    input:
    ------
    darknoise: [e-/pix/s]
        read noise
    npix [pix]
        number of pixels
    texp [s]
        exposure time in seconds
    """
    sig_dark = np.sqrt(darknoise * npix * texp)
    return sig_dark

def plot_noise_components(so):
    """
    plot spectra and transmission so know what we're dealing with
    """
    plt.figure()
    plt.plot(so.stel.v,so.hispec.ytransmit)

def plot_bg(so, v,instbg,skybg):
    fig, ax = plt.subplots(1,1, figsize=(8,5))  
    ax.plot(v,instbg+skybg)
    ax.set_xlim(900,2500)
    ax.set_ylim(0,0.5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Sky + Inst Bg (e-/s/pix)')
    ax2 = ax.twinx()
    #ax2.fill_between(so.filt.v,so.filt.s,facecolor='gray',edgecolor='black',alpha=0.2)
    #ax2.set_ylabel('Filter Response')
    # plot band
    ax2.fill_between(so.inst.y,0,1,facecolor='k',edgecolor='black',alpha=0.2)
    ax2.text(20+np.min(so.inst.y),0.9, 'y')
    ax2.fill_between(so.inst.J,0,1,facecolor='k',edgecolor='black',alpha=0.2)
    ax2.text(50+np.min(so.inst.J),0.9, 'J')
    ax2.fill_between(so.inst.H,0,1,facecolor='k',edgecolor='black',alpha=0.2)
    ax2.text(50+np.min(so.inst.H),0.9, 'H')
    ax2.fill_between(so.inst.K,0,1,facecolor='k',edgecolor='black',alpha=0.2)
    ax2.text(50+np.min(so.inst.K),0.9, 'K')
    ax2.set_ylim(0,1)

    fig, ax = plt.subplots(1,1, figsize=(8,5))  
    ax.plot(v,instbg)
    ax.set_xlim(900,2500)
    ax.set_ylim(0,0.5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Inst Bg (e-/s/pix)')
    ax2 = ax.twinx()
    #ax2.fill_between(so.filt.v,so.filt.s,facecolor='gray',edgecolor='black',alpha=0.2)
    #ax2.set_ylabel('Filter Response')
    # plot band
    ax2.fill_between(so.inst.y,0,1,facecolor='k',edgecolor='black',alpha=0.2)
    ax2.text(20+np.min(so.inst.y),0.9, 'y')
    ax2.fill_between(so.inst.J,0,1,facecolor='k',edgecolor='black',alpha=0.2)
    ax2.text(50+np.min(so.inst.J),0.9, 'J')
    ax2.fill_between(so.inst.H,0,1,facecolor='k',edgecolor='black',alpha=0.2)
    ax2.text(50+np.min(so.inst.H),0.9, 'H')
    ax2.fill_between(so.inst.K,0,1,facecolor='k',edgecolor='black',alpha=0.2)
    ax2.text(50+np.min(so.inst.K),0.9, 'K')
    ax2.set_ylim(0,1)


    fig, ax = plt.subplots(1,1, figsize=(8,5))  
    ax.plot(v,skybg)
    ax.set_xlim(900,2500)
    ax.set_ylim(0,0.5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Sky Bg (e-/s/pix)')
    ax2 = ax.twinx()
    #ax2.fill_between(so.filt.v,so.filt.s,facecolor='gray',edgecolor='black',alpha=0.2)
    #ax2.set_ylabel('Filter Response')
    # plot band
    ax2.fill_between(so.inst.y,0,1,facecolor='k',edgecolor='black',alpha=0.2)
    ax2.text(20+np.min(so.inst.y),0.9, 'y')
    ax2.fill_between(so.inst.J,0,1,facecolor='k',edgecolor='black',alpha=0.2)
    ax2.text(50+np.min(so.inst.J),0.9, 'J')
    ax2.fill_between(so.inst.H,0,1,facecolor='k',edgecolor='black',alpha=0.2)
    ax2.text(50+np.min(so.inst.H),0.9, 'H')
    ax2.fill_between(so.inst.K,0,1,facecolor='k',edgecolor='black',alpha=0.2)
    ax2.text(50+np.min(so.inst.K),0.9, 'K')
    ax2.set_ylim(0,1)

