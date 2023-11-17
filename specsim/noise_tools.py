##############################################################
# General functions for noise calculations
# contributors:
# Ashley Baker abaker@caltech.edu
# Huihao Zhang zhang.12043@osu.edu
# many functions based on psisim https://github.com/planetarysystemsimager/psisim/
###############################################################

import numpy as np
from scipy import interpolate

from astropy.modeling.models import BlackBody
from astropy import units as u
from astropy import constants as c 

from specsim.functions import tophat

from specsim.throughput_tools import get_emissivity, get_emissivities

all = {'get_sky_bg','get_inst_bg','sum_total_noise','plot_noise_components'}

def get_sky_bg(x,airmass=1.3,pwv=1.5,npix=3,R=100000,diam=10,area=76,skypath = '../../../../_DATA/sky/'):
    """
    generate sky background per reduced pixel, default is HIPSEC. 
    Source: DMawet jup. notebook

    inputs:
    -------
    x : array [nm]
        wavelength in nanometers

    airmass: float [1,inf)
        airmass of the observation. Defaults to 1.3

    pwv: float [mm] [0,inf)
        precipitable water vapor in millimeters during 
        the observation. Defaults to 1.5

    npix: integer 
        number of pixels, defaults to 3
    

    outputs:
    --------
    sky background (ph/s)
    """
    diam *= u.m
    area = area * u.m * u.m
    wave = x*u.nm

    pwv_rounded = np.round(pwv,1)
    airmass_rounded = np.round(airmass,1)
    
    fwhm = ((wave  / diam) * u.radian).to(u.arcsec)
    solidangle = fwhm**2 * 1.13 #corrected for Gaussian beam (factor 1.13)

    sky_background_MK_tmp  = np.genfromtxt(skypath+'mk_skybg_zm_'+str(pwv_rounded)+'_'+str(airmass_rounded)+'_ph.dat', skip_header=0)
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
    x : array [nm]
        wavelength in nanometers

    npix: integer
        number of pixels

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
        bbtemp_fxn  = BlackBody(temp * u.K, scale=1.0 * u.erg / (u.micron * u.s * u.cm**2 * u.arcsec**2)) 
        bbtemp      = bbtemp_fxn(wave) *  area.to(u.cm**2) * solidangle
        #bbtemp = blackbody_lambda(wave, temp).to(u.erg/(u.micron * u.s * u.cm**2 * u.arcsec**2)) * area.to(u.cm**2) * solidangle
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

    return em_total # units of ph/s/reduced_pix

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
    wave = x*u.nm

    fwhm *= u.arcsec
    solidangle = fwhm**2 * 1.13 #corrected for Gaussian beam (factor 1.13)

    sky_background_MK_tmp  = np.genfromtxt(skypath+'mk_skybg_zm_'+str(pwv)+'_'+str(airmass)+'_ph.dat', skip_header=0)
    sky_background_MK      = sky_background_MK_tmp[:,1] * u.photon/(u.s*u.arcsec**2*u.nm*u.m**2) 
    sky_background_MK_wave = sky_background_MK_tmp[:,0] * u.nm

    sky_background_interp=np.interp(wave, sky_background_MK_wave, sky_background_MK)
    sky_background_interp*= area * solidangle 
    
    return sky_background_interp.value # ph/s/nm

def get_inst_bg_tracking(x,pixel_size,npix,datapath='./data/throughput/hispec_subsystems_11032022/'):
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
    wave = x * u.nm
    window_temp = 277 * u.K # temperature of cryostat window
    pixel_size *= u.micron
    f_num = 8 # fnumber of cryostat window size to detector as of 8/15/23

    # Load blocking filter profile
    fx,fy = np.loadtxt(datapath + 'feicam/blocking_filter.TXT',skiprows=20).T
    f = interpolate.interp1d(fx[::-1]*u.nm,fy,bounds_error=False,fill_value=0)
    blocking_filter   = f(wave)/100
    
    # load window emissivity
    #fx,fy = np.loadtxt(datapath + 'feicam/Infrasil_Window.txt').T
    #f = interpolate.interp1d(fx[::-1]*u.nm,fy,bounds_error=False,fill_value=0)
    window_emissivity = 0.05 #1 - f(wave)/100
    
    # Create QE profile for H2RG matching cutoff
    QE = tophat(wave.value,600,2600,0.9) # sensitivity of h2rg

    area_times_omega = u.radian**2 * 1.13**2 * np.pi**2 * pixel_size**2 / 4 /f_num**2
    bbtemp_fxn  = BlackBody(window_temp, scale=1.0 * u.erg / (u.micron * u.s * u.cm**2 * u.arcsec**2)) 
    bb   = area_times_omega.to(u.cm**2 * u.arcsec**2) * bbtemp_fxn(wave)

    bb_spec_dens = bb.to(u.photon/u.s/u.nm, equivalencies=u.spectral_density(wave))
    
    # thermal spectrum over npix, then integrate
    thermal_spectrum = npix * QE * window_emissivity * blocking_filter * bb_spec_dens # units of ph/nm/s/pix
    thermal = np.trapz(thermal_spectrum,wave)

    return thermal_spectrum, thermal.value # units of ph/nm/s

def get_contrast(wave,pl_sep,tel_diam,seeing,strehl):
    """
    Gets the contrast for SMF positioned on planet of some
    distance from host star

    inputs
    ------
    wave         - [nm] A list of wavelengths [float length m]
    pl_sep       - [mas] separations at which to calculate the speckle noise in arcseconds
    tel_diam     - Telescope diameter [m]
    seeing       - seeing during observation [arcsec]
    strehl       - strehl of AO correction
    """

    p_law_kolmogorov = -11./3
    p_law_ao_coro_filter = -2 
    nactuators = 58             # number of actuators
    fiber_contrast_gain = 10.   # represents suppression thanks to fiber
    
    # apply units
    pl_sep   *= u.marcsec
    tel_diam *= u.m
    wvs = u.micron * wave.copy() /1000 # convert to microns
    seeing *= u.arcsec

    #compute r0
    r0 = 0.55e-6/(seeing.to(u.radian)) * u.m * u.radian #Easiest to ditch the seeing unit here. 

    #The AO control radius in units of lambda/D
    cutoff = nactuators/2

    contrast = np.zeros_like(wvs)

    #Dimitri to put in references to this math
    r0_sc = r0 * (wvs/(0.55*u.micron))**(6./5)
    w_halo = tel_diam / r0_sc

    ang_sep_resel_in = pl_sep.to(u.radian)*tel_diam /wvs.to(u.m) / u.radian #Convert separtiona from arcsec to units of lam/D. rid of radian unit

    contrast = np.pi*(1-strehl)*0.488/w_halo**2 * (1+11./6*(ang_sep_resel_in/w_halo)**2)**(-11/6.)

    contrast_at_cutoff = np.pi*(1-strehl)*0.488/w_halo**2 * (1+11./6*(cutoff/w_halo)**2)**(-11/6.)

    biggest_ang_sep = np.abs(ang_sep_resel_in - cutoff) == np.min(np.abs(ang_sep_resel_in - cutoff))

    contrast[ang_sep_resel_in < cutoff] = contrast_at_cutoff[ang_sep_resel_in < cutoff]*(ang_sep_resel_in[ang_sep_resel_in < cutoff]/cutoff)**p_law_ao_coro_filter

    #Apply the fiber contrast gain
    contrast /= fiber_contrast_gain

    #Make sure nothing is greater than 1. 
    contrast[contrast>1] = 1.

    return contrast


def get_speckle_noise_vfn(wave,ho_wfe,tt_dyn,pl_sep,mag,seeing,strehl,tel_diam,vortex_charge):
    """
    taken from https://github.com/planetarysystemsimager/psisim/blob/kpic/psisim/instruments/modhis.py#L441C1-L441C1
    planet is off axis, star gets reduction in throughput due to vortex

    inputs
    ------
    wave [nm]   - wavelength array
    ho_wfe [nm] - High order wave front error

    outupts
    -------
    contrast

    TODO
    ----
    need planet throughput to accompany it since off axis?
    """
    # apply units
    ho_wfe *= u.nm
    tt_dyn *= u.mas
    wvs = u.micron * wave.copy() /1000 # convert to microns
    tel_diam *= u.m
    host_diameter *=u.mas

    #-- Get Stellar leakage due to WFE
    #Pick the WFE coefficient based on the vortex charge. Coeff values emprically determined in simulation
    if vortex_charge == 1:
        wfe_coeff = 0.840       # Updated on 1/11/21 based on 6/17/19 pyWFS data
    elif vortex_charge == 2:
        wfe_coeff = 1.650       # Updated on 1/11/21 based on 6/17/19 pyWFS data

    #Approximate contrast from WFE
    contrast = (wfe_coeff * ho_wfe.to(u.micron) / wvs)**(2.) # * self.vortex_charge)

    #-- Get Stellar leakage due to Tip/Tilt Jitter
    # Convert jitter to lam/D
    ttlamD = tt_dyn.to(u.radian) / (wvs.to(u.m)/ tel_diam) / u.radian

    # Use leakage approx. from Ruane et. al 2019 
        # https://arxiv.org/pdf/1908.09780.pdf      Eq. 3
    ttnull = (ttlamD)**(2*vortex_charge)

    # Add to total contrast
    contrast += ttnull

    #-- Get Stellar leakage due to finite sized star (Geometric leakage)
      # Assumes user has already set host diameter with set_vfn_host_diameter()
      # Equation and coefficients are from Ruante et. al 2019
        # https://arxiv.org/pdf/1908.09780.pdf     fig 7c
    # Convert host_diameter to units of lambda/D
    host_diam_LoD = host_diameter.to(u.radian) / (wvs.to(u.m)/tel_diam) /u.radian

    # Define Coefficients for geometric leakage equation
    if vortex_charge == 1:
        geo_coeff = 3.5
    elif vortex_charge == 2:
        geo_coeff = 4.2
    
    # Compute leakage
    geonull = (host_diam_LoD / geo_coeff)**(2*vortex_charge)
    
    # Add to total contrast
    contrast += geonull
        
    #convert to ndarray for consistency with contrast returned by other modes
    contrast = np.array(contrast)
    
    #Make sure nothing is greater than 1. 
    contrast[contrast>1] = 1.
    
    return contrast



def sum_total_noise(flux,texp, nsamp, inst_bg, sky_bg, darknoise,readnoise,npix,speckle,noisefloor=None):
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
    speckle - array [e-]
        counts from speckle leakage from star. should be zeroes if on axis
    noisefloor - float or None (default: None)
        noise cap to be applied. Defined relative to flux such that 1/noisecap is the max SNR allowed
    
    outputs:
    -------
    noise: array [e-]
        total noise sampled on flux grid
    """
    # shot noise - array w/ wavelength or integrated over band
    sig_flux = np.sqrt(np.abs(flux))

    # speckle noise
    speckle_noise = np.sqrt(speckle)
    post_processing_gain = 100. # reduction of speckle related systematics in software

    # background (instrument and sky) - array w/ wavelength matching flux array sampling or integrated over band
    sig_bg   = background_noise(inst_bg,sky_bg, texp)

    # read noise  - reduces by number of ramps, limit to 6 at best
    sig_read = read_noise(np.max((3,(readnoise/np.sqrt(nsamp)))), npix)
    
    # dark current - times time and pixels
    sig_dark = dark_noise(darknoise,npix,texp) #* get dark noise every sample
    
    noise    = np.sqrt(sig_flux **2 + sig_bg**2 + sig_read**2 + sig_dark**2 + speckle_noise**2 + (speckle/post_processing_gain)**2) 

    # cap the noise if a number is provided
    if noisefloor is not None:
        noise[np.where(noise < noisefloor)] = noisefloor * flux # noisecap is fraction of flux, 1/noisecap gives max SNR

    return noise

def background_noise(inst_bg,sky_bg, texp):
    """
    Compute the noise due to instrument and sky background photons

    inputs
    ------
    inst_bg - float/array [photons/sec/reduced pixel]
        the instrument background flux 
    sky_bg  -  float/array [photons/sec/reduced pixel]
        the sky background flux 
    texp    - float [seconds]
        the exposure time

    returns
    -------
    float [photons]
        the standard deviation noise of sky and instrument thermal background thermal
    """
    total_bg = texp * (inst_bg + sky_bg) # per reduced pixel already so dont need to include vertical pixel extent
    
    return np.sqrt(np.abs(total_bg) )


def read_noise(rn,npix):
    """
    input:
    ------
    rn: [e-/pix]
        read noise
    npix [pix]
        number of pixels

    output:
    -------
    float [photons]
        the standard deviation of detector read noise over npix
    """
    return np.sqrt(npix * rn**2)

def dark_noise(darknoise,npix,texp):
    """
    Computs Poisson noise due to dark current

    input:
    ------
    darknoise: [e-/pix/s]
        read noise
    npix [pix]
        number of pixels
    texp [s]
        exposure time in seconds

    output:
    -------
    sig_dark [photons]
        the standard deviation of dark current photons over npix
    """
    sig_dark = np.sqrt(darknoise * npix * texp)
    return sig_dark



########### PLOT

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

