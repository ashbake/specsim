##############################################################
# General functions for noise calculations
# contributors:
# Ashley Baker abaker@caltech.edu
# Huihao Zhang zhang.12043@osu.edu
# many functions based on psisim https://github.com/planetarysystemsimager/psisim/
###############################################################

import numpy as np
from scipy import interpolate
import os
import pandas as pd

from astropy.modeling.models import BlackBody
from astropy import units as u
from astropy import constants as c 

from specsim.functions import tophat, resample
from specsim.throughput_tools import get_emissivity, get_emissivities

all = {'get_sky_bg','get_inst_bg','sum_total_noise','plot_noise_components'}

def get_sky_bg(x,airmass=1.3,pwv=1.5,npix=3,R=100000,diam=10,area=76,skypath = '../../../../_DATA/sky/'):
    """
    Generate sky background per reduced pixel, default is HISPEC.
    Loads the Mauna Kea sky emission model (OH lines + thermal continuum,
    in ph/s/arcsec^2/nm/m^2) for the tabulated (pwv, airmass) grid point
    nearest the requested values, interpolates it onto the input
    wavelength grid, and converts it to a photon count rate by
    multiplying by the telescope collecting area, the diffraction-limited
    beam solid angle (from wavelength/diameter, corrected for a Gaussian
    beam), and the wavelength width of one reduced-pixel resolution
    element (wave/R/npix).
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
    R: float
        resolving power of instrument, default is 100,000
    diam: float [m]
        diameter of telescope in meters
    area: float [m^2]
        area of telescope in meters squared
    skypath: string
        path to the directory containing the Mauna Kea sky background
        model files (mk_skybg_zm_<pwv>_<airmass>_ph.dat)

    outputs:
    --------
    array [ph/s]
        sky background photon rate per reduced pixel, sampled on the
        input wavelength grid x
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
    Generate instrument thermal background per reduced pixel, default to HISPEC.
    Loads the emissivity and physical temperature of each red-arm and
    blue-arm instrument subsystem (via get_emissivity), builds a
    Planck blackbody spectrum for each temperature scaled by the
    telescope area and the diffraction-limited beam solid angle, weights
    each blackbody by the corresponding subsystem emissivity, sums the
    contributions across subsystems, converts to a photon rate over one
    reduced-pixel resolution element (wave/R/npix), stitches the red and
    blue arm results together at 1.4 micron, and spline-interpolates the
    result back onto the input wavelength grid.
    Source: DMawet jup. notebook

    inputs:
    -------
    x : array [nm]
        wavelength in nanometers
    npix: integer
        number of pixels
    R: float
        resolving power of instrument, default is 100,000
    diam: float [m]
        diameter of telescope in meters
    area: float [m^2]
        area of telescope in meters squared
    datapath: string
        path to where throughput data in HISPEC format is

    outputs:
    --------
    array [ph/s]
        instrument thermal background photon rate per reduced pixel
        (already considering PSF sampling), sampled on the input
        wavelength grid x
    """
    em_red,em_blue, temps = get_emissivity(x,datapath=datapath)

    # assign units
    diam *= u.m
    area *= u.m * u.m
    wave = x*u.nm

    # compute pixel width in nanometers
    fwhm = ((wave  / diam) * u.radian).to(u.arcsec)
    solidangle = fwhm**2 * 1.13 #corrected for Gaussian beam (factor 1.13)
    pix_width_nm  = (wave/R/npix) #* u.nm 

    # step through temperatures and emissivities for red and blue
    # em_red and em_blue are indexed matching temp index
    for i,temp in enumerate(temps):
        bbtemp_fxn  = BlackBody(temp * u.K, scale=1.0 * u.erg / (u.micron * u.s * u.cm**2 * u.arcsec**2)) 
        bbtemp      = bbtemp_fxn(wave) *  area.to(u.cm**2) * solidangle
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

    # w,s
    w = np.concatenate([x[isubblue],x[isubred]])
    s = np.concatenate([em_blue_tot,em_red_tot])

    # interpolate onto input x array
    tck        = interpolate.splrep(w,s.value, k=2, s=0)
    em_total   = interpolate.splev(x,tck,der=0,ext=1)

    return em_total # units of ph/s/reduced_pix

def get_sky_bg_tracking(x,fwhm,airmass=1.5,pwv=1.5,area=76,skypath = '../../../../_DATA/sky/'):
    """
    Generate sky background per pixel for the tracking/acquisition camera,
    default to HISPEC. Loads the Mauna Kea sky emission model (OH lines +
    thermal continuum, in ph/s/arcsec^2/nm/m^2) for the given (pwv, airmass),
    interpolates it onto the input wavelength grid, and converts it to a
    photon count rate per nm by multiplying by the telescope collecting
    area and the PSF solid angle (from the supplied FWHM, corrected for a
    Gaussian beam). Unlike get_sky_bg, this does not divide by resolving
    power/npix, so the result is per nm rather than per reduced pixel.
    Source: DMawet jup. notebook

    inputs:
    -------
    x : array [nm]
        wavelength in nanometers
    fwhm: float [arcsec]
        full width at half maximum of the PSF on the tracking camera,
        used to set the solid angle subtended by one resolution element
    airmass: float [1,inf)
        airmass of the observation. Defaults to 1.5
    pwv: float [mm] [0,inf)
        precipitable water vapor in millimeters during the observation.
        Defaults to 1.5
    area: float [m^2]
        area of telescope in meters squared
    skypath: string
        path to the directory containing the Mauna Kea sky background
        model files (mk_skybg_zm_<pwv>_<airmass>_ph.dat)

    outputs:
    --------
    array [ph/s/nm]
        sky background photon rate per nm, sampled on the input
        wavelength grid x
    """
    area = area * u.m * u.m
    wave = x*u.nm

    fwhm *= u.arcsec
    solidangle = fwhm**2 * 1.13 #corrected for Gaussian beam (factor 1.13)

    sky_background_MK_tmp  = np.genfromtxt(skypath+'mk_skybg_zm_'+str(pwv)+'_'+str(round(airmass,1))+'_ph.dat', skip_header=0)
    sky_background_MK      = sky_background_MK_tmp[:,1] * u.photon/(u.s*u.arcsec**2*u.nm*u.m**2) 
    sky_background_MK_wave = sky_background_MK_tmp[:,0] * u.nm

    sky_background_interp=np.interp(wave, sky_background_MK_wave, sky_background_MK)
    sky_background_interp*= area * solidangle 
    
    return sky_background_interp.value # ph/s/nm

def get_inst_bg_tracking(x,pixel_size,npix,datapath='./data/throughput/hispec_subsystems_11032022/'):
    """
    Generate the instrument thermal background seen by the tracking camera,
    per pixel, default to HISPEC. Source: DMawet jup. notebook.
    Models the thermal emission of the cryostat window as a blackbody at a
    fixed temperature (277 K), attenuated by the blocking filter
    transmission, the (fixed, approximate) window emissivity, and the
    H2RG quantum efficiency (modeled as a tophat between 600-2600 nm).
    The blackbody is scaled by the effective area x solid angle set by
    the pixel size and the optical f-number, converted to a photon rate
    per nm, and multiplied by npix to get the total thermal spectrum for
    npix pixels; that spectrum is also integrated over wavelength to give
    a single total photon rate.
    change this to take emissivities and temps as inputs so dont
    have to rely on get_emissivities

    inputs:
    -------
    x : array [nm]
        wavelength in nanometers
    pixel_size: float [micron]
        physical size of one detector pixel
    npix: integer
        number of pixels over which the thermal background is summed
    datapath: string
        path to where throughput data in HISPEC format is (used here to
        load the blocking filter transmission curve)

    outputs:
    --------
    thermal_spectrum: array [ph/s/nm]
        instrument thermal background spectral photon rate for npix
        pixels, sampled on the input wavelength grid x
    thermal: float [ph/s]
        thermal_spectrum integrated over wavelength, i.e. the total
        instrument thermal background photon rate for npix pixels
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
    Gets the residual-speckle contrast (relative to the stellar peak) seen
    by a single-mode fiber (SMF) positioned on a planet at some angular
    separation from its host star, based on a Kolmogorov-turbulence halo
    model of the AO-corrected PSF. Computes the Fried parameter r0 from
    the seeing, scales it to the observing wavelength, converts the
    planet separation to units of lambda/D ("resels"), and evaluates the
    residual-halo power-law contrast at that separation (clipped to a
    power-law extrapolation inside the AO control radius set by the
    number of actuators). The result is reduced by an empirical
    single-mode-fiber suppression gain and clipped to a maximum of 1.

    inputs
    ------
    wave         - [nm] A list of wavelengths [float length m]
    pl_sep       - [mas] separations at which to calculate the speckle noise in arcseconds
    tel_diam     - Telescope diameter [m]
    seeing       - seeing during observation [arcsec]
    strehl       - strehl of AO correction

    outputs
    -------
    contrast - array, same shape as wave
        residual speckle contrast (dimensionless, relative to the
        unocculted stellar peak) at the given planet separation, as a
        function of wavelength. Values are clipped to be <= 1.
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

def get_MODHIS_contrast(folder, ao_mode, seeing, zenith_angle, magnitude, waves, radius):
    """Function to get contrast from a particular file at a given radius.
    Looks up the pre-computed MODHIS AO simulation contrast-vs-radius
    profile (a CSV of separation vs. azimuthally-summed annulus
    intensity) matching the requested AO mode, seeing percentile, zenith
    angle, and (rounded to nearest integer) stellar magnitude, once per
    near-IR band (y, J, H, K). Within each band the profile is linearly
    interpolated (with extrapolation) to the requested radius to get a
    single contrast value, which is then assigned to every wavelength
    that falls in that band. Wavelengths outside the defined y/J/H/K
    ranges are assigned a contrast of 1 (no attenuation/undefined).
    Rounds to the nearest magnitude, interpolates to the given radius.
    Uses the same calculated value for every wavelength in the same band.

    inputs
    ------
    folder         - string, folder containing the csv profiles
    ao_mode        - adaptive optics mode, can be NGS, LGS, off, or auto
    seeing         - string, seeing percentile. Can be good, average, or bad
    zenith_angle   - float, zenith angle of observation
    magnitude      - float, stellar magnitude
    waves          - [nm] A list of wavelengths [float length m]
    radius         - float [mas], radius at which to get contrast in milliarcseconds
        (converted internally to arcseconds to match the CSV profiles)

    outputs
    -------
    overall_contrast - array, same shape as waves
        contrast (dimensionless, from the annulus-summed-intensity
        profile) at the given radius for each wavelength, using the
        band-matched interpolated value; 1 for wavelengths outside the
        y/J/H/K band definitions
    """

    ao_mode_map = {'NGS': 'ngsao_ngsao', 'LGS_ON': 'mcao_pyttf'}
    seeing_map = {'0.6': '25', '0.8': '50', '1.1': '75'} # Different conversion made my load_inputs for the seeing. Good/average/bad is already a number
    # seeing_map = {'good': '25', 'average': '50', 'bad': '75'}
    
    ao_mode = ao_mode_map.get(ao_mode, ao_mode)
    seeing_str = str(seeing)
    seeing = seeing_map[seeing_str]
    # seeing = seeing_map.get(seeing_str, seeing)

    zenith_angle = str(int(zenith_angle))
    magnitude = round(magnitude)
    radius = radius / 1000  # Convert radius to arcseconds
    overall_contrast = np.zeros_like(waves, dtype=float)
    csv_filename_skeleton = '%s_%sp_za%s_mag%s_evlpsfcl_1_x0_y0_%s.csv'

    # Define each band, with no gaps in between
    bands = [
        ('K', (1865, 2460)),
        ('H', (1410, 1865)),
        ('J', (1120, 1410)),
        ('y', (970, 1120))
    ]

    # Dictionary to store contrast values for each band
    band_contrast = {}

    # Iterate over each band and calculate the contrast once per band
    for band_name, (start, end) in bands:
        # Filter the wavelengths that fall into the current band
        wave_indices = np.where((waves >= start) & (waves < end))[0]

        if wave_indices.size == 0:
            continue  # Skip if no wavelengths are in this band

        full_file = os.path.join(folder, csv_filename_skeleton % (ao_mode, seeing, zenith_angle, magnitude, band_name))

        # Take out the error handling. If the file is missing, load_inputs will skip this function and use the old get_contrast
        df = pd.read_csv(full_file)
        radii = df.iloc[:, 0].values  # First column is radius
        contrast = df.iloc[:, 1].values  # Second column is contrast (sum of intensity in annulus)
        
        interpolation_function = interpolate.interp1d(radii, contrast, kind='linear', fill_value='extrapolate')
        contrast_value = interpolation_function(radius).item()

        # Store the contrast value for this band
        band_contrast[band_name] = contrast_value

    # Assign contrast values to each wavelength based on their band
    for i, wavelength in enumerate(waves):
        for band_name, (start, end) in bands:
            if start <= wavelength < end:
                overall_contrast[i] = band_contrast.get(band_name, 1)
                break
        else:
            overall_contrast[i] = 1

    return overall_contrast


def get_speckle_noise_vfn(wave,ho_wfe,tt_dyn,pl_sep,mag,seeing,strehl,tel_diam,vortex_charge):
    """
    Estimate residual on-axis stellar leakage (contrast) for a vector
    vortex fiber nulling (VFN) coronagraph, i.e. the planet is off axis
    while the star is (imperfectly) nulled by the vortex.
    Sums three leakage terms, each approximated as a power law in units
    of lambda/D and calibrated against simulations/references: (1)
    leakage from high-order wavefront error (quasi-static/AO residual
    speckles), using an empirically-fit coefficient set by the vortex
    charge; (2) leakage from dynamic tip/tilt jitter, using the
    approximation of Ruane et al. 2019 (Eq. 3); and (3) geometric
    leakage from the finite angular size of the host star, using the
    fit of Ruane et al. 2019 (Fig. 7c). The three terms are summed and
    the result is clipped to a maximum contrast of 1.
    taken from https://github.com/planetarysystemsimager/psisim/blob/kpic/psisim/instruments/modhis.py#L441C1-L441C1
    planet is off axis, star gets reduction in throughput due to vortex

    inputs
    ------
    wave [nm]     - wavelength array
    ho_wfe [nm]   - High order wave front error (quasi-static/AO residual
                    wavefront error) used to estimate the WFE-driven
                    stellar leakage term
    tt_dyn [mas]  - Dynamic tip/tilt jitter amplitude, used to estimate
                    the tip/tilt-driven stellar leakage term
    pl_sep [mas]  - angular separation of the planet from the host star
                    (carried through for context; not currently used in
                    the leakage calculation below)
    mag           - stellar magnitude (carried through for context; not
                    currently used in the leakage calculation below)
    seeing [arcsec] - seeing during the observation (carried through for
                    context; not currently used in the leakage
                    calculation below)
    strehl        - Strehl ratio of the AO correction (carried through
                    for context; not currently used in the leakage
                    calculation below)
    tel_diam [m]  - telescope diameter
    vortex_charge - integer topological charge of the vortex coronagraph
                    (1 or 2); selects the empirical WFE and geometric
                    leakage coefficients

    outputs
    -------
    contrast - array, same shape as wave
        total estimated on-axis stellar leakage contrast (dimensionless,
        sum of WFE, tip/tilt, and geometric leakage terms), clipped to
        be <= 1

    TODO
    ----
    need planet throughput to accompany it since off axis?
    note: this function references an undefined `host_diameter`
    variable (not one of the listed inputs) for the geometric leakage
    term; see function body.
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
    sig_read = read_noise(np.max((6,(readnoise/np.sqrt(nsamp)))), npix)
    
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
    Compute the total detector read noise contribution over npix pixels
    by adding the per-pixel read noise in quadrature (rn * sqrt(npix)).

    input:
    ------
    rn: [e-/pix]
        read noise per pixel (per read/ramp, already reduced by number
        of samples by the caller if applicable)
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
    Computes Poisson noise due to dark current, i.e. the standard
    deviation of the Poisson-distributed dark current counts accumulated
    over npix pixels during the exposure time (sqrt(darknoise * npix * texp)).

    input:
    ------
    darknoise: [e-/pix/s]
        dark current rate per pixel
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
    """
    Plot combined sky + instrument background versus wavelength,
    with instrument bands overlaid

    input
    -----
    so - storage object
        used to get the instrument band definitions to overlay
    v - array [nm]
        wavelength array
    instbg - array [e-/s/pix]
        instrument background
    skybg - array [e-/s/pix]
        sky background
    """
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

