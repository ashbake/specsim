##############################################################
# Spectrograph: the science detector -- throughput, orders, and the exposure
# taken through them
###############################################################
#
# Merges the old instrument.Spectrograph (hardware) with observation.Observation
# (one exposure). They were split, but TrackingCamera already did both in one
# class, so the science and guide detectors now have the same shape:
#
#     Spectrograph(...).load(x, ao).observe(x, star, atm, ao, texp=...)
#     TrackingCamera(...).observe(x, star, atm, ao, spectrograph)
#
# The background/contrast helpers that only this detector uses live here too,
# mirroring get_sky_bg_tracking/get_inst_bg_tracking in trackingcamera.py.
#
# Note two same-named quantities that are NOT the same array:
#   .ytransmit         total throughput (base x coupling x dichroic) on grid x
#   .base_throughput_v base throughput resampled onto the observed grid .v
# The second was called `ytransmit` on the old Observation; it is renamed here
# because observe() reads .ytransmit to build the stellar flux and would
# otherwise clobber it.

import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy import interpolate
from astropy import units as u
from astropy.modeling.models import BlackBody

from specsim.aosystem import AOSystem
from specsim.atmosphere import Atmosphere
from specsim.functions import calc_strehl_marechal, degrade_spec, resample, sum_total_noise
from specsim.star import Star

def get_sky_bg(x,sky_bg_v,sky_bg,npix=3,R=100000,diam=10,area=76):
    """
    Generate sky background per reduced pixel, default is HISPEC.
    Takes an already-loaded Mauna Kea sky emission model (OH lines +
    thermal continuum, in ph/s/arcsec^2/nm/m^2 -- see
    atmosphere.load_sky_background/Atmosphere.sky_bg), interpolates
    it onto the input wavelength grid, and converts it to a photon count
    rate by multiplying by the telescope collecting area, the
    diffraction-limited beam solid angle (from wavelength/diameter,
    corrected for a Gaussian beam), and the wavelength width of one
    reduced-pixel resolution element (wave/R/npix).
    Source: DMawet jup. notebook

    inputs:
    -------
    x : array [nm]
        wavelength in nanometers to evaluate/interpolate onto
    sky_bg_v : array [nm]
        wavelength grid sky_bg is sampled on (e.g. Atmosphere.v)
    sky_bg : array [ph/s/arcsec^2/nm/m^2]
        sky background surface brightness, sampled on sky_bg_v (e.g.
        Atmosphere.sky_bg, from atmosphere.load_sky_background)
    npix: integer
        number of pixels, defaults to 3
    R: float
        resolving power of instrument, default is 100,000
    diam: float [m]
        diameter of telescope in meters
    area: float [m^2]
        area of telescope in meters squared

    outputs:
    --------
    array [ph/s]
        sky background photon rate per reduced pixel, sampled on the
        input wavelength grid x
    """
    diam *= u.m
    area = area * u.m * u.m
    wave = x*u.nm

    fwhm = ((wave  / diam) * u.radian).to(u.arcsec)
    solidangle = fwhm**2 * 1.13 #corrected for Gaussian beam (factor 1.13)

    pix_width_nm  = (wave/R/npix) #* u.nm
    sky_background_interp=np.interp(wave.value, sky_bg_v, sky_bg) * u.photon/(u.s*u.arcsec**2*u.nm*u.m**2) * area * solidangle * pix_width_nm

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


def get_order_bounds(filename):
    """
    open order bounds file

    input
    -----
    filename - name of order file containing wavelength [nm], order width [nm] comma delimited

    output
    ------
    cenlam - order center wavelength [nm]
    width  - order width [nm]
    """
    f = np.loadtxt(filename,delimiter=',')
    cenlam, width = f.T[0],f.T[1]
    return cenlam, width


##############################################################
# Instrument throughput and fiber coupling, read off disk
###############################################################

def pick_coupling_rounded(transmission_path,w,ho_wfe, tt_dynamic, lo_wfe=50, tt_static=0, defocus=0, atm=1,adc=1,pl_on=1,piaa_boost=1.3):
    """
    Look up fiber injection/coupling efficiency by rounding the requested
    wavefront-error and tip-tilt parameters to the nearest values available
    in the pre-computed coupling grid (a set of CSV files, one per parameter
    combination), then loading that single file and interpolating it onto
    the requested wavelength grid. High-order WFE is not rounded; instead
    it is converted analytically to a Strehl ratio (Marechal approximation)
    and multiplied onto the tabulated (rounded-grid) coupling.

    inputs
    ------
    transmission_path : string
        path to the directory containing the 'coupling/' subfolder with the
        couplingEff_atm%s_adc%s_PL%s_defoc%snmRMS_LO%snmRMS_ttStatic%smas_ttDynamic%smasRMS.csv
        grid files
    w : array
        wavelength array, in nm if max(w) >= 10, otherwise assumed to be in
        microns and converted to nm internally
    ho_wfe : float or array [nm]
        high-order wavefront error, used to compute the Strehl ratio applied
        multiplicatively to the tabulated coupling (not used to select the
        grid file)
    tt_dynamic : float [mas]
        dynamic tip-tilt RMS; rounded to the nearest 0.5 mas (grid sampling)
        to select the coupling file, and clipped to the 19.5 mas file if the
        rounded value is >= 20
    lo_wfe : float, optional [nm]
        low-order wavefront error RMS; rounded to the nearest 25 nm to select
        the coupling file. Default 50
    tt_static : float, optional [mas]
        static tip-tilt; rounded to the nearest 0.5 mas to select the
        coupling file. Default 0
    defocus : float, optional [nm]
        defocus term RMS; rounded to the nearest 25 nm to select the
        coupling file. Default 0
    atm : int, optional
        0 or 1, whether the atmosphere was included in the simulation grid
        used to select the coupling file. Default 1
    adc : int, optional
        0 or 1, whether the ADC (atmospheric dispersion corrector) was
        included in the simulation grid used to select the coupling file.
        Default 1
    pl_on : int, optional
        0 or 1, whether the photonic lantern is on. If on, the coupling
        efficiencies of the three PL output modes are summed; if off, only
        mode 1 (single-mode fiber) is used. Default 1
    piaa_boost : float, optional
        multiplicative coupling boost factor from the PIAA lens, applied on
        top of the tabulated coupling and Strehl. Default 1.3

    outputs
    -------
    coupling : array
        coupling efficiency vs wavelength (tabulated grid value, interpolated
        onto w, times ho_strehl times piaa_boost)
    ho_strehl : array
        Strehl ratio computed from ho_wfe via the Marechal approximation,
        same wavelength grid as w
    """
    if np.max(w) < 10: 
        wave=w.copy() * 1000
    else:
        wave=w.copy()
    
    filename_skeleton = 'coupling/couplingEff_atm%s_adc%s_PL%s_defoc%snmRMS_LO%snmRMS_ttStatic%smas_ttDynamic%smasRMS.csv'
    tt_dynamic_rounded = np.round(2 * tt_dynamic) / 2 # round to neared 0.5 because grid is sampled to 0.5mas
    lo_wfe_rounded = int(100*np.round(4*(lo_wfe/100))/4) # round to nearest 25
    tt_static_rounded = np.round(tt_static*2)/2
    if int(tt_static_rounded)==tt_static_rounded: tt_static_rounded  = int(tt_static_rounded)
    if int(tt_dynamic_rounded)==tt_dynamic_rounded: tt_dynamic_rounded  = int(tt_dynamic_rounded)
    defocus_rounded =  int(100*np.round(4*(defocus/100))/4)

    if tt_dynamic_rounded < 20:
        f = pd.read_csv(transmission_path+filename_skeleton%(int(atm),int(adc),int(pl_on),defocus_rounded,lo_wfe_rounded,tt_static_rounded,tt_dynamic_rounded)) # load file
    else:
        f = pd.read_csv(transmission_path+filename_skeleton%(int(atm),int(adc),int(pl_on),defocus_rounded,lo_wfe_rounded,tt_static_rounded,19.5)) # load file

    if pl_on:
        coupling_data_raw = f['coupling_eff_mode1'] + f['coupling_eff_mode2'] + f['coupling_eff_mode3']
    else:
        coupling_data_raw = f['coupling_eff_mode1']

    # interpolate onto self.x
    finterp = interpolate.interp1d(1000*f['wavelength_um'].values,coupling_data_raw,bounds_error=False,fill_value=0)
    coupling_data = finterp(wave)

    #piaa_boost = 1.3 # based on Gary's sims, but needs updating because will be less for when Photonic lantern is being used
    ho_strehl  = calc_strehl_marechal(ho_wfe,wave)
    coupling   = coupling_data  * ho_strehl * piaa_boost

    return coupling, ho_strehl


def pick_coupling_interpolate(w,dynwfe,ttStatic,ttDynamic,LO=50,PLon=0,piaa_boost=1.3,points=None,values=None):
    """
    Compute fiber injection/coupling efficiency by N-D interpolation (via
    scipy.interpolate.interpn) of the pre-computed coupling grid, rather
    than rounding to the nearest tabulated point as pick_coupling_rounded
    does. The grid ('points') and its tabulated values ('values') must be
    supplied by the caller, typically from grid_interp_coupling(). High-order
    WFE is applied analytically as a Strehl factor on top of the
    interpolated coupling, exactly as in pick_coupling_rounded.

    Note: docstring reflects current behavior; a TODO in the original code
    notes that full interpolation (vs. rounding) was still being implemented.

    inputs
    ------
    w : array
        wavelength array. If min(w) > 10 it is assumed to be in nm and is
        divided by 1000 to get microns (used to build the 'point' passed to
        interpn against the wavelength axis of 'points'); the working array
        is converted back to nm afterward (if still < 10) before computing
        the Strehl ratio, to match the nm units expected for dynwfe
    dynwfe : float or array [nm]
        high-order/dynamic wavefront error, used to compute ho_strehl via
        exp(-(2*pi*dynwfe/wave)^2) and applied multiplicatively to the
        interpolated coupling
    ttStatic : float [mas]
        static tip-tilt; must be in range 0-10 or a ValueError is raised
    ttDynamic : float [mas]
        dynamic tip-tilt; must be in range 0-20 or a ValueError is raised
    LO : float, optional [nm]
        low-order wavefront error RMS; must be in range 0-100 or a
        ValueError is raised. Default 50
    PLon : int, optional
        0 or 1, whether the photonic lantern is on; coerced to int and must
        be <= 1 or a ValueError is raised. If on, the three PL output-mode
        coupling efficiencies are interpolated separately and recombined
        (with an extra 0.95 recombination-loss factor applied below 1.4 um);
        if off, only the single-mode-fiber coupling (mode 1) is used.
        Default 0
    piaa_boost : float, optional
        multiplicative coupling boost factor from the PIAA lens. Default 1.3
    points : tuple of arrays, optional
        grid axis values (LO, ttStatic, ttDynamic, wavelength) defining the
        coupling table, as returned by grid_interp_coupling()
    values : tuple of arrays, optional
        tabulated coupling efficiency array(s) on the 'points' grid, as
        returned by grid_interp_coupling(); one array if PLon is off, three
        (mode1, mode2, mode3) if PLon is on

    outputs
    -------
    coupling : array
        interpolated coupling efficiency vs wavelength, times ho_strehl
        times piaa_boost
    ho_strehl : array
        Strehl ratio computed from dynwfe, same wavelength grid as w
    """
    PLon = int(PLon)

    waves = w.copy()
    if np.min(waves) > 10:
        waves/=1000 # convert nm to um

    # check range of each variable
    if ttStatic > 10 or ttStatic < 0:
        raise ValueError('ttStatic is out of range, 0-10')
    if ttDynamic > 20 or ttDynamic < 0:
        raise ValueError('ttDynamic is out of range, 0-10')
    if LO > 100 or LO < 0:
        raise ValueError('LO is out of range,0-100')
    if PLon >1:
        raise ValueError('PL is out of range')

    if PLon:
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
        losses = np.ones_like(mode1) # due to PL imperfection
        losses[np.where(waves< 1.400)[0]] = 0.95 # only apply to y band
        raw_coupling = losses*(mode1+mode2+mode3) # do dumb things for now #0.95 is a recombination loss term 
    else:
        values_1 = values[0]
        #points, values_1 = grid_interp_coupling(PLon)
        point = (LO,ttStatic,ttDynamic,waves)
        raw_coupling = interpolate.interpn(points, values_1, point,bounds_error=False,fill_value=0)

    if np.max(waves) < 10:
        waves*=1000 # nm to match dynwfe

    ho_strehl =  np.exp(-(2*np.pi*dynwfe/waves)**2) # computed per wavelength as grid
    coupling  = raw_coupling * piaa_boost * ho_strehl
    
    return coupling, ho_strehl


def grid_interp_coupling(PLon=1,path='./data/instrument/hispec/throughput/coupling/',atm=1,adc=1):
    """
    Build the N-D coupling-efficiency grid (axes: low-order WFE, static
    tip-tilt, dynamic tip-tilt, wavelength) used by pick_coupling_interpolate
    for true interpolation, as opposed to the nearest-grid-point rounding
    done in pick_coupling_rounded. Loops over every combination of LO
    (0-100 nm, step 25), ttStatic (0-10 mas, step 1), and ttDynamic
    (0-20 mas, step 0.5), reads the corresponding
    couplingEff_atm%s_adc%s_PL%s_defoc0nmRMS_LO%snmRMS_ttStatic%smas_ttDynamic%smasRMS.csv
    file (defocus fixed at 0), and stacks the per-mode coupling efficiency
    columns into 4-D arrays suitable for scipy.interpolate.interpn.

    inputs
    ------
    PLon : int, optional
        0 or 1, whether the photonic lantern is on. If on, the coupling
        efficiencies for all three PL output modes (mode1, mode2, mode3)
        are loaded into separate grids; if off, only mode1 (single-mode
        fiber) is loaded. Default 1
    path : string, optional
        directory containing the coupling grid CSV files
    atm : int, optional
        0 or 1, whether the atmosphere was included in the simulation grid
        being loaded (selects file name). Default 1
    adc : int, optional
        0 or 1, whether the ADC (atmospheric dispersion corrector) was
        included in the simulation grid being loaded (selects file name).
        Default 1

    outputs
    -------
    points : tuple of arrays
        grid axis values (LOs, ttStatics, ttDynamics, wavelength_um) defining
        the coordinates of the 'values' arrays, for use with interpn
    values_1 : array [len(LOs), len(ttStatics), len(ttDynamics), n_wave]
        tabulated coupling efficiency for PL output mode 1 (or the only
        mode, if PLon is 0)
    values_2 : array, only returned if PLon
        tabulated coupling efficiency for PL output mode 2
    values_3 : array, only returned if PLon
        tabulated coupling efficiency for PL output mode 3
    """
    LOs = np.arange(0,125,25)
    ttStatics = np.arange(11)
    ttDynamics = np.arange(0,20.5,0.5)
    
    filename_skeleton = 'couplingEff_atm%s_adc%s_PL%s_defoc0nmRMS_LO%snmRMS_ttStatic%smas_ttDynamic%smasRMS.csv'

    # to dfine values, must open up each file. not sure if can deal w/ wavelength
    values_1 = np.zeros((len(LOs),len(ttStatics),len(ttDynamics),100))
    values_2 = np.zeros((len(LOs),len(ttStatics),len(ttDynamics),100))
    values_3 = np.zeros((len(LOs),len(ttStatics),len(ttDynamics),100))  
    for i,LO in enumerate(LOs):
        for j,ttStatic in enumerate(ttStatics):
            for k,ttDynamic in enumerate(ttDynamics):
                if round(ttDynamic)==ttDynamic: ttDynamic=round(ttDynamic)
                f = pd.read_csv(path+filename_skeleton%(atm,adc,PLon,LO,ttStatic,ttDynamic))
                if PLon:
                    values_1[i,j,k,:]=f['coupling_eff_mode1'] #what to fill here?
                    values_2[i,j,k,:]=f['coupling_eff_mode2'] #what to fill here?
                    values_3[i,j,k,:]=f['coupling_eff_mode3'] #what to fill here?
                else:
                    values_1[i,j,k,:]=f['coupling_eff_mode1'] #what to fill here?

                #values_hk[i,j,k]=f['coupling_eff_mode1'][50] #what to fill here?
    
    points = (LOs, ttStatics, ttDynamics,f['wavelength_um'].values)

    if PLon:
        return points,values_1,values_2,values_3
    else:
        return points,values_1


def get_emissivity(wave,datapath='./data/instrument/hispec/throughput/'):
    """
    Load and interpolate the per-surface emissivity curves for each optical
    element in the red and blue optical paths (excluding fiber coupling),
    onto the requested wavelength grid, along with the assumed physical
    temperature of each surface. Fiber contributions ('fib*') are doubled
    to account for the integrating-sphere measurement setup used to derive
    those emissivity files.

    inputs
    ------
    wave : array
        wavelength array to sample emissivity on; converted from nm to
        microns internally if min(wave) > 10
    datapath : string, optional
        path to the directory containing per-surface subfolders
        (tel/ao/feicom/feired/feiblue/fibred/fibblue/rspec), each with a
        '<surface>_emissivity.csv' file (columns: wavelength_um, emissivity)

    outputs
    -------
    em_red : list of arrays
        emissivity vs wave for each surface in the red path
        (['tel','ao','feicom','feired','fibred','rspec']), in that order
    em_blue : list of arrays
        emissivity vs wave for each surface in the blue path
        (['tel','ao','feicom','feiblue','fibblue','bspec']), in that order
    temps : list of floats [K]
        assumed physical temperature for each of the 6 surface slots,
        [276,276,276,276,276,77] (thermal background surfaces at ambient,
        detector/cold stage at 77 K)
    """
    x = wave.copy()
    if np.min(x) > 10:
        x/=1000 #convert nm to um

    red_include = ['tel', 'ao', 'feicom', 'feired','fibred','rspec']#,'coupling']
    blue_include = ['tel', 'ao', 'feicom', 'feiblue','fibblue','bspec']
    temps = [276,276,276,276,276,77]

    em_red, em_blue = [],[]
    for i in red_include:
        wtemp, stemp = np.loadtxt(datapath + i + '/%s_emissivity.csv'%i, delimiter=',',skiprows=1).T
        f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
        if i.startswith('fib'):
            em_red.append(2*f(x)) # count fib twice because of integrating sphere
        else:
            em_red.append(f(x))

    for i in blue_include:
        wtemp, stemp = np.loadtxt(datapath + i + '/%s_emissivity.csv'%i, delimiter=',',skiprows=1).T
        f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
        if i.startswith('fib'):
            em_blue.append(2*f(x)) # count fib twice bc of integrating sphere
        else:
            em_blue.append(f(x)) #

    return em_red,em_blue,temps


def get_emissivities(wave,surfaces=['tel'],datapath='./data/instrument/hispec/throughput/'):
    """
    Derive per-surface emissivity as (1 - throughput) for an arbitrary list
    of named surfaces, by loading each surface's '<surface>_throughput.csv'
    file and interpolating it onto the requested wavelength grid. Unlike
    get_emissivity(), this does not use dedicated emissivity CSV files or
    apply the fiber integrating-sphere doubling factor, and the caller
    supplies the list of surfaces to include.

    inputs
    ------
    wave : array
        wavelength array to sample emissivity on; converted from nm to
        microns internally if min(wave) > 10
    surfaces : list of strings, optional
        names of the subfolders/surfaces to load, each expected to contain a
        '<surface>_throughput.csv' file (columns: wavelength_um, throughput).
        Default ['tel']
    datapath : string, optional
        path to the directory containing the per-surface subfolders

    outputs
    -------
    em : list of arrays
        1 - throughput vs wave, one array per entry in 'surfaces', in the
        same order
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


def get_base_throughput(wave,datapath='./data/instrument/hispec/throughput/'):
    """
    Compute the total instrument throughput excluding fiber coupling, by
    multiplying together the per-surface throughput curves along the red
    path (['tel','ao','feicom','feired','fibred','rspec']) for wavelengths
    > 1.4 um and along the blue path
    (['tel','ao','feicom','feiblue','fibblue','bspec']) for wavelengths
    < 1.4 um, then concatenating the two bands into a single blue-to-red
    array on the input wavelength grid. To plot the result, use
    plot.plot_base_throughput().

    inputs
    ------
    wave - array
        wavelength array [nm] to sample throughput on (converted to microns
        internally if min(wave) > 10)
    ploton - Bool
        default is False, whether to plot throughput (blue and red curves
        vs wavelength) and save the figure to './base_throughput.png'
    datapath - string
        path to throughput files in special HISPEC/MODHIS structure, with
        one subfolder per surface each containing a '<surface>_throughput.csv'
        file (columns: wavelength_um, throughput)

    outputs:
    ---------
    s - array
        total base throughput, sampled on wave grid, blue band
        (wave < 1.4 um) followed by red band (wave > 1.4 um)
    data - dict
        nested dict {'red': {surface: throughput_array, ...},
        'blue': {surface: throughput_array, ...}} holding the individual
        per-surface throughput curves (each interpolated onto wave) used to
        build snew
    """
    # wavelength array to um
    x = wave.copy()
    if np.min(x) > 10:
        x/=1000 #convert nm to um

    data={}
    data['red']  = {}
    data['blue'] = {}
    #plt.figure()
    for spec in ['red','blue']:
        if spec=='red':
            include = ['tel', 'ao', 'feicom', 'feired','fibred','rspec']#,'coupling']
        if spec=='blue':
            include = ['tel', 'ao', 'feicom', 'feiblue','fibblue','bspec']#,'coupling']

        for i in include:
            if i==include[0]:
                wtemp, stemp = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
                s = f(x)
                #plt.plot(w,s,label=i)
            else:
                wtemp, stemp = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                # interpolate onto s
                f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
                s*=f(x)
                #plt.plot(w,s,label=i)
            # store throughput in dictionary
            data[spec][i] = f(x)

        if spec=='red':
            isub = np.where(x > 1.4) 
            wred = x[isub]
            specred = s[isub]
        if spec=='blue':
            isub = np.where(x<1.4)
            specblue = s[isub]
            wblue = x[isub]
    
    w = np.concatenate([wblue,wred])
    s = np.concatenate([specblue,specred])

    return s, data


def load_photonic_lantern():
    """
    Load the photonic lantern's mode-transfer (unitary) matrices, which map
    input modes to output single-mode fibers, from a fixed .npy file, along
    with the wavelength grid they were computed on.

    inputs
    ------
    None

    outputs
    -------
    wavearr : array [nm]
        20-point wavelength grid spanning 970-1350 nm on which the unitary
        matrices are defined
    data : array
        unitary transfer matrices loaded from
        './data/throughput/photonic_lantern/unitary_matrices.npy', one
        matrix per wavelength in wavearr (shape depends on the saved file,
        e.g. [n_wave, n_mode, n_mode])
    """
    wavearr = np.linspace(970,1350,20)
    data = np.load('./data/throughput/photonic_lantern/unitary_matrices.npy')
    
    return wavearr,data


class Spectrograph:
    """
    Spectrograph wavelength range, resolution, detector properties, echelle
    order geometry, and total optical throughput, plus the telescope
    collecting area/diameter (area_m2/diameter_m) it's paired with -- these
    are fixed per instrument (e.g. HISPEC/Keck vs. MODHIS/TMT), not
    independently variable, so they live here rather than on a separate
    Telescope object. (Renamed from Instrument, since AOSystem/TrackingCamera
    are also part of "the instrument" and now live in this same module.)
    """

    def __init__(self, l0: float = 900, l1: float = 2500, res: float = 100000, res_samp: float = 3,
                 pix_vert: float = 4, extraction_frac: float = 0.925,
                 saturation: float = 100000, readnoise: float = 12, darknoise: float = 0.01,
                 pl_on: int = 1, rv_floor: float = 0.5, atm: int = 1, adc: int = 1,
                 transmission_path: Optional[str] = None, transmission_file: Optional[str] = None,
                 order_bounds_file: Optional[str] = None,
                 area_m2: float = 76, diameter_m: float = 10):
        self.l0 = l0
        self.l1 = l1
        self.res = res
        self.res_samp = res_samp
        self.pix_vert = pix_vert
        self.extraction_frac = extraction_frac
        self.saturation = saturation
        self.readnoise = readnoise
        self.darknoise = darknoise
        self.pl_on = pl_on
        self.rv_floor = rv_floor
        self.atm = atm
        self.adc = adc
        self.transmission_path = transmission_path
        self.transmission_file = transmission_file
        self.order_bounds_file = order_bounds_file
        self.area_m2 = area_m2        # telescope collecting area [m^2] -- tied to the instrument's telescope, not independently variable
        self.diameter_m = diameter_m  # telescope diameter [m]

        # derived state, set by load()
        self.order_cens: Optional[np.ndarray] = None
        self.order_widths: Optional[np.ndarray] = None
        self.sig: Optional[np.ndarray] = None
        self.base_throughput: Optional[np.ndarray] = None
        self.coupling: Optional[np.ndarray] = None
        self.xtransmit: Optional[np.ndarray] = None
        self.ytransmit: Optional[np.ndarray] = None    # total throughput on grid x (base x coupling x dichroic)
        # derived state, set by observe()
        self.star: Optional[Star] = None
        self.atmosphere = None
        self.ao_system = None
        self.texp: Optional[float] = None
        self.texp_frame_set = 'default'
        self.nsamp: Optional[int] = None
        self.zenith_angle: Optional[float] = None
        self.companion: Optional[Star] = None
        self.pl_sep: float = 0
        self.texp_frame: Optional[float] = None
        self.nframes: Optional[int] = None
        self.frame_phot_per_nm: Optional[np.ndarray] = None
        self.frame_phot_per_nm_pl: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None                    # wavelength grid of the observed spectrum [nm]
        self.s_frame_star: Optional[np.ndarray] = None
        self.s_frame: Optional[np.ndarray] = None
        self.contrast: Optional[np.ndarray] = None
        self.speckle_frame: Optional[np.ndarray] = None
        self.s: Optional[np.ndarray] = None                    # observed spectrum, all frames [photons]
        self.base_throughput_v: Optional[np.ndarray] = None    # base_throughput resampled onto self.v (NOT ytransmit, which is the total throughput on x)
        self.sky_bg_ph: Optional[np.ndarray] = None
        self.inst_bg_ph: Optional[np.ndarray] = None
        self.noise_frame: Optional[np.ndarray] = None
        self.noise: Optional[np.ndarray] = None
        self.snr: Optional[np.ndarray] = None
        self.v_res_element: Optional[np.ndarray] = None
        self.snr_res_element: Optional[np.ndarray] = None
        self.snr_max_orders: Optional[np.ndarray] = None
        self.snr_mean_orders: Optional[np.ndarray] = None
        self.order_inds: Optional[list] = None
        self.ind_filter: Optional[np.ndarray] = None

    def load(self, x: np.ndarray, ao_system: AOSystem) -> "Spectrograph":
        """
        Load the echelle order geometry, the per-pixel wavelength sampling,
        and the total instrument throughput curve (base optical/detector
        throughput times fiber coupling efficiency times the AO dichroic).
        If transmission_file is set (and loadable), a user-supplied total
        throughput curve is used directly instead. Depends on ao_system
        already having been built (needs ho_wfe/tt_static/tt_dynamic/
        defocus/pywfs_dichroic).

        inputs
        ------
        x - array, shared wavelength grid [nm]
        ao_system - AOSystem, already .select()-ed

        output
        ------
        self, with order_cens/order_widths/sig/base_throughput/coupling/
        xtransmit/ytransmit set
        """
        self.order_cens, self.order_widths = get_order_bounds(self.order_bounds_file)
        self.sig = x / self.res / self.res_samp  # lambda/res = dlambda, nm per pixel

        try:  # if a custom transmission file is given and loadable, use it, otherwise load HISPEC/MODHIS version
            thput_x, thput_y = np.loadtxt(self.transmission_file, delimiter=',').T
            if np.max(thput_x) < 5: thput_x *= 1000  # convert to nanometers
            tck_thput = interpolate.splrep(thput_x, thput_y, k=1, s=0)
            self.xtransmit = x
            self.ytransmit = interpolate.splev(x, tck_thput, der=0, ext=1)
            self.ytransmit = np.where(self.ytransmit < 0, 0, self.ytransmit)  # make negative throughput values to 0
            self.base_throughput = self.ytransmit.copy()
            print('Loaded Custom Transmission File')
        except Exception:
            self.base_throughput, _ = get_base_throughput(x, datapath=self.transmission_path)  # everything except coupling
            self.base_throughput = np.where(self.base_throughput < 0, 0, self.base_throughput)  # make negative throughput values to 0

            self.coupling, _ = pick_coupling_rounded(
                self.transmission_path, x, ao_system.ho_wfe, ao_system.tt_dynamic,
                lo_wfe=ao_system.lo_wfe, tt_static=ao_system.tt_static, defocus=ao_system.defocus,
                atm=self.atm, adc=self.adc, pl_on=self.pl_on)

            self.xtransmit = x
            self.ytransmit = self.base_throughput * self.coupling * ao_system.pywfs_dichroic  # pywfs not being considered typically so pywfs_dichroic is one here

        return self

    def observe(self, x: np.ndarray, star: Star, atmosphere: Atmosphere, ao_system: AOSystem,
                texp: float = 900, texp_frame_set='default', nsamp: int = 1,
                zenith_angle: float = 45, companion: Optional[Star] = None,
                pl_sep: float = 0) -> "Spectrograph":
        """
        Compute the flux reaching the spectrometer (stellar spectrum x
        telescope area x spectrograph throughput x telluric transmission),
        pick the per-frame exposure time to avoid saturation (or use a
        user-set value), degrade and resample the spectrum onto the
        spectrograph's resolution/pixel grid, add sky and spectrograph thermal
        background, and compute the total photon and read/dark noise per
        frame and across all frames. From that, derive the SNR spectrum
        per pixel (v, snr) and per resolution element (v_res_element,
        snr_res_element), plus max/mean SNR per echelle order.

        If pl_sep>0 (off-axis companion), additionally computes the
        companion flux and the stellar speckle contribution at the
        companion's separation (via ao_system.contrast_profile_path/MODHIS
        contrast calculator, falling back to an analytic contrast model),
        and s/snr then refer to the companion signal with the star's
        speckle halo as an added noise/background term.

        Requires .load() to have been called first (needs ytransmit/sig/
        order_cens). Returns self, so calls can be chained:
        Spectrograph(...).load(x, ao).observe(x, star, atm, ao).
        """
        self.star, self.atmosphere, self.ao_system = star, atmosphere, ao_system
        self.texp, self.texp_frame_set, self.nsamp = texp, texp_frame_set, nsamp
        self.zenith_angle, self.companion, self.pl_sep = zenith_angle, companion, pl_sep
        spec, atm, aos = self, atmosphere, ao_system

        # flux density is stellar flux * telescope area * spectrograph throughput * atmospheric absorption
        # If planet separation is >0, compute for the planet also
        phot_per_sec_nm = star.s * spec.area_m2 * spec.ytransmit * np.abs(atm.s)
        if self.pl_sep > 0:
            phot_per_sec_nm_pl = self.companion.s * spec.area_m2 * spec.ytransmit * np.abs(atm.s)
            try:
                contrast = get_MODHIS_contrast(aos.contrast_profile_path, aos.mode_chosen, atm.seeing,
                                                            self.zenith_angle, star.params.mag, x, self.pl_sep)  # new version, specific to MODHIS
                print("Using new MODHIS contrast calculator with radial profile database.")
            except Exception as e:
                print(f"Warning: {e}, using old contrast calculator with analytic method.")
                contrast = get_contrast(x, self.pl_sep, spec.diameter_m, atm.seeing, aos.strehl)  # old version

        # Figure out the exposure time per frame to avoid saturation
        # Default case takes 900s as maximum frame exposure time length
        if self.texp_frame_set == 'default':
            if self.pl_sep > 0:  # use estimated planet flux if off axis mode
                max_ph_per_s = np.max((phot_per_sec_nm_pl + contrast * phot_per_sec_nm) * spec.sig)
            else:
                max_ph_per_s = np.max(phot_per_sec_nm * spec.sig)
            # set text frame
            if self.texp < 900:
                texp_frame_tmp = np.min((self.texp, spec.saturation / max_ph_per_s))
            else:
                texp_frame_tmp = np.min((900, spec.saturation / max_ph_per_s))
            self.nframes = int(np.ceil(self.texp / texp_frame_tmp))
            print('Nframes set to %s' % self.nframes)
            self.texp_frame = np.round(self.texp / self.nframes, 2)
            print('Texp per frame set to %s' % self.texp_frame)
        # user defined exposure time per frame case:
        else:
            if self.texp < self.texp_frame_set:
                print('Exposure time is less than the set exposure time per frame, will set frame time to the total exposure time')
            self.texp_frame = np.min((self.texp_frame_set, self.texp))
            self.nframes = int(np.ceil(self.texp / self.texp_frame))
            print('Texp per frame set to user defined value %s' % self.texp_frame)
            print('Nframes set to %s' % self.nframes)

        # Degrade to spectrograph resolution after applying frame exposure time
        self.frame_phot_per_nm = phot_per_sec_nm * self.texp_frame
        s_ccd_lores = degrade_spec(star.v, self.frame_phot_per_nm, spec.res)

        if self.pl_sep > 0:
            self.frame_phot_per_nm_pl = phot_per_sec_nm_pl * self.texp_frame
            s_ccd_lores_pl = degrade_spec(star.v, self.frame_phot_per_nm_pl, spec.res)

        # Resample onto res element grid - new wavelength grid self.v
        self.v, self.s_frame_star = resample(star.v, s_ccd_lores, sig=np.mean(spec.sig), dx=0, eta=1, mode='fast')
        self.s_frame_star *= spec.extraction_frac
        # remove negatives from star spectrum
        self.s_frame_star = np.where(self.s_frame_star < 0, 0, self.s_frame_star)
        if self.pl_sep > 0:
            _, self.s_frame = resample(star.v, s_ccd_lores_pl, sig=np.mean(spec.sig), dx=0, eta=1, mode='fast')
            self.s_frame *= spec.extraction_frac  # extraction fraction, reduce photons to mimic spectral extraction imperfection

            # interpolate contrast curve onto new low res array
            spec_contrast_interp = interpolate.interp1d(spec.xtransmit, contrast)
            self.contrast = spec_contrast_interp(self.v)
            # speckle is the star flux times contrast
            self.speckle_frame = self.contrast * self.s_frame_star
        else:  # sframe is the star when on axis, speckle is zeros
            self.s_frame = self.s_frame_star
            self.speckle_frame = np.zeros_like(self.s_frame)

        # Get total spectrum for all frames
        # save planet spectrum as main science spectrum
        self.s = self.s_frame * self.nframes

        # Resample throughput for applying to sky background
        base_throughput_interp = interpolate.interp1d(spec.xtransmit, spec.base_throughput)
        self.base_throughput_v = base_throughput_interp(self.v)  # base throughput resampled onto self.v

        # Load background spectrum - sky is top of telescope and will be reduced by spec BASE throughput.
        # Coupling already accounted for in solid angle of fiber. Does spec bkg needs partial throughput
        # applied - ignored for now to be conservative
        self.sky_bg_ph = self.base_throughput_v * get_sky_bg(self.v, atm.v, atm.sky_bg, npix=spec.pix_vert,
                                                                  R=spec.res, diam=spec.diameter_m, area=spec.area_m2)
        self.inst_bg_ph = get_inst_bg(self.v, npix=spec.pix_vert, R=spec.res, diam=spec.diameter_m,
                                                    area=spec.area_m2, datapath=spec.transmission_path)

        # Calculate noise
        if spec.pl_on:  # 3 port lantern hack
            # need to figure out what to do for sky and spec bkg bc depends on coupling
            noise_frame_yJ = np.sqrt(3) * sum_total_noise(
                self.s_frame / 3, self.texp_frame, self.nsamp, self.inst_bg_ph / np.sqrt(3), self.sky_bg_ph / np.sqrt(3),
                spec.darknoise, spec.readnoise, spec.pix_vert, self.speckle_frame)  # flux split evenly over 3 traces for each of 3 PL outputs
            noise_frame = sum_total_noise(
                self.s_frame, self.texp_frame, self.nsamp, self.inst_bg_ph, self.sky_bg_ph,
                spec.darknoise, spec.readnoise, spec.pix_vert, self.speckle_frame)
            yJ_sub = np.where(self.v < 1400)[0]
            noise_frame[yJ_sub] = noise_frame_yJ[yJ_sub]  # fill in yj with sqrt(3) times noise in PL case
        else:
            noise_frame = sum_total_noise(
                self.s_frame, self.texp_frame, self.nsamp, self.inst_bg_ph, self.sky_bg_ph,
                spec.darknoise, spec.readnoise, spec.pix_vert, self.speckle_frame)

        # Remove nans and 0s from noise frame, make these infinite
        noise_frame[np.where(np.isnan(noise_frame))] = np.inf
        noise_frame[np.where(noise_frame == 0)] = np.inf

        # Combine noise in quadrature for all frames
        self.noise_frame = noise_frame
        self.noise = np.sqrt(self.nframes) * noise_frame

        # Compute snr and resample to get SNR per res element (assumes flux in the number of pixels
        # spanning a res element (3 for hispec/modhis) combine in quadrature)
        self.snr = self.s / self.noise
        self.v_res_element, self.snr_res_element = resample(
            self.v, self.snr, sig=spec.res_samp, dx=0, eta=1 / np.sqrt(spec.res_samp), mode='pixels')

        # compute median and max snr per order
        order_snrs_mean, order_snrs_max, order_inds = [], [], []
        for i, lam_cen in enumerate(spec.order_cens):
            order_ind = np.where((self.v_res_element > lam_cen - 0.9 * spec.order_widths[i] / 2) &
                                  (self.v_res_element < lam_cen + 0.9 * spec.order_widths[i] / 2))[0]
            order_inds.append(order_ind)
            if np.nanmean(self.snr_res_element[order_ind]) > 0.001:
                order_snrs_mean.append(np.nanmean(self.snr_res_element[order_ind]))
                order_snrs_max.append(np.nanmax(self.snr_res_element[order_ind]))
            else:
                order_snrs_mean.append(np.nan)
                order_snrs_max.append(np.nan)

        self.snr_max_orders = np.array(order_snrs_max)
        self.snr_mean_orders = np.array(order_snrs_mean)
        self.order_inds = order_inds

        # define indices in passbands that actually fall on detectors (TODO should tweak these?)
        ind_yj = np.where((self.v > 980) & (self.v < 1335))[0]
        ind_hk = np.where((self.v > 1480) & (self.v < 2450))[0]
        self.ind_filter = np.array(ind_yj.tolist() + ind_hk.tolist())

        return self
