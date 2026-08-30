##############################################################
# Observation: ties Star + Spectrograph + Atmosphere + AOSystem together for
# a single exposure and derives the resulting SNR
###############################################################
#
# Replaces fill_data.observe() (load_inputs.py). Domain objects are
# constructor args (an Observation is the thing most likely to get rebuilt
# repeatedly -- texp/mag sweeps -- while Star/Spectrograph/Atmosphere/AOSystem
# stay fixed); x is a .run() argument since it's a pipeline-wide grid
# decided by the caller (the future Simulate builder), not a property of
# the observation itself. Telescope area/diameter live on Spectrograph (see
# specsim/instrument.py) rather than a separate Telescope object.

from typing import Optional

import os

import numpy as np
import pandas as pd
from scipy import interpolate
from astropy import units as u
from astropy.modeling.models import BlackBody

from specsim.atmosphere import Atmosphere
from specsim.functions import degrade_spec, resample, sum_total_noise
from specsim.instrument import AOSystem, Spectrograph
from specsim.star import Star
from specsim.throughput_tools import get_emissivity


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


class Observation:
    """
    A single exposure: computed flux, background, noise, and SNR spectra
    for an on-axis (or off-axis, with a companion) star observed through a
    given Spectrograph/Atmosphere/AOSystem.
    """

    def __init__(self, star: Star, spectrograph: Spectrograph, atmosphere: Atmosphere,
                 ao_system: AOSystem,
                 texp: float = 900, texp_frame_set='default', nsamp: int = 1,
                 zenith_angle: float = 45,
                 companion: Optional[Star] = None, pl_sep: float = 0):
        self.star = star
        self.spectrograph = spectrograph
        self.atmosphere = atmosphere
        self.ao_system = ao_system
        self.texp = texp
        self.texp_frame_set = texp_frame_set
        self.nsamp = nsamp
        self.zenith_angle = zenith_angle
        self.companion = companion
        self.pl_sep = pl_sep

        # derived state, set by run()
        self.texp_frame: Optional[float] = None
        self.nframes: Optional[int] = None
        self.frame_phot_per_nm: Optional[np.ndarray] = None
        self.frame_phot_per_nm_pl: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.s_frame_star: Optional[np.ndarray] = None
        self.s_frame: Optional[np.ndarray] = None
        self.contrast: Optional[np.ndarray] = None
        self.speckle_frame: Optional[np.ndarray] = None
        self.s: Optional[np.ndarray] = None
        self.ytransmit: Optional[np.ndarray] = None
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
        self.order_cens: Optional[np.ndarray] = None
        self.ind_filter: Optional[np.ndarray] = None

    def run(self, x: np.ndarray) -> "Observation":
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

        Returns self, so calls can be chained: Observation(...).run(x).
        """
        star, spec, atm, aos = self.star, self.spectrograph, self.atmosphere, self.ao_system

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
        self.ytransmit = base_throughput_interp(self.v)  # save throughput sampled to final spectrum

        # Load background spectrum - sky is top of telescope and will be reduced by spec BASE throughput.
        # Coupling already accounted for in solid angle of fiber. Does spec bkg needs partial throughput
        # applied - ignored for now to be conservative
        self.sky_bg_ph = self.ytransmit * get_sky_bg(self.v, atm.v, atm.sky_bg, npix=spec.pix_vert,
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
        self.order_cens = spec.order_cens.copy()  # nice to have this in Observation too

        # define indices in passbands that actually fall on detectors (TODO should tweak these?)
        ind_yj = np.where((self.v > 980) & (self.v < 1335))[0]
        ind_hk = np.where((self.v > 1480) & (self.v < 2450))[0]
        self.ind_filter = np.array(ind_yj.tolist() + ind_hk.tolist())

        return self
