##############################################################
# Instrument hardware: AOSystem, Spectrograph, TrackingCamera
###############################################################
#
# These three classes together make up "the instrument" (HISPEC/MODHIS):
# the AO system that feeds a corrected beam to both the spectrograph and
# the tracking camera, the spectrograph itself (wavelength range/
# resolution/detector/throughput -- what used to be called `Instrument`),
# and the acquisition/tracking camera. Kept in one module since they're
# instrument-hardware siblings that reference each other (Spectrograph.load()
# needs an already-.select()-ed AOSystem; TrackingCamera.observe() needs
# both). Order in this file matters: AOSystem is defined first (Spectrograph
# forward-referenced as a string type hint since it's defined after),
# then Spectrograph, then TrackingCamera (needs both, already defined).

from typing import Optional

import numpy as np
import pandas as pd
from scipy import interpolate
from astropy import units as u
from astropy.modeling.models import BlackBody

from specsim import throughput_tools
from specsim.atmosphere import Atmosphere
from specsim.bandpass import Bandpass
from specsim.functions import (calc_plate_scale, calc_strehl, calc_strehl_marechal,
                               sum_total_noise, tophat, tt_to_strehl)
from specsim.star import Star, StarParams


def load_WFE(ho_wfe_file, tt_wfe_file, zenith_angle, seeing):
    """
    Load new-format WFE files where AO performance (high order WFE and
    tip/tilt residual vs. guide star magnitude) is tabulated per AO mode,
    seeing, and zenith angle, and reshape it into a per-mode dictionary

    inputs
    ------
    ho_wfe_file : str
        path to csv of high order WFE [nm] vs. mag, multi-indexed by
        seeing, zenith angle [deg], and AO mode
    tt_wfe_file : str
        path to csv of tip/tilt WFE [nm] vs. mag, same format as ho_wfe_file
    zenith_angle : int
        zenith angle of observation [deg], must be one of 0, 30, 45, 60
    seeing : str
        seeing condition, must be one of 'good', 'average', 'bad'

    output
    ------
    data : dict
        keyed by AO mode name, each entry a dict with:
            'band'   - str, magnitude band the WFE values are defined in
            'ho_wfe' - array, high order WFE [nm] sampled at 'ho_mag'
            'tt_wfe' - array, tip/tilt WFE [nm] sampled at 'tt_mag'
            'ho_mag' - array, magnitudes corresponding to 'ho_wfe'
            'tt_mag' - array, magnitudes corresponding to 'tt_wfe'
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


class AOSystem:
    """
    AO mode selection and the resulting wavefront error / Strehl / dichroic
    transmission for the on-axis star.
    """

    def __init__(self, mode: str = 'auto', tt_static: float = 2, lo_wfe: float = 50, defocus: float = 25,
                 ho_wfe_file: Optional[str] = None, tt_dynamic_file: Optional[str] = None,
                 ho_wfe_set: Optional[float] = None, tt_dynamic_set: Optional[float] = None,
                 mag='default', mag_band='default', teff='default',
                 contrast_profile_path: Optional[str] = None):
        self.mode = mode
        self.tt_static = tt_static
        self.lo_wfe = lo_wfe
        self.defocus = defocus
        self.ho_wfe_file = ho_wfe_file
        self.tt_dynamic_file = tt_dynamic_file
        self.ho_wfe_set = ho_wfe_set
        self.tt_dynamic_set = tt_dynamic_set
        self.mag = mag
        self.mag_band = mag_band
        self.teff = teff
        self.contrast_profile_path = contrast_profile_path  # MODHIS-only; passed through for Observation's contrast calc

        # derived state, set by select()
        self.mode_chosen: Optional[str] = None
        self.ho_wfe: Optional[float] = None
        self.tt_dynamic: Optional[float] = None
        self.ao_mag: Optional[float] = None
        self.strehl: Optional[float] = None
        self.strehl_array: Optional[np.ndarray] = None
        self.band: Optional[str] = None
        self.ao_modes: Optional[np.ndarray] = None
        self.ho_strehl: Optional[np.ndarray] = None  # Strehl(wfe=ho_wfe) across the full x grid, used for fiber coupling
        self.pywfs_dichroic: Optional[np.ndarray] = None
        self.ao_star: Optional[Star] = None  # only set when teff != 'default' (a fresh model is loaded for the AO star)

    def select(self, x: np.ndarray, star: Star, filt: Bandpass, filter_path: str, zp_file: str,
               spectrograph: "Spectrograph", zenith_angle: float, seeing_set: str, bands: dict) -> "AOSystem":
        """
        Determine the AO correction quality (high-order and tip-tilt
        wavefront error, and resulting Strehl) for the on-axis star. If
        ho_wfe_set/tt_dynamic_set are both given (floats or both file
        paths), those user-defined values are used directly and
        mode_chosen is set to 'User Defined'. Otherwise, WFE lookup tables
        (ho_wfe_file/tt_dynamic_file) are loaded for every available AO
        mode as a function of guide-star magnitude, seeing, and zenith
        angle; the guide-star magnitude used to sample each mode is
        computed either from the on-axis star's spectrum (teff/mag ==
        'default') or from a freshly loaded model at teff/mag otherwise.
        The Strehl for each candidate mode is computed from its HO WFE
        (Marechal approximation) and TT WFE, and either the mode with the
        highest Strehl is picked (mode=='auto') or the requested mode is
        used. Also builds the dichroic transmission applied to the
        tracking light path (pywfs_dichroic) and the per-wavelength Strehl
        used later for fiber coupling (ho_strehl).

        inputs
        ------
        x - array, shared wavelength grid [nm]
        star - Star, the already-loaded on-axis star
        filt - Bandpass, the on-axis star's photometric filter
        filter_path, zp_file - str, needed to load a Bandpass for
            mag_band/a WFE mode's native band if different from filt's
        spectrograph - Spectrograph (constructed, not necessarily .load()-ed yet); only .diameter_m is used
        zenith_angle - float [deg]
        seeing_set - str ('good'/'average'/'bad')
        bands - dict of {band: [lo, hi]} wavelength edges (yJHK), used to
            build the dichroic tophat when a PyWFS-sharing mode is chosen

        output
        ------
        self, with mode_chosen/ho_wfe/tt_dynamic/ao_mag/strehl/
        strehl_array/band/ao_modes/ho_strehl/pywfs_dichroic set
        """
        mag_filt = None
        if self.mag_band != 'default':
            # mag is quoted in a different band than filt -- load just that
            # band's curve + zp/dl_l to scale to (Bandpass.load derives the
            # filter family from the band: R->Johnson, JHK->2mass, y->cfht)
            mag_filt = Bandpass.load(filter_path, zp_file, self.mag_band, x=x)

        if self.teff == 'default':
            # reuse the on-axis star's already-loaded model grid; rescale
            # factor_0 instead of reloading if only the mag differs (or
            # recompute it via mag_filt's bandpass if mag is in a different band)
            if self.mag == 'default':
                ao_star = star
            else:
                ao_star = star.rescaled(self.mag, filt=mag_filt)
        else:  # if new teff, load new model
            ao_star = Star(StarParams(teff=self.teff, mag=self.mag, vsini=0, rv=0, logg=star.params.logg,
                                       phoenix_folder=star.params.phoenix_folder, sonora_folder=star.params.sonora_folder)
                           ).load(x, mag_filt if mag_filt is not None else filt)
            self.ao_star = ao_star

        if self.tt_dynamic_set is not None or self.ho_wfe_set is not None:
            # requires either both to be text file or both to be floats
            if type(self.ho_wfe_set) != type(self.tt_dynamic_set):
                raise ValueError('HO WFE and TT Dynamic must *both* be set to float values or both to file paths to WFE files')
            self.mode_chosen = 'User Defined HO and TT values'
            self.band = 'N/A'
        else:
            data = load_WFE(self.ho_wfe_file, self.tt_dynamic_file, zenith_angle, seeing_set)
            ao_modes = np.array(list(data.keys()))
            strehl, ho_wfes, tt_wfes, aomags = [], [], [], []
            for ao_mode in ao_modes:
                # get magnitude in band the AO mode is defined in
                wfe_bp = Bandpass.load(filter_path, zp_file, data[ao_mode]['band'], x=x)
                wfe_mag = ao_star.magnitude_in_band(wfe_bp)
                aomags.append(wfe_mag)
                # interpolate over WFEs and sample HO and TT at correct mag
                f_howfe = interpolate.interp1d(data[ao_mode]['ho_mag'], data[ao_mode]['ho_wfe'], bounds_error=False, fill_value=10000)
                f_ttwfe = interpolate.interp1d(data[ao_mode]['tt_mag'], data[ao_mode]['tt_wfe'], bounds_error=False, fill_value=10000)
                ho_wfe = float(f_howfe(wfe_mag))
                tt_wfe = float(f_ttwfe(wfe_mag))

                # compute strehl and save total
                strehl_ho = calc_strehl_marechal(ho_wfe, filt.center_wavelength)
                strehl_tt = tt_to_strehl(tt_wfe, filt.center_wavelength, spectrograph.diameter_m)
                strehl.append(strehl_ho * strehl_tt)
                ho_wfes.append(ho_wfe)
                tt_wfes.append(tt_wfe)

            self.strehl_array = np.array(strehl)
            # if user wants the code to pick best mode:
            if self.mode == 'auto' or self.mode == 'Auto':
                print('Auto AO Mode')
                i_AO = np.argmax(np.array(strehl))
            # if the user selected a specific mode:
            else:
                if self.mode in ao_modes:
                    i_AO = np.where(self.mode == ao_modes)[0][0]
                else:
                    raise ValueError('AO mode chosen not a mode! Modes: auto or %s' % ao_modes)

            self.mode_chosen = ao_modes[i_AO]
            self.ho_wfe = ho_wfes[i_AO]
            self.tt_dynamic = tt_wfes[i_AO]
            self.ao_mag = aomags[i_AO]
            self.strehl = strehl[i_AO]
            self.band = data[self.mode_chosen]['band']
            self.ao_modes = ao_modes.copy()

            print(f'AO mag is {round(self.ao_mag,2)} in {self.band} band for {self.teff} Kelvin AO star)')

            # TYPICALLY BAND SHOULD MATCH CHOSEN AO - RAISE WARNING IF NOT
            mag_band_used = filt.band if self.mag_band == 'default' else self.mag_band
            if mag_band_used != self.band:
                print("WARNING:  The temperature of the ao star will matter! mag is specified in '%s' band but the chosen AO mode (%s) is "
                      "natively defined in '%s' band -- e.g. LGS+STRAP modes are typically R band while "
                      "mag defaults to the science star's band ('%s')."
                      % (mag_band_used, self.mode_chosen, self.band, filt.band))

        print('AO mode chosen: %s' % self.mode_chosen)
        print('HO WFE is %s' % round(self.ho_wfe))
        print('tt dynamic is %s' % round(self.tt_dynamic, 2))

        # per-wavelength Strehl for fiber coupling; only needs ho_wfe/x, so
        # computed here rather than in Spectrograph (see module docstring)
        self.ho_strehl = calc_strehl_marechal(self.ho_wfe, x)

        # consider throughput impact of ao mode here
        # dichroic gets applied to science; pywfs_dichroic gets applied to tracking
        if '100H' in self.mode_chosen:
            self.pywfs_dichroic = 1 - tophat(x, bands['H'][0], bands['H'][1], 1)
            print('Selected a 100H mode, applying dichroic to science path')
        elif '100J' in self.mode_chosen:
            self.pywfs_dichroic = 1 - tophat(x, bands['J'][0], bands['J'][1], 1)
            print('Selected a 100J mode, applying dichroic to science path')
        elif '80J' in self.mode_chosen:
            self.pywfs_dichroic = 1 - 0.8 * tophat(x, bands['J'][0], bands['J'][1], 1)
            print('Selected an 80J mode, applying dichroic to science path')
        elif '80H' in self.mode_chosen:
            self.pywfs_dichroic = 1 - 0.8 * tophat(x, bands['H'][0], bands['H'][1], 1)
            print('Selected an80H mode, applying dichroic to science path')
        else:
            self.pywfs_dichroic = np.ones_like(x)

        return self


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
        self.ytransmit: Optional[np.ndarray] = None

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
            self.base_throughput, _ = throughput_tools.get_base_throughput(x, datapath=self.transmission_path)  # everything except coupling
            self.base_throughput = np.where(self.base_throughput < 0, 0, self.base_throughput)  # make negative throughput values to 0

            self.coupling, _ = throughput_tools.pick_coupling_rounded(
                self.transmission_path, x, ao_system.ho_wfe, ao_system.tt_dynamic,
                lo_wfe=ao_system.lo_wfe, tt_static=ao_system.tt_static, defocus=ao_system.defocus,
                atm=self.atm, adc=self.adc, pl_on=self.pl_on)

            self.xtransmit = x
            self.ytransmit = self.base_throughput * self.coupling * ao_system.pywfs_dichroic  # pywfs not being considered typically so pywfs_dichroic is one here

        return self


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


def get_tracking_cam(camera='h2rg',x=None):
    """
    Return the detector properties of the selected tracking/guide camera.

    Gary assumes 0.9 for the QE of the H2RG, so qe_mod modifies the
    throughput model accordingly. For the cred2-family cameras, qe_mod is
    instead a tophat scaling over the array x (980-1650nm) since their QE
    profile differs from the H2RG.

    inputs:
    -------
    camera - str
        tracking camera to select. One of 'h2rg', 'perfect', 'cred2_kpic',
        'cred2', 'cred2_rn25', 'cred2_rn20' (default 'h2rg')

    x - array or None
        wavelength array [nm], used only to build the qe_mod tophat for the
        cred2-family cameras. If None, qe_mod defaults to 1 for those
        cameras.

    returns:
    --------
    rn - float
        read noise [e-]

    pixel_pitch - float
        pixel pitch [um]

    qe_mod - float or array
        QE scale factor relative to the throughput model, unitless (1 means
        no modification; may be a tophat array over x for cred2 cameras)

    dark - float
        dark current [e-/s/pix]

    saturation - float
        full well/saturation level [e-]
    """
    if camera=='h2rg':
        rn = 12 #e-
        pixel_pitch = 18 #um
        qe_mod = 0.888 # relative to what is assumed in the throughput model
        dark=0.8 #e-/s/pix
        saturation = 80000

    if camera=='perfect':
        rn = 0 #e-
        pixel_pitch = 18 #um
        qe_mod = 1 # relative to what is assumed in the throughput model
        dark=0 #e-/s/pix
        saturation = 80000

    if camera=='cred2_kpic':
        rn = 45 #e- Calvin measured 40e-, spec is 30e-, ashley measured 35 cds without 8th column noise
        pixel_pitch = 15 #um https://www.axiomoptics.com/products/c-red-2/
        if np.any(x==None): qe_mod=1
        else: qe_mod = tophat(x,980,1650,1) # scale  
        dark=315 #e-/s liquid cooling mode -40, calvin measured 450e-, spec sheet is 600e-,measured 315e- from KPIC C-RED2
        saturation = 33000

    if camera=='cred2':
        rn = 30 #e- Calvin measured 40e-, spec is 30e-, ashley measured 35 cds without 8th column noise
        pixel_pitch = 15 #um https://www.axiomoptics.com/products/c-red-2/
        if np.any(x==None): qe_mod=1
        else: qe_mod = tophat(x,980,1650,1) # scale  
        dark=315 #e-/s liquid cooling mode -40, calvin measured 450e-, spec sheet is 600e-,measured 315e- from KPIC C-RED2
        saturation = 33000

    if camera=='cred2_rn25':
        rn = 25 #e- spie paper for 20 reads which is max for 27 FPS integration time
        pixel_pitch = 15 #um https://www.axiomoptics.com/products/c-red-2/
        if np.any(x==None): qe_mod=1
        else: qe_mod = tophat(x,980,1650,1) # scale  
        dark=315 #e-/s liquid cooling mode -40, calvin measured 450e-, spec sheet is 600e-,measured 315e- from KPIC C-RED2
        saturation = 33000

    if camera=='cred2_rn20':
        rn = 20 #e- spie paper for 20 reads which is max for 27 FPS integration time
        pixel_pitch = 15 #um https://www.axiomoptics.com/products/c-red-2/
        if np.any(x==None): qe_mod=1
        else: qe_mod = tophat(x,980,1650,1) # scale  
        dark=315 #e-/s liquid cooling mode -40, calvin measured 450e-, spec sheet is 600e-,measured 315e- from KPIC C-RED2
        saturation = 33000

    if camera=='cred2_rn20':
        rn = 20 #e- spie paper for 20 reads which is max for 27 FPS integration time
        pixel_pitch = 15 #um https://www.axiomoptics.com/products/c-red-2/
        if np.any(x==None): qe_mod=1
        else: qe_mod = tophat(x,980,1650,1) # scale  
        dark=315 #e-/s liquid cooling mode -40, calvin measured 450e-, spec sheet is 600e-,measured 315e- from KPIC C-RED2
        saturation = 33000

    return rn, pixel_pitch, qe_mod, dark, saturation


def get_tracking_band(wave,band):
    """
    Build a tophat bandpass for the requested tracking-camera filter band
    and return its center wavelength.

    Band edges are hard-coded approximations of the named photometric/
    engineering bands (some tuned for KPIC/cred2 considerations rather than
    strict MKO definitions); see
    https://home.ifa.hawaii.edu/users/tokunaga/MKO-NIR_filter_set.html#yfilter
    for the standard definitions this should eventually be updated to match.

    inputs:
    -------
    wave - array
        wavelength array [nm] over which to evaluate the bandpass

    band - str
        name of the tracking band to select. One of 'z', 'y', 'JHgap',
        'JHgapKPIC', 'JHgap_minus', 'J', 'Jplus', 'Hplus', 'H', 'Hplus50',
        'JHplus20', 'JHplus', 'K', 'Hkpic', 'yJH', 'yJ', 'JHgap20J', 'JHgap20H'
        'JHgap_narrowed' (default 'JHgap')

    returns:
    --------
    bandpass - array
        tophat transmission evaluated at wave, unitless (0-1; 'Hplus50' and
        'JHplus20' use reduced flat throughput of 0.5 and 0.2 respectively)

    center_wavelength - float
        midpoint wavelength of the selected band [nm]
    """
    if band=='z':
        l0,lf = 820,970
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1) #make up fake band

    if band=='y':
        l0,lf = 970,1070
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1) #make up fake band

    if band=='JHgap':
        l0,lf= 1335,1453
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='JHgap20J':
        l0,lf= 1335,1490  # make these narrower bc this is pyramid case that steals a little bit
        center_wavelength = (l0+lf)/2
        bandpassJHgap = tophat(wave,l0,lf,1)

        l0,lf= 1170,1330 #J
        center_wavelength =  (l0+lf)/2
        bandpassJ = tophat(wave,l0,lf,1)

        bandpass = bandpassJHgap + 0.2 * bandpassJ

    if band=='JHgap20H':
        l0,lf= 1335,1490  # make these narrower bc this is pyramid case that steals a little bit
        center_wavelength = (l0+lf)/2
        bandpassJHgap = tophat(wave,l0,lf,1)

        l0,lf= 1490,1780
        center_wavelength =  (l0+lf)/2
        bandpassH = tophat(wave,l0,lf,1)

        bandpass = bandpassJHgap + 0.2 * bandpassH

    if band=='JHgap_narrowed':
        l0,lf= 1350,1450 # less bc pyramid takes some- double check how much 
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='JHgapKPIC':
        l0,lf= 1450-25,1450+25
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='JHgap_minus':
        l0,lf= 1400,1490
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='J':
        l0,lf= 1170,1330 #
        center_wavelength =  (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='Jplus':
        l0,lf= 1170,1490  # J plus jh gap
        center_wavelength =  (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='Hplus':
        l0,lf= 1330,1780 #1450 cuts into jh gap a little, 1950 before instrument bkg turn on
        center_wavelength =  (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='H':
        l0,lf= 1490,1780
        center_wavelength =  (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='Hplus50':
        # for consideration for c-red2
        l0,lf= 1330,1780
        center_wavelength =  (l0+lf)/2
        bandpass = tophat(wave,l0,lf,0.5)
        #bandpass[np.where(wave >1490)]*=0.5 # could make jhgap 1 but would make filter more difficult maybe to make

    if band=='JHplus20':
        # for consideration for c-red2 ....meh
        l0,lf= 1170,1780
        center_wavelength =  (l0+lf)/2
        bandpass = tophat(wave,l0,lf,0.2)
        #bandpass[np.where(wave >1490)]=1 # could make jhgap 1 but would make filter more difficult maybe to make

    if band=='JHplus':
        # J and H bands
        l0,lf= 1170,1780
        center_wavelength =  (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)
        #bandpass[np.where(wave >1490)]=1 # could make jhgap 1 but would make filter more difficult maybe to make
        
    if band=='K':
        l0,lf= 1950,2460
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='Hkpic':
        l0,lf= 1500,1650
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='yJH':
        l0,lf= 980,1780
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)
    
    if band=='yJ':
        l0,lf= 980,1490
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    return bandpass, center_wavelength


def get_sky_bg_tracking(x,fwhm,sky_bg_v,sky_bg,area=76):
    """
    Generate sky background per pixel for the tracking/acquisition camera,
    default to HISPEC. Takes an already-loaded Mauna Kea sky emission
    model (OH lines + thermal continuum, in ph/s/arcsec^2/nm/m^2 -- see
    atmosphere.load_sky_background/Atmosphere.sky_bg), interpolates
    it onto the input wavelength grid, and converts it to a photon count
    rate per nm by multiplying by the telescope collecting area and the
    PSF solid angle (from the supplied FWHM, corrected for a Gaussian
    beam). Unlike get_sky_bg, this does not divide by resolving
    power/npix, so the result is per nm rather than per reduced pixel.
    Source: DMawet jup. notebook

    inputs:
    -------
    x : array [nm]
        wavelength in nanometers to evaluate/interpolate onto
    fwhm: float [arcsec]
        full width at half maximum of the PSF on the tracking camera,
        used to set the solid angle subtended by one resolution element
    sky_bg_v : array [nm]
        wavelength grid sky_bg is sampled on (e.g. Atmosphere.v)
    sky_bg : array [ph/s/arcsec^2/nm/m^2]
        sky background surface brightness, sampled on sky_bg_v (e.g.
        Atmosphere.sky_bg, from atmosphere.load_sky_background)
    area: float [m^2]
        area of telescope in meters squared

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

    sky_background_interp = np.interp(wave.value, sky_bg_v, sky_bg) * u.photon/(u.s*u.arcsec**2*u.nm*u.m**2)
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
    window_temp = 277 * u.K # temperature of cryostat window will be close to AO room temperature
    pixel_size *= u.micron
    f_num = 6 # fnumber of cold snout
    fudge_factor = 5 # ATC will view more than just the warm window, so include multiplicative factor to be conservative. 
    # also we measure the thermal background to be 500ish e-/s as of 8/11/26 - TBU once done with ATC testing with painted cold snout

    # Load blocking filter profile
    fx,fy = np.loadtxt(datapath + 'feicam/blocking_filter.TXT',skiprows=20).T
    f = interpolate.interp1d(fx[::-1]*u.nm,fy[::-1],bounds_error=False,fill_value=0)
    blocking_filter   = f(wave)/100 # convert to fraction from percent
    
    # load window emissivity
    #fx,fy = np.loadtxt(datapath + 'feicam/Infrasil_Window.txt').T
    #f = interpolate.interp1d(fx[::-1]*u.nm,fy,bounds_error=False,fill_value=0)
    window_emissivity = 0.05 #1 - f(wave)/100, this is conservative. emissivity should be max 0.01 but there will be other factors
    
    # Create QE profile for H2RG matching cutoff
    QE = tophat(wave.value,600,2550,0.8) # sensitivity of h2rg, cuts off at 2.5um

    area_times_omega = u.radian**2 * 1.13**2 * np.pi**2 * pixel_size**2 / 4 /f_num**2
    bbtemp_fxn  = BlackBody(window_temp, scale=1.0 * u.erg / (u.micron * u.s * u.cm**2 * u.arcsec**2)) 
    bb   = area_times_omega.to(u.cm**2 * u.arcsec**2) * bbtemp_fxn(wave)

    bb_spec_dens = bb.to(u.photon/u.s/u.nm, equivalencies=u.spectral_density(wave))
    
    # thermal spectrum over npix, then integrate
    thermal_spectrum = fudge_factor * npix * QE * window_emissivity * blocking_filter * bb_spec_dens # units of ph/nm/s/pix
    thermal = np.trapz(thermal_spectrum,wave)

    return thermal_spectrum, thermal.value # spectrum in units of ph/nm/s


class TrackingCamera:
    """
    Acquisition/tracking camera: detector properties, throughput, PSF FWHM,
    background, and the resulting SNR/centroid error of the tracking
    measurement.
    """

    def __init__(self, camera: str = 'h2rg', band: str = 'JHgap', fratio: float = 35,
                 texp: float = 1, field_r: float = 0,
                 transmission_file=None, aberrations_file: Optional[str] = None):
        self.camera = camera
        self.band = band
        self.fratio = fratio
        self.texp = texp
        self.field_r = field_r
        self.transmission_file = transmission_file
        self.aberrations_file = aberrations_file

        # derived state, set by observe()
        self.pixel_pitch: Optional[float] = None
        self.dark: Optional[float] = None
        self.rn: Optional[float] = None
        self.qe_mod: Optional[float] = None
        self.saturation: Optional[float] = None
        self.xtransmit: Optional[np.ndarray] = None
        self.ytransmit: Optional[np.ndarray] = None
        self.platescale: Optional[float] = None
        self.platescale_units = 'arcsec/pixel'
        self.center_wavelength: Optional[float] = None
        self.bandpass: Optional[np.ndarray] = None
        self.fwhm: Optional[float] = None
        self.fwhm_units = 'pixel'
        self.npix: Optional[float] = None
        self.strehl: Optional[float] = None
        self.sky_bg_spec: Optional[np.ndarray] = None
        self.sky_bg_ph: Optional[float] = None
        self.inst_bg_spec: Optional[np.ndarray] = None
        self.inst_bg_ph: Optional[float] = None
        self.signal_spec: Optional[np.ndarray] = None
        self.nphot_nocap: Optional[float] = None
        self.od: Optional[float] = None
        self.signal: Optional[float] = None
        self.saturation_flag: Optional[bool] = None
        self.noise: Optional[float] = None
        self.snr: Optional[float] = None
        self.centroid_err: Optional[float] = None

    def observe(self, x: np.ndarray, star: Star, atmosphere: Atmosphere, ao_system: AOSystem,
                spectrograph: Spectrograph) -> "TrackingCamera":
        """
        Load the tracking camera detector properties and throughput curve,
        compute the plate scale and PSF FWHM (from ao_system.ho_wfe/
        tt_dynamic via load_WFE), the sky and instrument background seen
        by the camera, and the stellar photon signal integrated over the
        tracking band and PSF core. From the resulting SNR, derive the
        centroid error used to characterize tracking/guiding precision. If
        the peak flux would saturate the tracking detector, an equivalent
        neutral-density filter (od) is applied to cap the signal.

        Returns self, so calls can be chained: TrackingCamera(...).observe(...).
        """
        # pick guide camera - eventually settle on one and put params in config file!
        rn, pixel_pitch, qe_mod, dark, saturation = get_tracking_cam(camera=self.camera, x=x)
        self.pixel_pitch = pixel_pitch
        self.dark = dark
        self.rn = rn
        self.qe_mod = qe_mod  # to switch cameras, wont need this later bc qe will match throughput model
        self.saturation = saturation  # to switch cameras, wont need this later bc qe will match throughput model

        # load and store tracking camera throughput - file structure hard coded
        if type(self.transmission_file) == float:
            self.xtransmit, self.ytransmit = x, np.ones_like(x) * self.transmission_file * self.qe_mod
        else:
            xtemp, ytemp = np.loadtxt(self.transmission_file, delimiter=',').T  # microns!
            f = interpolate.interp1d(xtemp * 1000, ytemp, kind='linear', bounds_error=False, fill_value=0)
            self.xtransmit, self.ytransmit = x, f(x) * self.qe_mod

        # get plate scale
        self.platescale = calc_plate_scale(self.pixel_pitch, D=spectrograph.diameter_m, fratio=self.fratio)

        # load tracking band
        bandpass, self.center_wavelength = get_tracking_band(x, self.band)
        self.bandpass = bandpass * ao_system.pywfs_dichroic

        # get fwhm (in pixels)
        self.fwhm = float(self.get_fwhm(ao_system.ho_wfe, ao_system.tt_dynamic,
                                         self.center_wavelength, spectrograph.diameter_m))
        self.npix = np.pi * (self.fwhm / 2) ** 2  # only take noise in circle of diameter FWHM
        print('Tracking FWHM=%spix' % self.fwhm)

        self.strehl = calc_strehl(ao_system.ho_wfe, self.center_wavelength)

        # get sky background and instrument background, spec is ph/nm/s
        # fwhm must be in arcsec
        self.sky_bg_spec = get_sky_bg_tracking(x, self.fwhm * self.platescale, atmosphere.v,
                                                            atmosphere.sky_bg, area=spectrograph.area_m2)
        self.sky_bg_ph = np.trapz(self.sky_bg_spec * self.bandpass * self.ytransmit, x)  # sky bkg needs mult by throughput and bandpass profile

        # get background spec (takes thermal emission from warm cryostat window)
        # units of ph/nm/s for spectrum and ph/s for inst_bg_ph
        self.inst_bg_spec, self.inst_bg_ph = get_inst_bg_tracking(x, self.pixel_pitch, self.npix, datapath=spectrograph.transmission_path)

        # get photons in band
        self.signal_spec = star.s * self.texp * spectrograph.area_m2 * self.ytransmit * np.abs(atmosphere.s)

        # fac is empirically the fraction of light approx under 2D gaussian of FWHM~4pix,
        # which scales npix. This was tuned based on a actual toy centroiding model fit and gets results to match
        fac = 0.5
        nphot = fac * self.strehl * np.trapz(self.signal_spec * self.bandpass, star.v)

        # get noise
        self.noise = sum_total_noise(nphot, self.texp, 1, self.inst_bg_ph, self.sky_bg_ph, self.dark, self.rn, self.npix, 0)
        print(f'Tracking noise: {self.noise} e-')

        # get centroid error, cap if saturated
        # peak of 2D Gaussian 4pix wide will be 1/10th of the flux in a 4pix diameter aperture (empirically derived)
        flux_in_peak = nphot / 10  # nphot is already times 0.5 to give flux in a 4 pix diameter aperture
        if flux_in_peak > self.saturation:
            # pick an ND filter
            # compute OD of filter needed (neg neg computes ceiling)
            self.od = np.max((-1 * round(-1 * np.log10(flux_in_peak / self.saturation), 0), 0))
            # Apply chosen nd filter
            self.signal = 10 ** (-1 * self.od) * nphot  # cap nphot
            self.noise = sum_total_noise(self.signal, self.texp, 1, self.inst_bg_ph, self.sky_bg_ph, self.dark, self.rn, self.npix, 0)
            # save things related to saturation
            self.saturation_flag = True
            self.nphot_nocap = nphot
            print(f'Tracking OD needed {self.od}, nphot capped to {self.signal} e-')
        else:
            self.od = 0.0
            self.nphot_nocap = nphot
            self.signal = nphot  # no blocking needed
            self.saturation_flag = False
            print(f'Tracking photons: {self.signal} e-')

        self.snr = self.signal / self.noise
        # for centroid error, care about the SNR in the peak
        self.centroid_err = (1 / np.pi) * self.fwhm / self.snr  # same fwhm but snr is reduced to not saturate like if used an ND filter
        print(f'Tracking SNR: {self.snr}, centroid error: {self.centroid_err} pix')

        return self

    def get_tracking_optics_aberrations(self, field_r=None):
        """
        Load PSF size of the tracking camera optics as a function of field
        position and interpolate to the requested field radius.

        Reads a file of field position vs RMS spot size (per wavelength) for the
        tracking optics, converts the RMS at 1400nm from um to pixels (using the
        pixel pitch of the selected camera), and linearly interpolates
        (extrapolating outside the tabulated range) to field_r. Wavelength
        dependence is ignored beyond picking the 1400nm column, since the RMS
        does not vary much with wavelength across the tabulated columns.

        inputs:
        --------
        field_r (float, 0-3) [arcsec]
            field radius position of tracking star on guide camera field.
            Defaults to this camera's configured field_r.

        The camera (for the um->pixel conversion) and the aberrations file are
        read off self. To plot spot RMS vs field, use
        plot.plot_tracking_cam_spot_rms().

        returns:
        -------
        RMS of the PSF due to optical aberrations in pixels (radius rms), float,
        interpolated (or extrapolated) to field_r
        """
        field_r = self.field_r if field_r is None else field_r

        # todo, make this a try statement, so if a file isn't available it just assumes 0 added aberrations
        try:
            f = np.loadtxt(self.aberrations_file)
        except FileNotFoundError:
            print(f"Aberrations file not found: {self.aberrations_file}")
            f = np.array([[0, 0, 0, 0, 0, 0, 0, 0]])
        field, rmstot, rms900,rms1000,rms1200,rms1400,rms1600,rms2200  = f.T #field [deg], rms [um]
        _,pixel_pitch,_,_,_ = get_tracking_cam(camera=self.camera,x=None)

        # should interpolate across wavelength but theyre not so different so just use 1400nm
        # multiply rms by 2 to get diameter (closer to FWHM)
        f = interpolate.interp1d(field * 3600, rms1400/pixel_pitch,bounds_error=False,fill_value='extrapolate')

        return f(field_r)

    def get_fwhm(self, wfe, tt_resid, wavelength, diam, getall=False):
        """
        Compute the total image FWHM on the tracking camera by combining the
        diffraction-limited spot (broadened by high-order wavefront error via
        the Strehl ratio), the tip/tilt residual, and off-axis aberrations from
        the tracking camera optics, all added in quadrature.

        inputs:
        -------
        wfe - float
            high-order (non tip/tilt) residual wavefront error [nm]

        tt_resid - float
            residual tip/tilt error [mas]

        wavelength - float
            observing wavelength [nm]

        diam - float
            telescope diameter [m] (converted to nm internally to match
            wavelength in the diffraction-limit formula)

        getall - bool
            if True, also return the intermediate strehl/FWHM component terms
            (default False)

        The plate scale [arcsec/pixel], field radius, camera name and
        aberrations file are read off self. If the aberrations file cannot be
        found, the off-axis contribution falls back to 0.5 pixels with a warning.

        to do:
        check how RMS relates to FWHM

        returns:
        --------
        fwhm - float
            total image FWHM [pixels] combining diffraction/high-order WFE,
            tip/tilt, and off-axis aberration terms in quadrature

        if getall is True, also returns (in this order, with some values
        duplicated): strehl (Strehl ratio, unitless), diffraction_spot_pix
        (diffraction-limited spot size [pixels]), fwhm_ho (FWHM from
        diffraction + high-order WFE [pixels]), fwhm_tt (FWHM from tip/tilt
        residual [pixels]), fwhm_offaxis (FWHM from off-axis camera aberrations
        [pixels]), followed by a repeat of strehl, diffraction_spot_pix,
        fwhm_ho, fwhm_tt, fwhm_offaxis
        """
        platescale = self.platescale

        rms_to_fwhm = 1/0.44 # from KAON, not too off from gaussian 1sig to FWHM factor
        radius_to_diam = 2
    
        # get WFE
        strehl = np.exp(-(2*np.pi*wfe/wavelength)**2)

        # Diffraction limited spot with High Order WFE FWHM
        diffraction_spot_arcsec = 206265 * wavelength/ (diam * 10**9) # arcsec
        diffraction_spot_pix = diffraction_spot_arcsec / platescale
        fwhm_ho = diffraction_spot_pix / strehl**(1/4) # 1/strehl**.25 from dimitri, to account for broadening deviation from diffraction limit

        # Tip Tilt FWHM in pixels
        fwhm_tt = rms_to_fwhm * tt_resid*1e-3/platescale 

        # FWHM from off axis aberrations in camera optics
        try:
            fwhm_offaxis     = radius_to_diam * self.get_tracking_optics_aberrations() # times 2 to get radius
        except:
            fwhm_offaxis = 0.5
            print('Cant find file %s' %self.aberrations_file)
            print('Warning: no tracking camera aberrations file found, assuming 0.5')
    
        fwhm = np.sqrt(fwhm_tt**2 + fwhm_ho**2 + fwhm_offaxis**2)

        if getall:
            return fwhm, strehl, diffraction_spot_pix, fwhm_ho, fwhm_tt, fwhm_offaxis, strehl, diffraction_spot_pix, fwhm_ho, fwhm_tt, fwhm_offaxis
        else:
            return fwhm
