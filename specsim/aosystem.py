##############################################################
# AOSystem: AO mode selection and the resulting wavefront error / Strehl
###############################################################
#
# Split out of instrument.py, which still holds the Spectrograph and
# TrackingCamera it feeds. Nothing here imports them back: select() takes the
# telescope diameter as a constructor parameter rather than reaching for it on
# a Spectrograph, so this module has no dependency on the detector modules.
#
# AOSystem is the first thing built in a scene that
# actually depends on the star: the guide-star magnitude picks the AO mode,
# the mode sets the high-order WFE and tip-tilt residual, and those set the
# Strehl that Spectrograph.load() turns into fiber coupling.
#
# load_WFE() stays a module-level function here (like load_phoenix in star.py
# and load_telluric_transmission in atmosphere.py) -- it is pure file IO over
# the instrument's WFE tables and needs no AOSystem to be useful.

from typing import Optional

import numpy as np
import pandas as pd
from scipy import interpolate

from specsim.bandpass import Bandpass
from specsim.functions import calc_strehl_marechal, tophat, tt_to_strehl
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
                 contrast_profile_path: Optional[str] = None,
                 area_m2: float = 76, diameter_m: float = 10):
        self.mode = mode
        self.area_m2 = area_m2        # telescope collecting area [m^2]; carried for symmetry with the detectors
        self.diameter_m = diameter_m  # telescope diameter [m], used to turn tip-tilt residual into a Strehl term
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
               zenith_angle: float, seeing_set: str, bands: dict) -> "AOSystem":
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
                strehl_tt = tt_to_strehl(tt_wfe, filt.center_wavelength, self.diameter_m)
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

            # The AO mode's WFE table is indexed by magnitude in self.band, so the
            # AO star's magnitude is colour-converted into that band by
            # magnitude_in_band() above -- report what was actually used.
            mag_band_used = filt.band if self.mag_band == 'default' else self.mag_band
            print("AO mag is %s in %s band (from %s = %s, Teff = %sK)"
                  % (round(self.ao_mag, 2), self.band, mag_band_used,
                     round(ao_star.params.mag, 2), int(ao_star.params.teff)))

            # Only a problem if the user explicitly quoted the AO magnitude in a
            # band that isn't the one the chosen mode is tabulated in -- then the
            # conversion leans on the assumed AO star Teff, which they also set.
            # When mag_band is 'default' the magnitude is inherited from the
            # science star and converted from its own spectrum, which is fine.
            if self.mag_band != 'default' and self.mag_band != self.band:
                print("WARNING: AO mag was given in '%s' band but the chosen AO mode (%s) is natively "
                      "defined in '%s' band, so the magnitude was colour-converted assuming Teff = %sK. "
                      "Set [ao] mag_band to '%s' (or check [ao] teff) if that isn't what you want."
                      % (self.mag_band, self.mode_chosen, self.band, int(ao_star.params.teff), self.band))

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
