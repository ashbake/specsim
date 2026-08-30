##############################################################
# Atmosphere: telluric transmission + sky background + seeing for a given
# observing setup, plus the loaders that read them off disk
###############################################################
#
# Replaces fill_data.telluric() (load_inputs.py). zenith_angle is a .load()
# argument rather than constructor state -- it's an observation-level input,
# not a property of the atmosphere itself, so the same Atmosphere (and its
# already-loaded TAPAS file) can be reused across observations that only
# change texp/zenith angle, without rereading the fits file.
#
# load_telluric_transmission/load_sky_background stay module-level functions
# (like load_phoenix/load_sonora in star.py) -- they're pure file IO plus
# PWV/airmass scaling, useful without an Atmosphere, and they take no
# instance state beyond what's passed in.
#
# load_sky_background() was previously duplicated inside
# get_sky_bg()/get_sky_bg_tracking(), each re-reading the file
# from disk given raw pwv/airmass/skypath. It depends only on pwv/airmass
# (both Atmosphere properties), so the load lives here; get_sky_bg/
# get_sky_bg_tracking now take the already-loaded spectrum and do only the
# telescope/instrument-specific conversion to a photon rate.

from typing import Optional

import numpy as np
from astropy.io import fits
from scipy import interpolate

SEEING_MAP = {'good': 0.6, 'average': 0.8, 'bad': 1.1}

# TAPAS fits column name -> lowercase key used on Atmosphere/returned here.
# H2O scales with pwv*airmass (relative to the file's reference PWV/airmass);
# every other species scales with airmass alone.
_SPECIES_COLUMNS = ['Rayleigh', 'O3', 'O2', 'N2', 'CO', 'CH4', 'CO2', 'N2O']


def load_telluric_transmission(x, telluric_file, pwv, airmass):
    """
    Load a TAPAS telluric transmission model and scale each
    molecular/scattering component from the file's reference PWV/airmass to
    the requested pwv/airmass, resampled onto x.

    inputs
    ------
    x - array, wavelength grid to resample onto [nm]
    telluric_file - str, TAPAS fits file with PWV/AIRMASS header keywords
        giving the file's reference conditions
    pwv - float, requested precipitable water vapor [mm]
    airmass - float, requested airmass

    output
    ------
    dict with keys 'h2o', 'rayleigh', 'o3', 'o2', 'n2', 'co', 'ch4', 'co2',
    'n2o' (each an array on x, scaled to pwv/airmass) and 's' (the product
    of all of them -- the total telluric transmission spectrum)
    """
    data = fits.getdata(telluric_file)
    pwv0 = fits.getheader(telluric_file)['PWV']
    airmass0 = fits.getheader(telluric_file)['AIRMASS']

    _, ind = np.unique(data['Wave/freq'], return_index=True)

    def _scale(column, exponent):
        tck = interpolate.splrep(data['Wave/freq'][ind], data[column][ind] ** exponent, k=2, s=0)
        return interpolate.splev(x, tck, der=0, ext=1)

    result = {'h2o': _scale('H2O', pwv * airmass / pwv0 / airmass0)}
    for column in _SPECIES_COLUMNS:
        result[column.lower()] = _scale(column, airmass / airmass0)

    result['s'] = (result['h2o'] * result['rayleigh'] * result['o3'] * result['o2'] *
                   result['n2'] * result['co'] * result['ch4'] * result['co2'] * result['n2o'])
    return result


def load_sky_background(x, pwv, airmass, skypath):
    """
    Load the Mauna Kea sky emission model (OH lines + thermal continuum,
    in ph/s/arcsec^2/nm/m^2) for the tabulated (pwv, airmass) grid point
    nearest the requested values, and resample it onto x.

    inputs
    ------
    x - array, wavelength grid to resample onto [nm]
    pwv - float, requested precipitable water vapor [mm]
    airmass - float, requested airmass
    skypath - str, path to the directory containing the Mauna Kea sky
        background model files (mk_skybg_zm_<pwv>_<airmass>_ph.dat)

    output
    ------
    array, sky background surface brightness [ph/s/arcsec^2/nm/m^2],
    resampled onto x
    """
    pwv_rounded = np.round(pwv, 1)
    airmass_rounded = np.round(airmass, 1)
    data = np.genfromtxt(f'{skypath}mk_skybg_zm_{pwv_rounded}_{airmass_rounded}_ph.dat', skip_header=0)
    return np.interp(x, data[:, 0], data[:, 1])


class Atmosphere:
    """
    Telluric transmission spectrum (total + per-species), sky background
    surface brightness, and seeing for a given precipitable water vapor /
    zenith angle.
    """

    def __init__(self, telluric_file: str, sky_path: Optional[str] = None,
                 pwv: float = 1.3, seeing_set: str = 'average'):
        self.telluric_file = telluric_file
        self.sky_path = sky_path
        self.pwv = pwv
        self.seeing_set = seeing_set

        # derived state, set by load()
        self.airmass: Optional[float] = None
        self.v: Optional[np.ndarray] = None
        self.s: Optional[np.ndarray] = None
        self.h2o: Optional[np.ndarray] = None
        self.rayleigh: Optional[np.ndarray] = None
        self.o3: Optional[np.ndarray] = None
        self.o2: Optional[np.ndarray] = None
        self.n2: Optional[np.ndarray] = None
        self.co: Optional[np.ndarray] = None
        self.ch4: Optional[np.ndarray] = None
        self.co2: Optional[np.ndarray] = None
        self.n2o: Optional[np.ndarray] = None
        self.seeing: Optional[float] = None
        self.sky_bg: Optional[np.ndarray] = None  # sky background surface brightness [ph/s/arcsec^2/nm/m^2], on self.v

    def load(self, x: np.ndarray, zenith_angle: float) -> "Atmosphere":
        """
        Compute airmass from zenith_angle, load+scale the telluric
        transmission model onto x (see load_telluric_transmission), load the
        sky background model onto x (see load_sky_background), and map
        seeing_set to a numeric seeing [arcsec] via SEEING_MAP.

        Returns self, so calls can be chained: Atmosphere(...).load(x, zenith_angle).
        """
        self.airmass = 1 / np.cos(np.pi * zenith_angle / 180.)
        self.v = x

        # Load telluric transmission (total + per-species) from TAPAS file
        comps = load_telluric_transmission(x, self.telluric_file, self.pwv, self.airmass)
        self.h2o, self.rayleigh, self.o3 = comps['h2o'], comps['rayleigh'], comps['o3']
        self.o2, self.n2, self.co = comps['o2'], comps['n2'], comps['co']
        self.ch4, self.co2, self.n2o = comps['ch4'], comps['co2'], comps['n2o']
        self.s = comps['s']

        # Load Sky Background
        self.sky_bg = load_sky_background(x, self.pwv, self.airmass, self.sky_path)

        # Set the seeing
        if self.seeing_set in SEEING_MAP:
            self.seeing = SEEING_MAP[self.seeing_set]
        else:
            print('seeing_set must be good, average, or bad')

        return self
