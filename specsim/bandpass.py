##############################################################
# Bandpass: a photometric filter curve loaded from disk, plus its
# zeropoint and dlambda/lambda, optionally resampled onto a shared
# wavelength grid.
###############################################################
#
# Replaces the ad-hoc SimpleNamespace/duck-typed "filt" objects that used
# to get built inline wherever a magnitude needed to be interpreted in a
# band other than so.filt's (the AO star's mag_band, or a WFE mode's
# native band) -- one object, one loader, cached so the same (family,
# band) requested repeatedly (e.g. once per AO mode) doesn't re-read the
# filter curve or zeropoint table from disk each time.
#
# The disk readers (load_zp_table, get_zp, load_filter) are module-level
# functions, like load_phoenix in star.py and load_telluric_transmission in
# atmosphere.py. The band->family convention (family_for_band) stays on the
# class since it reads nothing. They all used to sit in source_tools.py,
# whose stellar-model half now lives in star.py.
#
# Star.load()/rescaled()/magnitude_in_band() accept these (or anything
# duck-typed the same way, e.g. so.filt, a FILTER instance -- see
# objects.py) wherever a bandpass is needed.

import glob
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np
from scipy import interpolate

from specsim.functions import integrate


# yJHK wavelength band edges [nm], used by AOSystem (dichroic tophat) and plotting.
YJHK = {
    'y': [980, 1100],
    'J': [1170, 1327],
    'H': [1490, 1780],
    'K': [1990, 2460],
}


@lru_cache(maxsize=None)
def load_zp_table(zp_file):
    "Cached read of a zeropoint lookup table -- avoids re-reading (and re-warning on) the same file every call."
    return np.loadtxt(zp_file, dtype=str).T


def get_zp(zp_file, family, band):
    """
    Zeropoint flux [Jy] for a given filter family/band, read from zp_file
    (family, band, zp columns). The file itself is cached across calls, so
    looking this up repeatedly (e.g. once per AO mode) doesn't re-read (or
    re-warn on) it from disk each time.
    """
    zps = load_zp_table(zp_file)
    izp = np.where((zps[0] == family) & (zps[1] == band))[0]
    return float(zps[2][izp][0])


@lru_cache(maxsize=None)
def load_filter(filter_path, zp_file, family, band):
    """
    Load a photometric filter transmission curve from disk, together with
    everything derivable from it alone: its zeropoint, its dlambda/lambda,
    and its transmission-weighted center wavelength.

    Searches filter_path for a file matching '*<family>*<band>.dat' and
    loads its two columns as wavelength [nm] and transmission. Filter
    files store wavelength in different units depending on family
    (e.g. Angstrom for Johnson, micron for 2MASS/cfht/decam), so the
    units are auto-detected from the mean of the wavelength column:
    mean > 3000 is assumed to be Angstrom (divided by 10 to get nm),
    mean < 10 is assumed to be micron (multiplied by 1000 to get nm).

    Cached on (filter_path, zp_file, family, band): this is the expensive,
    x-grid-independent half of building a Bandpass, so requesting the same
    band again (e.g. once per candidate AO mode) reuses the read rather
    than hitting disk. Resampling onto a particular x grid is not cached --
    see Bandpass.resample().

    inputs:
    -------
    filter_path - str
        directory to search for the filter file

    zp_file - str
        path to the zeropoint table (family, band, zp columns)

    family - str
        filter family name (e.g. 'Johnson', 'SLOAN'), matched as a
        substring of the filter filename

    band - str
        filter band name (e.g. 'V', 'uprime_filter'), matched as a
        substring of the filter filename

    returns:
    --------
    xraw - array
        filter wavelength grid [nm]

    yraw - array
        filter transmission, unitless (out of 1)

    zp - float
        zeropoint flux at m=0 [Jy]

    dl_l - float
        delta lambda over lambda for the passband, unitless

    center_wavelength - float
        transmission-weighted center wavelength [nm]
    """
    filter_file    = glob.glob(filter_path + '*' + family + '*' + band + '.dat')[0]
    xraw, yraw     = np.loadtxt(filter_file).T # units vary by file (Angstrom or micron)
    if np.mean(xraw) > 3000: xraw = xraw / 10    # Angstrom -> nm
    if np.mean(xraw) < 10: xraw = xraw * 1000    # micron -> nm

    zp = get_zp(zp_file, family, band)
    dl_l = np.mean(integrate(xraw, yraw) / xraw)
    center_wavelength = integrate(xraw, yraw * xraw) / integrate(xraw, yraw)
    return xraw, yraw, zp, dl_l, center_wavelength


@dataclass
class Bandpass:
    """
    A photometric filter bandpass: raw transmission curve (xraw/yraw),
    zeropoint flux [Jy], dlambda/lambda, and transmission-weighted center
    wavelength -- everything a Star's scale_stellar()/calc_nphot()/
    magnitude_in_band() need -- plus, once resampled, the transmission
    on a shared wavelength grid (x/y).
    """
    family: str
    band: str
    xraw: np.ndarray
    yraw: np.ndarray
    zp: float
    dl_l: float
    center_wavelength: float
    x: Optional[np.ndarray] = field(default=None, repr=False)
    y: Optional[np.ndarray] = field(default=None, repr=False)

    @staticmethod
    def family_for_band(band):
        """
        Photometric filter family conventionally used for a given band, so
        callers don't have to specify/guess it alongside the band: 2MASS for
        J/H/K, CFHT for y, Johnson otherwise (U/B/V/R/I).
        """
        if band in ('J', 'H', 'K'):
            return '2mass'
        if band == 'y':
            return 'cfht'
        return 'Johnson'

    @classmethod
    def load(cls, filter_path: str, zp_file: str, band: str, family: Optional[str] = None,
             x: Optional[np.ndarray] = None) -> "Bandpass":
        """
        Build a Bandpass for band from its filter curve/zeropoint/dl_l/
        center_wavelength (see load_filter(), which caches the disk read).
        If x is given, also resamples the transmission onto it (see
        .resample()) -- this step always runs fresh, so a stale resample
        from a previous x grid is never returned.

        family is derived from band via family_for_band() (2MASS for J/H/K,
        CFHT for y, Johnson otherwise), so callers never need to supply it.
        Pass it explicitly only for a band whose conventional family is not
        the one you want -- e.g. the SLOAN uprime_filter, decam y, or TESS
        curves in data/filters/.
        """
        if family is None:
            family = cls.family_for_band(band)
        xraw, yraw, zp, dl_l, center_wavelength = load_filter(filter_path, zp_file, family, band)
        bp = cls(family=family, band=band, xraw=xraw, yraw=yraw, zp=zp, dl_l=dl_l, center_wavelength=center_wavelength)
        if x is not None:
            bp.resample(x)
        return bp

    def resample(self, x: np.ndarray) -> "Bandpass":
        "Interpolate the transmission curve onto x."
        self.x, self.y = x, self.interp()(x)
        return self

    def interp(self):
        "Interpolating function over the raw transmission curve, evaluable at any wavelength grid (e.g. a star's raw model grid)."
        return interpolate.interp1d(self.xraw, self.yraw, bounds_error=False, fill_value=0)
