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
# Star.load()/rescaled()/magnitude_in_band() accept these (or anything
# duck-typed the same way, e.g. so.filt, a FILTER instance -- see
# objects.py) wherever a bandpass is needed.

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np
from scipy import interpolate

from specsim import source_tools
from specsim.functions import integrate


@lru_cache(maxsize=None)
def _load_raw(filter_path, zp_file, family, band):
    """
    Cached disk read of a filter curve + everything derivable from it alone
    (zeropoint, dl_l, center_wavelength) -- the expensive, x-grid-independent
    part, computed once per (family, band) no matter how many times it's
    requested (e.g. once per candidate AO mode) or resampled onto different x.
    """
    xraw, yraw = source_tools.load_filter(filter_path, family, band)
    zp = source_tools.get_zp(zp_file, family, band)
    dl_l = np.mean(integrate(xraw, yraw) / xraw)
    center_wavelength = integrate(xraw, yraw * xraw) / integrate(xraw, yraw)
    return xraw, yraw, zp, dl_l, center_wavelength


@dataclass
class Bandpass:
    """
    A photometric filter bandpass: raw transmission curve (xraw/yraw),
    zeropoint flux [Jy], dlambda/lambda, and transmission-weighted center
    wavelength -- everything source_tools.scale_stellar()/calc_nphot() and
    Star.magnitude_in_band() need -- plus, once resampled, the transmission
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

    @classmethod
    def load(cls, filter_path: str, zp_file: str, family: str, band: str, x: Optional[np.ndarray] = None) -> "Bandpass":
        """
        Load (family, band)'s filter curve/zeropoint/dl_l/center_wavelength
        -- cached by (filter_path, zp_file, family, band), so requesting the
        same band again (e.g. once per candidate AO mode) reuses the cached
        read instead of hitting disk again. If x is given, also resamples
        the transmission onto it (see .resample()) -- this step always runs
        fresh (not cached), so a stale resample from a previous x grid is
        never returned.
        """
        xraw, yraw, zp, dl_l, center_wavelength = _load_raw(filter_path, zp_file, family, band)
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
