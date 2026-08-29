##############################################################
# Star: owns a star's parameters plus the spectrum derived from them
###############################################################
#
# Replaces the orchestration that used to live in
# source_tools.load_stellar_model(x, mag, teff, vsini, so, rv), which took
# a `so` storage object and reached into so.filt/so.stel itself. Here the
# Filter dependency is passed explicitly to load(), and the derived state
# (loaded spectrum, scale factor, which model grid was used) lives on the
# instance instead of being unpacked into so.stel.* by the caller.
#
# The underlying stateless helpers (load_phoenix, load_sonora, load_filter,
# scale_stellar, _lsf_rotate) are unchanged and still live in
# source_tools.py -- they don't need `self` and are reused here rather than
# duplicated.

from dataclasses import dataclass, replace
from typing import Optional, Literal

import numpy as np
from scipy import interpolate
from astropy.convolution import convolve

from specsim import source_tools
from specsim.bandpass import Bandpass
from specsim.functions import integrate

SPEEDOFLIGHT = 2.998e8  # m/s
PHOT_PER_S_M2_PER_JY = 1.51e7  # http://astroweb.case.edu/ssm/ASTR620/mags.html
WAV_MARGIN_NM = 5  # nm; buffer added past a filter's exact bounds when picking a
                    # model-grid load range, since load_phoenix/load_sonora clip
                    # with a strict inequality and never return a point exactly
                    # at the requested wav_start/wav_end


@dataclass
class StarParams:
    "User-facing inputs for a single star's spectrum (on-axis star, companion, or AO reference star all use this)."
    teff: float = 3600          # K
    mag: float = 10             # mag, in the Filter's bandpass passed to load()
    vsini: float = 0            # km/s
    rv: float = 0                # km/s, used to Doppler-shift the spectrum (e.g. to offset from tellurics)
    logg: float = 4.5            # used for PHOENIX models only
    phoenix_folder: Optional[str] = None   # required if teff >= 2300
    sonora_folder: Optional[str] = None    # required if teff < 2300


class Star:
    """
    A star's spectrum: model selection (PHOENIX/Sonora), magnitude scaling,
    rotational broadening, and RV shift, plus the resulting arrays.
    """

    def __init__(self, params: StarParams):
        self.params = params

        # derived state, set by load()
        self.v: Optional[np.ndarray] = None            # wavelength grid the spectrum is interpolated onto [nm]
        self.s: Optional[np.ndarray] = None             # scaled/broadened/shifted spectrum [phot/s/m2/nm]
        self.vraw: Optional[np.ndarray] = None           # raw model wavelength grid [nm]
        self.sraw: Optional[np.ndarray] = None           # raw model flux [phot/m2/s/nm]
        self.model: Optional[Literal['phoenix', 'sonora']] = None
        self.stel_file: Optional[str] = None
        self.factor_0: Optional[float] = None            # scale factor applied to sraw to match params.mag in filt

    def load(self, x: np.ndarray, filt: Bandpass, rv: Optional[float] = None) -> "Star":
        """
        Load the model grid, scale to this star's magnitude in `filt`,
        rotationally broaden by vsini, apply an rv Doppler shift, and
        interpolate the result onto x. Mirrors
        source_tools.load_stellar_model(x, mag, teff, vsini, so, rv), with
        `filt` -- a Bandpass (see specsim.bandpass), or anything duck-typed
        the same way, e.g. so.filt -- passed explicitly instead of reached
        into via `so`.

        Returns self, so calls can be chained: Star(params).load(x, filt).
        """
        p = self.params
        rv = p.rv if rv is None else rv

        l0 = min(np.min(x), np.min(filt.xraw) - WAV_MARGIN_NM)
        l1 = max(np.max(x), np.max(filt.xraw) + WAV_MARGIN_NM)

        self._load_model_grid(p.teff, p.logg, l0, l1)
        self.factor_0 = source_tools.scale_stellar(filt, self.vraw, self.sraw, p.mag)

        tck_stel = interpolate.splrep(self.vraw, self.sraw, k=2, s=0)
        s = self.factor_0 * interpolate.splev(x, tck_stel, der=0, ext=1)

        if p.vsini > 0:
            s = self._broaden(x, s, p.vsini)

        if rv != 0:
            s = self._doppler_shift(x, s, rv)

        s[s < 0] = 0  # interpolation artifacts
        self.v, self.s = x, s
        return self

    def rescaled(self, mag: float, filt: Optional[Bandpass] = None) -> "Star":
        """
        Return a new Star reusing this star's already-loaded model grid
        (no reload of the PHOENIX/Sonora file, unless `filt` is given and
        its bandpass isn't already covered by the loaded grid), rescaled
        to a different magnitude. No .v/.s are set on the result -- only
        vraw/sraw/model/stel_file/factor_0, which is all
        magnitude_in_band() needs. Used for an AO reference star assumed
        to share the on-axis star's Teff but not necessarily its
        magnitude or the band that magnitude is quoted in.

        If `filt` is None, `mag` is assumed to be in the same band this
        star's factor_0 was already scaled to, and the rescale is a plain
        ratio: factor_0 * 10**(0.4*(old_mag - new_mag)).

        If `filt` is given (a different bandpass than the original
        scaling), factor_0 is instead recomputed via
        source_tools.scale_stellar(filt, vraw, sraw, mag) -- real
        synthetic photometry in the new band -- reloading the model grid
        first (over the union of the old range and filt's range) if
        filt's wavelength range isn't already covered.
        """
        if self.factor_0 is None:
            raise RuntimeError("call load() before rescaled()")
        new_star = Star(replace(self.params, mag=mag))
        new_star.model, new_star.stel_file = self.model, self.stel_file

        if filt is None:
            new_star.vraw, new_star.sraw = self.vraw, self.sraw
            new_star.factor_0 = self.factor_0 * 10 ** (0.4 * (self.params.mag - mag))
            return new_star

        vraw, sraw = self.vraw, self.sraw
        if (np.min(filt.xraw) < np.min(vraw)) or (np.max(filt.xraw) > np.max(vraw)):
            wav_start = min(np.min(filt.xraw), np.min(vraw)) - WAV_MARGIN_NM
            wav_end = max(np.max(filt.xraw), np.max(vraw)) + WAV_MARGIN_NM
            if self.model == 'phoenix':
                vraw, sraw = source_tools.load_phoenix(self.stel_file, self.params.phoenix_folder,
                                                         wav_start=wav_start, wav_end=wav_end)
            else:
                vraw, sraw = source_tools.load_sonora(self.stel_file, wav_start=wav_start, wav_end=wav_end)

        new_star.vraw, new_star.sraw = vraw, sraw
        new_star.factor_0 = source_tools.scale_stellar(filt, vraw, sraw, mag)
        return new_star

    def magnitude_in_band(self, bandpass: Bandpass) -> float:
        """
        Apparent magnitude of this (already-loaded) star in a different
        photometric filter band, given as a Bandpass (see
        specsim.bandpass). Mirrors source_tools.get_band_mag, reading
        vraw/sraw/model/stel_file/factor_0 off self instead of needing them
        passed in explicitly alongside a `so`.
        """
        if self.factor_0 is None:
            raise RuntimeError("call load() (or rescaled()) before magnitude_in_band()")

        filt_interp = bandpass.interp()

        vraw, sraw = self.vraw, self.sraw
        if (np.min(bandpass.xraw) < np.min(vraw)) or (np.max(bandpass.xraw) > np.max(vraw)):
            wav_start = np.min(bandpass.xraw) - WAV_MARGIN_NM
            wav_end = np.max(bandpass.xraw) + WAV_MARGIN_NM
            if self.model == 'phoenix':
                vraw, sraw = source_tools.load_phoenix(self.stel_file, self.params.phoenix_folder,
                                                         wav_start=wav_start, wav_end=wav_end)
            elif self.model == 'sonora':
                vraw, sraw = source_tools.load_sonora(self.stel_file,
                                                        wav_start=wav_start, wav_end=wav_end)

        filtered = self.factor_0 * sraw * filt_interp(vraw)
        flux = integrate(vraw, filtered)
        flux_Jy = flux / PHOT_PER_S_M2_PER_JY / bandpass.dl_l

        return -2.5 * np.log10(flux_Jy / bandpass.zp)

    def _load_model_grid(self, teff, logg, wav_start, wav_end):
        "Pick Sonora (teff < 2300K) vs PHOENIX and set vraw/sraw/model/stel_file."
        if teff < 2300:
            g = '316'  # mks units, log10(316*100)=4.5, matches phoenix logg convention used below
            self.stel_file = self.params.sonora_folder + 'sp_t%sg%snc_m0.0' % (int(teff), g)
            self.vraw, self.sraw = source_tools.load_sonora(self.stel_file, wav_start=wav_start, wav_end=wav_end)
            self.model = 'sonora'
        else:
            teff_str = str(int(teff)).zfill(5)
            logg_str = '{:.2f}'.format(logg)
            self.stel_file = 'lte%s-%s-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits' % (teff_str, logg_str)
            self.vraw, self.sraw = source_tools.load_phoenix(self.stel_file, self.params.phoenix_folder,
                                                               wav_start=wav_start, wav_end=wav_end)
            self.model = 'phoenix'

    def _broaden(self, x, s, vsini):
        dwvl_mean = np.abs(np.nanmean(np.diff(x)))
        dvel_mean = (dwvl_mean / np.nanmean(x)) * SPEEDOFLIGHT / 1e3
        kernel, _ = source_tools._lsf_rotate(dvel_mean, vsini, epsilon=0.6)
        return convolve(s, kernel, normalize_kernel=True)

    def _doppler_shift(self, x, s, rv):
        doppler_factor = 1.0 + (rv * 1000) / SPEEDOFLIGHT
        tck = interpolate.splrep(x * doppler_factor, s, k=3, s=0)
        return interpolate.splev(x, tck, der=0, ext=1)
