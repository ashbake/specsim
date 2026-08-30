##############################################################
# Star: owns a star's parameters plus the spectrum derived from them
###############################################################
#
# Replaces the orchestration that used to live in the old
# source_tools.load_stellar_model(x, mag, teff, vsini, so, rv), which took
# a `so` storage object and reached into so.filt/so.stel itself. Here the
# Filter dependency is passed explicitly to load(), and the derived state
# (loaded spectrum, scale factor, which model grid was used) lives on the
# instance instead of being unpacked into so.stel.* by the caller.
#
# The model-grid readers (load_phoenix, load_sonora) stay module-level
# functions below -- they're pure file IO and are useful without a Star.
# calc_nphot/scale_stellar are methods, since both read the star's own
# magnitude (and, for scale_stellar, its loaded vraw/sraw) and take the
# bandpass as an argument, exactly like magnitude_in_band. The rotation
# kernel is a pure math helper with no star state, so it lives in
# functions.py as lsf_rotate() alongside define_lsf(). All of these used
# to live in source_tools.py, which no longer exists. The filter-side
# helpers (load_filter/get_zp/family_for_band) live on Bandpass.

import os
from dataclasses import dataclass, replace
from typing import Optional, Literal

import numpy as np
from scipy import interpolate
from astropy.io import fits
from astropy.convolution import convolve

from specsim.bandpass import Bandpass
from specsim.functions import SPEEDOFLIGHT, integrate, lsf_rotate

PHOT_PER_S_M2_PER_JY = 1.51e7  # http://astroweb.case.edu/ssm/ASTR620/mags.html
WAV_MARGIN_NM = 5  # nm; buffer added past a filter's exact bounds when picking a
                    # model-grid load range, since load_phoenix/load_sonora clip
                    # with a strict inequality and never return a point exactly
                    # at the requested wav_start/wav_end


def load_phoenix(stelname,stelpath,wav_start=750,wav_end=780):
    """
    Load a PHOENIX stellar spectrum fits file and return the requested
    wavelength subarray, converted to photon flux units.

    http://phoenix.astro.physik.uni-goettingen.de/?page_id=15

    Converts flux from ergs/s/cm2/cm to phot/cm2/s/nm using
    https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf
    then to phot/m2/s/nm.

    inputs:
    -------
    stelname - str
        filename of the PHOENIX spectrum fits file (flux only; the
        wavelength grid is loaded separately from
        'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits' in stelpath)

    stelpath - str
        directory path containing stelname and the PHOENIX wavelength file

    wav_start - float
        lower wavelength bound of the returned subarray [nm] (default 750)

    wav_end - float
        upper wavelength bound of the returned subarray [nm] (default 780)

    returns:
    --------
    wavelength - array
        wavelength subarray [nm]

    flux - array
        photon flux subarray [phot/m2/s/nm]
    """
    # conversion factor

    f = fits.open(stelpath + stelname)
    spec = f[0].data / (1e8) # ergs/s/cm2/cm to ergs/s/cm2/Angstrom for conversion
    f.close()
    
    wave_file = os.path.join(stelpath + 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits') #assume wave in same folder
    f = fits.open(wave_file)
    lam = f[0].data # angstroms
    f.close()
    
    # Convert
    conversion_factor = 5.03*10**7 * lam #lam in angstrom here
    spec *= conversion_factor # phot/cm2/s/angstrom
    
    # Take subarray requested
    isub = np.where((lam > wav_start*10.0) & (lam < wav_end*10.0))[0]

    # Convert 
    return lam[isub]/10.0,spec[isub] * 10 * 100**2 #nm, phot/m2/s/nm


def load_sonora(stelname,wav_start=750,wav_end=780):
    """
    Load a Sonora stellar/substellar atmosphere model file and return the
    requested wavelength subarray, converted to photon flux units.

    The file's wavelength column is loaded in microns, high to low, and is
    reversed and converted to Angstroms internally. Flux is converted from
    erg/cm2/s/Hz to erg/cm2/s/Angstrom (via c/lambda^2) and then to
    phot/cm2/s/Angstrom using
    https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf, before being
    returned as phot/m2/s/nm.

    inputs:
    -------
    stelname - str
        path to the Sonora model file (whitespace-delimited, 2 header rows,
        columns: wavelength [micron], flux [erg/cm2/s/Hz])

    wav_start - float
        lower wavelength bound of the returned subarray [nm] (default 750)

    wav_end - float
        upper wavelength bound of the returned subarray [nm] (default 780)

    returns:
    --------
    wavelength - array
        wavelength subarray [nm]

    flux - array
        photon flux subarray [phot/m2/s/nm]
    """
    f = np.loadtxt(stelname,skiprows=2)

    lam  = 10000* f[:,0][::-1] #microns to angstroms, needed for conversiosn
    spec = f[:,1][::-1] # erg/cm2/s/Hz
    
    spec *= 3e18/(lam**2)# convert spec to erg/cm2/s/angstrom
    
    conversion_factor = 5.03*10**7 * lam #lam in angstrom here
    spec *= conversion_factor # phot/cm2/s/angstrom
    
    isub = np.where( (lam > wav_start*10.0) & (lam < wav_end*10.0))[0]

    return lam[isub]/10.0,spec[isub] * 10 * 100**2 #nm, phot/m2/s/nm (my fave)


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
        the old source_tools.load_stellar_model(x, mag, teff, vsini, so, rv), with
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
        self.factor_0 = self.scale_stellar(filt, p.mag)

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
        scaling), factor_0 is instead recomputed via the new star's
        scale_stellar(filt, mag) -- real synthetic photometry in the new
        band -- reloading the model grid first (over the union of the old
        range and filt's range) if filt's wavelength range isn't already
        covered.
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
                vraw, sraw = load_phoenix(self.stel_file, self.params.phoenix_folder,
                                                         wav_start=wav_start, wav_end=wav_end)
            else:
                vraw, sraw = load_sonora(self.stel_file, wav_start=wav_start, wav_end=wav_end)

        new_star.vraw, new_star.sraw = vraw, sraw
        new_star.factor_0 = new_star.scale_stellar(filt, mag)
        return new_star

    def magnitude_in_band(self, bandpass: Bandpass) -> float:
        """
        Apparent magnitude of this (already-loaded) star in a different
        photometric filter band, given as a Bandpass (see
        specsim.bandpass). Mirrors the old source_tools.get_band_mag, reading
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
                vraw, sraw = load_phoenix(self.stel_file, self.params.phoenix_folder,
                                                         wav_start=wav_start, wav_end=wav_end)
            elif self.model == 'sonora':
                vraw, sraw = load_sonora(self.stel_file,
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
            self.vraw, self.sraw = load_sonora(self.stel_file, wav_start=wav_start, wav_end=wav_end)
            self.model = 'sonora'
        else:
            teff_str = str(int(teff)).zfill(5)
            logg_str = '{:.2f}'.format(logg)
            self.stel_file = 'lte%s-%s-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits' % (teff_str, logg_str)
            self.vraw, self.sraw = load_phoenix(self.stel_file, self.params.phoenix_folder,
                                                               wav_start=wav_start, wav_end=wav_end)
            self.model = 'phoenix'

    def _broaden(self, x, s, vsini):
        dwvl_mean = np.abs(np.nanmean(np.diff(x)))
        dvel_mean = (dwvl_mean / np.nanmean(x)) * SPEEDOFLIGHT / 1e3
        kernel, _ = lsf_rotate(dvel_mean, vsini, epsilon=0.6)
        return convolve(s, kernel, normalize_kernel=True)

    def _doppler_shift(self, x, s, rv):
        doppler_factor = 1.0 + (rv * 1000) / SPEEDOFLIGHT
        tck = interpolate.splrep(x * doppler_factor, s, k=3, s=0)
        return interpolate.splev(x, tck, der=0, ext=1)

    def calc_nphot(self, bandpass: Bandpass, mag: Optional[float] = None) -> float:
        """
        Photon flux [phot/s/m2] this star would deliver at the top of Earth's
        atmosphere through `bandpass`, from its magnitude alone (no model
        spectrum involved). The exact inverse of magnitude_in_band(), and
        shaped the same way: the bandpass supplies dl_l/zp, the star supplies
        the magnitude.

        http://astroweb.case.edu/ssm/ASTR620/mags.html -- see the table there
        for dl_l/zp values for some standard bands.

        inputs:
        -------
        bandpass - Bandpass
            a specsim.bandpass.Bandpass (or anything duck-typed the same
            way); supplies dl_l (delta lambda over lambda, unitless) and zp
            (flux at m=0 in Jansky)

        mag - float, optional
            magnitude in bandpass. Defaults to this star's params.mag, which
            is what the scaling path wants; pass it explicitly to ask what a
            different magnitude would deliver through the same bandpass.

        outputs:
        --------
        photon flux: float, photons per second per square meter [phot/s/m2]
        at the top of Earth's atmosphere
        """
        mag = self.params.mag if mag is None else mag

        return bandpass.dl_l * bandpass.zp * 10 ** (-0.4 * mag) * PHOT_PER_S_M2_PER_JY

    def scale_stellar(self, filt: Bandpass, mag: Optional[float] = None) -> float:
        """
        Compute the scale factor needed to normalize this star's loaded model
        spectrum (vraw/sraw) so that its integrated flux through `filt`
        matches a given magnitude.

        Interpolates the filter transmission onto the stellar wavelength grid,
        integrates the filtered stellar spectrum, and compares it to the
        expected photon flux for the requested magnitude (via calc_nphot) to
        derive a single multiplicative scale factor. Raises a Warning if the
        stellar model does not fully cover the filter bandpass.

        inputs:
        -------
        filt - Bandpass
            a specsim.bandpass.Bandpass (or anything duck-typed the same way);
            must provide xraw/yraw (filter wavelength [nm] and transmission)
            as well as dl_l and zp attributes used by calc_nphot

        mag - float, optional
            desired magnitude of the star in the filter bandpass. Defaults to
            this star's params.mag.

        returns:
        --------
        factor - float
            multiplicative scale factor to apply to sraw so that its
            integrated flux through the filter matches mag
        """
        if self.vraw is None:
            raise RuntimeError("no model grid loaded -- call load() first")
        mag = self.params.mag if mag is None else mag

        if (np.min(filt.xraw) < np.min(self.vraw)) or (np.max(filt.xraw) > np.max(self.vraw)):
            raise Warning('Check that stellar model in scale_stellar extends past filter profile')

        filtered_stellar   = self.sraw * filt.interp()(self.vraw)   # filter profile resampled to phoenix times phoenix flux density
        nphot_expected_0   = self.calc_nphot(filt, mag)              # what's the integrated flux supposed to be in photons/m2/s?
        nphot_model        = integrate(self.vraw, filtered_stellar)  # what's the integrated flux now? in same units as ^

        return nphot_expected_0/nphot_model
