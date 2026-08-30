##############################################################
# Simulate: top-level entry point tying Star/Bandpass/Atmosphere/AOSystem/
# Spectrograph/Observation together into one user-facing object
###############################################################
#
# Replaces the fill_data(so)-then-plot(so) pattern: the user builds a
# Simulate from typed domain objects (StarParams, Spectrograph, Atmosphere,
# AOSystem, ...) and calls .snr()/.rv_precision()/.ccf_snr()/
# .exposure_time_for_snr() to get results, with no `so` anywhere. See
# simulate_from_config() (specsim/config.py) for building one from the
# existing flat .cfg files. Telescope area/diameter live on Spectrograph
# (see specsim/instrument.py) rather than a separate Telescope object.

from dataclasses import replace
from typing import Optional

import numpy as np

from specsim.analyze import Analyze
from specsim.atmosphere import Atmosphere
from specsim.bandpass import Bandpass, YJHK
from specsim.instrument import AOSystem, Spectrograph, TrackingCamera
from specsim.observation import Observation
from specsim.star import Star, StarParams


class Simulate:
    """
    Builds the scene (Bandpass, Star(s), Atmosphere, AOSystem, Spectrograph)
    from user inputs, then exposes SNR/RV-precision/CCF-SNR/exposure-time
    calculations as methods. An Observation is built lazily on first use
    and cached; call one of the set_* methods to change an input and
    invalidate only the cached objects that actually depend on it.
    """

    def __init__(self, *, star: StarParams, spectrograph: Spectrograph, atmosphere: Atmosphere,
                 ao_system: AOSystem,
                 filt_band: str = 'H', filt_family: Optional[str] = None,
                 filter_path: str, zp_file: str,
                 texp: float = 900, texp_frame_set='default', nsamp: int = 1, zenith_angle: float = 45,
                 companion: Optional[StarParams] = None, pl_sep: float = 0,
                 tracking_camera: Optional[TrackingCamera] = None):
        self.spectrograph = spectrograph
        self.atmosphere = atmosphere
        self.ao_system = ao_system
        self.filt_family = filt_family
        self.filt_band = filt_band
        self.filter_path = filter_path
        self.zp_file = zp_file
        self.texp = texp
        self.texp_frame_set = texp_frame_set
        self.nsamp = nsamp
        self.zenith_angle = zenith_angle
        self.pl_sep = pl_sep
        self.tracking_camera_config = tracking_camera

        self.x = np.arange(spectrograph.l0, spectrograph.l1, 0.0005)
        self.filt = Bandpass.load(filter_path, zp_file, filt_band, filt_family, x=self.x)
        self.star = Star(star).load(self.x, self.filt)
        self.companion = Star(companion).load(self.x, self.filt) if companion is not None else None

        self.atmosphere.load(self.x, self.zenith_angle)
        self.ao_system.select(self.x, self.star, self.filt, self.filter_path, self.zp_file,
                               self.spectrograph, self.zenith_angle, self.atmosphere.seeing_set, YJHK)
        self.spectrograph.load(self.x, self.ao_system)

        self.tracking_camera: Optional[TrackingCamera] = None
        self._observation: Optional[Observation] = None

    def _get_observation(self) -> Observation:
        if self._observation is None:
            self._observation = Observation(
                self.star, self.spectrograph, self.atmosphere, self.ao_system,
                texp=self.texp, texp_frame_set=self.texp_frame_set, nsamp=self.nsamp,
                zenith_angle=self.zenith_angle, companion=self.companion, pl_sep=self.pl_sep,
            ).run(self.x)
        return self._observation

    def snr(self) -> Observation:
        "Return the Observation (per-pixel/per-resolution-element/per-order SNR), computing it on first call."
        return self._get_observation()

    @property
    def analysis(self) -> Analyze:
        "Analyze bound to the current Observation and scene. Rebuilt on each access (it only stores references); the expensive Observation underneath is still cached."
        return Analyze(self._get_observation(), self.spectrograph, self.atmosphere, self.star)

    def rv_precision(self, telluric_cutoff: float = 0.01, velocity_cutoff: float = 30):
        "Achievable RV precision (analyze.RVPrecisionResult) for the current scene."
        return self.analysis.rv_precision(telluric_cutoff=telluric_cutoff, velocity_cutoff=velocity_cutoff)

    def ccf_snr(self, model=None, systematics_residuals: float = 0.01, kernel_size: int = 201, norm_cutoff: float = 0.95):
        "Matched-filter CCF SNR (analyze.CCFSNRResult), full spectrum and per yJHK band."
        return self.analysis.ccf_snr(model=model, systematics_residuals=systematics_residuals,
                                      kernel_size=kernel_size, norm_cutoff=norm_cutoff)

    def exposure_time_for_snr(self, target_snr: float):
        "Exposure time (analyze.ETCResult) needed to reach target_snr, per pixel/resolution-element and per order."
        return self.analysis.exposure_time_for_snr(target_snr)

    def exposure_time_for_ccf_snr(self, goal_ccf: float, systematics_residuals: float = 0.01,
                                   kernel_size: int = 201, norm_cutoff: float = 0.95):
        "Exposure time (dict of {'y','J','H','K': seconds}) needed to reach goal_ccf CCF SNR in each band."
        return self.analysis.exposure_time_for_ccf_snr(goal_ccf, systematics_residuals=systematics_residuals,
                                                        kernel_size=kernel_size, norm_cutoff=norm_cutoff)

    def tracking(self) -> TrackingCamera:
        "Return the TrackingCamera observation, computing it on first call. Raises if no tracking_camera was configured."
        if self.tracking_camera_config is None:
            raise ValueError("no tracking_camera was passed to Simulate()")
        if self.tracking_camera is None:
            self.tracking_camera = self.tracking_camera_config
            self.tracking_camera.observe(self.x, self.star, self.atmosphere, self.ao_system, self.spectrograph)
        return self.tracking_camera

    def set_star_mag(self, mag: float):
        "Reload the on-axis star at a new magnitude (same band), then re-select the AO mode and reload the spectrograph coupling, since both can depend on the science star. Invalidates the cached Observation/tracking. (Uses Star.load(), not Star.rescaled() -- rescaled() skips setting .v/.s, which Observation needs.)"
        self.star = Star(replace(self.star.params, mag=mag)).load(self.x, self.filt)
        self.ao_system.select(self.x, self.star, self.filt, self.filter_path, self.zp_file,
                               self.spectrograph, self.zenith_angle, self.atmosphere.seeing_set, YJHK)
        self.spectrograph.load(self.x, self.ao_system)
        self._observation = None
        self.tracking_camera = None

    def set_star_teff(self, teff: float):
        "Reload the on-axis star at a new effective temperature (same magnitude/band), then re-select the AO mode and reload the spectrograph coupling. Teff changes the star's colour, so its magnitude in the AO mode's native band -- and hence the WFE and coupling -- changes too. Invalidates the cached Observation/tracking. Requires a model grid file for the requested teff (PHOENIX for teff >= 2300K, Sonora below)."
        self.star = Star(replace(self.star.params, teff=teff)).load(self.x, self.filt)
        self.ao_system.select(self.x, self.star, self.filt, self.filter_path, self.zp_file,
                               self.spectrograph, self.zenith_angle, self.atmosphere.seeing_set, YJHK)
        self.spectrograph.load(self.x, self.ao_system)
        self._observation = None
        self.tracking_camera = None

    def set_ao_mode(self, mode: str):
        "Change the AO mode and reload the spectrograph coupling (which depends on the chosen mode's ho_wfe/tt_dynamic). Invalidates the cached Observation/tracking."
        self.ao_system.mode = mode
        self.ao_system.select(self.x, self.star, self.filt, self.filter_path, self.zp_file,
                               self.spectrograph, self.zenith_angle, self.atmosphere.seeing_set, YJHK)
        self.spectrograph.load(self.x, self.ao_system)
        self._observation = None
        self.tracking_camera = None

    def set_texp(self, texp: float):
        "Change the total exposure time. Invalidates only the cached Observation -- cheapest setter, no AO/spectrograph recompute."
        self.texp = texp
        self._observation = None
