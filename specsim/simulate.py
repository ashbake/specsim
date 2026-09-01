##############################################################
# Simulate: top-level entry point tying Star/Bandpass/Atmosphere/AOSystem/
# Spectrograph/TrackingCamera together into one user-facing object
###############################################################
#
# Replaces the fill_data(so)-then-plot(so) pattern: the user builds a
# Simulate from typed domain objects (StarParams, Spectrograph, Atmosphere,
# AOSystem, ...) and calls .snr()/.rv_precision()/.ccf_snr()/
# .exposure_time_for_snr() to get results, with no `so` anywhere. See
# simulate_from_config() (specsim/config.py) for building one from the
# existing flat .cfg files. Telescope area/diameter live on Spectrograph
# (see specsim/spectrograph.py) rather than a separate Telescope object.

from dataclasses import replace
from typing import Optional

import numpy as np

from specsim.analyze import Analyze
from specsim.atmosphere import Atmosphere
from specsim.bandpass import Bandpass, YJHK
from specsim.aosystem import AOSystem
from specsim.spectrograph import Spectrograph
from specsim.trackingcamera import TrackingCamera
from specsim.star import Star, StarParams


class _Unset:
    "Type of the UNSET sentinel below."
    def __repr__(self):
        return '<unset>'


# Default for every Simulate.set_*() argument, marking one the caller didn't
# pass. A plain None default wouldn't do: None and 'default' are both real,
# distinct values here -- set_ao(ho_wfe=None) clears the WFE override, and
# set_ao(mag='default') goes back to inheriting the science star's magnitude.
UNSET = _Unset()


def _given(**kwargs):
    "Drop the arguments the caller didn't pass, keeping the real values (including None and 'default')."
    return {name: value for name, value in kwargs.items() if value is not UNSET}


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
                               self.zenith_angle, self.atmosphere.seeing_set, YJHK)
        self.spectrograph.load(self.x, self.ao_system)

        self.tracking_camera: Optional[TrackingCamera] = None
        self._observed = False

    # Read off the Bandpass rather than stored separately: set_star(band=...)
    # rebuilds self.filt, and a second copy of the band would silently drift
    # out of step with it.
    @property
    def filt_band(self) -> str:
        "Photometric band the science star's magnitude is defined in. Change it with set_star(band=...)."
        return self.filt.band

    @property
    def filt_family(self) -> str:
        "Filter family behind filt_band (e.g. '2mass' for H). Derived from the band unless one was set explicitly."
        return self.filt.family

    def _get_observation(self) -> Spectrograph:
        "Run the exposure on the spectrograph if it hasn't been run since the last input change, and return it."
        if not self._observed:
            self.spectrograph.observe(
                self.x, self.star, self.atmosphere, self.ao_system,
                texp=self.texp, texp_frame_set=self.texp_frame_set, nsamp=self.nsamp,
                zenith_angle=self.zenith_angle, companion=self.companion, pl_sep=self.pl_sep)
            self._observed = True
        return self.spectrograph

    def snr(self) -> Spectrograph:
        "Return the observed Spectrograph (per-pixel/per-resolution-element/per-order SNR on .snr/.snr_res_element/.snr_max_orders), computing the exposure on first call."
        return self._get_observation()

    @property
    def analysis(self) -> Analyze:
        "Analyze bound to the observed spectrograph and the rest of the scene. Rebuilt on each access (it only stores references); the exposure itself is only re-run when an input changes."
        return Analyze(self._get_observation(), self.atmosphere, self.star)

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
            self.tracking_camera.load(self.x, self.ao_system).observe(self.x, self.star, self.atmosphere)
        return self.tracking_camera

    # ---- setters: one per domain object, changing any subset of its inputs ----
    #
    # The scene builds in one direction -- star -> AO -> fiber coupling ->
    # exposure -- so each setter reloads its own object and everything
    # downstream of it, then marks the cached exposure stale. Arguments left
    # out are unchanged; passing several at once does the reload work once
    # instead of once per parameter.

    def _invalidate(self):
        "Mark the cached exposure and tracking observation stale, so the next snr()/tracking() recomputes them."
        self._observed = False
        self.tracking_camera = None

    def _reselect_ao(self):
        "Re-run AO mode selection and reload the fiber coupling that depends on its WFE, then invalidate. Every setter that touches something upstream of the AO ends here."
        self.ao_system.select(self.x, self.star, self.filt, self.filter_path, self.zp_file,
                               self.zenith_angle, self.atmosphere.seeing_set, YJHK)
        self.spectrograph.load(self.x, self.ao_system)
        self._invalidate()

    def set_star(self, *, mag=UNSET, teff=UNSET, vsini=UNSET, rv=UNSET, logg=UNSET,
                 phoenix_folder=UNSET, sonora_folder=UNSET,
                 band=UNSET, family=UNSET) -> "Simulate":
        """
        Change the on-axis star and reload its spectrum, then re-select the
        AO mode and reload the spectrograph coupling -- both depend on the
        science star, since its magnitude (and, through its colour, its
        Teff) sets the guide-star magnitude the WFE tables are sampled at.

        inputs (all optional; anything not passed is left unchanged)
        ------
        mag - float, apparent magnitude in the `band` below
        teff - float [K], needs a model grid file present (PHOENIX for
            teff >= 2300, Sonora below)
        vsini - float [km/s], rotational broadening
        rv - float [km/s], Doppler shift (e.g. to move lines off tellurics)
        logg - float, surface gravity; PHOENIX models only
        phoenix_folder, sonora_folder - str, where to read model grids
            from, if not the folders the config pointed at
        band - str, the photometric band `mag` is quoted in (e.g. 'K').
            Unlike the arguments above, this isn't a StarParams field --
            it rebuilds the scene's Bandpass. It lives here because a
            magnitude and the band it's quoted in are one statement, so
            set_star(mag=12, band='K') is a single reload rather than two.
        family - str, filter family for `band`, for the cases where the
            conventional one isn't wanted (e.g. 'decam' rather than 'cfht'
            for y). Passing `band` alone RESETS the family to the
            conventional one for that band; pass both to keep a
            non-conventional one.

        Changing the band REINTERPRETS the magnitude rather than
        colour-converting it: an H=10 star becomes a K=10 star, so its
        physical flux -- and the SNR -- change. Two knock-on effects worth
        knowing, both correct rather than bugs:
          - a companion, whose magnitude is quoted in the same band, is
            renormalised too;
          - [ao] mag_band='default' MEANS "the science band", so an AO
            guide magnitude left at default follows the band as well, and
            with mode='auto' a different AO mode can win (filt's center
            wavelength sets the Strehl, and the high-order and tip-tilt
            terms scale differently with wavelength).

        output
        ------
        self, so calls can be chained: sim.set_star(mag=12).snr()
        """
        if band is not UNSET or family is not UNSET:
            new_band = self.filt.band if band is UNSET else band
            # a new band re-derives the family, so switching away from an
            # explicit 'decam' y to K doesn't go looking for a 'decam K' curve
            new_family = family if family is not UNSET else (None if band is not UNSET else self.filt.family)
            if self.ao_system.mag_band == 'default' and self.ao_system.mag != 'default':
                print("WARNING: [ao] mag=%s was quoted in the '%s' band and mag_band is 'default', so it is now "
                      "being read as '%s'. Set [ao] mag_band to pin it." % (self.ao_system.mag, self.filt.band, new_band))
            # assign only once the load succeeds, so a bad band leaves the scene usable
            self.filt = Bandpass.load(self.filter_path, self.zp_file, new_band, new_family, x=self.x)
            if self.companion is not None:
                # its magnitude is quoted in the same band, so it renormalises too
                self.companion.load(self.x, self.filt)

        # Star.load(), not Star.rescaled() -- rescaled() skips setting .v/.s, which the exposure needs.
        # Unconditional and after the filter rebuild, so changing band and star params together is one reload.
        updates = _given(mag=mag, teff=teff, vsini=vsini, rv=rv, logg=logg,
                          phoenix_folder=phoenix_folder, sonora_folder=sonora_folder)
        self.star = Star(replace(self.star.params, **updates)).load(self.x, self.filt)
        self._reselect_ao()
        return self

    def set_filter(self, *, band=UNSET, family=UNSET) -> "Simulate":
        "Change the photometric band the science magnitude is defined in. Alias for set_star(band=..., family=...) -- see there for the semantics, which are not a colour conversion."
        return self.set_star(band=band, family=family)

    def set_ao(self, *, mode=UNSET, mag=UNSET, mag_band=UNSET, teff=UNSET,
               ho_wfe=UNSET, tt_dynamic=UNSET) -> "Simulate":
        """
        Change the AO system and re-run mode selection, then reload the
        spectrograph coupling (which depends on the resulting WFE).

        inputs (all optional; anything not passed is left unchanged)
        ------
        mode - str, 'auto' to pick the highest-Strehl mode, or a mode name
            from the instrument's WFE tables (e.g. 'NGS', 'LGS_ON')
        mag - float, guide-star magnitude, or 'default' to inherit the
            science star's
        mag_band - str, the band `mag` is quoted in (e.g. 'R'), or
            'default' to treat it as the science star's [filt] band.
            specsim colour-converts from here into whatever band the
            chosen mode's WFE table is indexed by.
        teff - float [K], assumed guide-star temperature, or 'default' to
            reuse the science star's model. Only affects that colour
            conversion; a new model grid is loaded when it isn't 'default'.
        ho_wfe - float [nm], pins the high-order WFE instead of reading it
            off the chosen mode. Must be passed together with tt_dynamic.
            Pass ho_wfe=None, tt_dynamic=None to go back to the mode lookup.
        tt_dynamic - float [mas], pins the dynamic tip-tilt residual.
            Keep this on the instrument's tabulated coupling grid (MODHIS
            ships 0-4.5 mas in 0.5 mas steps); off-grid values fail on the
            missing coupling file.

        output
        ------
        self, so calls can be chained: sim.set_ao(mode='NGS').snr()
        """
        for name, value in _given(mode=mode, mag=mag, mag_band=mag_band, teff=teff,
                                   ho_wfe_set=ho_wfe, tt_dynamic_set=tt_dynamic).items():
            setattr(self.ao_system, name, value)
        self._reselect_ao()
        return self

    def set_atmosphere(self, *, pwv=UNSET, seeing_set=UNSET, zenith_angle=UNSET) -> "Simulate":
        """
        Change the observing conditions and reload the atmosphere (telluric
        transmission and sky background, both scaled by airmass).

        Seeing and zenith angle also index the AO WFE tables, so changing
        either re-runs AO selection and reloads the coupling as well;
        changing pwv alone does not, since it doesn't reach the AO.

        inputs (all optional; anything not passed is left unchanged)
        ------
        pwv - float [mm], precipitable water vapour
        seeing_set - str, 'good', 'average' or 'bad'
        zenith_angle - float [deg], sets the airmass; the WFE tables are
            only tabulated at 0, 30, 45 and 60

        output
        ------
        self, so calls can be chained: sim.set_atmosphere(pwv=3).snr()
        """
        conditions = _given(pwv=pwv, seeing_set=seeing_set)
        for name, value in conditions.items():
            setattr(self.atmosphere, name, value)
        if zenith_angle is not UNSET:
            self.zenith_angle = zenith_angle
        self.atmosphere.load(self.x, self.zenith_angle)

        if 'seeing_set' in conditions or zenith_angle is not UNSET:
            self._reselect_ao()
        else:
            self._invalidate()  # pwv only reaches the exposure, not the AO
        return self

    def set_obs(self, *, texp=UNSET, texp_frame_set=UNSET, nsamp=UNSET) -> "Simulate":
        """
        Change the exposure. Cheapest setter: it only marks the cached
        exposure stale, with no star/AO/coupling recompute, and leaves the
        tracking camera alone (it has its own, independent texp).

        inputs (all optional; anything not passed is left unchanged)
        ------
        texp - float [s], total integration time
        texp_frame_set - float [s] per frame, or 'default' to let the
            spectrograph pick a frame time that avoids saturation
        nsamp - int, samples up the ramp

        output
        ------
        self, so calls can be chained: sim.set_obs(texp=1800).snr()
        """
        for name, value in _given(texp=texp, texp_frame_set=texp_frame_set, nsamp=nsamp).items():
            setattr(self, name, value)
        self._observed = False
        return self
