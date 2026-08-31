##############################################################
# Analyze: operations on the output of a Simulate run -- RV precision,
# matched-filter CCF SNR, and exposure-time calculations
###############################################################
#
# Merges the old ccf_tools.py and etc_tools.py. All four calculations took
# the same (spectrograph, atmosphere, star) bundle as leading
# arguments, so they are methods on one class that holds that bundle.
# Simulate.rv_precision()/ccf_snr()/exposure_time_for_snr()/
# exposure_time_for_ccf_snr() delegate here, or use sim.analysis directly.
#
# The generic spectrum math these used to carry (gaussian_fwhm, spec_make,
# spec_rv_noise_calc, doppler) now lives in functions.py, and
# get_order_bounds -- which parses a spectrograph's order file -- lives in
# instrument.py next to its main caller, Spectrograph.load().
#
# taken from Sam Halverson and Arpita Roys code

from dataclasses import dataclass

import numpy as np
from scipy import interpolate, signal

from specsim.functions import SPEEDOFLIGHT, degrade_spec, resample


def make_telluric_mask(v,s,cutoff=0.01,velocity_cutoff=5):
    """
    input
    -----
    v - array[nm]
        wavelength array of telluric spectrum
    s - array [transmission 0-1]
        spectrum of telluric spectrum
    cutoff - float
        cutoff (0-1) in what lines to mask. default is 0.01 (mask down to 1%)
    velocity_cutoff - float [km/s] 
        velocity around each telluric feature to mask out. Assumes 1 pix ~= km/s

    output
    ------
    telluric_mask - array
        mask corresponding sampled at v array
    """
    telluric_mask = np.ones_like(s)
    telluric_mask[np.where(s < (1-cutoff))[0]] = 0
    for iroll in range(velocity_cutoff): # assume one pixel is 1km/s approx
        telluric_mask[np.where(np.roll(s,iroll) < (1-cutoff))[0]] = 0
        telluric_mask[np.where(np.roll(s,-1*iroll) < (1-cutoff))[0]] = 0

    return telluric_mask


def get_rv_precision(v,s,n,order_cens,order_widths,noise_floor=0.5,mask=None):
    """
    Compute the photon-limited RV precision achievable from a stellar
    spectrum, evaluated per spectral order and combined, following the
    spectral information-content method of Murphy et al. 2007 (weighting
    each pixel by the square of the local flux slope divided by its noise
    variance).

    inputs
    ------
    v - array [nm]
        wavelength array
    s - array
        stellar spectrum (no other sources in it)
    n - array
        noise array (1-sigma uncertainty on s), same length/sampling as v and s
    order_cens - array [nm]
        center wavelength of each spectral order over which to compute RV
        precision
    order_widths - array [nm]
        full width of each spectral order, same length as order_cens (in
        the same order). Only the central 90% of each order's width is
        used when summing the RV information content, to avoid noisy
        order edges
    noise_floor - float, [m/s]
        RV noise floor added in quadrature to each order's photon-limited
        precision to represent additional non-photon error terms (e.g.
        wavelength calibration floor). Default is 0.5 m/s
    mask - array or None
        optional per-pixel weighting mask (same length/sampling as v, s,
        n), e.g. to zero out telluric-contaminated or otherwise unwanted
        pixels before computing RV information content. If None
        (default), no masking is applied (all pixels weighted equally)

    output
    ------
    dv_tot -  array, [m/s]
        per order rv precision with rv floor added
    dv_spec - float, [m/s]
        combined order velocities, no floor added
    dv_vals - array [m/s]
        per order rv precision, no floor added
    """
    # generate rv information content
    flux_interp = interpolate.InterpolatedUnivariateSpline(v,s, k=1)
    dflux = flux_interp.derivative()
    spec_deriv = dflux(v)
    sigma_ord = np.abs(n) #np.abs(s) ** 0.5 # np.abs(n)
    sigma_ord[np.where(sigma_ord ==0)] = 1e10
    all_w = (v ** 2.) * (spec_deriv ** 2.) / sigma_ord ** 2. # include read noise and dark here!!
    
    # make mask if none provided
    if np.any(mask==None):
        mask = np.ones_like(all_w)

    # go through each order
    dv_vals = np.zeros_like(order_cens)
    for i,lam_cen in enumerate(order_cens):
        order_ind   = np.where((v > lam_cen - 0.9*order_widths[i]/2) & (v < lam_cen + 0.9*order_widths[i]/2))[0]
        w_ord       = all_w[order_ind] * mask[order_ind]
        denom       = (np.nansum(w_ord[1:-1])**0.5) # m/s
        dv_order    = SPEEDOFLIGHT / (denom + 0.000001)
        dv_vals[i]  = dv_order
    
    dv_vals[np.where(dv_vals>1e4)[0]] = np.inf # where denom was 0 make inf

    dv_tot  = np.sqrt(dv_vals**2 + noise_floor**2)
    dv_spec  = 1. / (np.nansum(1./dv_vals**2.))**0.5
    dv_spec_floor  = 1. / (np.nansum(1./dv_tot**2.))**0.5

    return dv_tot,dv_spec,dv_vals


def _matched_filter_snr(v, signal_spec, noise, star_v, tel_s, tel_rayleigh, res,
                         model=None, systematics_residuals=0.01, kernel_size=201, norm_cutoff=0.95):
    """
    Shared matched-filter CCF SNR calculation used by both compute_ccf_snr
    (full multi-frame spectrum) and compute_ccf_snr_etc (single frame) --
    previously duplicated near-verbatim between the two. A telluric
    transmission spectrum is built from the H2O/Rayleigh components, the
    model (or signal/sky_trans if no model given) is median-filtered to
    remove the continuum leaving just high-frequency spectral features,
    deep telluric regions (below norm_cutoff) are excluded, and the
    matched filter equation is evaluated using the per-pixel noise
    variance.

    inputs
    ------
    v - array [nm], wavelength grid of signal_spec/noise
    signal_spec - array, the spectrum (full multi-frame or single-frame)
    noise - array, 1-sigma noise, same grid as signal_spec
    star_v - array [nm], the star's hi-res wavelength grid (star.v)
    tel_s, tel_rayleigh - arrays on star_v, atmosphere.s/atmosphere.rayleigh
    res - float, spectrograph.res (resolving power), used to degrade the
        telluric spectrum to spectrograph resolution before resampling onto v
    model - array or None, template spectrum; if None, defaults to
        signal_spec/sky_trans
    systematics_residuals, kernel_size, norm_cutoff - see compute_ccf_snr

    output
    ------
    CCFSNRResult
    """
    signal_spec = signal_spec.copy()
    noise = noise.copy()

    # make telluric spec sampled to v
    tel_rayleigh = tel_rayleigh.copy()
    tel_rayleigh[tel_rayleigh == 0] = np.inf
    telluric_spec = tel_s / tel_rayleigh  # h2o only
    telluric_spec[np.where(np.isnan(telluric_spec))] = 0
    telluric_spec_lores = degrade_spec(star_v, telluric_spec, res)
    filt_interp = interpolate.interp1d(star_v, telluric_spec_lores, bounds_error=False, fill_value=0)
    sky_trans = filt_interp(v) / np.max(filt_interp(v))  # filter profile resampled to phoenix times phoenix flux density

    # Get the noise variance
    total_noise_var = noise ** 2
    bad_noise = np.isnan(total_noise_var)
    total_noise_var[bad_noise] = np.inf

    # Calculate some normalization factor
    norm = (1 - systematics_residuals) * sky_trans

    # Get a median-filtered version of your model spectrum
    # smaller kernel size speeds up calculation, seems a little conservative (lower ccf snr out) bc doesnt smooth as well maybe
    if np.any(model == None): model = signal_spec / sky_trans  # default to this bc at R~100k this is good enough and adds simplicity
    model_medfilt = signal.medfilt(model, kernel_size=kernel_size)  # finds continuum of spectrum
    # Subtract the median version from the original model, effectively high-pass filtering the model
    model_filt = model - model_medfilt
    model_filt[np.isnan(model_filt)] = 0.  # set nans to 0
    model_filt[norm < norm_cutoff] = 0.     # set deep tellurics to 0
    model_filt[bad_noise] = 0.            # set where noise is nan to 0

    # Divide out the sky transmission
    normed_signal = signal_spec / norm
    # High-pass filter like with the model
    signal_filt = normed_signal - model_medfilt / np.max(norm)  # subtract off model_medfilt instead to speed things up, gets very close
    signal_filt[np.isnan(signal_filt)] = 0.
    signal_filt[norm < norm_cutoff] = 0.
    signal_filt[bad_noise] = 0.

    def _ccf_snr(sub):
        return np.sqrt((np.sum(signal_filt[sub] * model_filt[sub] / total_noise_var[sub])) ** 2 /
                       np.sum(model_filt[sub] * model_filt[sub] / total_noise_var[sub]))

    ccf_snr = np.sqrt((np.sum(signal_filt * model_filt / total_noise_var)) ** 2 /
                       np.sum(model_filt * model_filt / total_noise_var))
    sub_y = np.where(v < 1100)[0]
    sub_J = np.where((v > 1100) & (v < 1327))[0]
    sub_H = np.where((v > 1490) & (v < 1780))[0]
    sub_K = np.where((v > 1990) & (v < 2460))[0]

    return CCFSNRResult(ccf_snr=ccf_snr, ccf_snr_y=_ccf_snr(sub_y), ccf_snr_J=_ccf_snr(sub_J),
                         ccf_snr_H=_ccf_snr(sub_H), ccf_snr_K=_ccf_snr(sub_K))


@dataclass
class RVPrecisionResult:
    "Per-order and total RV precision, plus the intermediate spectra used to compute them."
    rv_order: np.ndarray       # per-order RV precision including the spectrograph/telluric noise floor [m/s]
    rv_tot: float              # total RV precision across the full spectrum, including the noise floor [m/s]
    telluric_mask: np.ndarray  # boolean/weight mask excluding regions near deep telluric lines
    s_telcont_free: np.ndarray  # stellar spectrum with spectrograph throughput continuum and telluric absorption removed, resampled onto spectrograph.v


@dataclass
class CCFSNRResult:
    "Matched-filter CCF SNR for the full spectrum and restricted to each photometric band."
    ccf_snr: float
    ccf_snr_y: float
    ccf_snr_J: float
    ccf_snr_H: float
    ccf_snr_K: float


@dataclass
class ETCResult:
    "Exposure time needed to reach a target SNR, per pixel/resolution-element and per order."
    etc: np.ndarray             # array, exposure time [s] needed at each wavelength/resolution element
    etc_order_max: np.ndarray   # array, exposure time [s] per order at that order's max SNR wavelength
    etc_order_mean: np.ndarray  # array, exposure time [s] per order at that order's mean SNR


class Analyze:
    """
    Post-processing of one exposure: RV precision, matched-filter CCF SNR,
    and exposure times. Holds the observed Spectrograph (which carries both
    the hardware and the resulting spectra) plus the Atmosphere and Star the
    exposure was taken through, since every calculation needs some
    combination of them. Cheap to construct -- it only stores references.
    """

    def __init__(self, spectrograph, atmosphere, star):
        self.spectrograph = spectrograph   # already .load()-ed and .observe()-d
        self.atmosphere = atmosphere
        self.star = star

    def rv_precision(self, telluric_cutoff=0.01, velocity_cutoff=30):
        """
        Compute the achievable radial velocity precision for this observation.
        Builds a "telluric/continuum-free" version of the observed spectrum
        (throughput continuum and telluric absorption divided out) so the RV
        information content reflects only the stellar lines, builds a mask
        that excludes wavelengths near deep telluric lines (deeper than
        telluric_cutoff, masked out to +/-velocity_cutoff in velocity space),
        and passes these along with the noise spectrum to get_rv_precision to
        get the RV uncertainty per order and for the full spectrum, adding the
        spectrograph/telluric systematic noise floor (self.spectrograph.rv_floor) in
        quadrature.

        inputs
        ------
        telluric_cutoff - float
            telluric line depth (0-1) below which wavelengths start being
            masked out of the RV calculation
        velocity_cutoff - float, km/s
            velocity window around each masked telluric line to also exclude

        output
        ------
        RVPrecisionResult
        """
        # Create spectrum with continuum removed and tellurics removed
        # the noise spectrum will consider tellurics but shouldnt be in the spectrum for computing RV
        continuum = self.spectrograph.ytransmit / np.max(self.spectrograph.ytransmit)
        # guard 0/0 (-> nan, which would smear via degrade_spec's convolution below) wherever
        # there's no throughput/telluric transmission to divide out in the first place
        continuum_safe = np.where(continuum == 0, np.inf, continuum)
        tel_s_safe = np.where(self.atmosphere.s == 0, np.inf, np.abs(self.atmosphere.s))
        if self.spectrograph.pl_sep > 0:
            telcont_free_hires = self.spectrograph.nframes * self.spectrograph.frame_phot_per_nm_pl / continuum_safe / tel_s_safe
        else:
            telcont_free_hires = self.spectrograph.nframes * self.spectrograph.frame_phot_per_nm / continuum_safe / tel_s_safe

        # remove telurics
        telcont_free_lores = degrade_spec(self.star.v, telcont_free_hires, self.spectrograph.res)
        v, telcont_free = resample(self.star.v, telcont_free_lores, sig=np.mean(self.spectrograph.sig), dx=0, eta=1, mode='fast')
        telcont_free[np.where(np.isnan(telcont_free))] = 0
        f_interp = interpolate.interp1d(v, telcont_free, bounds_error=False, fill_value=0)
        s_telcont_free = f_interp(self.spectrograph.v)

        # make telluric only spectrum, resample onto spectrograph.v to match spectrograph.s
        self.atmosphere.rayleigh[self.atmosphere.rayleigh == 0] = np.inf
        telluric_spec = self.atmosphere.s / self.atmosphere.rayleigh / self.atmosphere.o3  # no continuum altering things!
        telluric_spec[np.where(np.isnan(telluric_spec))] = 0
        telluric_spec_lores = degrade_spec(self.star.v, telluric_spec, self.spectrograph.res)
        v, telluric_spec_lores_resamp = resample(self.star.v, telluric_spec_lores, sig=np.mean(self.spectrograph.sig), dx=0, eta=1, mode='fast')
        tel_interp = interpolate.interp1d(v, telluric_spec_lores_resamp, bounds_error=False, fill_value=0)
        s_tel = tel_interp(self.spectrograph.v) / np.max(tel_interp(self.spectrograph.v))

        # run radial velocity precision
        telluric_mask = make_telluric_mask(self.spectrograph.v, s_tel, cutoff=telluric_cutoff, velocity_cutoff=velocity_cutoff)
        dv_tot, dv_spec, dv_vals = get_rv_precision(self.spectrograph.v, s_telcont_free, self.spectrograph.noise,
                                                     self.spectrograph.order_cens, self.spectrograph.order_widths,
                                                     noise_floor=self.spectrograph.rv_floor, mask=telluric_mask)

        rv_order = dv_tot  # per order rv with noise floor
        rv_tot = np.sqrt(dv_spec ** 2 + self.spectrograph.rv_floor ** 2)  # add noise floor

        return RVPrecisionResult(rv_order=rv_order, rv_tot=rv_tot, telluric_mask=telluric_mask, s_telcont_free=s_telcont_free)

    def ccf_snr(self, model=None, systematics_residuals=0.01, kernel_size=201, norm_cutoff=0.95):
        '''
        Calculates the cross-correlation function (CCF) signal-to-noise ratio
        using a matched-filter formalism, i.e. the SNR that would be obtained
        by cross-correlating the observed spectrum against a stellar/telluric
        template (as used for high-resolution spectroscopy detections/RV
        work), for the full spectrograph.s spectrum and separately for the
        y/J/H/K bands.

        Inputs:
        model       - Your model spectrum, default None and divides signal by telluric spec
        systematics_residuals - A multiplicative factor that estimates the residual level of the host star spectrum and telluric lines in your signal (Default of 1%)
        kernel_size  - The default high-pass filter size.
        norm_cutoff  - A cutoff below which we don't calculate the ccf-snr

        Output:
        CCFSNRResult

        references:
        -----------
        https://github.com/planetarysystemsimager/psisim/blob/kpic/psisim/signal.py
        https://arxiv.org/pdf/1909.07571.pdf
        https://arxiv.org/pdf/2305.19355.pdf
        '''
        return _matched_filter_snr(self.spectrograph.v, self.spectrograph.s, self.spectrograph.noise, self.star.v, self.atmosphere.s, self.atmosphere.rayleigh,
                                    self.spectrograph.res, model=model, systematics_residuals=systematics_residuals,
                                    kernel_size=kernel_size, norm_cutoff=norm_cutoff)

    def exposure_time_for_snr(self, target_snr):
        """
        Given the per-frame SNR already computed by Observation.run()
        (self.spectrograph.s_frame/self.spectrograph.noise_frame, scaled to a resolution
        element), scale by (target_snr/snr_frame)^2 to derive the total
        exposure time needed to reach target_snr, per pixel/resolution-element
        and per order using both the per-order max and mean SNR.

        inputs
        ------
        target_snr - float, desired signal to noise ratio

        output
        ------
        ETCResult
        """
        snr_frame = np.sqrt(self.spectrograph.res_samp) * self.spectrograph.s_frame / self.spectrograph.noise_frame  # per resolution element
        # make 0s nans so doesnt blow up
        inan = np.where(snr_frame == 0)[0]
        snr_frame[inan] = np.nan

        # result is in seconds
        etc = self.spectrograph.texp_frame * (target_snr / snr_frame) ** 2  # texp per frame times nframes - per snr element
        etc_order_max = self.spectrograph.texp_frame * (target_snr / (self.spectrograph.snr_max_orders / np.sqrt(self.spectrograph.nframes))) ** 2  # per order max
        etc_order_mean = self.spectrograph.texp_frame * (target_snr / (self.spectrograph.snr_mean_orders / np.sqrt(self.spectrograph.nframes))) ** 2

        return ETCResult(etc=etc, etc_order_max=etc_order_max, etc_order_mean=etc_order_mean)

    def exposure_time_for_ccf_snr(self, goal_ccf, systematics_residuals=0.01, kernel_size=201, norm_cutoff=0.95):
        '''
        Calculates the exposure time required to achieve a desired CCF SNR
        (goal_ccf) with a matched filter. This is the same matched-filter
        calculation as compute_ccf_snr but run on a single frame's
        signal/noise (self.spectrograph.s_frame, self.spectrograph.noise_frame) instead of
        the full multi-frame spectrum; the model is always signal/sky_trans
        (no user-supplied model option, unlike compute_ccf_snr). Since CCF SNR
        scales as sqrt(exposure time), the per-frame CCF SNR in each band is
        scaled by (goal_ccf/ccf_snr)^2 x self.spectrograph.texp_frame to get the
        needed total exposure time, computed separately for the y/J/H/K
        bands. Does not currently account for systematics_residuals scaling
        with exposure time.

        Inputs:
        --------
        goal_ccf    - CCF SNR for which exposure time will be computed
        systematics_residuals - A multiplicative factor that estimates the residual level of the host star spectrum and telluric lines in your signal (Default of 1%)
        kernel_size  - The default high-pass filter size.
        norm_cutoff  - A cutoff below which we don't calculate the ccf-snr

        Output:
        --------
        dict with keys 'y', 'J', 'H', 'K' - float, exposure time [s] needed in
        each band to reach goal_ccf
        '''
        # TODO: This function does not account for systematics at the moment
        # To account for read_noise, we need to change how the number of frames is done in PSISIM
        # For systematics, we need to find a nice way to invert the CCF SNR equation when systematics are present
        result = _matched_filter_snr(self.spectrograph.v, self.spectrograph.s_frame, self.spectrograph.noise_frame, self.star.v, self.atmosphere.s, self.atmosphere.rayleigh,
                                      self.spectrograph.res, model=None, systematics_residuals=systematics_residuals,
                                      kernel_size=kernel_size, norm_cutoff=norm_cutoff)

        return {band: self.spectrograph.texp_frame * goal_ccf ** 2 / getattr(result, f'ccf_snr_{band}') ** 2
                for band in ('y', 'J', 'H', 'K')}
