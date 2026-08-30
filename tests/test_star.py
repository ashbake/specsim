"""
Tests for specsim/star.py: loading PHOENIX/Sonora model spectra, scaling
them to a target magnitude, and recovering that magnitude by integrating
the flux back through a filter bandpass (Star/Bandpass supersede the old
source_tools.load_stellar_model/get_band_mag).

These use the real model files and filter curves shipped in data/ rather
than mocks, since the whole point is to check the actual numeric pipeline
(unit conversions, integration, scaling) with real spectra.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from specsim.functions import integrate
from specsim.star import Star, StarParams, load_phoenix, load_sonora

from conftest import PHOENIX_DIR, SONORA_DIR


# ---------------------------------------------------------------------------
# calc_nphot: photon flux <-> magnitude conversion
# ---------------------------------------------------------------------------

def _fake_bandpass(dl_l, zp):
    "Minimal duck-typed stand-in: calc_nphot only reads .dl_l and .zp."
    return SimpleNamespace(dl_l=dl_l, zp=zp)


def test_calc_nphot_zero_mag_matches_zeropoint_formula():
    zp = 1000.0  # Jy
    dl_l = 0.15
    nphot = Star(StarParams(mag=0)).calc_nphot(_fake_bandpass(dl_l, zp))
    expected = dl_l * zp * 1.51e7
    assert nphot == pytest.approx(expected, rel=1e-12)


def test_calc_nphot_scales_correctly_with_magnitude():
    bp = _fake_bandpass(0.2, 1500.0)
    star = Star(StarParams(mag=0))
    nphot0 = star.calc_nphot(bp, mag=0)
    nphot5 = star.calc_nphot(bp, mag=5)
    # every 5 magnitudes fainter is 100x fewer photons
    assert nphot0 / nphot5 == pytest.approx(100.0, rel=1e-8)


def test_calc_nphot_defaults_to_the_stars_own_magnitude():
    bp = _fake_bandpass(0.2, 1500.0)
    star = Star(StarParams(mag=7.5))
    assert star.calc_nphot(bp) == pytest.approx(star.calc_nphot(bp, mag=7.5), rel=1e-12)


# ---------------------------------------------------------------------------
# raw model loaders: sanity-check units/shape of what's read off disk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("teff", [2300, 5800])
def test_load_phoenix_returns_sane_h_band_spectrum(teff):
    stel_file = f"lte{teff:05d}-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    wave, flux = load_phoenix(stel_file, PHOENIX_DIR, wav_start=1490, wav_end=1780)

    assert len(wave) > 100
    assert np.all(np.diff(wave) > 0)  # strictly increasing wavelength
    assert wave.min() >= 1490 and wave.max() <= 1780
    assert np.all(np.isfinite(flux))
    assert np.all(flux >= 0)
    assert np.median(flux) > 0


def test_load_sonora_returns_sane_h_band_spectrum():
    stel_file = SONORA_DIR + "sp_t1400g316nc_m0.0"
    wave, flux = load_sonora(stel_file, wav_start=1490, wav_end=1780)

    assert len(wave) > 10
    assert np.all(np.diff(wave) > 0)
    assert wave.min() >= 1490 and wave.max() <= 1780
    assert np.all(np.isfinite(flux))
    assert np.all(flux >= 0)
    assert np.median(flux) > 0


# ---------------------------------------------------------------------------
# Integrated-flux / magnitude round trip: scale a model to a target
# magnitude in a bandpass (Star.load), then independently integrate the
# scaled flux back through that bandpass (Star.magnitude_in_band, which
# reloads the model file itself) and check the magnitude comes back out.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "teff,band,mag",
    [
        (5800, "J", 8.0),
        (5800, "H", 10.0),
        (5800, "K", 10.0),
        (2300, "H", 12.0),  # boundary teff that should still pick phoenix
    ],
)
def test_phoenix_model_scaled_to_mag_recovers_mag_in_bandpass(make_bandpass, teff, band, mag):
    bp = make_bandpass(band)
    x = np.arange(bp.xraw.min() - 50, bp.xraw.max() + 50, 0.01)

    star = Star(StarParams(teff=teff, mag=mag, phoenix_folder=PHOENIX_DIR, sonora_folder=SONORA_DIR)).load(x, bp)
    assert star.model == "phoenix"

    recovered_mag = star.magnitude_in_band(bp)
    assert recovered_mag == pytest.approx(mag, abs=1e-6)


@pytest.mark.parametrize(
    "teff,band,mag",
    [
        (1400, "H", 14.0),
        (1400, "K", 14.0),
    ],
)
def test_sonora_model_scaled_to_mag_recovers_mag_in_bandpass(make_bandpass, teff, band, mag):
    bp = make_bandpass(band)
    x = np.arange(bp.xraw.min() - 50, bp.xraw.max() + 50, 0.01)

    star = Star(StarParams(teff=teff, mag=mag, phoenix_folder=PHOENIX_DIR, sonora_folder=SONORA_DIR)).load(x, bp)
    assert star.model == "sonora"

    recovered_mag = star.magnitude_in_band(bp)
    assert recovered_mag == pytest.approx(mag, abs=1e-6)


def test_star_magnitude_in_band_reloads_model_when_band_not_covered(make_bandpass):
    # Star.magnitude_in_band should reload the model file from disk (over
    # the requested bandpass's own wavelength range) when the star's
    # already-loaded grid doesn't fully cover it -- exercise that fallback
    # path explicitly, and check it agrees with a star loaded wide enough
    # from the start that no reload is needed.
    filt_j = make_bandpass("J")
    filt_h = make_bandpass("H")

    narrow_x = np.arange(filt_j.xraw.min() - 50, filt_j.xraw.max() + 50, 0.01)
    narrow_star = Star(StarParams(teff=5800, mag=9.0, phoenix_folder=PHOENIX_DIR, sonora_folder=SONORA_DIR)).load(narrow_x, filt_j)
    assert filt_h.xraw.max() > narrow_star.vraw.max()  # confirm H isn't covered by the loaded (J-band) grid
    mag_h_via_reload = narrow_star.magnitude_in_band(filt_h)

    wide_x = np.arange(filt_j.xraw.min() - 50, filt_h.xraw.max() + 50, 0.01)
    wide_star = Star(StarParams(teff=5800, mag=9.0, phoenix_folder=PHOENIX_DIR, sonora_folder=SONORA_DIR)).load(wide_x, filt_j)
    mag_h_no_reload = wide_star.magnitude_in_band(filt_h)

    assert mag_h_via_reload == pytest.approx(mag_h_no_reload, abs=1e-6)


def test_scale_stellar_raises_if_model_does_not_cover_filter(make_bandpass):
    bp = make_bandpass("H")
    star = Star(StarParams(mag=10))
    # model grid narrower than the filter on both sides
    star.vraw = np.linspace(bp.xraw.min() + 10, bp.xraw.max() - 10, 50)
    star.sraw = np.ones_like(star.vraw)
    with pytest.raises(Warning):
        star.scale_stellar(bp)


def test_scale_stellar_raises_if_no_model_loaded(make_bandpass):
    with pytest.raises(RuntimeError):
        Star(StarParams(mag=10)).scale_stellar(make_bandpass("H"))


# ---------------------------------------------------------------------------
# Rotational broadening should redistribute flux within the bandpass, not
# create or destroy it.
# ---------------------------------------------------------------------------

def test_vsini_broadening_conserves_band_integrated_flux(make_bandpass):
    bp = make_bandpass("H")
    x = np.arange(bp.xraw.min() - 20, bp.xraw.max() + 20, 0.005)
    mag = 10.0

    s_unbroadened = Star(StarParams(teff=5800, mag=mag, vsini=0, phoenix_folder=PHOENIX_DIR, sonora_folder=SONORA_DIR)).load(x, bp).s
    s_broadened = Star(StarParams(teff=5800, mag=mag, vsini=15, phoenix_folder=PHOENIX_DIR, sonora_folder=SONORA_DIR)).load(x, bp).s

    filt_interp = bp.interp()
    flux_unbroadened = integrate(x, s_unbroadened * filt_interp(x))
    flux_broadened = integrate(x, s_broadened * filt_interp(x))

    assert flux_broadened == pytest.approx(flux_unbroadened, rel=0.02)
