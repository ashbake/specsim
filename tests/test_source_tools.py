"""
Tests for specsim/source_tools.py: loading PHOENIX/Sonora model spectra,
scaling them to a target magnitude, and recovering that magnitude by
integrating the flux back through a filter bandpass.

These use the real model files and filter curves shipped in data/ rather
than mocks, since the whole point is to check the actual numeric pipeline
(unit conversions, integration, scaling) with real spectra.
"""
import numpy as np
import pytest
from scipy.interpolate import interp1d

from specsim import source_tools
from specsim.functions import integrate

from conftest import PHOENIX_DIR, SONORA_DIR


# ---------------------------------------------------------------------------
# calc_nphot: photon flux <-> magnitude conversion
# ---------------------------------------------------------------------------

def test_calc_nphot_zero_mag_matches_zeropoint_formula():
    zp = 1000.0  # Jy
    dl_l = 0.15
    nphot = source_tools.calc_nphot(dl_l, zp, mag=0)
    expected = dl_l * zp * 1.51e7
    assert nphot == pytest.approx(expected, rel=1e-12)


def test_calc_nphot_scales_correctly_with_magnitude():
    nphot0 = source_tools.calc_nphot(0.2, 1500.0, mag=0)
    nphot5 = source_tools.calc_nphot(0.2, 1500.0, mag=5)
    # every 5 magnitudes fainter is 100x fewer photons
    assert nphot0 / nphot5 == pytest.approx(100.0, rel=1e-8)


# ---------------------------------------------------------------------------
# raw model loaders: sanity-check units/shape of what's read off disk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("teff", [2300, 5800])
def test_load_phoenix_returns_sane_h_band_spectrum(teff):
    stel_file = f"lte{teff:05d}-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    wave, flux = source_tools.load_phoenix(stel_file, PHOENIX_DIR, wav_start=1490, wav_end=1780)

    assert len(wave) > 100
    assert np.all(np.diff(wave) > 0)  # strictly increasing wavelength
    assert wave.min() >= 1490 and wave.max() <= 1780
    assert np.all(np.isfinite(flux))
    assert np.all(flux >= 0)
    assert np.median(flux) > 0


def test_load_sonora_returns_sane_h_band_spectrum():
    stel_file = SONORA_DIR + "sp_t1400g316nc_m0.0"
    wave, flux = source_tools.load_sonora(stel_file, wav_start=1490, wav_end=1780)

    assert len(wave) > 10
    assert np.all(np.diff(wave) > 0)
    assert wave.min() >= 1490 and wave.max() <= 1780
    assert np.all(np.isfinite(flux))
    assert np.all(flux >= 0)
    assert np.median(flux) > 0


# ---------------------------------------------------------------------------
# Integrated-flux / magnitude round trip: scale a model to a target
# magnitude in a bandpass, then independently integrate the scaled flux
# back through that bandpass (via get_band_mag, which reloads the filter
# and model file itself) and check the magnitude comes back out.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "teff,family,band,mag",
    [
        (5800, "2mass", "J", 8.0),
        (5800, "2mass", "H", 10.0),
        (5800, "2mass", "K", 10.0),
        (2300, "2mass", "H", 12.0),  # boundary teff that should still pick phoenix
    ],
)
def test_phoenix_model_scaled_to_mag_recovers_mag_in_bandpass(make_so, teff, family, band, mag):
    so = make_so(family, band)
    x = np.arange(so.filt.xraw.min() - 50, so.filt.xraw.max() + 50, 0.01)

    _, vraw, sraw, model, stel_file, factor_0 = source_tools.load_stellar_model(x, mag, teff, 0, so)
    assert model == "phoenix"

    recovered_mag = source_tools.get_band_mag(so, vraw, sraw, model, stel_file, family, band, factor_0)
    assert recovered_mag == pytest.approx(mag, abs=1e-6)


@pytest.mark.parametrize(
    "teff,family,band,mag",
    [
        (1400, "2mass", "H", 14.0),
        (1400, "2mass", "K", 14.0),
    ],
)
def test_sonora_model_scaled_to_mag_recovers_mag_in_bandpass(make_so, teff, family, band, mag):
    so = make_so(family, band)
    x = np.arange(so.filt.xraw.min() - 50, so.filt.xraw.max() + 50, 0.01)

    _, vraw, sraw, model, stel_file, factor_0 = source_tools.load_stellar_model(x, mag, teff, 0, so)
    assert model == "sonora"

    recovered_mag = source_tools.get_band_mag(so, vraw, sraw, model, stel_file, family, band, factor_0)
    assert recovered_mag == pytest.approx(mag, abs=1e-6)


def test_get_band_mag_reloads_model_when_given_range_is_too_narrow(make_so):
    # get_band_mag should reload the model file from disk (over the filter's
    # own wavelength range) when the vraw/sraw handed to it don't fully
    # cover the filter -- exercise that fallback path explicitly.
    so = make_so("2mass", "H")
    mag = 9.0
    x = np.arange(so.filt.xraw.min() - 50, so.filt.xraw.max() + 50, 0.01)
    _, vraw, sraw, model, stel_file, factor_0 = source_tools.load_stellar_model(x, mag, 5800, 0, so)

    mid = len(vraw) // 2
    narrow_vraw = vraw[mid - 100 : mid + 100]
    narrow_sraw = sraw[mid - 100 : mid + 100]
    assert narrow_vraw.max() - narrow_vraw.min() < (so.filt.xraw.max() - so.filt.xraw.min())

    recovered_mag = source_tools.get_band_mag(
        so, narrow_vraw, narrow_sraw, model, stel_file, "2mass", "H", factor_0
    )
    assert recovered_mag == pytest.approx(mag, abs=1e-6)


def test_scale_stellar_raises_if_model_does_not_cover_filter(make_so):
    so = make_so("2mass", "H")
    stelv = np.linspace(so.filt.xraw.min() + 10, so.filt.xraw.max() - 10, 50)
    stels = np.ones_like(stelv)
    with pytest.raises(Warning):
        source_tools.scale_stellar(so.filt, stelv, stels, mag=10)


# ---------------------------------------------------------------------------
# Rotational broadening should redistribute flux within the bandpass, not
# create or destroy it.
# ---------------------------------------------------------------------------

def test_vsini_broadening_conserves_band_integrated_flux(make_so):
    so = make_so("2mass", "H")
    x = np.arange(so.filt.xraw.min() - 20, so.filt.xraw.max() + 20, 0.005)
    mag = 10.0

    s_unbroadened, *_ = source_tools.load_stellar_model(x, mag, 5800, 0, so)
    s_broadened, *_ = source_tools.load_stellar_model(x, mag, 5800, 15, so)

    filt_interp = interp1d(so.filt.xraw, so.filt.yraw, bounds_error=False, fill_value=0)
    flux_unbroadened = integrate(x, s_unbroadened * filt_interp(x))
    flux_broadened = integrate(x, s_broadened * filt_interp(x))

    assert flux_broadened == pytest.approx(flux_unbroadened, rel=0.02)
