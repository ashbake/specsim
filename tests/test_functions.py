"""
Unit tests for the small numeric utilities in specsim/functions.py that the
rest of the pipeline (resampling, degrading spectra to instrument
resolution, unit conversions) is built on.
"""
import numpy as np
import pytest

from specsim.functions import (
    degrade_spec,
    gaussian,
    integrate,
    resample,
    setup_band,
    tophat,
    vac_to_stand,
)


def test_integrate_of_constant():
    x = np.linspace(0, 10, 1000)
    y = np.full_like(x, 3.0)
    assert integrate(x, y) == pytest.approx(30.0, rel=1e-9)


def test_integrate_of_linear_ramp():
    # trapezoidal rule is exact for a linear function
    x = np.linspace(0, 5, 50)
    y = 2 * x + 1
    expected = np.trapz(y, x)
    assert integrate(x, y) == pytest.approx(expected, rel=1e-12)


def test_gaussian_has_unit_area():
    x = np.linspace(-10, 10, 20001)
    g = gaussian(x, shift=0, sig=1.0)
    assert integrate(x, g) == pytest.approx(1.0, rel=1e-6)


def test_gaussian_peak_is_centered_at_shift():
    x = np.linspace(-10, 10, 2001)
    g = gaussian(x, shift=2.5, sig=1.0)
    assert x[np.argmax(g)] == pytest.approx(2.5, abs=0.02)


def test_tophat_is_zero_outside_and_constant_inside():
    x = np.linspace(0, 10, 1001)
    y = tophat(x, l0=3, lf=7, throughput=0.8)
    inside = (x > 3) & (x < 7)
    assert np.all(y[inside] == pytest.approx(0.8))
    assert np.all(y[~inside] == 0)


def test_setup_band_is_centered_tophat():
    x = np.linspace(-5, 5, 1001)
    y = setup_band(x, x0=1.0, sig=2.0, eta=0.5)
    inside = (x > 0.0) & (x < 2.0)
    assert np.all(y[inside] == pytest.approx(0.5))
    assert np.all(y[~inside] == 0)


def test_vac_to_stand_air_wavelength_is_shorter_than_vacuum():
    # index of refraction of air is > 1, so air (standard) wavelength
    # should be slightly shorter than the vacuum wavelength
    wave_vac = np.array([5000.0, 10000.0, 15000.0])  # Angstrom
    wave_air = vac_to_stand(wave_vac)
    assert np.all(wave_air < wave_vac)
    # the correction should be small (a few parts in 10^4), not huge
    assert np.all((wave_vac - wave_air) / wave_vac < 1e-3)


def test_resample_fast_rejects_sigma_smaller_than_sampling():
    x = np.arange(0, 10, 0.5)
    y = np.ones_like(x)
    with pytest.raises(ValueError):
        resample(x, y, sig=0.1, mode="fast")


def test_resample_fast_flux_matches_window_integral():
    x = np.arange(0, 100, 0.1)
    y = np.ones_like(x)
    sig = 1.0
    _, y_resampled = resample(x, y, sig=sig, dx=0, eta=1, mode="fast")

    # resample() derives its own pixel spacing from median(diff(x)) (which
    # can differ from the nominal 0.1 by float accumulation error in
    # np.arange) and truncates sig/dlam to an integer window size -- mirror
    # that exact arithmetic here rather than assuming the idealized sig, so
    # this documents the actual (self-consistent) behavior of resample().
    dlam = np.median(np.diff(x))
    nsamp = int(sig / dlam)
    expected = dlam * nsamp
    interior = y_resampled[5:-5]
    assert np.allclose(interior, expected, rtol=1e-9)


def test_resample_pixels_mode_bins_by_integer_pixel_count():
    x = np.arange(0, 50, 1.0)
    y = np.ones_like(x)
    _, y_resampled = resample(x, y, sig=4, dx=0, eta=1, mode="pixels")
    interior = y_resampled[2:-2]
    assert np.allclose(interior, 4.0, rtol=1e-9)


def test_degrade_spec_conserves_total_flux():
    # Spike well away from the array edges; degrading (convolving with a
    # normalized LSF) should redistribute flux, not create or destroy it.
    x = np.arange(999.0, 1001.0, 0.001)
    y = np.zeros_like(x)
    y[len(y) // 2] = 1000.0

    y_low = degrade_spec(x, y, res=20000)

    assert np.sum(y_low) == pytest.approx(np.sum(y), rel=1e-3)
    # flux should have spread out from the single spike into neighbors
    assert np.count_nonzero(y_low) > 1


def test_degrade_spec_raises_for_lsf_too_coarse_for_sampling():
    # resolving power far too high for this wavelength sampling -> LSF
    # kernel would be under 20 samples wide, which define_lsf rejects
    x = np.arange(1000.0, 1010.0, 1.0)
    y = np.ones_like(x)
    with pytest.raises(ValueError):
        degrade_spec(x, y, res=100000)
