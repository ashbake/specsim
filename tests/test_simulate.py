"""
Regression test for the Simulate/simulate_from_config pipeline
(specsim/simulate.py, specsim/config.py) against known-good numeric
outputs (SNR, RV precision, CCF SNR, ETC), computed with
configs/modhis_snr.cfg (the config the example scripts already use).

The golden values were originally captured from the pre-refactor
fill_data(so) pipeline and cross-checked against Simulate/
simulate_from_config bit-for-bit (see git history for
tests/test_characterization.py, this file's predecessor). They were
regenerated once since, to reflect a real bugfix: Observation.run()'s
call to get_sky_bg() previously never passed npix/R/diam/area,
so the sky-background contribution to the noise budget silently used the
function's HISPEC-shaped defaults (diam=10m, area=76m^2) regardless of
the configured instrument, rather than MODHIS's actual 30m/655m^2. The
numeric shift for this particular (bright-star, short-exposure) config is
tiny, since sky background is a small fraction of the total noise budget
here.
"""
import os

import numpy as np
import pytest

from specsim.config import simulate_from_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(REPO_ROOT, "configs", "modhis_snr.cfg")
GOLDEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "modhis_snr_golden.npz")


@pytest.fixture(scope="module")
def golden():
    return np.load(GOLDEN_FILE)


@pytest.fixture(scope="module")
def sim():
    return simulate_from_config(CONFIG_FILE)


@pytest.fixture(scope="module")
def observation(sim):
    return sim.snr()


def test_snr(observation, golden):
    dec = int(golden["decimation"])
    assert np.allclose(observation.snr[::dec], golden["snr"], rtol=1e-6)
    assert np.allclose(observation.snr_res_element[::dec], golden["snr_res_element"], rtol=1e-6)
    assert np.allclose(observation.v_res_element[::dec], golden["v_res_element"], rtol=1e-6)


def test_order_cens(observation, golden):
    assert np.allclose(observation.order_cens, golden["order_cens"], rtol=1e-6)


def test_rv_precision(sim, observation, golden):
    rv = sim.rv_precision(telluric_cutoff=0.2, velocity_cutoff=2)
    assert np.allclose(rv.rv_order, golden["rv_order"], rtol=1e-6, equal_nan=True)
    assert rv.rv_tot == pytest.approx(float(golden["rv_tot"]), rel=1e-6)


def test_ccf_snr(sim, observation, golden):
    ccf = sim.ccf_snr()
    assert ccf.ccf_snr == pytest.approx(float(golden["ccf_snr"]), rel=1e-6)


def test_etc(sim, observation, golden):
    etc = sim.exposure_time_for_snr(100)
    assert np.allclose(etc.etc_order_mean, golden["etc_order_mean"], rtol=1e-6)
    assert np.allclose(etc.etc_order_max, golden["etc_order_max"], rtol=1e-6)
