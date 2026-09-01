"""
Regression test for the Simulate/simulate_from_config pipeline
(specsim/simulate.py, specsim/config.py) against known-good numeric
outputs (SNR, RV precision, CCF SNR, ETC), computed with
configs/modhis_snr.cfg (the config the example scripts already use).

The golden values were originally captured from the pre-refactor
fill_data(so) pipeline and cross-checked against Simulate/
simulate_from_config bit-for-bit (see git history for
tests/test_characterization.py, this file's predecessor). They were
regenerated once since, to reflect a real bugfix: the exposure's
call to get_sky_bg() previously never passed npix/R/diam/area,
so the sky-background contribution to the noise budget silently used the
function's HISPEC-shaped defaults (diam=10m, area=76m^2) regardless of
the configured instrument, rather than MODHIS's actual 30m/655m^2. The
numeric shift for this particular (bright-star, short-exposure) config is
tiny, since sky background is a small fraction of the total noise budget
here.

Regenerated a second time when the throughput continuum was removed from
the RV signal in Analyze.rv_precision(). The spectrum had been divided by
both telluric transmission and the throughput continuum, while the noise
passed to get_rv_precision() was the noise on the *uncorrected* spectrum.
Dividing a spectrum by a transmission must divide its noise by the same
factor, so the information content was inflated by 1/continuum^2 and
low-throughput orders looked artificially precise. Only rv_order/rv_tot
moved (rv_tot 0.6249 -> 0.7292 m/s); snr, ccf_snr and the ETC arrays in
the golden file are byte-identical.

Regenerated a third time, immediately after, for the matching half of that
same fix: the noise is now divided by the telluric transmission the signal
was divided by. This moved rv_tot 0.7292 -> 0.7359 m/s -- small because
deep telluric lines are masked out anyway -- and the change is
concentrated where it should be, penalising the most telluric-affected
quartile of orders by ~8% and the cleanest quartile by ~0.4%.
"""
import inspect
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from specsim.bandpass import Bandpass
from specsim.config import simulate_from_config
from specsim.simulate import Simulate
from specsim.paths import SPECSIM_ROOT, resolve

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


# ---- setters ----
#
# Each of these builds its own Simulate rather than using the module-scoped
# `sim` fixture, which the tests above share and expect unmutated. Values are
# kept on the grids the shipped data files cover: tt_dynamic on MODHIS's
# coupling grid (0-4.5 mas, 0.5 mas steps), and pwv/zenith_angle on the sky
# background grid (only pwv/airmass 1.5/1.2, 1.5/1.4 and 1.8/1.4 are shipped).


@pytest.fixture
def fresh_sim():
    return simulate_from_config(CONFIG_FILE)


def median_snr(sim):
    return np.nanmedian(sim.snr().snr_res_element)


def test_set_ao_wfe(fresh_sim):
    "set_ao() pins ho_wfe/tt_dynamic instead of reading them off the chosen AO mode, degrading them lowers the Strehl and the SNR, and clearing them restores the mode lookup."
    sim = fresh_sim
    mode, ho_wfe, tt_dynamic = sim.ao_system.mode_chosen, sim.ao_system.ho_wfe, sim.ao_system.tt_dynamic
    snr_auto = median_snr(sim)

    sim.set_ao(ho_wfe=190, tt_dynamic=2.0)
    assert sim.ao_system.mode_chosen == "User Defined HO and TT values"
    assert (sim.ao_system.ho_wfe, sim.ao_system.tt_dynamic) == (190, 2.0)
    snr_190 = median_snr(sim)

    sim.set_ao(ho_wfe=300, tt_dynamic=3.0)
    assert (sim.ao_system.ho_wfe, sim.ao_system.tt_dynamic) == (300, 3.0)
    assert sim.ao_system.strehl < 1
    assert median_snr(sim) < snr_190  # worse AO -> worse coupling -> lower SNR

    sim.set_ao(ho_wfe=None, tt_dynamic=None)  # back to picking a mode from the WFE tables
    assert sim.ao_system.mode_chosen == mode
    assert (sim.ao_system.ho_wfe, sim.ao_system.tt_dynamic) == (ho_wfe, tt_dynamic)
    assert median_snr(sim) == pytest.approx(snr_auto, rel=1e-9)


def test_set_ao_wfe_requires_both(fresh_sim):
    with pytest.raises(ValueError, match="both"):
        fresh_sim.set_ao(ho_wfe=190)


def test_set_ao_mag(fresh_sim):
    "A fainter AO guide star gets a worse-corrected mode, and 'default' returns to inheriting the science star's magnitude."
    sim = fresh_sim
    snr_auto, ho_wfe_auto = median_snr(sim), sim.ao_system.ho_wfe

    sim.set_ao(mag=18, mag_band='R')
    assert sim.ao_system.ao_mag == pytest.approx(18, rel=1e-6)  # R is the WFE table's own band, so no colour conversion
    assert sim.ao_system.ho_wfe > ho_wfe_auto  # fainter guide star -> more residual wavefront error
    assert median_snr(sim) < snr_auto

    sim.set_ao(mag='default', mag_band='default')
    assert sim.ao_system.ho_wfe == pytest.approx(ho_wfe_auto, rel=1e-9)
    assert median_snr(sim) == pytest.approx(snr_auto, rel=1e-9)


def test_set_ao_teff_without_mag(fresh_sim):
    "AO teff with mag left at 'default' means 'as bright as the science star, but this colour' -- it must not trip over the 'default' string."
    sim = fresh_sim
    sim.set_ao(teff=2300)
    # science star is H=10/5800K; a 2300K star of the same H mag is much fainter in R
    assert sim.ao_system.ao_star.params.mag == sim.star.params.mag
    assert sim.ao_system.ao_mag > sim.star.params.mag
    assert np.isfinite(median_snr(sim))


def test_set_ao_mode(fresh_sim):
    sim = fresh_sim
    auto_mode = sim.ao_system.mode_chosen
    assert sim.set_ao(mode=auto_mode).ao_system.mode_chosen == auto_mode
    with pytest.raises(ValueError, match="not a mode"):
        sim.set_ao(mode='NotAMode')


def test_set_star_combined(fresh_sim):
    "set_star() applies several parameters in one reload, and matches applying them one at a time."
    sim = fresh_sim
    sim.set_star(teff=2300, mag=11, vsini=5)
    assert (sim.star.params.teff, sim.star.params.mag, sim.star.params.vsini) == (2300, 11, 5)
    snr_combined = median_snr(sim)

    stepwise = simulate_from_config(CONFIG_FILE)
    stepwise.set_star(teff=2300)
    stepwise.set_star(mag=11)
    stepwise.set_star(vsini=5)
    assert median_snr(stepwise) == pytest.approx(snr_combined, rel=1e-9)


def test_set_obs(fresh_sim):
    "texp is the cheap setter: no AO/coupling recompute, and the AO state is untouched."
    sim = fresh_sim
    snr_900, ho_wfe = median_snr(sim), sim.ao_system.ho_wfe
    sim.set_obs(texp=3600)
    assert sim.texp == 3600
    assert sim.ao_system.ho_wfe == ho_wfe
    assert median_snr(sim) > snr_900  # longer exposure -> more signal


def test_set_atmosphere(fresh_sim):
    "Seeing feeds the AO WFE lookup, so changing it re-selects the AO; pwv alone does not touch the AO."
    sim = fresh_sim
    ho_wfe_good = sim.ao_system.ho_wfe

    sim.set_atmosphere(seeing_set='bad')
    assert sim.atmosphere.seeing_set == 'bad'
    assert sim.ao_system.ho_wfe > ho_wfe_good  # worse seeing -> more residual wavefront error

    sim.set_atmosphere(seeing_set='good')
    assert sim.ao_system.ho_wfe == pytest.approx(ho_wfe_good, rel=1e-9)

    # zenith angle also indexes the WFE tables, so it re-selects the AO too
    sim.set_atmosphere(pwv=1.8, zenith_angle=45)
    assert (sim.atmosphere.pwv, sim.zenith_angle) == (1.8, 45)
    assert sim.ao_system.ho_wfe > ho_wfe_good  # further from zenith -> more atmosphere -> more WFE
    assert np.isfinite(median_snr(sim))


def test_set_atmosphere_pwv_leaves_ao_alone(fresh_sim):
    "pwv only reaches the exposure (telluric transmission + sky background), so the AO solution is untouched by it."
    sim = fresh_sim
    sim.set_atmosphere(zenith_angle=45)  # airmass 1.4, the only one with both pwv grid points shipped
    ho_wfe, strehl = sim.ao_system.ho_wfe, sim.ao_system.strehl
    snr_dry = median_snr(sim)

    sim.set_atmosphere(pwv=1.8)  # wetter sky: more telluric absorption, same AO correction
    assert sim.atmosphere.pwv == 1.8
    assert (sim.ao_system.ho_wfe, sim.ao_system.strehl) == (ho_wfe, strehl)
    assert median_snr(sim) < snr_dry


# ---- running from another directory ----
#
# Config paths like './data/...' are written relative to the specsim tree, not
# to wherever the user is running, and the instrument YAML falls back to the
# bundled one. Together that means a user only has to write a .cfg, anywhere.


def test_runs_from_another_directory(tmp_path, monkeypatch):
    "A .cfg copied somewhere else, with no instruments/ dir beside it, gives the same answer as the in-repo one."
    expected = median_snr(simulate_from_config(CONFIG_FILE))

    elsewhere = tmp_path / "some_project"
    elsewhere.mkdir()
    shutil.copy(CONFIG_FILE, elsewhere / "my_run.cfg")
    monkeypatch.chdir(elsewhere)

    sim = simulate_from_config("./my_run.cfg")
    assert median_snr(sim) == pytest.approx(expected, rel=1e-9)


def test_config_paths_anchor_to_specsim_not_cwd(tmp_path, monkeypatch):
    "A relative data path resolves against the specsim tree even when the CWD has a same-named decoy directory."
    decoy = tmp_path / "data"
    decoy.mkdir()  # would shadow specsim's data/ if paths were CWD-relative
    monkeypatch.chdir(tmp_path)

    sim = simulate_from_config(CONFIG_FILE)
    assert sim.filter_path.startswith(SPECSIM_ROOT)
    assert os.path.isfile(sim.zp_file)


def test_unknown_instrument_lists_bundled_ones(tmp_path):
    cfg = tmp_path / "bad.cfg"
    cfg.write_text("[run]\ninstrument=notreal\n")
    with pytest.raises(FileNotFoundError, match="hispec, modhis"):
        simulate_from_config(str(cfg))


def test_local_instrument_yaml_wins_over_bundled(tmp_path, monkeypatch):
    "An instruments/ dir beside the user's .cfg takes precedence, so a project can override the bundled instrument."
    project = tmp_path / "project"
    (project / "instruments").mkdir(parents=True)
    shutil.copy(CONFIG_FILE, project / "my_run.cfg")

    bundled = os.path.join(SPECSIM_ROOT, "configs", "instruments", "modhis.yaml")
    override = yaml.safe_load(open(bundled))
    override["spectrograph"]["readnoise"] = 99  # a value the bundled file doesn't have
    (project / "instruments" / "modhis.yaml").write_text(yaml.safe_dump(override))

    monkeypatch.chdir(project)
    assert simulate_from_config("./my_run.cfg").spectrograph.readnoise == 99


def test_model_folders_come_from_instrument_yaml(tmp_path, monkeypatch):
    "A user .cfg needn't mention phoenix_folder/sonora_folder at all -- they default from the instrument YAML to specsim's own models."
    elsewhere = tmp_path / "project"
    elsewhere.mkdir()
    shutil.copy(CONFIG_FILE, elsewhere / "my_run.cfg")
    monkeypatch.chdir(elsewhere)

    assert "phoenix_folder" not in (elsewhere / "my_run.cfg").read_text()
    sim = simulate_from_config("./my_run.cfg")
    assert sim.star.params.phoenix_folder == os.path.join(SPECSIM_ROOT, "data", "stel", "phoenix") + os.sep
    assert os.path.isdir(sim.star.params.phoenix_folder)
    assert os.path.isdir(sim.star.params.sonora_folder)


def test_model_folder_override_from_cfg(tmp_path, monkeypatch):
    "An absolute phoenix_folder under [stel] in the user's own .cfg still wins over the instrument YAML."
    my_models = tmp_path / "my_models"
    my_models.mkdir()
    # symlink specsim's models in, so the only thing under test is which folder gets used
    for model in (Path(SPECSIM_ROOT) / "data" / "stel" / "phoenix").iterdir():
        (my_models / model.name).symlink_to(model)
    cfg = tmp_path / "my_run.cfg"
    cfg.write_text(open(CONFIG_FILE).read().replace(
        "[stel]\n", f"[stel]\nphoenix_folder={my_models}{os.sep}\n"))
    monkeypatch.chdir(tmp_path)

    sim = simulate_from_config(str(cfg))
    assert sim.star.params.phoenix_folder == str(my_models) + os.sep
    assert sim.star.params.sonora_folder.startswith(SPECSIM_ROOT)  # untouched keys still come from the YAML


def test_resolve_preserves_trailing_separator():
    "sonora_folder is turned into a filename by string concatenation, so a trailing separator has to survive resolution."
    assert resolve("./data/stel/sonora/").endswith(os.sep)
    assert not resolve("./data/filters/zeropoints.txt").endswith(os.sep)
    assert resolve("/already/absolute/") == "/already/absolute/"


# ---- filter band ----
#
# The band is the one input that sits above the star in the build chain: it
# renormalises the star (and any companion), and it moves filt.center_wavelength,
# which is the reference wavelength for the Strehl terms feeding the coupling.
# Changing it REINTERPRETS the magnitude (an H=10 star becomes a K=10 star)
# rather than colour-converting it.


def bandpass_for(sim, band, family=None):
    return Bandpass.load(sim.filter_path, sim.zp_file, band, family)


def test_set_star_band_reinterprets_the_magnitude(fresh_sim):
    "The magnitude NUMBER is held fixed and re-read in the new band. This test fails under colour-conversion semantics."
    sim = fresh_sim  # H = 10, 5800 K
    sim.set_star(band='R')  # R, not K -- a ~1 mag colour lever, so the assertion can't go flaky

    assert sim.star.params.mag == 10  # the number itself is untouched
    assert (sim.filt.band, sim.filt.family) == ('R', 'Johnson')
    assert sim.star.magnitude_in_band(bandpass_for(sim, 'R')) == pytest.approx(10, abs=1e-3)
    # a 5800 K star that is R = 10 is substantially brighter in H
    assert sim.star.magnitude_in_band(bandpass_for(sim, 'H')) < 9.5


def test_set_star_band_matches_building_at_that_band():
    "The runtime path must recompute everything construction does -- the strongest guarantee here."
    switched = simulate_from_config(CONFIG_FILE).set_star(band='K')
    built = simulate_from_config(CONFIG_FILE, filt_band='K')

    assert switched.star.factor_0 == pytest.approx(built.star.factor_0, rel=1e-12)
    assert switched.ao_system.ho_wfe == pytest.approx(built.ao_system.ho_wfe, rel=1e-12)
    assert median_snr(switched) == pytest.approx(median_snr(built), rel=1e-9)


def test_set_star_band_changes_the_snr(fresh_sim):
    "Pin the AO first so this is a purely photometric assertion, not a coupled AO+photometry one."
    sim = fresh_sim.set_ao(ho_wfe=190, tt_dynamic=2.0)
    snr_h = median_snr(sim)
    sim.set_star(band='K')
    assert abs(median_snr(sim) - snr_h) / snr_h > 0.01
    assert np.isfinite(median_snr(sim))


def test_set_star_band_round_trips(fresh_sim):
    sim = fresh_sim
    snr_h = median_snr(sim)
    assert median_snr(sim.set_star(band='K').set_star(band='H')) == pytest.approx(snr_h, rel=1e-9)


def test_set_star_band_and_mag_apply_in_one_reload(fresh_sim):
    "Mirrors test_set_star_combined: together must equal one at a time."
    fresh_sim.set_star(mag=9, band='K')
    stepwise = simulate_from_config(CONFIG_FILE).set_star(band='K').set_star(mag=9)
    assert median_snr(fresh_sim) == pytest.approx(median_snr(stepwise), rel=1e-9)


def test_set_star_family_override(fresh_sim):
    "band alone re-derives the family; pass both to keep a non-conventional one."
    sim = fresh_sim
    assert sim.set_star(band='y').filt.family == 'cfht'
    decam = sim.set_star(band='y', family='decam')
    assert decam.filt.family == 'decam'
    assert sim.set_star(band='y').filt.family == 'cfht'  # re-derived, not sticky
    # and switching band without a family must not carry 'decam' into a band it has no curve for
    assert sim.set_star(band='y', family='decam').set_star(band='K').filt.family == '2mass'


def test_set_star_unknown_band_leaves_the_scene_intact(fresh_sim):
    "The filter is assigned only after the load succeeds, so a typo doesn't wedge the object."
    sim = fresh_sim
    snr_before = median_snr(sim)
    with pytest.raises(ValueError, match="No filter curve"):
        sim.set_star(band='Q')
    assert sim.filt.band == 'H'
    assert median_snr(sim) == pytest.approx(snr_before, rel=1e-12)


def test_filt_band_properties_track_the_bandpass(fresh_sim):
    "filt_band/filt_family read off the Bandpass, so they can't drift out of step with it."
    sim = fresh_sim
    assert (sim.filt_band, sim.filt_family) == ('H', '2mass')
    sim.set_star(band='K')
    assert (sim.filt_band, sim.filt_family) == ('K', '2mass')


def test_set_filter_matches_set_star_band():
    a = simulate_from_config(CONFIG_FILE).set_filter(band='K')
    b = simulate_from_config(CONFIG_FILE).set_star(band='K')
    assert median_snr(a) == pytest.approx(median_snr(b), rel=1e-12)


def test_set_star_band_can_reselect_the_ao(fresh_sim):
    "The band moves filt.center_wavelength, so the AO solution legitimately moves too. Don't pin a mode name -- which mode wins is a property of the WFE tables."
    sim = fresh_sim
    strehl_h = sim.ao_system.strehl
    sim.set_star(band='K')
    assert sim.ao_system.mode_chosen in sim.ao_system.ao_modes
    assert sim.ao_system.strehl != strehl_h


def test_set_star_band_reloads_the_companion(tmp_path, monkeypatch):
    "A companion's magnitude is quoted in the same band, so it must renormalise too. Before this was handled, it stayed scaled in the old band and pl_sep>0 results were silently wrong."
    cfg = tmp_path / "companion.cfg"
    cfg.write_text(open(CONFIG_FILE).read().replace("pl_sep=0", "pl_sep=100").replace("pl_mag=19", "pl_mag=15"))
    monkeypatch.chdir(tmp_path)

    sim = simulate_from_config(str(cfg))
    factor_h = sim.companion.factor_0
    sim.set_star(band='K')

    assert sim.companion.factor_0 != factor_h
    assert sim.companion.magnitude_in_band(bandpass_for(sim, 'K')) == pytest.approx(15, abs=1e-3)


def test_band_defaults_to_h_when_the_cfg_omits_it(tmp_path, monkeypatch):
    "Simulate.__init__ and simulate_from_config must agree on the fallback band -- they used to disagree ('H' vs 'J'), which would silently change every result for a cfg with no [filt] band."
    cfg = tmp_path / "noband.cfg"
    cfg.write_text(re.sub(r'^band=H\s*$', '', open(CONFIG_FILE).read(), flags=re.M))
    monkeypatch.chdir(tmp_path)

    assert simulate_from_config(str(cfg)).filt_band == 'H'
    assert inspect.signature(Simulate).parameters['filt_band'].default == 'H'


# ---- RV precision sanity ----
#
# test_rv_precision above only pins rv_order against stored values, so it would
# happily lock in physically backwards numbers. These check the SIGN of the
# relationships instead, which is what a real error would break.


def order_diagnostics(sim):
    "Per-order median SNR, mean telluric transmission, and median throughput, aligned with spectrograph.order_cens."
    sp, atm = sim.spectrograph, sim.atmosphere
    tel = np.interp(sp.v_res_element, atm.v, atm.s)
    thr = np.interp(sp.v_res_element, sim.star.v, sp.ytransmit)
    snr_o, tel_o, thr_o = [], [], []
    for ind in sp.order_inds:
        snr_o.append(np.nanmedian(sp.snr_res_element[ind]))
        tel_o.append(np.nanmean(tel[ind]))
        thr_o.append(np.nanmedian(thr[ind]))
    return np.array(snr_o), np.array(tel_o), np.array(thr_o)


def test_rv_error_falls_with_snr_and_cleanliness(fresh_sim):
    "More photons -> better RV precision, and cleaner orders -> better RV precision. Both correlations must be negative; a sign flip here is the symptom of an inverted information-content weighting."
    sim = fresh_sim
    sim.snr()
    rv = sim.rv_precision(telluric_cutoff=0.2, velocity_cutoff=2)
    snr_o, tel_o, _ = order_diagnostics(sim)
    ok = np.isfinite(rv.rv_order) & np.isfinite(snr_o)
    assert ok.sum() > 20

    assert np.corrcoef(snr_o[ok], rv.rv_order[ok])[0, 1] < -0.2
    assert np.corrcoef(tel_o[ok], rv.rv_order[ok])[0, 1] < -0.2


def test_rv_error_improves_with_exposure_time(fresh_sim):
    "Photon-limited precision scales as 1/sqrt(t), so 4x the exposure should approach 2x better -- bounded loosely since the rv_floor is added in quadrature."
    sim = fresh_sim
    sim.snr()
    rv_900 = sim.rv_precision(telluric_cutoff=0.2, velocity_cutoff=2).rv_tot
    sim.set_obs(texp=3600)
    sim.snr()
    rv_3600 = sim.rv_precision(telluric_cutoff=0.2, velocity_cutoff=2).rv_tot
    assert rv_3600 < rv_900
    assert 1.2 < rv_900 / rv_3600 < 2.2


def test_rv_error_worsens_for_a_fainter_star(fresh_sim):
    sim = fresh_sim
    sim.snr()
    bright = sim.rv_precision(telluric_cutoff=0.2, velocity_cutoff=2).rv_tot
    sim.set_star(mag=14)
    sim.snr()
    assert sim.rv_precision(telluric_cutoff=0.2, velocity_cutoff=2).rv_tot > bright


def test_rv_signal_is_not_divided_by_the_throughput_continuum(fresh_sim):
    """
    The RV spectrum has tellurics divided out but NOT the throughput
    continuum: the continuum is smooth, so dividing by it removes no line
    structure and only rescales the signal without rescaling the noise,
    which inflates the information content by 1/continuum^2 and makes
    low-throughput orders look artificially precise.

    Checked by construction rather than by value: s_tel_free must track the
    observed photon spectrum up to the telluric division alone.
    """
    sim = fresh_sim
    sim.snr()
    rv = sim.rv_precision(telluric_cutoff=0.2, velocity_cutoff=2)
    sp = sim.spectrograph

    cont = np.interp(sp.v, sim.star.v, sp.ytransmit / np.max(sp.ytransmit))
    live = (rv.telluric_mask > 0) & (rv.s_tel_free > 0) & (cont > 0.05)
    # if the continuum had been divided out, s_tel_free would be anti-correlated
    # with throughput; with it left in, the two must rise together
    assert np.corrcoef(cont[live], rv.s_tel_free[live])[0, 1] > 0


def test_rv_noise_is_divided_by_the_same_telluric_as_the_signal(fresh_sim):
    """
    The RV signal has telluric transmission divided out, so the noise must be
    divided by it too -- dividing a spectrum by a transmission T divides its
    noise by T. Skipping it inflates W = lam^2 (dS/dlam)^2 / sigma^2 by 1/T^2
    and makes absorbed regions look artificially precise.

    Checked by consequence: with the noise left inconsistent, telluric-affected
    orders come out better than they should, so making it consistent must
    penalise them more than it penalises clean orders.
    """
    import specsim.analyze as az
    sim = fresh_sim
    sim.snr()
    consistent = sim.rv_precision(telluric_cutoff=0.2, velocity_cutoff=2)

    # recompute with the old, inconsistent noise by intercepting get_rv_precision
    captured = {}
    real = az.get_rv_precision

    def inconsistent(v, s, n, *a, **kw):
        captured['scaled_noise'] = n
        return real(v, s, sim.spectrograph.noise, *a, **kw)  # un-scaled noise, as before

    try:
        az.get_rv_precision = inconsistent
        before = sim.rv_precision(telluric_cutoff=0.2, velocity_cutoff=2)
    finally:
        az.get_rv_precision = real

    # the noise actually handed to get_rv_precision must have been scaled up
    assert np.nanmedian(captured['scaled_noise']) > np.nanmedian(sim.spectrograph.noise)

    _, tel_o, _ = order_diagnostics(sim)
    ok = np.isfinite(consistent.rv_order) & np.isfinite(before.rv_order)
    lo, hi = np.quantile(tel_o[ok], [0.25, 0.75])
    dirty, clean = ok & (tel_o <= lo), ok & (tel_o >= hi)

    ratio_dirty = np.median(consistent.rv_order[dirty] / before.rv_order[dirty])
    ratio_clean = np.median(consistent.rv_order[clean] / before.rv_order[clean])
    assert ratio_dirty > 1.0          # telluric orders get (correctly) worse
    assert ratio_dirty > ratio_clean  # and more so than clean ones
