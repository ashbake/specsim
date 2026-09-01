"""
Tests for specsim/bandpass.py: the loaded-band registry, instance
isolation, and the errors raised for a band that can't be loaded.

Note Bandpass._loaded is class-level state shared across the whole
session. clear_cache() is called only inside this file -- an autouse
clearing fixture would force every other test module to re-read filter
curves from disk for no benefit.
"""
import os

import numpy as np
import pytest

from specsim.bandpass import Bandpass, available_bands, load_filter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILTER_DIR = os.path.join(REPO_ROOT, "data", "filters") + os.sep
ZP_FILE = os.path.join(REPO_ROOT, "data", "filters", "zeropoints.txt")


@pytest.fixture
def clean_registry():
    Bandpass.clear_cache()
    yield
    Bandpass.clear_cache()


def load(band, family=None, x=None):
    return Bandpass.load(FILTER_DIR, ZP_FILE, band, family, x=x)


def test_loaded_records_each_band(clean_registry):
    load('H')
    load('K')
    assert set(Bandpass.loaded()) == {('2mass', 'H'), ('2mass', 'K')}


def test_registry_entries_are_never_resampled(clean_registry):
    "The aliasing regression test: what's handed out may be resampled, what's stored must stay pristine."
    grid = np.arange(1000, 2000, 0.5)
    bp = load('H', x=grid)
    assert bp.x is grid
    assert Bandpass.loaded()[('2mass', 'H')].x is None


def test_two_loads_are_independent(clean_registry):
    "AOSystem.select() loads bands in a loop; resample() mutates, so callers must not share an instance."
    g1, g2 = np.arange(1000, 2000, 0.5), np.arange(1200, 1800, 0.5)
    a, b = load('H', x=g1), load('H', x=g2)
    assert a is not b
    assert a.x is g1 and b.x is g2
    assert np.array_equal(a.xraw, b.xraw)  # the raw curve is still shared, which is fine -- nothing mutates it


def test_loaded_returns_copies(clean_registry):
    "Mutating what loaded() hands back must not corrupt the registry."
    load('H')
    Bandpass.loaded()[('2mass', 'H')].resample(np.arange(1000, 2000, 0.5))
    assert Bandpass.loaded()[('2mass', 'H')].x is None


def test_clear_cache_empties_registry_and_disk_caches(clean_registry):
    load('H')
    assert Bandpass.loaded()
    Bandpass.clear_cache()
    assert Bandpass.loaded() == {}
    assert load_filter.cache_info().currsize == 0


def test_family_is_derived_from_band(clean_registry):
    assert load('H').family == '2mass'
    assert load('y').family == 'cfht'
    assert load('R').family == 'Johnson'
    assert load('y', family='decam').family == 'decam'


def test_unknown_band_error_names_available_bands(clean_registry):
    with pytest.raises(ValueError, match="No filter curve") as exc:
        load('Q')
    assert '2mass H' in str(exc.value)  # the message lists what can be loaded


def test_missing_zeropoint_error(tmp_path, clean_registry):
    """
    A curve present on disk with no zeropoint row. Built explicitly rather
    than relying on the shipped TESS mismatch, whose failure mode differs
    between case-sensitive and case-insensitive filesystems.
    """
    filters = tmp_path / "filters"
    filters.mkdir()
    (filters / "Fake.Q.dat").write_text("1000 0.0\n1500 1.0\n2000 0.0\n")
    with pytest.raises(ValueError, match="No zeropoint"):
        Bandpass.load(str(filters) + os.sep, ZP_FILE, 'Q', family='Fake')


def test_available_bands_lists_the_zeropoint_table():
    pairs = available_bands(ZP_FILE)
    assert ('2mass', 'H') in pairs and ('cfht', 'y') in pairs
    assert pairs == sorted(pairs)
