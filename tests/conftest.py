"""
Shared fixtures for the specsim test suite.

Tests exercise the real data files shipped in data/ (PHOENIX/Sonora model
spectra, filter curves, zeropoints) rather than mocking them, since the
whole point of specsim is numeric behavior tied to those files.
"""
import glob
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from specsim.functions import integrate
from specsim.objects import storage_object

DATA_DIR = os.path.join(REPO_ROOT, "data")
PHOENIX_DIR = os.path.join(DATA_DIR, "stel", "phoenix") + os.sep
SONORA_DIR = os.path.join(DATA_DIR, "stel", "sonora") + os.sep
FILTER_DIR = os.path.join(DATA_DIR, "filters") + os.sep
ZP_FILE = os.path.join(DATA_DIR, "filters", "zeropoints.txt")


def _load_filt(so, family, band):
    """Populate so.filt the way fill_data.filter() does (load_inputs.py),
    minus resampling the transmission curve onto an instrument grid, which
    the tests below don't need."""
    zps = np.loadtxt(so.filt.zp_file, dtype=str).T
    izp = np.where((zps[0] == family) & (zps[1] == band))[0]
    if len(izp) == 0:
        raise ValueError(f"No zeropoint found for {family}/{band} in {so.filt.zp_file}")
    so.filt.zp = float(zps[2][izp])
    so.filt.family = family
    so.filt.band = band
    so.filt.filter_file = glob.glob(so.filt.filter_path + "*" + family + "*" + band + ".dat")[0]
    so.filt.xraw, so.filt.yraw = np.loadtxt(so.filt.filter_file).T
    if np.max(so.filt.xraw) > 5000:
        so.filt.xraw = so.filt.xraw / 10  # Angstrom -> nm
    if np.max(so.filt.xraw) < 10:
        so.filt.xraw = so.filt.xraw * 1000  # micron -> nm
    so.filt.dl_l = np.mean(integrate(so.filt.xraw, so.filt.yraw) / so.filt.xraw)
    so.filt.center_wavelength = integrate(so.filt.xraw, so.filt.yraw * so.filt.xraw) / integrate(
        so.filt.xraw, so.filt.yraw
    )
    return so


@pytest.fixture
def make_so():
    """Factory fixture: build a minimal storage_object with a photometric
    filter band loaded and stellar model paths set, ready to pass into
    specsim.source_tools functions directly (without running the full
    fill_data pipeline)."""

    def _make(family="2mass", band="H"):
        so = storage_object()
        so.filt.zp_file = ZP_FILE
        so.filt.filter_path = FILTER_DIR
        so.stel.phoenix_folder = PHOENIX_DIR
        so.stel.sonora_folder = SONORA_DIR
        so.stel.logg = 4.5
        _load_filt(so, family, band)
        return so

    return _make
