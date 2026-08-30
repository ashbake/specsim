"""
Shared fixtures for the specsim test suite.

Tests exercise the real data files shipped in data/ (PHOENIX/Sonora model
spectra, filter curves, zeropoints) rather than mocking them, since the
whole point of specsim is numeric behavior tied to those files.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from specsim.bandpass import Bandpass

DATA_DIR = os.path.join(REPO_ROOT, "data")
PHOENIX_DIR = os.path.join(DATA_DIR, "stel", "phoenix") + os.sep
SONORA_DIR = os.path.join(DATA_DIR, "stel", "sonora") + os.sep
FILTER_DIR = os.path.join(DATA_DIR, "filters") + os.sep
ZP_FILE = os.path.join(DATA_DIR, "filters", "zeropoints.txt")


@pytest.fixture
def make_bandpass():
    "Factory fixture: load a Bandpass for the given band, ready to pass into specsim.star functions. family is derived from the band unless overridden."

    def _make(band="H", family=None):
        return Bandpass.load(FILTER_DIR, ZP_FILE, band, family)

    return _make
