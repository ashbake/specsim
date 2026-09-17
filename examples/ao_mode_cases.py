# Compare SNR across three AO guide-star strategies for one science target.
#
# Each case is the same star, exposure and conditions, observed with a
# different AO mode fed by a different (off-axis) guide star. Only the [ao]
# inputs change between cases, so sim.set_ao() re-runs mode selection and
# reloads the fiber coupling, and everything upstream is reused.
import os, sys
from pathlib import Path
try:
	_root = str(Path(__file__).resolve().parent.parent)
except NameError: # running interactively; assume cwd is examples/
	_root = str(Path.cwd().parent)
sys.path.insert(0, _root)

import numpy as np

from specsim.config import simulate_from_config

# Data outside the specsim tree. These are handed straight to the spectrograph
# and the star rather than going through a config file, so specsim's own
# path resolution doesn't apply -- anchor them here so this runs from any folder.
def _sibling(*parts):
	return os.path.join(_root, '..', '..', *parts) + os.sep

# The AO modes below land on tip-tilt residuals of 5.5-7.5 mas, and the repo
# only ships the 3 mas coupling table, so point the spectrograph at the full
# coupling grid in the HISPEC throughput tree instead. Base throughput still
# comes from the repo's transmission_file -- only the coupling grid is swapped.
COUPLING_PATH = _sibling('HISPEC', '_data', 'throughput', 'hispec_subsystems', 'coupling')

# Likewise, the repo only ships 2300 K and 5800 K PHOENIX models; the target
# temperatures below need the fuller local grid.
PHOENIX_FOLDER = _sibling('_data', 'phoenix')

# Grid of science targets and default conditions
TEFFS = [3000, 9600]     # K
KMAGS = [10, 15]         # mag
TEXPS = [900, 14400]     # s
ZENITH_ANGLE = 30        # deg
SEEING = 'average'       # 'median' seeing in the old config is 'average' here

# AO cases: the guide star each mode is fed by, and how far off-axis it sits.
# Separation isn't a specsim input -- the WFE tables already fold the
# anisoplanatism into each mode -- so it's carried as a comment only.
AO_CASES = [
	dict(mode='LGS_STRAP',  mag=13.7, mag_band='V'),  # LGS + STRAP, 19" off-axis
	dict(mode='LGS_100J',   mag=14.2, mag_band='J'),  # LGS + IWA J, 5.5" off-axis
	dict(mode='PyWFS_100H', mag=9.3,  mag_band='H'),  # NGS IWA H, 5.5" off-axis
]

if __name__ == '__main__':
	i = 1  # which science target

	# filt band is fixed at build time (the whole scene is built on its
	# bandpass), so it's a constructor override rather than a setter
	sim = simulate_from_config(os.path.join(_root, 'configs', 'hispec_snr.cfg'), filt_band='K')
	sim.spectrograph.coupling_path = COUPLING_PATH

	# Pin the AO mode *before* moving the star. The config leaves mode='auto',
	# and set_star() re-runs mode selection -- under 'auto' that picks the best
	# mode for the new magnitude and loads its coupling table, which is wasted
	# work here, since none of the cases below use it.
	sim.set_ao(**AO_CASES[0])
	sim.set_star(teff=TEFFS[i], mag=KMAGS[i], phoenix_folder=PHOENIX_FOLDER)
	sim.set_obs(texp=TEXPS[i])
	sim.set_atmosphere(seeing_set=SEEING, zenith_angle=ZENITH_ANGLE)

	snrs = []
	for case in AO_CASES:
		obs = sim.set_ao(**case).snr()
		snrs.append([obs.snr_res_element, obs.snr_max_orders, obs.snr_mean_orders])
		print('%-12s %s=%-5s -> HO WFE %3s nm, tt %4s mas, median SNR/res element %s'
		      % (case['mode'], case['mag_band'], case['mag'], round(sim.ao_system.ho_wfe),
		         round(sim.ao_system.tt_dynamic, 2), round(float(np.nanmedian(obs.snr_res_element)), 1)))

	snr1, snr2, snr3 = snrs
