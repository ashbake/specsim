# Fiber coupling vs. wavefront error, and AO tip/tilt performance vs. guide star
# magnitude, for a generic HISPEC setup.
#
# Three plots:
#   1. coupling vs. wavelength for a few high-order WFE values
#   2. coupling vs. wavelength for a few dynamic tip/tilt values
#   3. tip/tilt residual vs. guide star magnitude for LGS_STRAP and LGS_100J
#
# The two sweeps behave differently under the hood, which is worth knowing before
# picking values. High-order WFE is applied analytically (Marechal), so any value
# works. Dynamic tip/tilt is a lookup: it is rounded to the nearest 0.5 mas and
# the matching pre-computed coupling file has to exist on disk. The repo ships
# exactly one of those files (3 mas), so this script points at the full HISPEC
# coupling grid instead -- see COUPLING_PATH.
import os, sys
from pathlib import Path
try:
	_root = str(Path(__file__).resolve().parent.parent)
except NameError: # running interactively; assume cwd is examples/
	_root = str(Path.cwd().parent)
sys.path.insert(0, _root)

import matplotlib.pylab as plt
import numpy as np
from scipy import interpolate

from specsim.aosystem import load_WFE
from specsim.bandpass import Bandpass
from specsim.config import simulate_from_config
from specsim.star import Star

# The full HISPEC coupling grid, which carries the whole 0-20 mas range rather
# than the single table the repo ships. Base throughput still comes from the
# repo's transmission_file. Point this at your own copy if it lives elsewhere.
COUPLING_PATH = os.path.join(_root, '..', '..', 'HISPEC', '_data', 'throughput', 'hispec_subsystems', 'coupling') + os.sep

CONFIG = os.path.join(_root, 'configs', 'hispec_snr.cfg')
SAVEPATH = './output/'

# The AO modes are tabulated against magnitude in their own band (LGS_STRAP in V,
# LGS_100J in J), so comparing them needs a common axis. Everything is put on J
# by colour-converting through an assumed guide star -- a 2700 K M dwarf here,
# which is a very red V-J ~ 5.8, so the conversion moves LGS_STRAP a long way.
# The 2700 K PHOENIX model is not one of the two the repo ships.
COLOR_TEFF = 2700
COLOR_LOGG = 4.5
PHOENIX_FOLDER = os.path.join(_root, '..', '..', '_data', 'phoenix') + os.sep

# Sequential sweeps get a sequential colormap (viridis: perceptually uniform and
# colorblind-safe), since the parameter is ordered rather than categorical.
SWEEP_CMAP = plt.get_cmap('viridis')
# The two AO modes are categorical -- a fixed, CVD-checked pair, not a cycled default.
MODE_COLORS = {'LGS_STRAP': '#1f77b4', 'LGS_100J': '#d95f02'}

HO_WFES = [90, 130, 170, 220, 280]        # nm RMS, at fixed tip/tilt
TT_DYNAMICS = [0.5, 1.0, 2.0, 4.0, 8.0]   # mas RMS, at fixed high-order WFE
REF_HO_WFE = 130                          # nm, held fixed for the tip/tilt sweep
REF_TT_DYNAMIC = 2.0                      # mas, held fixed for the WFE sweep
DECIMATE = 200                            # x is ~3.7M points; thin it for plotting
SENTINEL_TT = 100                         # tables use ~1000 mas to mean 'AO does not close here'


def load_hispec():
	"Build the HISPEC scene and point it at the full coupling grid."
	sim = simulate_from_config(CONFIG)
	sim.spectrograph.coupling_path = COUPLING_PATH
	return sim


def _coupling(sim, ho_wfe, tt_dynamic):
	"""
	Coupling on the (decimated) wavelength grid for one WFE/tip-tilt pair,
	trimmed to where the coupling tables are actually defined. The grid runs
	wider than the tabulated bandpass and is zero-filled outside it, which
	would otherwise draw a cliff down to zero at each end.
	"""
	sim.set_ao(ho_wfe=ho_wfe, tt_dynamic=tt_dynamic)
	x, coupling = sim.x[::DECIMATE], sim.spectrograph.coupling[::DECIMATE]
	valid = coupling > 0
	return x[valid], coupling[valid]


def plot_coupling_vs_ho_wfe(sim, ho_wfes=HO_WFES, tt_dynamic=REF_TT_DYNAMIC, savepath=SAVEPATH):
	"""
	Fiber coupling vs. wavelength for several high-order WFE values, at fixed
	dynamic tip/tilt. Coupling falls with WFE through the Marechal Strehl term,
	and the falloff is wavelength dependent -- exp(-(2*pi*wfe/lam)^2) bites
	hardest at the blue end.

	inputs
	------
	sim - Simulate, from load_hispec()
	ho_wfes - list of float [nm RMS]
	tt_dynamic - float [mas RMS], held fixed
	savepath - str, directory to write the figure into (must exist)

	output
	------
	fig, ax
	"""
	fig, ax = plt.subplots(figsize=(8, 5))
	for i, ho_wfe in enumerate(ho_wfes):
		x, coupling = _coupling(sim, ho_wfe, tt_dynamic)
		color = SWEEP_CMAP(i / max(len(ho_wfes) - 1, 1))
		# 5 series, so the legend carries identity; direct labels would collide at the curve ends
		ax.plot(x, coupling, lw=2, color=color, label='%s nm' % ho_wfe)

	ax.set_xlabel('Wavelength [nm]')
	ax.set_ylabel('Fiber coupling')
	ax.set_title('HISPEC coupling vs. high-order WFE (tip/tilt fixed at %s mas)' % tt_dynamic)
	ax.legend(title='HO WFE', fontsize=9)
	ax.grid(alpha=0.3)
	ax.set_ylim(0, None)
	fig.tight_layout()
	fig.savefig(os.path.join(savepath, 'coupling_vs_ho_wfe.png'), dpi=150)
	return fig, ax


def plot_coupling_vs_tt_dynamic(sim, tt_dynamics=TT_DYNAMICS, ho_wfe=REF_HO_WFE, savepath=SAVEPATH):
	"""
	Fiber coupling vs. wavelength for several dynamic tip/tilt values, at fixed
	high-order WFE. Unlike the WFE sweep, each value here is a lookup into the
	pre-computed coupling grid, rounded to the nearest 0.5 mas -- so values off
	that grid, or outside what is on disk, raise a missing-file error.

	inputs
	------
	sim - Simulate, from load_hispec()
	tt_dynamics - list of float [mas RMS], on the 0.5 mas grid
	ho_wfe - float [nm RMS], held fixed
	savepath - str, directory to write the figure into (must exist)

	output
	------
	fig, ax
	"""
	fig, ax = plt.subplots(figsize=(8, 5))
	for i, tt_dynamic in enumerate(tt_dynamics):
		x, coupling = _coupling(sim, ho_wfe, tt_dynamic)
		color = SWEEP_CMAP(i / max(len(tt_dynamics) - 1, 1))
		ax.plot(x, coupling, lw=2, color=color, label='%s mas' % tt_dynamic)

	ax.set_xlabel('Wavelength [nm]')
	ax.set_ylabel('Fiber coupling')
	ax.set_title('HISPEC coupling vs. dynamic tip/tilt (HO WFE fixed at %s nm)' % ho_wfe)
	ax.legend(title='TT dynamic', fontsize=9)
	ax.grid(alpha=0.3)
	ax.set_ylim(0, None)
	fig.tight_layout()
	fig.savefig(os.path.join(savepath, 'coupling_vs_tt_dynamic.png'), dpi=150)
	return fig, ax


def band_minus_j(sim, band, teff=COLOR_TEFF, phoenix_folder=PHOENIX_FOLDER):
	"""
	Colour (m_band - m_J) of an assumed guide star, used to put AO modes
	tabulated in different bands onto a common J axis. A star with J = m has
	m_band = m + this.

	Built from a standalone Star rather than sim.set_star(), so it doesn't drag
	the AO/coupling chain along just to read a colour off a model spectrum.

	inputs
	------
	sim - Simulate, for its wavelength grid and filter paths
	band - str, the band to convert from (returns exactly 0 for 'J')
	teff - float [K], assumed guide star temperature; the colour depends
		strongly on this (V-J is ~1.0 at 5800 K but ~5.8 at 2700 K)
	phoenix_folder - str, model grid to read teff from

	output
	------
	float, m_band - m_J
	"""
	if band == 'J':
		return 0.0
	bp_band = Bandpass.load(sim.filter_path, sim.zp_file, band)
	bp_j = Bandpass.load(sim.filter_path, sim.zp_file, 'J')
	# start from the scene star so the model folders come along, then override
	# everything that defines this hypothetical guide star. PHOENIX_FOLDER only
	# has to be set when the wanted teff isn't in the folder the scene uses.
	star = Star(sim.star.params, teff=teff, logg=COLOR_LOGG, mag=10, vsini=0, rv=0,
	             phoenix_folder=phoenix_folder or sim.star.params.phoenix_folder).load(sim.x, bp_j)
	return star.magnitude_in_band(bp_band) - star.magnitude_in_band(bp_j)


def plot_tt_dynamic_vs_mag(sim, modes=('LGS_STRAP', 'LGS_100J'), mags_j=np.arange(5, 16.01, 0.25),
                            teff=COLOR_TEFF, savepath=SAVEPATH):
	"""
	Tip/tilt residual vs. J magnitude for a set of AO modes, read off the
	instrument's WFE tables (the same interpolation AOSystem.select() does) at
	the scene's zenith angle and seeing.

	The tables are indexed by magnitude in each mode's OWN band -- LGS_STRAP in
	V, LGS_100J in J -- which makes the modes incomparable as tabulated. Each
	is colour-converted onto a common J axis for an assumed guide star of
	`teff`: a mode tabulated in band B is sampled at B = J + (B - J).

	The conversion is only as good as that Teff assumption, and for a red star
	it is a big shift -- at 2700 K, V - J is about 5.8 mag, so LGS_STRAP's
	V-band table is sampled ~6 mag fainter than the J value plotted.

	Note the tables' column header reads 'WFE[nm]', but specsim consumes these
	values as a tip/tilt angle in mas (they feed tt_to_strehl and the
	ttDynamic<x>masRMS coupling lookup), so that is how they are plotted here.

	inputs
	------
	sim - Simulate, from load_hispec(); supplies the WFE file paths, zenith
		angle and seeing
	modes - iterable of str, AO mode names present in the WFE tables
	mags_j - array of guide star J magnitude to evaluate
	teff - float [K], assumed guide star temperature for the colour conversion
	savepath - str, directory to write the figure into (must exist)

	output
	------
	fig, ax
	"""
	data = load_WFE(sim.ao_system.ho_wfe_file, sim.ao_system.tt_dynamic_file,
	                 sim.zenith_angle, sim.atmosphere.seeing_set)

	fig, ax = plt.subplots(figsize=(8, 5))
	for mode in modes:
		if mode not in data:
			raise ValueError('%r is not in the WFE tables. Modes: %s' % (mode, ', '.join(data)))
		entry = data[mode]
		colour = band_minus_j(sim, entry['band'], teff=teff)
		native_mags = mags_j + colour  # the magnitude the table is actually indexed by

		f_tt = interpolate.interp1d(entry['tt_mag'], entry['tt_wfe'], bounds_error=False, fill_value=np.nan)
		tt = f_tt(native_mags)
		tt[tt >= SENTINEL_TT] = np.nan  # the tables park failed AO at a placeholder value
		color = MODE_COLORS.get(mode)
		label = mode if entry['band'] == 'J' else '%s (%s = J %+.1f)' % (mode, entry['band'], colour)
		ax.plot(mags_j, tt, lw=2, color=color, label=label)
		finite = np.where(np.isfinite(tt))[0]
		if len(finite):
			ax.annotate(mode, (mags_j[finite[-1]], tt[finite[-1]]), xytext=(5, 0),
			            textcoords='offset points', va='center', fontsize=9, color=color)

	ax.set_xlabel('Guide star J magnitude')
	ax.set_ylabel('Tip/tilt residual [mas RMS]')
	ax.set_title('AO tip/tilt performance vs. J magnitude\n'
	              '(%s seeing, %s deg zenith, colours for an assumed %sK guide star)'
	              % (sim.atmosphere.seeing_set, sim.zenith_angle, teff))
	ax.legend(fontsize=9)
	ax.grid(alpha=0.3)
	ax.set_xlim(mags_j.min(), mags_j.max() + 0.8)  # room for the direct labels
	fig.tight_layout()
	fig.savefig(os.path.join(savepath, 'tt_dynamic_vs_mag.png'), dpi=150)
	return fig, ax


if __name__ == '__main__':
	os.makedirs(SAVEPATH, exist_ok=True)
	sim = load_hispec()

	plot_coupling_vs_ho_wfe(sim)
	plot_coupling_vs_tt_dynamic(sim)
	plot_tt_dynamic_vs_mag(sim)

	print('wrote %s to %s' % (', '.join(sorted(os.listdir(SAVEPATH))), os.path.abspath(SAVEPATH)))
	plt.show()
