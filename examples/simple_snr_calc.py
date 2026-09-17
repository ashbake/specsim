# calc signal to noise
#%matplotlib
#
import sys,matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pylab as plt

font = {'size'   : 14}
matplotlib.rc('font', **font)

from pathlib import Path
try:
	ROOT = Path(__file__).resolve().parent.parent
except NameError: # running interactively; assume cwd is examples/
	ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))

from specsim.config import simulate_from_config
from specsim import plot
import os

plt.ion()

if __name__=='__main__':
	#load inputs
	configfile = str(ROOT / 'configs' / 'modhis_snr.cfg')
	sim = simulate_from_config(configfile)

	observation = sim.snr()
	plot.plot_snr_orders(sim.spectrograph, sim.ao_system, sim.filt, sim.star, snrtype='res_element', mode='peak',savepath='./output/')
	#plt.axhline(30,c='k',ls='--')
	#plot.plot_snr(sim.spectrograph, sim.ao_system, sim.filt, sim.star, snrtype='res_element', savepath='./output/')
	#plt.axhline(30,c='k',ls='--')
	rv = sim.rv_precision(telluric_cutoff=0.2, velocity_cutoff=2)
	plot.plot_rv_err(sim.spectrograph, rv, sim.atmosphere, sim.star, sim.filt, savefig=True, savepath='./output/')
