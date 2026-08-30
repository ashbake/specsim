# calc signal to noise

import os,sys,matplotlib
import matplotlib.pylab as plt


font = {'size'   : 14}
matplotlib.rc('font', **font)

from pathlib import Path
try: sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
except: pass # if running in terminal must run from main directory to see specsim folder

from specsim.config import simulate_from_config

plt.ion()

if __name__=='__main__':
	#load inputs
	print("Current working directory:", os.getcwd())
	# Change current directory to parent directory
	if os.getcwd().split('/')[-1] == 'examples': os.chdir('..')

	configfile = './configs/hispec_etc_onaxis.cfg'
	sim = simulate_from_config(configfile)

	target_snr = 100
	etc = sim.exposure_time_for_snr(target_snr)

	# plot
	plt.figure()
	plt.plot(sim.spectrograph.order_cens,etc.etc_order_mean)
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('t (s) to SNR %s' %target_snr)
	plt.title('ETC Result')
