# calc signal to noise

import sys,matplotlib
import matplotlib.pylab as plt


font = {'size'   : 14}
matplotlib.rc('font', **font)

sys.path.append('../')
from specsim.objects import load_object
from specsim.load_inputs import fill_data
from specsim import plot_tools
import os

plt.ion()

if __name__=='__main__':
	#load inputs
	print("Current working directory:", os.getcwd())
	# Change current directory to parent directory
	if os.getcwd().split('/')[-1] == 'examples': os.chdir('..')
	
	configfile = './configs/hispec_etc_onaxis.cfg'
	so    = load_object(configfile)
	cload = fill_data(so) # put coupling files in load and wfe stuff too


	# plot
	plt.figure()
	plt.plot(so.obs.order_cens,so.obs.etc_order_mean)
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('t (s) to SNR %s' %so.obs.target_snr)
	plt.title('ETC Result')


	
