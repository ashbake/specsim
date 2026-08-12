# calc signal to noise
#%matplotlib
#
import sys,matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pylab as plt

font = {'size'   : 14}
matplotlib.rc('font', **font)

from pathlib import Path
try: sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
except: pass # if running in terminal must run from main directory to see specsim folder

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
	
	configfile = './configs/modhis_snr.cfg'
	so    = load_object(configfile)
	cload = fill_data(so) # put coupling files in load and wfe stuff too

	#plot_tools.plot_snr_orders(so,snrtype='res_element',mode='peak',savepath='./output/')
	#plt.axhline(30,c='k',ls='--')
	#plot_tools.plot_snr(so,snrtype='res_element',savepath='./output/')
	#plt.axhline(30,c='k',ls='--')
	cload.compute_rv(so,telluric_cutoff=0.2,velocity_cutoff=2)
	plot_tools.plot_rv_err(so,savefig=True,savepath='./')
	
