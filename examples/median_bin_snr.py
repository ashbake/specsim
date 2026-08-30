# make median bin SNR in 4 hours plot
import os,sys
from pathlib import Path
try:
	_root = str(Path(__file__).resolve().parent.parent)
except NameError: # running interactively; assume cwd is examples/
	_root = str(Path.cwd().parent)
sys.path.insert(0, _root)
os.chdir(_root)

import numpy as np
import matplotlib.pylab as plt

from specsim.config import simulate_from_config
from specsim.bandpass import YJHK

#plt.ion()

#load inputs
configfile = './configs/modhis_snr.cfg'
sim = simulate_from_config(configfile)

# step through magnitudes
mag_arr= np.arange(8,22)
snr_arr = [] # snr
for mag in mag_arr:
	sim.set_star_mag(mag)
	snr_arr.append(sim.snr().snr_res_element)


# plot
plt.figure()
xextent = YJHK[sim.filt.band]
v_res_element = sim.snr().v_res_element
iband = np.where((v_res_element > xextent[0]) & (v_res_element < xextent[1]))[0]
plt.semilogy(mag_arr,np.median(np.array(snr_arr)[:,iband],axis=1),label=sim.filt.band) # sqrt 3 hack to get res element snr
plt.plot(mag_arr,mag_arr*0 + 30,'k--')
plt.legend()
plt.xlabel('Magnitude')
plt.ylabel('Median bin SNR')
plt.title('MODHIS SNR in 4 hours')

my_xticks = mag_arr
plt.xticks(mag_arr, mag_arr)

plt.ylim(1,10000)
my_yticks = [10,100,1000]
plt.yticks(my_yticks,my_yticks)
plt.subplots_adjust(left=0.15,bottom=0.15)
plt.grid()
plt.text(9,33,'SNR=30')
plt.savefig('./examples/output/median_bin_snr_per_band.png')

plt.show()
