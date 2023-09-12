# make median bin SNR in 4 hours plot
import os,sys
sys.path.append('../')
os.chdir('../')

import numpy as np
import matplotlib.pylab as plt

from specsim.load_inputs import fill_data
from specsim.objects import load_object
from specsim.functions import *

#plt.ion()

#load inputs
configfile = './configs/modhis_snr.cfg'
so    = load_object(configfile)
cload = fill_data(so) # put coupling files in load and wfe stuff too

# step through magnitudes
mag_arr= np.arange(8,22)
snr_arr = [] # snr 
for mag in mag_arr:
	cload.set_mag(so, mag)
	snr_arr.append(so.obs.snr_res_element)


# plot
plt.figure()
exec('xextent = so.inst.' + so.filt.band)
iband = np.where((so.obs.v_res_element > xextent[0]) & (so.obs.v_res_element <xextent[1]))[0]
plt.semilogy(mag_arr,np.median(np.array(snr_arr)[:,iband],axis=1),label=so.filt.band) # sqrt 3 hack to get res element snr
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
plt.savefig('./output/median_bin_snr_per_band.png')


