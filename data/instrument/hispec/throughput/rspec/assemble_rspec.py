# need to use tracking camera throughput arm
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, signal

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

SMALL_SIZE = 32
MEDIUM_SIZE = 40
BIGGER_SIZE = 48

plt.rcParams['font.size'] = '14'
plt.rcParams['font.family'] = 'sans'
plt.rcParams['axes.linewidth'] = '1.3'
fontname = 'Arial Narrow'


# modify echelle.csv with the canon throughput
wech,sech= np.loadtxt('echelle.csv', delimiter=',',skiprows=1).T
wcan,scan= np.loadtxt('echelle_canon_raw.csv', delimiter=',',skiprows=1).T

# find peaks of canon sim
test = signal.find_peaks(scan, height=0.59,width=0.001,rel_height = 0.1)
ipeaks = test[0]

# interp through peaks
def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

smooth_peaks = butter_lowpass_filter(scan[ipeaks], 2, 30,2)
f = interpolate.interp1d(wcan[ipeaks], smooth_peaks,bounds_error=False,fill_value="extrapolate")
echelle_data = 0.95 * f(wech) * sech 
np.savetxt('echelle_canon.csv',np.vstack((wech,echelle_data)).T,fmt='%f',delimiter=',')


# spec
au_file = 'protected_au.csv'
include = ['col1',  'col2',  'col3' , 'coldstop', 'echelle',      'xdis',    'cam1', 'cam2','cam3',   'qe']
files   = [au_file,au_file,au_file, 0.94,   'echelle_canon.csv', 'cx.csv', au_file, au_file, au_file, 0.9]
w   = np.arange(0.9,2.5,0.001)
s   = np.ones_like(w)

for i in files:
    if type(i) is float:
        s *= i
    else:
        wtemp,stemp= np.loadtxt(i, delimiter=',',skiprows=1).T
        f = interpolate.interp1d(wtemp, stemp,bounds_error=False,fill_value=0)
        s*=f(w)

plt.figure('RSPEC')
plt.plot(w,s)
plt.grid(True)
plt.xlabel('Wavelength (um)')
plt.ylabel('Transmission')
plt.title("HISPEC RSpec")
plt.savefig('rspec_throughput_new.png')

np.savetxt('rspec_throughput.csv',np.vstack((w,s)).T,fmt='%f',header="wavelength_um, throughput",delimiter=',')




