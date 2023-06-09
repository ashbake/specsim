import sys
import matplotlib
import numpy as np
import matplotlib.pylab as plt
from scipy import interpolate

font = {'size'   : 14}
matplotlib.rc('font', **font)

sys.path.append('./utils/')
from objects import load_object
from load_inputs import fill_data, load_filter
from functions import *
from noise_tools import get_sky_bg, get_inst_bg, sum_total_noise
from throughput_tools import pick_coupling, get_band_mag, get_base_throughput
from wfe_tools import get_tip_tilt_resid, get_HO_WFE
import pandas as pd
plt.ion()


def compute_ccf_snr_matchedfilter(signal, model, total_noise, sky_trans, systematics_residuals=0.01,kernel_size=501,norm_cutoff=0.8):
    '''
    Calculate the Cross-correlation function signal to noise ration with a matched filter
    Inputs:
    signal      - Your observed spectrum
    model       - Your model spectrum
    total_noise - Your total noise estimate in the same units as your signal
    sky_trans   - The sky transmission
    systematics_residuals - A multiplicative factor that estimates the residual level of the host star spectrum and telluric lines in your signal (Default of 1%)
    kernel_size  - The default high-pass filter size.
    norm_cutoff  - A cutoff below which we don't calculate the ccf-snr
    '''
    #Get the noise variance
    total_noise_var = total_noise**2 
    bad_noise = np.isnan(total_noise_var)
    total_noise_var[bad_noise]=np.inf

    #Calculate some normalization factor
    #Dimitri to explain this better. 
    norm = ((1-systematics_residuals)*sky_trans)
    
    #Get a median-filtered version of your model spectrum
    model_medfilt = medfilt(model,kernel_size=kernel_size)
    #Subtract the median version from the original model, effectively high-pass filtering the model
    model_filt = model-model_medfilt*model.unit
    model_filt[np.isnan(model_filt)] = 0.
    model_filt[norm<norm_cutoff] = 0.
    model_filt[bad_noise] = 0.

    #Divide out the sky transmision
    normed_signal = signal/norm
    #High-pass filter like with the model
    signal_medfilt = medfilt(normed_signal,kernel_size=kernel_size)
    signal_filt = normed_signal-signal_medfilt*normed_signal.unit
    signal_filt[np.isnan(signal_filt)] = 0.
    signal_filt[norm<norm_cutoff] = 0.
    signal_filt[bad_noise] = 0.
    
    #Now the actual ccf_snr
    ccf_snr = np.sqrt((np.sum(signal_filt * model_filt/total_noise_var))**2 / np.sum(model_filt * model_filt/total_noise_var))

    return ccf_snr



if __name__=='__main__':
	#load inputs
	configfile = './configs/ilocater_hd189733.cfg'
	so	= load_object(configfile)
	cload = fill_data(so) # put coupling files in load and wfe stuff too






