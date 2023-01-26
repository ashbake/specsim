##############################################################
# General functions for calc_snr_max
###############################################################

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy import signal

all = {}

def get_tip_tilt_resid(Vmag, mode):
    """
    load data from haka sims, spit out tip tilt
    """
    modes = np.array(['K','SH','80J','80H','80JK','LGS'])# corresponding modes to match assumptions of text files 
    imode = np.where(modes==mode)[0]

    #load data file
    f = np.loadtxt('./data/WFE/HAKA/Kstar_tiptilt.txt').T
    vmags = f[0]
    tip_tilts = f[1:][imode] # switch to take best mode for observing case

    #interpolate
    f_tt = interpolate.interp1d(vmags,tip_tilts, bounds_error=False,fill_value=10000)

    return f_tt(Vmag)

def get_HO_WFE(Vmag, mode):
    """
    load data from haka sims, spit out tip tilt
    """
    modes = np.array(['K','SH','80J','80H','80JK','LGS'])# corresponding modes to match assumptions of text files 
    imode = np.where(modes==mode)[0][0]

    #load data file
    f = np.loadtxt('./data/WFE/HAKA/Kstar_HOwfe.txt').T
    vmags = f[0]
    wfes = f[1:][imode]

    #interpolate
    f_wfe = interpolate.interp1d(vmags,wfes, bounds_error=False,fill_value=10000)

    return f_wfe(Vmag)


#
#
#
#
#
#
# OLD PAST HERE
def get_dyn_wfe(Rmag):
    """
    ***fix this - only return wfe that is important for centroiding based
    on josh's simulations*** maybe it's all but probs should take out tip/tilt
    
    # R mag inputed!
    https://www2.keck.hawaii.edu/optics/lgsao/performance.html

    should get this soon from rich
    """
    mag_arr = np.linspace(0,20,21) #sampling to match following arrays
    # old haka - to update
    aowfe_ngs=np.array([133,134,134,134,135,137,140,144,153,162,182,209,253,311,398,510,671,960,1446,2238,3505])
    aowfe_lgs=np.array([229,229,229,229,229,229,230,230,231,231,232,235,238,247,255,281,301,345,434,601,889])

    f_ngs = interpolate.interp1d(mag_arr,aowfe_ngs, bounds_error=False,fill_value=10000)
    f_lgs = interpolate.interp1d(mag_arr,aowfe_lgs, bounds_error=False,fill_value=10000)

    wfe = np.min([f_ngs(Rmag), f_lgs(Rmag)])

    if Rmag>20:
        print("WARNING, Rmag outside bounds of WFE data")

    return wfe



