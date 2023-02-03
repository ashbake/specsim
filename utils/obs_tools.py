##############################################################
# General functions for calc_snr_max
###############################################################

import numpy as np
from scipy.integrate import trapz
from scipy import signal
from scipy import signal, interpolate

from functions import tophat

all = {}


def calc_plate_scale(pixel_pitch, D=10, fratio=35):
    """
    D: diameter in meters
    fratio: 35 default

    return :
    -------
    platescale_arcsec_pix
    """
    platescale_arcsec_um = 206265 / fratio / (D * 10**6) #arc/um
    platescale_arcsec_pix = platescale_arcsec_um * pixel_pitch
    return platescale_arcsec_pix


def get_tracking_cam(camera='h2rg'):
    """
    gary assumes 0.9 for the QE of the H2RG, so modify throughput accordingly
    """
    if camera=='h2rg':
        rn = 12 #e-
        pixel_pitch = 18 #um
        qe_mod = 1 # relative to what is assumed in the throughput model
        dark=0.8 #e-/s

    if camera=='alladin':
        rn = 43 #e- ??????
        pixel_pitch = 18 #um ?????
        qe_mod = 0.8/0.9 # ???? not sure if we know this
        dark = 0.2 # e-/s

    if camera=='cred2':
        rn = 40 #e-  what is claimed online
        pixel_pitch = 15 #um https://www.axiomoptics.com/products/c-red-2/
        qe_mod = 0.8/.9 # 
        dark=600 #e-/s liquid cooling mode -40

    if camera=='cred2_xswir':
        #extended NIR
        rn = 50 #e-  what is claimed online
        pixel_pitch = 15 #um https://www.axiomoptics.com/products/c-red-2/
        qe_mod = 0.8 /0.9 # 
        dark=10000 #e-/s liquid cooling mode -40

    return rn, pixel_pitch, qe_mod, dark

def get_tracking_band(wave,band):
    """
    pick tracking band and get some stats on it

    update to 
    https://home.ifa.hawaii.edu/users/tokunaga/MKO-NIR_filter_set.html#yfilter
    """
    if band=='z':
        l0,lf = 800,950
        center_wavelength = 875
        bandpass = tophat(wave,l0,lf,1) #make up fake band

    if band=='y':
        l0,lf = 970,1070
        center_wavelength = 1050
        bandpass = tophat(wave,l0,lf,1) #make up fake band

    if band=='JHgap':
        l0,lf= 1335,1485
        center_wavelength = 1400
        bandpass = tophat(wave,l0,lf,1)

    if band=='J' or band=='H' or band=='K':
        # UPDATE THIS
        x,y = load_filter(so,'Johnson',1)
        y[np.where(y>0.2)] =1
        f = interpolate.interp1d(x,y, bounds_error=False,fill_value=0)
        bandpass = f(wave)
        center_wavelength = np.average(x,weights=y)

    return bandpass, center_wavelength

def get_fwhm(wfe,tt_resid,wavelength,diam,platescale,offaxis=0,getall=False):
    """
    combine DL by strehlt and tip/tilt error and off axis

    inputs:
    -------
    platescale: [arcsec/pixel]
        plate scale of image

    to do:
    update fwhm_tt and fwhm_offaxis
    """
    # get WFE
    strehl = np.exp(-(2*np.pi*wfe/wavelength)**2)

    # Diffraction limited spot with High Order WFE FWHM
    diffraction_spot_arcsec = 206265 * wavelength/ (diam * 10**9) # arcsec
    diffraction_spot_pix = diffraction_spot_arcsec / platescale
    fwhm_ho = diffraction_spot_pix / strehl**(1/4) # 1/strehl**.25 from dimitri, to account for broadening deviation from diffraction limit

    # Tip Tilt FWHM in pixels
    fwhm_tt = tt_resid*10**-3/platescale # assume 10 [arcsec] for now

    # FWHM from off axis aberrations in camera optics
    fwhm_offaxis=offaxis *4 # [pix] if off axis input is 1, assume at edge of field where RMS is 4 pix
    # ^ need to use mm instead of 4pix, and load camera pixel pitch
    # ^ also should just load the curve from Mitsuko of RMS diameter vs field angle
    
    fwhm = np.sqrt(fwhm_tt**2 + fwhm_ho**2 + fwhm_offaxis**2)

    if getall:
        return fwhm, strehl, diffraction_spot_pix, fwhm_ho, fwhm_tt, fwhm_offaxis, strehl, diffraction_spot_pix, fwhm_ho, fwhm_tt, fwhm_offaxis
    else:
        return fwhm



