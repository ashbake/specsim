##############################################################
# General functions for calc_snr_max
###############################################################

import numpy as np
from scipy.integrate import trapz
from scipy import signal
from scipy import signal, interpolate

from functions import tophat
import load_inputs

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


def get_tracking_cam(camera='h2rg',x=None):
    """
    gary assumes 0.9 for the QE of the H2RG, so modify throughput accordingly

    this function is fed an x array which is used only in the cred2 selection because the QE profile
    is different from the h2rg
    """
    if camera=='h2rg':
        rn = 12 #e-
        pixel_pitch = 18 #um
        qe_mod = 1 # relative to what is assumed in the throughput model
        dark=0.8 #e-/s/pix
        saturation = 80000

    if camera=='perfect':
        rn = 0 #e-
        pixel_pitch = 18 #um
        qe_mod = 1 # relative to what is assumed in the throughput model
        dark=0 #e-/s/pix
        saturation = 80000

    if camera=='cred2_kpic':
        rn = 45 #e- Calvin measured 40e-, spec is 30e-, ashley measured 35 cds without 8th column noise
        pixel_pitch = 15 #um https://www.axiomoptics.com/products/c-red-2/
        if np.any(x==None): qe_mod=1
        else: qe_mod = tophat(x,980,1650,1) # scale  
        dark=315 #e-/s liquid cooling mode -40, calvin measured 450e-, spec sheet is 600e-,measured 315e- from KPIC C-RED2
        saturation = 33000

    if camera=='cred2':
        rn = 30 #e- Calvin measured 40e-, spec is 30e-, ashley measured 35 cds without 8th column noise
        pixel_pitch = 15 #um https://www.axiomoptics.com/products/c-red-2/
        if np.any(x==None): qe_mod=1
        else: qe_mod = tophat(x,980,1650,1) # scale  
        dark=315 #e-/s liquid cooling mode -40, calvin measured 450e-, spec sheet is 600e-,measured 315e- from KPIC C-RED2
        saturation = 33000

    if camera=='cred2_rn25':
        rn = 25 #e- spie paper for 20 reads which is max for 27 FPS integration time
        pixel_pitch = 15 #um https://www.axiomoptics.com/products/c-red-2/
        if np.any(x==None): qe_mod=1
        else: qe_mod = tophat(x,980,1650,1) # scale  
        dark=315 #e-/s liquid cooling mode -40, calvin measured 450e-, spec sheet is 600e-,measured 315e- from KPIC C-RED2
        saturation = 33000

    if camera=='cred2_rn20':
        rn = 20 #e- spie paper for 20 reads which is max for 27 FPS integration time
        pixel_pitch = 15 #um https://www.axiomoptics.com/products/c-red-2/
        if np.any(x==None): qe_mod=1
        else: qe_mod = tophat(x,980,1650,1) # scale  
        dark=315 #e-/s liquid cooling mode -40, calvin measured 450e-, spec sheet is 600e-,measured 315e- from KPIC C-RED2
        saturation = 33000

    if camera=='cred2_rn20':
        rn = 20 #e- spie paper for 20 reads which is max for 27 FPS integration time
        pixel_pitch = 15 #um https://www.axiomoptics.com/products/c-red-2/
        if np.any(x==None): qe_mod=1
        else: qe_mod = tophat(x,980,1650,1) # scale  
        dark=315 #e-/s liquid cooling mode -40, calvin measured 450e-, spec sheet is 600e-,measured 315e- from KPIC C-RED2
        saturation = 33000

    return rn, pixel_pitch, qe_mod, dark, saturation


def get_tracking_optics_aberrations(field_r,camera,ploton=False,filepath=None):
    """
    loads PSF size of tracking optics 

    intputs:
    --------
    field_r (float, 0-3) [arcsec]
        field radius position of tracking star on guide camera field
    
    camera (str, 'h2rg' or 'cred2')
        camera to assume for converting um to pixels

    ploton (bool)
        plots the psf RMS vs field position in arcsec

    filepath (str)
        path and filename to file containing optics aberrations in field position and rms per wavelengths
    returns:
    -------
    RMS of the PSF due to optical aberrations in pixels (radius rms)
    """
    f = np.loadtxt(filepath)
    field, rmstot, rms900,rms1000,rms1200,rms1400,rms1600,rms2200  = f.T #field [deg], rms [um]
    _,pixel_pitch,_,_,_ = get_tracking_cam(camera=camera,x=None)

    # should interpolate across wavelength but theyre not so different so just use 1400nm
    # multiply rms by 2 to get diameter (closer to FWHM)
    f = interpolate.interp1d(field * 3600, rms1400/pixel_pitch,bounds_error=False,fill_value='extrapolate')

    if ploton:
        plt.figure()
        # multiply rms by sqrt (2) to get a diagonal cut, multiple by 2 to get diameter
        plt.plot(field*3600,np.sqrt(2) * rmstot/pixel_pitch,label='total') 
        plt.plot(field*3600,np.sqrt(2) * rms900/pixel_pitch,label='900nm')
        plt.plot(field*3600,np.sqrt(2) * rms2200/pixel_pitch,label='2200nm')
        plt.xlabel('Field [arcsec]')
        plt.ylabel('RMS Radius [pix]')
        plt.title('Tracking Camera Spot RMS')
        plt.legend()

    return f(field_r)


def get_tracking_band(wave,band):
    """
    pick tracking band and get some stats on it

    update to 
    https://home.ifa.hawaii.edu/users/tokunaga/MKO-NIR_filter_set.html#yfilter
    """
    if band=='z':
        l0,lf = 820,970
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1) #make up fake band

    if band=='y':
        l0,lf = 970,1070
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1) #make up fake band

    if band=='JHgap':
        l0,lf= 1335,1490
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='JHgap_minus':
        l0,lf= 1400,1490
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='J':
        l0,lf= 1170,1330 #
        center_wavelength =  (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='Jplus':
        l0,lf= 1130,1490 #might help cred2
        center_wavelength =  (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='Hplus':
        l0,lf= 1330,1780 #1450 cuts into jh gap a little, 1950 before instrument bkg turn on
        center_wavelength =  (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='H':
        l0,lf= 1490,1780
        center_wavelength =  (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='Hplus50':
        # for consideration for c-red2
        l0,lf= 1330,1780
        center_wavelength =  (l0+lf)/2
        bandpass = tophat(wave,l0,lf,0.5)
        #bandpass[np.where(wave >1490)]*=0.5 # could make jhgap 1 but would make filter more difficult maybe to make

    if band=='JHplus20':
        # for consideration for c-red2 ....meh
        l0,lf= 1170,1780
        center_wavelength =  (l0+lf)/2
        bandpass = tophat(wave,l0,lf,0.2)
        #bandpass[np.where(wave >1490)]=1 # could make jhgap 1 but would make filter more difficult maybe to make

    if band=='JHplus':
        # for consideration for c-red2 ....meh
        l0,lf= 1170,1780
        center_wavelength =  (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)
        #bandpass[np.where(wave >1490)]=1 # could make jhgap 1 but would make filter more difficult maybe to make
        
    if band=='K':
        l0,lf= 1950,2460
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='Hkpic':
        l0,lf= 1500,1650
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='yJH':
        l0,lf= 980,1780
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)
    
    if band=='yJ':
        l0,lf= 980,1490
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    return bandpass, center_wavelength

def get_fwhm(wfe,tt_resid,wavelength,diam,platescale,field_r=0,camera='h2rg',getall=False,aberrations_file=None):
    """
    combine DL by strehlt and tip/tilt error and off axis

    inputs:
    -------
    platescale: [arcsec/pixel]
        plate scale of image

    to do:
    check how RMS relates to FWHM
    """
    rms_to_fwhm = 1/0.44 # from KAON, not too off from gaussian 1sig to FWHM factor
    radius_to_diam = 2
    
    # get WFE
    strehl = np.exp(-(2*np.pi*wfe/wavelength)**2)

    # Diffraction limited spot with High Order WFE FWHM
    diffraction_spot_arcsec = 206265 * wavelength/ (diam * 10**9) # arcsec
    diffraction_spot_pix = diffraction_spot_arcsec / platescale
    fwhm_ho = diffraction_spot_pix / strehl**(1/4) # 1/strehl**.25 from dimitri, to account for broadening deviation from diffraction limit

    # Tip Tilt FWHM in pixels
    fwhm_tt = rms_to_fwhm * tt_resid*1e-3/platescale 

    # FWHM from off axis aberrations in camera optics
    try:
        fwhm_offaxis     = radius_to_diam * get_tracking_optics_aberrations(field_r,camera,filepath=aberrations_file) # times to to get radius
    except:
        fwhm_offaxis = 0.5
        print('Cant find file %s' %aberrations_file)
        print('Warning: no tracking camera aberrations file found, assuming 0.5')
    
    fwhm = np.sqrt(fwhm_tt**2 + fwhm_ho**2 + fwhm_offaxis**2)

    if getall:
        return fwhm, strehl, diffraction_spot_pix, fwhm_ho, fwhm_tt, fwhm_offaxis, strehl, diffraction_spot_pix, fwhm_ho, fwhm_tt, fwhm_offaxis
    else:
        return fwhm

def compute_band_photon_counts():
    """
    """
    newmags = []
    all_bands = []
    Johnson_bands = ['U','B','V','R','I','J','H','K']
    for i,band in enumerate(Johnson_bands):
        newmags.append(load_inputs.get_band_mag(so,'Johnson',band,so.stel.factor_0))
        all_bands.append(band)

    #newmags.append(load_inputs.get_band_mag(so,'Sloan','uprime_filter',so.stel.factor_0))
    #all_bands.append('uprime_filter')

    load_inputs.get_band_mag(so,'SLOAN','uprime_filter',so.stel.factor_0)


def get_order_value(so,v,snr,height=0.055,distance=2e4,prominence=0.01):
    """
    given array, return max and mean of snr per order
    """
    order_peaks      = signal.find_peaks(so.inst.base_throughput,height=height,distance=distance,prominence=prominence)
    order_cen_lam    = so.stel.v[order_peaks[0]]
    blaze_angle      =  76
    snr_peaks = []
    snr_means = []
    for i,lam_cen in enumerate(order_cen_lam):
        line_spacing = 0.02 if lam_cen < 1475 else 0.01
        m = np.sin(blaze_angle*np.pi/180) * 2 * (1/line_spacing)/(lam_cen/1000)
        fsr  = lam_cen/m
        isub_test= np.where((so.stel.v> (lam_cen - fsr/2)) & (so.stel.v < (lam_cen+fsr/2))) #FINISH THIS
        #plt.plot(so.stel.v[isub_test],total_throughput[isub_test],'k--')
        sub_snr = snr[np.where((v > (lam_cen - 1.3*fsr/2)) & (v < (lam_cen+1.3*fsr/2)))[0]] #FINISH THIS]
        snr_peaks.append(np.nanmax(sub_snr))
        snr_means.append(np.nanmean(sub_snr))

    return np.array(order_cen_lam), np.array(snr_peaks), np.array(snr_means)

def get_order_value2(v,base_throughput,x,y):
    """
    given array, return max and mean of snr per order
    """
    order_peaks      = signal.find_peaks(base_throughput,height=0.055,distance=2e4,prominence=0.01)
    order_cen_lam    = v[order_peaks[0]]
    blaze_angle      =  76
    snr_peaks = []
    snr_means = []
    for i,lam_cen in enumerate(order_cen_lam):
        line_spacing = 0.02 if lam_cen < 1475 else 0.01
        m = np.sin(blaze_angle*np.pi/180) * 2 * (1/line_spacing)/(lam_cen/1000)
        fsr  = lam_cen/m
        isub_test= np.where((v> (lam_cen - fsr/2)) & (v < (lam_cen+fsr/2))) #FINISH THIS
        #plt.plot(so.stel.v[isub_test],total_throughput[isub_test],'k--')
        sub_snr = y[np.where((x > (lam_cen - 1.3*fsr/2)) & (x < (lam_cen+1.3*fsr/2)))[0]] #FINISH THIS]
        snr_peaks.append(np.nanmax(sub_snr))
        snr_means.append(np.nanmean(sub_snr))

    return np.array(order_cen_lam), np.array(snr_peaks), np.array(snr_means)


def air_index_refraction(lam,p,t):
    """
    https://iopscience.iop.org/article/10.1088/0026-1394/30/3/004/pdf
    edlen https://iopscience.iop.org/article/10.1088/0026-1394/2/2/002/pdf 
    P: torr
    t: celcius
    """
    sig = 10**7/lam * (1e-4) # 1e-4 cm/micron
    ns = 1 + (1/1e8) * (8342.13 + 2406030*(130 - sig**2)**(-1) + 15997*(38.9 -sig**2)**-1)
    n = 1 + (p * (ns -1)/ 720.775) * (1 + p*(0.817 - 0.0133*t)*(10**-6))/(1 + 0.0036610*t)
    return n


def load_confirmed_planets(planets_filename = './data/populations/confirmed_planets_PS_2023.01.12_16.07.07.csv'):
    planet_data =  pd.read_csv(planets_filename,delimiter=',',comment='#')
    # add brown dwarfs!
    hmags = planet_data['sy_hmag']
    teffs = planet_data['st_teff']
    return hmags,teffs


