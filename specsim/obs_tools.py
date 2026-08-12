##############################################################
# General functions for calc_snr_max
###############################################################

import numpy as np
from scipy import signal
from scipy import signal, interpolate

from specsim.functions import tophat
from specsim import load_inputs
from specsim.ccf_tools import get_order_bounds
all = {}


def calc_plate_scale(pixel_pitch, D=10, fratio=35):
    """
    Compute the on-sky plate scale of a detector given its pixel size and
    the telescope/beam geometry.

    inputs:
    -------
    pixel_pitch - float
        detector pixel pitch [um]

    D - float
        telescope diameter [m] (default 10)

    fratio - float
        beam focal ratio (f-number), unitless (default 35)

    return:
    -------
    platescale_arcsec_pix - float
        plate scale [arcsec/pixel]
    """
    platescale_arcsec_um = 206265 / fratio / (D * 10**6) #arc/um
    platescale_arcsec_pix = platescale_arcsec_um * pixel_pitch
    return platescale_arcsec_pix


def get_tracking_cam(camera='h2rg',x=None):
    """
    Return the detector properties of the selected tracking/guide camera.

    Gary assumes 0.9 for the QE of the H2RG, so qe_mod modifies the
    throughput model accordingly. For the cred2-family cameras, qe_mod is
    instead a tophat scaling over the array x (980-1650nm) since their QE
    profile differs from the H2RG.

    inputs:
    -------
    camera - str
        tracking camera to select. One of 'h2rg', 'perfect', 'cred2_kpic',
        'cred2', 'cred2_rn25', 'cred2_rn20' (default 'h2rg')

    x - array or None
        wavelength array [nm], used only to build the qe_mod tophat for the
        cred2-family cameras. If None, qe_mod defaults to 1 for those
        cameras.

    returns:
    --------
    rn - float
        read noise [e-]

    pixel_pitch - float
        pixel pitch [um]

    qe_mod - float or array
        QE scale factor relative to the throughput model, unitless (1 means
        no modification; may be a tophat array over x for cred2 cameras)

    dark - float
        dark current [e-/s/pix]

    saturation - float
        full well/saturation level [e-]
    """
    if camera=='h2rg':
        rn = 12 #e-
        pixel_pitch = 18 #um
        qe_mod = 0.888 # relative to what is assumed in the throughput model
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
    Load PSF size of the tracking camera optics as a function of field
    position and interpolate to the requested field radius.

    Reads a file of field position vs RMS spot size (per wavelength) for the
    tracking optics, converts the RMS at 1400nm from um to pixels (using the
    pixel pitch of the selected camera), and linearly interpolates
    (extrapolating outside the tabulated range) to field_r. Wavelength
    dependence is ignored beyond picking the 1400nm column, since the RMS
    does not vary much with wavelength across the tabulated columns.

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
    RMS of the PSF due to optical aberrations in pixels (radius rms), float,
    interpolated (or extrapolated) to field_r
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
    Build a tophat bandpass for the requested tracking-camera filter band
    and return its center wavelength.

    Band edges are hard-coded approximations of the named photometric/
    engineering bands (some tuned for KPIC/cred2 considerations rather than
    strict MKO definitions); see
    https://home.ifa.hawaii.edu/users/tokunaga/MKO-NIR_filter_set.html#yfilter
    for the standard definitions this should eventually be updated to match.

    inputs:
    -------
    wave - array
        wavelength array [nm] over which to evaluate the bandpass

    band - str
        name of the tracking band to select. One of 'z', 'y', 'JHgap',
        'JHgapKPIC', 'JHgap_minus', 'J', 'Jplus', 'Hplus', 'H', 'Hplus50',
        'JHplus20', 'JHplus', 'K', 'Hkpic', 'yJH', 'yJ', 'JHgap20J', 'JHgap20H'
        'JHgap_narrowed' (default 'JHgap')

    returns:
    --------
    bandpass - array
        tophat transmission evaluated at wave, unitless (0-1; 'Hplus50' and
        'JHplus20' use reduced flat throughput of 0.5 and 0.2 respectively)

    center_wavelength - float
        midpoint wavelength of the selected band [nm]
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

    if band=='JHgap20J':
        l0,lf= 1335,1490  # make these narrower bc this is pyramid case that steals a little bit
        center_wavelength = (l0+lf)/2
        bandpassJHgap = tophat(wave,l0,lf,1)

        l0,lf= 1170,1330 #J
        center_wavelength =  (l0+lf)/2
        bandpassJ = tophat(wave,l0,lf,1)

        bandpass = bandpassJHgap + 0.2 * bandpassJ

    if band=='JHgap20H':
        l0,lf= 1335,1490  # make these narrower bc this is pyramid case that steals a little bit
        center_wavelength = (l0+lf)/2
        bandpassJHgap = tophat(wave,l0,lf,1)

        l0,lf= 1490,1780
        center_wavelength =  (l0+lf)/2
        bandpassH = tophat(wave,l0,lf,1)

        bandpass = bandpassJHgap + 0.2 * bandpassH

    if band=='JHgap_narrowed':
        l0,lf= 1350,1470 # less bc pyramid takes some- double check how much 
        center_wavelength = (l0+lf)/2
        bandpass = tophat(wave,l0,lf,1)

    if band=='JHgapKPIC':
        l0,lf= 1450-25,1450+25
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
        l0,lf= 1170,1490  # J plus jh gap
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
        # J and H bands
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
    Compute the total image FWHM on the tracking camera by combining the
    diffraction-limited spot (broadened by high-order wavefront error via
    the Strehl ratio), the tip/tilt residual, and off-axis aberrations from
    the tracking camera optics, all added in quadrature.

    inputs:
    -------
    wfe - float
        high-order (non tip/tilt) residual wavefront error [nm]

    tt_resid - float
        residual tip/tilt error [mas]

    wavelength - float
        observing wavelength [nm]

    diam - float
        telescope diameter [m] (converted to nm internally to match
        wavelength in the diffraction-limit formula)

    platescale - float
        plate scale of the image [arcsec/pixel]

    field_r - float
        field radius position of the tracking star on the guide camera,
        [arcsec] (default 0)

    camera - str
        tracking camera name passed to get_tracking_optics_aberrations /
        get_tracking_cam, e.g. 'h2rg' or 'cred2' (default 'h2rg')

    getall - bool
        if True, also return the intermediate strehl/FWHM component terms
        (default False)

    aberrations_file - str or None
        path to the tracking optics aberrations file passed to
        get_tracking_optics_aberrations; if the file cannot be found, the
        off-axis contribution falls back to 0.5 pixels with a warning

    to do:
    check how RMS relates to FWHM

    returns:
    --------
    fwhm - float
        total image FWHM [pixels] combining diffraction/high-order WFE,
        tip/tilt, and off-axis aberration terms in quadrature

    if getall is True, also returns (in this order, with some values
    duplicated): strehl (Strehl ratio, unitless), diffraction_spot_pix
    (diffraction-limited spot size [pixels]), fwhm_ho (FWHM from
    diffraction + high-order WFE [pixels]), fwhm_tt (FWHM from tip/tilt
    residual [pixels]), fwhm_offaxis (FWHM from off-axis camera aberrations
    [pixels]), followed by a repeat of strehl, diffraction_spot_pix,
    fwhm_ho, fwhm_tt, fwhm_offaxis
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
    Not fully implemented / currently broken helper intended to compute the
    Johnson U/B/V/R/I/J/H/K magnitudes (and eventually Sloan u') of the
    loaded stellar model for a given scaling factor.

    Takes no parameters, but its body references a module-level `so`
    (storage object) and `so.stel.factor_0` that are never defined or
    passed in, so as written this function will raise a NameError if
    called. It builds up `newmags`/`all_bands` lists via
    load_inputs.get_band_mag but does not return them, and the Sloan u'
    call at the end is unused (its result is discarded). This function
    appears to be a work-in-progress / unused stub rather than a working
    utility.

    returns:
    --------
    None (nothing is returned; the computed magnitudes are only kept in
    local lists that go out of scope)
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


def get_order_value2(so,v,snr,height=0.055,distance=2e4,prominence=0.01):
    """
    Identify spectral order centers from peaks in the instrument base
    throughput, estimate each order's free spectral range from a fixed
    grating equation, and return the peak and mean SNR within (a padded
    window around) each order.

    inputs:
    -------
    so - object
        storage object; uses so.inst.base_throughput (throughput array used
        to find order peaks) and so.stel.v (wavelength array [nm]
        corresponding to so.inst.base_throughput)

    v - array
        wavelength array [nm] corresponding to snr

    snr - array
        SNR spectrum evaluated at v, unitless

    height - float
        minimum peak height passed to scipy.signal.find_peaks when locating
        order centers in so.inst.base_throughput, unitless (default 0.055)

    distance - float
        minimum separation in samples between order peaks, passed to
        scipy.signal.find_peaks (default 2e4)

    prominence - float
        minimum peak prominence passed to scipy.signal.find_peaks, unitless
        (default 0.01)

    returns:
    --------
    order_cen_lam - array
        center wavelength [nm] of each identified order

    snr_peaks - array
        maximum SNR within +/-1.3*(FSR/2) of each order center, unitless

    snr_means - array
        mean SNR within +/-1.3*(FSR/2) of each order center, unitless
    """
    order_peaks      = signal.find_peaks(so.inst.base_throughput,height=height,distance=distance,prominence=prominence)
    order_cen_lam    = so.stel.v[order_peaks[0]]
    blaze_angle      = 76
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

def get_order_value(x,y,order_filename):
    """
    Given the order centers and free spectral ranges tabulated in
    order_filename, return the peak and mean of y (e.g. SNR) within (a
    padded window around) each order.

    inputs:
    -------
    x - array
        wavelength array [nm] corresponding to y

    y - array
        quantity to summarize per order (e.g. SNR), evaluated at x

    order_filename - str
        path to order bounds file (wavelength [nm], order width [nm],
        comma delimited) passed to ccf_tools.get_order_bounds

    returns:
    --------
    order_cen_lam - array
        center wavelength [nm] of each order, from order_filename

    snr_peaks - array
        maximum value of y within +/-1.3*(fsr/2) of each order center

    snr_means - array
        mean value of y within +/-1.3*(fsr/2) of each order center
    """
    order_cen_lam,fsr = get_order_bounds(order_filename)

    snr_peaks = []
    snr_means = []
    for i,lam_cen in enumerate(order_cen_lam):
        #plt.plot(so.stel.v[isub_test],total_throughput[isub_test],'k--')
        sub_snr = y[np.where((x > (lam_cen - 1.3*fsr[i]/2)) & (x < (lam_cen+1.3*fsr[i]/2)))[0]] #FINISH THIS]
        snr_peaks.append(np.nanmax(sub_snr))
        snr_means.append(np.nanmean(sub_snr))

    return np.array(order_cen_lam), np.array(snr_peaks), np.array(snr_means)


def air_index_refraction(lam,p,t):
    """
    Compute the index of refraction of air at the given wavelength,
    pressure, and temperature, using the (modified) Edlen equation.

    https://iopscience.iop.org/article/10.1088/0026-1394/30/3/004/pdf
    edlen https://iopscience.iop.org/article/10.1088/0026-1394/2/2/002/pdf

    inputs:
    -------
    lam - float or array
        wavelength [nm]

    p - float
        air pressure [torr]

    t - float
        air temperature [celsius]

    returns:
    --------
    n - float or array
        index of refraction of air, unitless
    """
    sig = 10**7/lam * (1e-4) # 1e-4 cm/micron
    ns = 1 + (1/1e8) * (8342.13 + 2406030*(130 - sig**2)**(-1) + 15997*(38.9 -sig**2)**-1)
    n = 1 + (p * (ns -1)/ 720.775) * (1 + p*(0.817 - 0.0133*t)*(10**-6))/(1 + 0.0036610*t)
    return n


def load_confirmed_planets(planets_filename = './data/populations/confirmed_planets_PS_2023.01.12_16.07.07.csv'):
    """
    Load host star H magnitudes and effective temperatures from a
    confirmed planets catalog (NASA Exoplanet Archive format)

    input
    -----
    planets_filename - str
        path to confirmed planets csv file

    output
    ------
    hmags - array
        host star H magnitudes
    teffs - array
        host star effective temperatures [K]
    """
    planet_data =  pd.read_csv(planets_filename,delimiter=',',comment='#')
    hmags = planet_data['sy_hmag']
    teffs = planet_data['st_teff']
    return hmags,teffs


