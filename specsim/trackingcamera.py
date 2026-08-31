##############################################################
# TrackingCamera: the guide detector -- its optics, backgrounds and exposure
###############################################################
#
# Split out of instrument.py alongside Spectrograph, which it mirrors: both
# take an already-.select()-ed AOSystem and .observe() a star through it.
# TrackingCamera also needs a Spectrograph, for the telescope diameter/area
# and the instrument throughput path.
#
# The camera-specific helpers below (detector properties, tracking bandpass,
# sky and thermal backgrounds) sit next to their only caller.

from typing import Optional

import numpy as np
from scipy import interpolate
from astropy import units as u
from astropy.modeling.models import BlackBody

from specsim.aosystem import AOSystem
from specsim.atmosphere import Atmosphere
from specsim.functions import calc_plate_scale, calc_strehl, sum_total_noise, tophat
from specsim.star import Star


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
        l0,lf= 1335,1453
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
        l0,lf= 1350,1450 # less bc pyramid takes some- double check how much 
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

def get_sky_bg_tracking(x,fwhm,sky_bg_v,sky_bg,area=76):
    """
    Generate sky background per pixel for the tracking/acquisition camera,
    default to HISPEC. Takes an already-loaded Mauna Kea sky emission
    model (OH lines + thermal continuum, in ph/s/arcsec^2/nm/m^2 -- see
    atmosphere.load_sky_background/Atmosphere.sky_bg), interpolates
    it onto the input wavelength grid, and converts it to a photon count
    rate per nm by multiplying by the telescope collecting area and the
    PSF solid angle (from the supplied FWHM, corrected for a Gaussian
    beam). Unlike get_sky_bg, this does not divide by resolving
    power/npix, so the result is per nm rather than per reduced pixel.
    Source: DMawet jup. notebook

    inputs:
    -------
    x : array [nm]
        wavelength in nanometers to evaluate/interpolate onto
    fwhm: float [arcsec]
        full width at half maximum of the PSF on the tracking camera,
        used to set the solid angle subtended by one resolution element
    sky_bg_v : array [nm]
        wavelength grid sky_bg is sampled on (e.g. Atmosphere.v)
    sky_bg : array [ph/s/arcsec^2/nm/m^2]
        sky background surface brightness, sampled on sky_bg_v (e.g.
        Atmosphere.sky_bg, from atmosphere.load_sky_background)
    area: float [m^2]
        area of telescope in meters squared

    outputs:
    --------
    array [ph/s/nm]
        sky background photon rate per nm, sampled on the input
        wavelength grid x
    """
    area = area * u.m * u.m
    wave = x*u.nm

    fwhm *= u.arcsec
    solidangle = fwhm**2 * 1.13 #corrected for Gaussian beam (factor 1.13)

    sky_background_interp = np.interp(wave.value, sky_bg_v, sky_bg) * u.photon/(u.s*u.arcsec**2*u.nm*u.m**2)
    sky_background_interp*= area * solidangle

    return sky_background_interp.value # ph/s/nm


def get_inst_bg_tracking(x,pixel_size,npix,blocking_filter_file):
    """
    Generate the instrument thermal background seen by the tracking camera,
    per pixel, default to HISPEC. Source: DMawet jup. notebook.
    Models the thermal emission of the cryostat window as a blackbody at a
    fixed temperature (277 K), attenuated by the blocking filter
    transmission, the (fixed, approximate) window emissivity, and the
    H2RG quantum efficiency (modeled as a tophat between 600-2600 nm).
    The blackbody is scaled by the effective area x solid angle set by
    the pixel size and the optical f-number, converted to a photon rate
    per nm, and multiplied by npix to get the total thermal spectrum for
    npix pixels; that spectrum is also integrated over wavelength to give
    a single total photon rate.
    change this to take emissivities and temps as inputs so dont
    have to rely on get_emissivities

    inputs:
    -------
    x : array [nm]
        wavelength in nanometers
    pixel_size: float [micron]
        physical size of one detector pixel
    npix: integer
        number of pixels over which the thermal background is summed
    blocking_filter_file: string
        path to the cold-snout blocking filter transmission curve
        (wavelength [nm], transmission [percent], 20 header rows)

    outputs:
    --------
    thermal_spectrum: array [ph/s/nm]
        instrument thermal background spectral photon rate for npix
        pixels, sampled on the input wavelength grid x
    thermal: float [ph/s]
        thermal_spectrum integrated over wavelength, i.e. the total
        instrument thermal background photon rate for npix pixels
    """
    wave = x * u.nm
    window_temp = 277 * u.K # temperature of cryostat window will be close to AO room temperature
    pixel_size *= u.micron
    f_num = 6 # fnumber of cold snout
    fudge_factor = 5 # ATC will view more than just the warm window, so include multiplicative factor to be conservative. 
    # also we measure the thermal background to be 500ish e-/s as of 8/11/26 - TBU once done with ATC testing with painted cold snout

    # Load blocking filter profile
    fx,fy = np.loadtxt(blocking_filter_file,skiprows=20).T
    f = interpolate.interp1d(fx[::-1]*u.nm,fy[::-1],bounds_error=False,fill_value=0)
    blocking_filter   = f(wave)/100 # convert to fraction from percent
    
    # load window emissivity
    #fx,fy = np.loadtxt(datapath + 'feicam/Infrasil_Window.txt').T
    #f = interpolate.interp1d(fx[::-1]*u.nm,fy,bounds_error=False,fill_value=0)
    window_emissivity = 0.05 #1 - f(wave)/100, this is conservative. emissivity should be max 0.01 but there will be other factors
    
    # Create QE profile for H2RG matching cutoff
    QE = tophat(wave.value,600,2550,0.8) # sensitivity of h2rg, cuts off at 2.5um

    area_times_omega = u.radian**2 * 1.13**2 * np.pi**2 * pixel_size**2 / 4 /f_num**2
    bbtemp_fxn  = BlackBody(window_temp, scale=1.0 * u.erg / (u.micron * u.s * u.cm**2 * u.arcsec**2)) 
    bb   = area_times_omega.to(u.cm**2 * u.arcsec**2) * bbtemp_fxn(wave)

    bb_spec_dens = bb.to(u.photon/u.s/u.nm, equivalencies=u.spectral_density(wave))
    
    # thermal spectrum over npix, then integrate
    thermal_spectrum = fudge_factor * npix * QE * window_emissivity * blocking_filter * bb_spec_dens # units of ph/nm/s/pix
    thermal = np.trapz(thermal_spectrum,wave)

    return thermal_spectrum, thermal.value # spectrum in units of ph/nm/s


class TrackingCamera:
    """
    Acquisition/tracking camera: detector properties, throughput, PSF FWHM,
    background, and the resulting SNR/centroid error of the tracking
    measurement.
    """

    def __init__(self, camera: str = 'h2rg', band: str = 'JHgap', fratio: float = 35,
                 texp: float = 1, field_r: float = 0,
                 transmission_file=None, aberrations_file: Optional[str] = None,
                 blocking_filter_file: Optional[str] = None,
                 area_m2: float = 76, diameter_m: float = 10):
        self.camera = camera
        self.band = band
        self.fratio = fratio
        self.texp = texp
        self.field_r = field_r
        self.transmission_file = transmission_file
        self.aberrations_file = aberrations_file
        self.blocking_filter_file = blocking_filter_file
        self.area_m2 = area_m2        # telescope collecting area [m^2] -- fed in directly rather than reached for on a Spectrograph
        self.diameter_m = diameter_m  # telescope diameter [m]

        # derived state, set by load()
        self.pixel_pitch: Optional[float] = None
        self.dark: Optional[float] = None
        self.rn: Optional[float] = None
        self.qe_mod: Optional[float] = None
        self.saturation: Optional[float] = None
        self.xtransmit: Optional[np.ndarray] = None
        self.ytransmit: Optional[np.ndarray] = None
        self.platescale: Optional[float] = None
        self.platescale_units = 'arcsec/pixel'
        self.center_wavelength: Optional[float] = None
        self.bandpass: Optional[np.ndarray] = None
        self.fwhm: Optional[float] = None
        self.fwhm_units = 'pixel'
        self.npix: Optional[float] = None
        self.strehl: Optional[float] = None
        self.sky_bg_spec: Optional[np.ndarray] = None
        self.sky_bg_ph: Optional[float] = None
        self.inst_bg_spec: Optional[np.ndarray] = None
        self.inst_bg_ph: Optional[float] = None
        self.signal_spec: Optional[np.ndarray] = None
        self.nphot_nocap: Optional[float] = None
        self.od: Optional[float] = None
        self.signal: Optional[float] = None
        self.saturation_flag: Optional[bool] = None
        self.noise: Optional[float] = None
        self.snr: Optional[float] = None
        self.centroid_err: Optional[float] = None

    def load(self, x: np.ndarray, ao_system: AOSystem) -> "TrackingCamera":
        """
        Set up the camera: detector properties for the selected camera, the
        throughput curve, the plate scale, the tracking bandpass, the PSF
        FWHM and Strehl implied by the AO correction, and the instrument
        thermal background those imply. Everything here depends only on the
        hardware and the AO system -- not on the star or the sky -- so it
        holds across exposures, mirroring Spectrograph.load().

        inputs
        ------
        x - array, shared wavelength grid [nm]
        ao_system - AOSystem, already .select()-ed (needs ho_wfe/tt_dynamic/
            pywfs_dichroic)

        Returns self, so calls can be chained:
        TrackingCamera(...).load(x, ao).observe(x, star, atm).
        """
        # pick guide camera - eventually settle on one and put params in config file!
        rn, pixel_pitch, qe_mod, dark, saturation = get_tracking_cam(camera=self.camera, x=x)
        self.pixel_pitch = pixel_pitch
        self.dark = dark
        self.rn = rn
        self.qe_mod = qe_mod  # to switch cameras, wont need this later bc qe will match throughput model
        self.saturation = saturation  # to switch cameras, wont need this later bc qe will match throughput model

        # load and store tracking camera throughput - file structure hard coded
        if type(self.transmission_file) == float:
            self.xtransmit, self.ytransmit = x, np.ones_like(x) * self.transmission_file * self.qe_mod
        else:
            xtemp, ytemp = np.loadtxt(self.transmission_file, delimiter=',').T  # microns!
            f = interpolate.interp1d(xtemp * 1000, ytemp, kind='linear', bounds_error=False, fill_value=0)
            self.xtransmit, self.ytransmit = x, f(x) * self.qe_mod

        # get plate scale
        self.platescale = calc_plate_scale(self.pixel_pitch, D=self.diameter_m, fratio=self.fratio)

        # load tracking band
        bandpass, self.center_wavelength = get_tracking_band(x, self.band)
        self.bandpass = bandpass * ao_system.pywfs_dichroic

        # get fwhm (in pixels)
        self.fwhm = float(self.get_fwhm(ao_system.ho_wfe, ao_system.tt_dynamic,
                                         self.center_wavelength, self.diameter_m))
        self.npix = np.pi * (self.fwhm / 2) ** 2  # only take noise in circle of diameter FWHM
        print('Tracking FWHM=%spix' % self.fwhm)

        self.strehl = calc_strehl(ao_system.ho_wfe, self.center_wavelength)

        # get background spec (takes thermal emission from warm cryostat window)
        # units of ph/nm/s for spectrum and ph/s for inst_bg_ph
        self.inst_bg_spec, self.inst_bg_ph = get_inst_bg_tracking(x, self.pixel_pitch, self.npix, blocking_filter_file=self.blocking_filter_file)

        return self

    def observe(self, x: np.ndarray, star: Star, atmosphere: Atmosphere) -> "TrackingCamera":
        """
        Expose on a star: the sky background through the tracking band, the
        stellar photon signal integrated over the band and PSF core, the
        total noise, and from those the SNR and centroid error that
        characterise tracking/guiding precision. If the peak flux would
        saturate the detector, an equivalent neutral-density filter (od) is
        applied to cap the signal.

        Requires .load() to have been called first (needs fwhm/npix/
        platescale/bandpass/ytransmit and the detector properties).

        inputs
        ------
        x - array, shared wavelength grid [nm]
        star - Star, already .load()-ed
        atmosphere - Atmosphere, already .load()-ed

        Returns self, so calls can be chained.
        """
        # get sky background, spec is ph/nm/s. fwhm must be in arcsec
        self.sky_bg_spec = get_sky_bg_tracking(x, self.fwhm * self.platescale, atmosphere.v,
                                                            atmosphere.sky_bg, area=self.area_m2)
        self.sky_bg_ph = np.trapz(self.sky_bg_spec * self.bandpass * self.ytransmit, x)  # sky bkg needs mult by throughput and bandpass profile

        # get photons in band
        self.signal_spec = star.s * self.texp * self.area_m2 * self.ytransmit * np.abs(atmosphere.s)

        # fac is empirically the fraction of light approx under 2D gaussian of FWHM~4pix,
        # which scales npix. This was tuned based on a actual toy centroiding model fit and gets results to match
        fac = 0.5
        nphot = fac * self.strehl * np.trapz(self.signal_spec * self.bandpass, star.v)

        # get noise
        self.noise = sum_total_noise(nphot, self.texp, 1, self.inst_bg_ph, self.sky_bg_ph, self.dark, self.rn, self.npix, 0)
        print(f'Tracking noise: {self.noise} e-')

        # get centroid error, cap if saturated
        # peak of 2D Gaussian 4pix wide will be 1/10th of the flux in a 4pix diameter aperture (empirically derived)
        flux_in_peak = nphot / 10  # nphot is already times 0.5 to give flux in a 4 pix diameter aperture
        if flux_in_peak > self.saturation:
            # pick an ND filter
            # compute OD of filter needed (neg neg computes ceiling)
            self.od = np.max((-1 * round(-1 * np.log10(flux_in_peak / self.saturation), 0), 0))
            # Apply chosen nd filter
            self.signal = 10 ** (-1 * self.od) * nphot  # cap nphot
            self.noise = sum_total_noise(self.signal, self.texp, 1, self.inst_bg_ph, self.sky_bg_ph, self.dark, self.rn, self.npix, 0)
            # save things related to saturation
            self.saturation_flag = True
            self.nphot_nocap = nphot
            print(f'Tracking OD needed {self.od}, nphot capped to {self.signal} e-')
        else:
            self.od = 0.0
            self.nphot_nocap = nphot
            self.signal = nphot  # no blocking needed
            self.saturation_flag = False
            print(f'Tracking photons: {self.signal} e-')

        self.snr = self.signal / self.noise
        # for centroid error, care about the SNR in the peak
        self.centroid_err = (1 / np.pi) * self.fwhm / self.snr  # same fwhm but snr is reduced to not saturate like if used an ND filter
        print(f'Tracking SNR: {self.snr}, centroid error: {self.centroid_err} pix')

        return self

    def get_tracking_optics_aberrations(self, field_r=None):
        """
        Load PSF size of the tracking camera optics as a function of field
        position and interpolate to the requested field radius.

        Reads a file of field position vs RMS spot size (per wavelength) for the
        tracking optics, converts the RMS at 1400nm from um to pixels (using the
        pixel pitch of the selected camera), and linearly interpolates
        (extrapolating outside the tabulated range) to field_r. Wavelength
        dependence is ignored beyond picking the 1400nm column, since the RMS
        does not vary much with wavelength across the tabulated columns.

        inputs:
        --------
        field_r (float, 0-3) [arcsec]
            field radius position of tracking star on guide camera field.
            Defaults to this camera's configured field_r.

        The camera (for the um->pixel conversion) and the aberrations file are
        read off self. To plot spot RMS vs field, use
        plot.plot_tracking_cam_spot_rms().

        returns:
        -------
        RMS of the PSF due to optical aberrations in pixels (radius rms), float,
        interpolated (or extrapolated) to field_r
        """
        field_r = self.field_r if field_r is None else field_r

        # todo, make this a try statement, so if a file isn't available it just assumes 0 added aberrations
        try:
            f = np.loadtxt(self.aberrations_file)
        except FileNotFoundError:
            print(f"Aberrations file not found: {self.aberrations_file}")
            f = np.array([[0, 0, 0, 0, 0, 0, 0, 0]])
        field, rmstot, rms900,rms1000,rms1200,rms1400,rms1600,rms2200  = f.T #field [deg], rms [um]
        _,pixel_pitch,_,_,_ = get_tracking_cam(camera=self.camera,x=None)

        # should interpolate across wavelength but theyre not so different so just use 1400nm
        # multiply rms by 2 to get diameter (closer to FWHM)
        f = interpolate.interp1d(field * 3600, rms1400/pixel_pitch,bounds_error=False,fill_value='extrapolate')

        return f(field_r)

    def get_fwhm(self, wfe, tt_resid, wavelength, diam, getall=False):
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

        getall - bool
            if True, also return the intermediate strehl/FWHM component terms
            (default False)

        The plate scale [arcsec/pixel], field radius, camera name and
        aberrations file are read off self. If the aberrations file cannot be
        found, the off-axis contribution falls back to 0.5 pixels with a warning.

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
        platescale = self.platescale

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
            fwhm_offaxis     = radius_to_diam * self.get_tracking_optics_aberrations() # times 2 to get radius
        except:
            fwhm_offaxis = 0.5
            print('Cant find file %s' %self.aberrations_file)
            print('Warning: no tracking camera aberrations file found, assuming 0.5')
    
        fwhm = np.sqrt(fwhm_tt**2 + fwhm_ho**2 + fwhm_offaxis**2)

        if getall:
            return fwhm, strehl, diffraction_spot_pix, fwhm_ho, fwhm_tt, fwhm_offaxis, strehl, diffraction_spot_pix, fwhm_ho, fwhm_tt, fwhm_offaxis
        else:
            return fwhm
