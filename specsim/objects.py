import configparser
import numpy as np
import os

from distutils.util import strtobool

all = {'storage_object','load_object'}


class storage_object():
    """
    Top-level container passed through the whole pipeline. Holds one
    sub-object per physical component (so.run, so.filt, so.stel, so.tel,
    so.inst, so.ao, so.obs, so.track), each of which starts with default
    values and is progressively filled in by load_object/fill_data.
    """
    def __init__(self):
        """
        Instantiate and attach all sub-storage objects (run, filt, stel,
        tel, inst, ao, obs, track) as attributes
        """
        # Classes
        self.run  = RUN()
        self.filt = FILTER() 
        self.stel = STELLAR()
        self.tel  = TELLURIC()
        self.inst = INSTRUMENT()
        self.ao   = AO()
        self.obs  = OBSERVATION() 
        self.track= TRACK()       
        # non class things
        self.info = "see objects.py in utils/ for info"


class RUN():
    "output/run settings: output path, plot tag, and base data folder"
    def __init__(self):
        """
        Set default run/output settings (outpath, tag, data_folder)
        """
        self.outpath      = './'   # output path
        self.tag          = 'test' # tag for plot saving
        self.data_folder  = './'   # base folder that all *_file/*_folder/*_path attrs (outside so.run) are joined onto, unless already absolute


class AO():
    "AO system parameters: mode, tip/tilt and high order WFE, coupling inputs"
    def __init__(self):
        """
        Set default AO system parameters (mode, static/dynamic tip-tilt,
        low order wfe, defocus, etc.) and placeholders for values filled
        in later by the code
        """
        # user defined
        self.mode        = 'auto'    # AO mode corresponding to ao wfe load fxn
        self.tt_static   = 2         # mas, static tip tilt error
        self.tt_dynamic_file  = None      # file with dynamic tip tilt error structured with seeing, ZA, AO mode
        self.ho_wfe_file      = None      # file with high order wfe error data structured with seeing, ZA, AO mode
        self.lo_wfe      = 50        # nm, low order 
        self.defocus     = 25        # nm, defocus error
        self.mag         = 'default' # magnitude of ao star, if 'default' uses mag of on axis star
        self.teff        = 'default' # teff of ao star, if 'default' uses teff of on axis star
        self.ho_wfe_set      = None      # high order wfe, to use instead of loading file
        self.tt_dynamic_set  = None      # dynamic tip tilt error, to use instead of loading file
        # filled in by code
        self.ho_wfe      = None      # high order wfe, will be filled in by code either from file or based on _set value
        self.tt_dynamic  = None      # dynamic tip tilt error, will be filled in by code either from file or based on _set value
        self.band        = None      # band of ao star
        self.dichroic    = None      # AO dichroic transmission, for HISPEC in case pyramid is used
        self.ho_strehl   = None      # high order strehl
        self.strehl_array= None      # strehl array as function of wavelength
        self.ao_mag      = None      # magnitude of ao star in band selected
        self.ao_modes    = None      # list of ao modes loaded from file
        self.mode_chosen = None      # mode chosen from ao_modes as best ao mode



class INSTRUMENT():
    "spectrograph parameters: wavelength range, resolution, detector, throughput"
    def __init__(self):
        """
        Set default instrument parameters (wavelength range, resolving
        power, detector properties, etc.) and placeholders for throughput
        and order information filled in later by the code
        """
        # user defined
        self.transmission_path = None # path to transmission files
        self.order_bounds_file = None # file with order bound information
        self.order_bounds      = None # order bounds of spectrograph
        self.atm = 1        # keyword for transmission file, HISPEC=1, MODHIS=0 for now
        self.adc = 1        # keyword for transmission file, HISPEC=1, MODHIS=0 for now
        self.l0   = 900     # nm, start of wavelengths to consider
        self.l1   = 2500    # nm, ending wavelength
        self.res  = 100000  # resolving power
        self.pix_vert = 4   # pixels, vertical extent of spectrum in cross dispersion
        self.extraction_frac = 0.925 # fraction of flux extracted for 4 vertical pixels, TODO should have code calculate it
        self.tel_area = 76 # m2, telescope area, keck is default
        self.tel_diam = 10 # m ,telescope diameter,  keck is default
        self.res_samp = 3  #pixels, sampling of resolution element
        self.saturation = 100000 # electrons, saturation limit of detector
        self.readnoise  = 12   # e-, default is CDS read noise of detector
        self.darknoise  = 0.01 # e-/pix/s, dark current to assume
        self.pl_on      = 1    # 0 or 1, if 1 it will assume photonic lantern in use for the blue channel
        self.rv_floor   = 0.5  # m/s, systematic noise floor of RV measurement for instrument and telluric systematics, 0.5m/s for hispec and modhis
        # code filled in values
        self.base_throughput = None # base throughput of instrument (no coupling)
        self.coupling        = None # coupling of fiber 
        self.order_cens      = None # order centers
        self.order_widths    = None # order widths
        self.sig             = None # resolution element in nm   
        self.transmission_file= None # transmission file name
        self.xtransmit      = None # x array of throughput [nm]
        self.ytransmit      = None # throughput of instrument [0,1]
        self.y              = None # y filter band
        self.J              = None # J filter band
        self.H              = None # H filter band
        self.K              = None # K filter band


class OBSERVATION():
    "observation setup (exposure time, zenith angle) and computed spectra/SNR/noise arrays"
    def __init__(self):
        """
        Set default observation parameters (exposure time, zenith angle,
        target SNR, etc.) and placeholders for spectra and noise arrays
        filled in later by the code
        """
        self.texp             = 900  # seconds, total integrated exposure time
        self.texp_frame_set   = 900  # seconds, maximum for a single exposure. default lets code choose it with max of 900
        self.nsamp            = 1    # number of up the ramp samples per frame exposure
        self.zenith_angle     = 45   # degrees, zenith angle of observation. Used to define airmass
        self.target_snr       = 100  # target snr for ETC calculation
        self.target_ccf_snr   = 5    # target ccf snr for ETC calculation
        # code filled in variables
        self.frame_phot_per_nm = None # photons per nm in a single frame of texp_frame seconds long
        self.inst_bg_ph    = None # background photons per nm in a single frame of texp_frame seconds long
        self.nframes       = None # number of frames to reach texp
        self.noise_frame   = None # noise per frame
        self.noise         = None # noise spectrum, all frames combined
        self.order_inds    = None # indices of each order of the spectrograph echelle
        self.v             = None # wavelength array
        self.s             = None # spectrum array
        self.snr           = None # snr array
        self.s_frame       = None # spectrum array per frame
        self.speckle_frame    = None # speckle noise per frame
        self.snr_max_orders   = None # max snr per order
        self.snr_mean_orders  = None # mean snr per order
        self.snr_res_element  = None # snr per resolution element
        self.v_res_element    = None # wavelength per resolution element
        self.texp_frame       = None # exposure time per frame



class FILTER():
    "photometric filter band/family selection, curve, and zeropoint"
    def __init__(self):
        """
        Set default photometric filter selection (band, family) and
        placeholders for the loaded filter curve and zeropoint
        """
        self.x    = None # wavelength array
        self.y    = None # filter transmission (fraction)
        self.zp   = None # zeropoints storage object - will be loaded
        self.filter_file=None
        self.zp_file = './data/filters/zeropoints.txt' #band zeropoints from: http://astroweb.case.edu/ssm/ASTR620/mags.html
        self.zp_unit = 'Jy' # jansky - units of file
        self.band    = 'J' # band to pick, yJHK
        self.family  = '2mass' # family of filter band, see zeropoints file 'cfht', '2mass' for JHK
        #zps    = np.loadtxt(self.zp_file,dtype=str).T
        #self.options =[zps[0],zps[1]] # returns options for bands to pick

class STELLAR():
    "star info and spectrum"
    def __init__(self):
        """
        Set default stellar/companion parameters (Teff, magnitude, vsini,
        rv, separation, etc.) and placeholders for the loaded spectrum
        filled in later by the code
        """
        # User optional define:
        self.phoenix_folder   = None  # Path to where Phoenix files live, T>=2300K objects
        self.sonora_folder    = None  # path to Sonora files, used for T<2300K objects
        self.vsini    = 0     # km/s, vsini of star
        self.mag      = 10    # mag, star magnitude defined in so.filt bandpass
        self.teff     = 3600  # K, star temperature
        self.rv       = 0     # absolute rv of system [km/s]
        self.pl_sep   = 0     # mas, if 0 it will assume on axis, if non zero it will assume off axis
        self.pl_teff  = 800   # K, planet temperature, used if pl_sep>0
        self.pl_mag   = 19    # mag, planet magnitude defined in same bandpass as star, used if pl_sep>0
        self.pl_vsini = 0     # km/s, planet vsini, used if pl_sep>0
        self.logg     = 4.5   # logg of star, default to 4.5
        # Filled in by code:
        self.vraw = None   # wavelength like normal (should match exoplanet and be in standard wavelength)
        self.sraw = None   # spectrum
        self.units = None  # units of sraw
        self.v = None      # wavelength
        self.s = None      # spectrum in photons
        self.model = None  # model chosen, 'phoenix' or 'sonora'
        self.factor_0 = None # factor to scale spectrum by to match magnitude
        self.star = None   # Star instance for the on-axis star, set by fill_data.stellar()
        self.pl_star = None # Star instance for the companion, set by fill_data.stellar() if pl_sep>0

class TELLURIC():
    "telluric transmission file, static"
    def __init__(self):
        """
        Set default telluric parameters (pwv, seeing) and placeholders
        for the loaded telluric transmission spectrum and its components
        """
        # User optional define:
        self.telluric_file   = None       # spec file name
        self.sky_path        = None       # path to sky emission files
        self.pwv             = 1.3        # mm
        self.seeing_set      = 'average'  # seeing to set: options of good (0.6), average (0.8), and bad (1.1) 
        # Filled in by code:
        self.airmass         = None      # gets converted from ZA
        self.v               = None      # wavelength 
        self.s               = None      # spectrum
        self.rayleigh        = None      # rayleigh scattering
        self.seeing          = None      # seeing corresponding to the set value
        self.h2o             = None      # water only transmission spectrum
        self.o3              = None      # ozone only transmission spectrum


class TRACK():
    "tracking camera storage"
    def __init__(self):
        """
        Set default tracking camera parameters (exposure time, f ratio,
        band, field radius) and placeholders for the loaded throughput
        """
        # User optional defined
        self.transmission_file = None # path to ATC transmission file
        self.texp      = 1    # exposure time of tracking camera [s]
        self.frat      = 35   # f ratio of tracking camera arm - 35 for HISPEC
        self.band      = 'JHgap' # band being used, [JHgap,z,y,J,H,K] see fxn in obs_tools.py for more options
        self.field_r   = 0    # radius across field for calculating aberrations
        # Filled in by code
        self.xtransmit = None # x array of throughput [nm]
        self.ytransmit = None # throughput of tracking camera [0,1]


def LoadConfig(configfile, config={}):
    """
    Read a configuration file 'XXX.cfg' into a flat dictionary

    inputs
    ------
    configfile : str
        path to a .cfg file with [section] headers and option=value lines
    config : dict, optional
        existing dict to extend (not mutated in place; a copy is returned)

    output
    ------
    config : dict
        keys of the form '<section>.<option>', values as stripped strings
    """
    config = config.copy(  )
    cp = configparser.ConfigParser(  )
    cp.read(configfile)
    for sec in cp.sections(  ):
        name = str(sec)
        for opt in cp.options(sec):
            config[name + "." + str(opt)] = str(
                cp.get(sec, opt)).strip()
    return config


PATH_ATTR_SUFFIXES = ('_file', '_folder', '_path')


def load_object(configfile):
    """
    Build a storage_object and populate it with user-defined values from
    a config file

    inputs
    ------
    configfile : str
        path to a .cfg file, parsed by LoadConfig into '<section>.<option>'
        keys matching the storage_object's sub-object attribute names
        (e.g. 'inst.res' sets so.inst.res)

    output
    ------
    so : storage_object
        with each config value set (as float where possible, else string),
        and any *_file/*_folder/*_path attribute (outside so.run) resolved
        relative to so.run.data_folder if not already absolute
    """
    if not os.path.isfile(configfile): raise Exception("Config File is Not Found!")
    config = LoadConfig(configfile)
    so     = storage_object()

    for key in config:
        s1,s2=key.split('.')
        try:
            setattr(getattr(so,s1),s2,float(config[key]))
        except ValueError:
            setattr(getattr(so,s1),s2,config[key])

    # Any attribute named *_file/*_folder/*_path (outside so.run) is treated
    # as living under so.run.data_folder unless it's already an absolute path.
    for section in (so.filt, so.stel, so.tel, so.inst, so.ao, so.obs, so.track):
        for attr, value in vars(section).items():
            if attr.endswith(PATH_ATTR_SUFFIXES) and isinstance(value, str) and not os.path.isabs(value):
                setattr(section, attr, os.path.join(so.run.data_folder, value))

    return so




