import configparser
import numpy as np
import os

from distutils.util import strtobool

all = {'storage_object','load_object'}


class storage_object():
    """
    Main storage object for organization
    """
    def __init__(self):
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
    "star info and spectrum"
    def __init__(self):
        self.outpath      = './'   # output path
        self.tag          = 'test' # tag for plot saving
        
class AO():
    "float values"
    def __init__(self):
        # user defined
        self.mode        = 'auto'    # AO mode corresponding to ao wfe load fxn
        self.tt_static   = 2         # mas, static tip tilt error
        self.tt_dynamic_set  = None  # file with dynamic tip tilt error structured with seeing, ZA, AO mode
        self.ho_wfe_set  = None      # file with high order wfe error data structured with seeing, ZA, AO mode
        self.lo_wfe      = 50        # nm, low order 
        self.defocus     = 25        # nm, defocus error
        self.mag         = 'default' # magnitude of ao star, if 'default' uses mag of on axis star
        self.teff        = 'default' # teff of ao star, if 'default' uses teff of on axis star
        # filled in by code
        self.band        = None      # band of ao star
        self.dichroic    = None      # AO dichroic transmission, for HISPEC in case pyramid is used
        self.ho_strehl   = None      # high order strehl
        self.ho_wfe      = None      # high order wfe
        self.tt_dynamic  = None      # dynamic tip tilt error
        self.strehl_array= None      # strehl array as function of wavelength
        self.ao_mag      = None      # magnitude of ao star in band selected
        self.ao_modes    = None      # list of ao modes loaded from file
        self.mode_chosen = None      # mode chosen from ao_modes as best ao mode



class INSTRUMENT():
    "float values"
    def __init__(self):
        # user defined
        self.transmission_path = None # path to transmission files
        self.order_bounds_file = None # file with order bound information
        self.order_bounds      = None # order bounds of spectrograph
        self.atm = 0        # keyword for transmission file, HISPEC=1, MODHIS=0 for now
        self.adc = 0        # keyword for transmission file, HISPEC=1, MODHIS=0 for now
        self.l0   = 900     # nm, start of wavelengths to consider
        self.l1   = 2500    # nm, ending wavelength
        self.res  = 100000  # resolving power
        self.pix_vert = 4   # pixels, vertical extent of spectrum in cross dispersion
        self.extraction_frac = 0.925 # fraction of flux extracted for 4 vertical pixels, should have code calculate it
        self.tel_area = 76 # m2, telescope area, keck is default
        self.tel_diam = 10 # m ,telescope diameter,  keck is default
        self.res_samp = 3  #pixels, sampling of resolution element
        self.saturation = 100000 # electrons, saturation limit of detector
        self.readnoise  = 12   # e-, CDS read noise of detector
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
    "float values, variables"
    def __init__(self):
        self.texp             = 900  # seconds, total integrated exposure time 
        self.texp_frame_set   = 900  # seconds, maximum for a single exposure. default lets code choose it with max of 900
        self.nsamp            = 1    # number of up the ramp samples per frame exposure
        self.zenith_angle     = 45   # degrees, zenith angle of observation. Used to define airmass
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
    "float values"
    def __init__(self):
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

class TELLURIC():
    "telluric transmission file, static"
    def __init__(self):
        # User optional define:
        self.telluric_file   = None       # spec file name
        self.skypath         = None       # path to sky emission files
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
        # User optional defined
        self.transmission_file = None # output name
        self.texp      = 1    # exposure time of tracking camera [s]
        self.frat      = 35   # f ratio of tracking camera arm - 35 for HISPEC
        self.band      = 'JHgap' # band being used, [JHgap,z,y,J,H,K] see fxn in obs_tools.py for more options
        self.field_r   = 0    # radius across field for calculating aberrations
        # Filled in by code
        self.xtransmit = None # x array of throughput [nm]
        self.ytransmit = None # throughput of tracking camera [0,1]


def LoadConfig(configfile, config={}):
    """
    Reads configuration file 'XXX.cfg'
    returns a dictionary with keys of the form
    <section>.<option> and the corresponding values
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


def load_object(configfile):
    """
    Loads config file as dictionary using LoadConfig function
    Then loads stoar_object and fills in user-defined
    quantities
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


    return so




