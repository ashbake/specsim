import configparser
import numpy as np
from distutils.util import strtobool

all = {'storage_object','load_object'}


class storage_object():
    """
    Main storage object for organization
    """
    def __init__(self):
        # Classes
        self.run  = RUN()
        self.const= CONSTANT()
        self.var  = VARIABLE()
        self.filt = FILTER()
        self.stel = STELLAR()
        self.tel  = TELLURIC()
        self.out  = OUTPUT()
        self.hispec   = HISPEC()
        self.kpf  = KPF()
        # non class things
        self.info = "see objects.py in utils/ for info"


class RUN():
    "star info and spectrum"
    def __init__(self):
        self.plot_prefix   = None       # stellar spec file name
        self.savename      = None # wavelength like normal (should match exoplanet and be in standard wavelength)

class CONSTANT():
    "float values"
    def __init__(self):
        self.l0   = None       # starting wavelength
        self.l1   = None       # ending wavelength
        self.res_hispec = 100000 # resolution

class VARIABLE():
    "float values, variables"
    def __init__(self):
        self.mag      = None       # v band magntidue
        self.teff     = None       # K, 100K steps
        self.exp_time = None       # seconds

class FILTER():
    "float values"
    def __init__(self):
        self.x    = None
        self.y    = None
        self.zp   = None # will be loaded http://astroweb.case.edu/ssm/ASTR620/mags.html
        self.filter_file=None
        self.zp_file = './data/filters/zeropoints.txt'
        self.zp_unit = 'Jy'
        self.band='J'

class STELLAR():
    "star info and spectrum"
    def __init__(self):
        # User optional define:
        self.phoenix_file   = None       # stellar spec file name, **make this take temp value in future
        # Filled in by code:
        self.vraw = None # wavelength like normal (should match exoplanet and be in standard wavelength)
        self.sraw = None #  spectrum
        self.vsini = 0 # km/s
        
class TELLURIC():
    "telluric transmission file, static"
    def __init__(self):
        # User optional define:
        self.telluric_file   = None       # spec file name
        # Filled in by code:
        self.v = None # wavelength 
        self.s = None #  spectrum


class OUTPUT():
    "output file info and storage"
    def __init__(self):
        # User optional defined
        self.savename = None # output name
        # Filled in by code
        self.spectrum = None # ca H&k spectrum

class HISPEC():	
    "HK data"
    def __init__(self):
        self.transmission_file=None


class KPF():
    "KPF data"
    def __init__(self):
        """
        KPF data
        """
        self.transmission_file=None
        self.res = None
        # defined here only
        self.order_wavelengths = np.array([448.10241617, 451.39728688, 454.74097048, 458.13455982,
        461.57918057, 465.07599254, 468.62619096, 472.23100781,
        475.8917133 , 479.60961731, 483.38607099, 487.22246838,
        491.12024812, 495.08089528, 499.10594321, 503.19697554,
        507.35562823, 511.58359179, 515.88261357, 520.25450013,
        524.70111979, 529.2244053 , 533.82635666, 538.50904399,
        543.27461076, 548.12527692, 553.06334248, 558.09119105,
        563.21129372, 568.4262131 , 573.73860762, 579.15123599,
        584.66696205, 590.28875976, 596.0197186 , 601.86304917,
        607.82208926, 613.90031015, 620.10132339, 626.42888791,
        632.88691768, 639.47948974, 646.21085279, 653.08543633,
        660.10786038, 667.28294582, 674.61572544, 682.11145573,
        689.77562939, 697.61398881, 705.63254041, 713.83756995,
        722.235659  , 730.83370256, 739.6389279 , 748.65891482,
        757.90161747, 767.37538769, 777.08900019, 787.05167968,
        797.27313007, 807.76356599, 818.53374687, 829.59501372,
        840.95932898, 852.63931966, 864.64832416])
        self.order_fsrs = np.array([])
        

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
    config = LoadConfig(configfile)
    so = storage_object()

    for key in config:
        s1,s2=key.split('.')
        try:
            setattr(getattr(so,s1),s2,float(config[key]))
        except ValueError:
            setattr(getattr(so,s1),s2,config[key])


    return so




