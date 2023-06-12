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
        self.outpath      = './' # stellar spec file name

class AO():
    "float values"
    def __init__(self):
        self.mode   = 'SH'     # AO mode corresponding to ao wfe load fxn
        self.tt_static   = 0    # mas, static tip tilt error
        self.tt_dynamic  = 'default' # mas or 'default', dynamic tip tilt error, default: takes from file based on stellar magnitdue
        self.lo_wfe = 50  # nm, low order 
        self.defocus = 25 #nm, low order
        self.v_mag   = 'default' # magnitude or 'defaul't, magnitude of AO star, default: uses Vmag of target star
        self.ho_wfe  = 'default' # nm or 'default', high order wave front error, default: loads from file


class INSTRUMENT():
    "float values"
    def __init__(self):
        self.l0   = 900     # nm, start of wavelengths to consider
        self.l1   = 2500    # nm, ending wavelength
        self.res  = 100000 # resolving power
        self.pix_vert = 3 # pixels, vertical extent of spectrum in cross dispersion
        self.tel_area = 76 # m2, telescope area, mauna kea is default
        self.tel_diam = 10 #m ,telescope diameter,  mauna kea default
        self.res_samp = 3 #pixels, sampling of resolution element
        self.saturation = 100000 # electrons, saturation limit ofr detector
        self.readnoise  = 12 # e-, CDS read noise of detector
        self.darknoise  = 0.01 # e-/pix/s, dark current to assume

class OBSERVATION():
    "float values, variables"
    def __init__(self):
        self.texp      = 900  # seconds, total integrated exposure time 
        self.texp_frame= 900  # seconds, maximum for a single exposure, 'max' will compute exposure to hit 50% full well 
        self.nsamp = 1        # number of up the ramp samples per frame exposure

class FILTER():
    "float values"
    def __init__(self):
        self.x    = None # wavelength array
        self.y    = None # filter transmission (fraction)
        self.zp   = None # zeropoints storage object - will be loaded
        self.filter_file=None
        self.zp_file = './data/filters/zeropoints.txt' #http://astroweb.case.edu/ssm/ASTR620/mags.html
        self.zp_unit = 'Jy'
        self.band   = 'J' # band to pick, yJHK
        self.family = '2mass' # family of filter band, see zeropoints file 'cfht', '2mass' for JHK
        #zps    = np.loadtxt(self.zp_file,dtype=str).T
        #self.options =[zps[0],zps[1]] # returns options for bands to pick

class STELLAR():
    "star info and spectrum"
    def __init__(self):
        # User optional define:
        self.phoenix_file   = None       # stellar spec file name, **make this take temp value in future
        # Filled in by code:
        self.vraw = None # wavelength like normal (should match exoplanet and be in standard wavelength)
        self.sraw = None #  spectrum
        self.vsini = 0 # km/s
        self.mag = 10
        
class TELLURIC():
    "telluric transmission file, static"
    def __init__(self):
        # User optional define:
        self.telluric_file   = None       # spec file name
        self.airmass = 1.5
        self.pwv     = 1.3
        # Filled in by code:
        self.v = None # wavelength 
        self.s = None #  spectrum


class TRACK():
    "tracking camera storage"
    def __init__(self):
        # User optional defined
        self.transmission_file = None # output name
        # Filled in by code
        self.xtransmit = None # x array of throughput [nm]
        self.ytransmit = None # throughput of tracking camera [0,1]
        self.texp      = None # exposure time of tracking camera [s]
        self.frat      = 40 # f ratio of tracking camera arm
        self.band      = 'JHgap' # band being used, [JHgap,z,y,J,H,K]
        self.offset    = 0 # offset of guide star to science target, [mas] # not implemented correctly yet


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
    so = storage_object()

    for key in config:
        s1,s2=key.split('.')
        try:
            setattr(getattr(so,s1),s2,float(config[key]))
        except ValueError:
            setattr(getattr(so,s1),s2,config[key])


    return so




