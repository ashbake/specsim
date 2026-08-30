##############################################################
# Two-file config -> Simulate translation layer
###############################################################
#
# Replaces specsim/objects.py's storage_object/load_object. Configuration
# is now split across two files:
#
# 1. A user-facing flat .cfg file (same '[section]\noption=value' format
#    as before) holding the parameters someone actually changes from run
#    to run: star magnitude/teff/vsini, exposure time, observing
#    conditions, AO star choice, and -- new -- [run] instrument, which
#    selects...
# 2. An instrument YAML file (configs/instruments/<instrument>.yaml)
#    holding parameters tied to the instrument/telescope: telescope area/
#    diameter, detector properties, AO/tracking-camera hardware and data
#    file paths. These rarely change and are shared across every run of
#    that instrument, so they're factored out into one file per
#    instrument rather than duplicated into every user .cfg.
#
# The two are merged into one flat '<section>.<option>' dict (user .cfg
# wins on any key collision) and translated into typed Simulate
# constructor kwargs exactly as before -- everything downstream of the
# merge is unchanged.

import configparser
import os

import yaml

from specsim.atmosphere import Atmosphere
from specsim.instrument import AOSystem, Spectrograph, TrackingCamera
from specsim.simulate import Simulate
from specsim.star import StarParams

PATH_ATTR_SUFFIXES = ('_file', '_folder', '_path')
INSTRUMENTS_SUBDIR = 'instruments'  # relative to the user .cfg's own directory


def load_config(configfile):
    """
    Read a configuration file 'XXX.cfg' into a flat dictionary.

    inputs
    ------
    configfile : str
        path to a .cfg file with [section] headers and option=value lines

    output
    ------
    config : dict
        keys of the form '<section>.<option>', values as stripped strings
    """
    cp = configparser.ConfigParser()
    cp.read(configfile)
    config = {}
    for sec in cp.sections():
        for opt in cp.options(sec):
            config[f'{sec}.{opt}'] = cp.get(sec, opt).strip()
    return config


def load_instrument_yaml(instrument_configfile):
    """
    Read an instrument YAML file (telescope area/diameter, detector
    properties, AO/tracking-camera hardware and data file paths) into the
    same flat '<section>.<option>' dict shape as load_config, so it can
    be merged with a user-facing .cfg the same way.

    inputs
    ------
    instrument_configfile : str
        path to a YAML file with top-level keys matching the .cfg section
        names (telescope, filt, atm, spectrograph, ao, track), each a mapping of
        option: value

    output
    ------
    config : dict
        keys of the form '<section>.<option>', values as read from YAML
        (already int/float/str as YAML parses them, unlike load_config's
        all-strings)
    """
    with open(instrument_configfile) as f:
        raw = yaml.safe_load(f) or {}
    config = {}
    for section, options in raw.items():
        for opt, value in (options or {}).items():
            config[f'{section}.{opt}'] = value
    return config


def _section(config, name):
    "dict of {option: value} for one '[name]' section, values coerced to float where possible"
    prefix = name + '.'
    out = {}
    for key, value in config.items():
        if key.startswith(prefix):
            opt = key[len(prefix):]
            try:
                out[opt] = float(value)
            except ValueError:
                out[opt] = value
    return out


def _resolve_paths(section, data_folder):
    "resolve any *_file/*_folder/*_path value in section against data_folder, unless already absolute"
    for key, value in section.items():
        if key.endswith(PATH_ATTR_SUFFIXES) and isinstance(value, str) and not os.path.isabs(value):
            section[key] = os.path.join(data_folder, value)
    return section


def simulate_from_config(configfile, instrument_configfile=None, **overrides):
    """
    Build a Simulate from a user-facing .cfg file plus an instrument YAML
    file. Section layout (after the two are merged):
    [stel] -> StarParams (+ pl_teff/pl_mag/pl_vsini/pl_sep for a companion)
    [filt] -> band (user; family is derived from it unless set explicitly)
              + filter_path/zp_file (instrument)
    [atm]  -> pwv/seeing_set (user) + telluric_file/sky_path (instrument)
    [telescope] -> merged into Spectrograph.area_m2/diameter_m (instrument-only)
    [spectrograph] -> Spectrograph (instrument-only)
    [ao]   -> AOSystem (mode/mag/teff/mag_band from user; tt_static/lo_wfe/
              defocus/ho_wfe_file/tt_dynamic_file/contrast_profile_path
              from instrument)
    [obs]  -> texp/texp_frame_set/nsamp/zenith_angle (user-only)
    [track] -> TrackingCamera (instrument-only; only built if present)

    inputs
    ------
    configfile : str
        path to the user-facing .cfg file. Its [run] section must set
        `instrument` (e.g. `instrument=hispec`) unless instrument_configfile
        is given explicitly.
    instrument_configfile : str, optional
        path to the instrument YAML file. If not given, defaults to
        '<dir of configfile>/instruments/<[run] instrument, lowercased>.yaml'
    **overrides
        Simulate constructor kwargs to override after translation, e.g.
        simulate_from_config(path, texp=1800)

    output
    ------
    Simulate
    """
    if not os.path.isfile(configfile):
        raise FileNotFoundError(f"Config file not found: {configfile}")
    user_config = load_config(configfile)
    run = _section(user_config, 'run')

    if instrument_configfile is None:
        instrument_name = run.get('instrument')
        if not instrument_name:
            raise ValueError(f"{configfile}: [run] section must set 'instrument' (e.g. instrument=hispec), "
                              "or pass instrument_configfile explicitly")
        instrument_configfile = os.path.join(os.path.dirname(os.path.abspath(configfile)),
                                              INSTRUMENTS_SUBDIR, f'{str(instrument_name).lower()}.yaml')
    if not os.path.isfile(instrument_configfile):
        raise FileNotFoundError(f"Instrument config file not found: {instrument_configfile}")

    instrument_config = load_instrument_yaml(instrument_configfile)
    config = {**instrument_config, **user_config}  # user .cfg wins on any key collision
    data_folder = run.get('data_folder', './')

    stel = _resolve_paths(_section(config, 'stel'), data_folder)
    filt = _resolve_paths(_section(config, 'filt'), data_folder)
    atm = _resolve_paths(_section(config, 'atm'), data_folder)
    telescope_cfg = _section(config, 'telescope')
    spectrograph_cfg = _resolve_paths(_section(config, 'spectrograph'), data_folder)
    ao = _resolve_paths(_section(config, 'ao'), data_folder)
    obs = _section(config, 'obs')
    track = _resolve_paths(_section(config, 'track'), data_folder)

    star = StarParams(teff=stel['teff'], mag=stel['mag'], vsini=stel.get('vsini', 0), rv=stel.get('rv', 0),
                       logg=stel.get('logg', 4.5), phoenix_folder=stel.get('phoenix_folder'),
                       sonora_folder=stel.get('sonora_folder'))

    pl_sep = stel.get('pl_sep', 0)
    companion = None
    if pl_sep:
        companion = StarParams(teff=stel['pl_teff'], mag=stel['pl_mag'], vsini=stel.get('pl_vsini', 0),
                                rv=stel.get('rv', 0), logg=stel.get('logg', 4.5),
                                phoenix_folder=stel.get('phoenix_folder'), sonora_folder=stel.get('sonora_folder'))

    atmosphere = Atmosphere(telluric_file=atm['telluric_file'], sky_path=atm.get('sky_path'),
                             pwv=atm.get('pwv', 1.3), seeing_set=atm.get('seeing_set', 'average'))

    ao_system = AOSystem(mode=ao.get('mode', 'auto'), tt_static=ao.get('tt_static', 2), lo_wfe=ao.get('lo_wfe', 50),
                         defocus=ao.get('defocus', 25), ho_wfe_file=ao.get('ho_wfe_file'),
                         tt_dynamic_file=ao.get('tt_dynamic_file'), ho_wfe_set=ao.get('ho_wfe_set'),
                         tt_dynamic_set=ao.get('tt_dynamic_set'), mag=ao.get('mag', 'default'),
                         mag_band=ao.get('mag_band', 'default'), teff=ao.get('teff', 'default'),
                         contrast_profile_path=ao.get('contrast_profile_path'))

    spectrograph = Spectrograph(l0=spectrograph_cfg.get('l0', 900), l1=spectrograph_cfg.get('l1', 2500), res=spectrograph_cfg.get('res', 100000),
                                res_samp=spectrograph_cfg.get('res_samp', 3), pix_vert=spectrograph_cfg.get('pix_vert', 4),
                                extraction_frac=spectrograph_cfg.get('extraction_frac', 0.925), saturation=spectrograph_cfg.get('saturation', 100000),
                                readnoise=spectrograph_cfg.get('readnoise', 12), darknoise=spectrograph_cfg.get('darknoise', 0.01),
                                pl_on=spectrograph_cfg.get('pl_on', 1), rv_floor=spectrograph_cfg.get('rv_floor', 0.5),
                                atm=spectrograph_cfg.get('atm', 1), adc=spectrograph_cfg.get('adc', 1),
                                transmission_path=spectrograph_cfg.get('transmission_path'), transmission_file=spectrograph_cfg.get('transmission_file'),
                                order_bounds_file=spectrograph_cfg.get('order_bounds_file'),
                                area_m2=telescope_cfg.get('area_m2', 76), diameter_m=telescope_cfg.get('diameter_m', 10))

    tracking_camera = None
    if track:
        tracking_camera = TrackingCamera(camera=track.get('camera', 'h2rg'), band=track.get('band', 'JHgap'),
                                         fratio=track.get('fratio', 35), texp=track.get('texp', 1),
                                         field_r=track.get('field_r', 0), transmission_file=track.get('transmission_file'),
                                         aberrations_file=track.get('aberrations_file'))

    kwargs = dict(star=star, spectrograph=spectrograph, atmosphere=atmosphere, ao_system=ao_system,
                  filt_band=filt.get('band', 'J'), filt_family=filt.get('family'),
                  filter_path=filt.get('filter_path'), zp_file=filt.get('zp_file'),
                  texp=obs.get('texp', 900), texp_frame_set=obs.get('texp_frame_set', 'default'),
                  nsamp=obs.get('nsamp', 1), zenith_angle=obs.get('zenith_angle', 45),
                  companion=companion, pl_sep=pl_sep, tracking_camera=tracking_camera)
    kwargs.update(overrides)
    return Simulate(**kwargs)
