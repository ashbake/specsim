"""
specsim: an SNR/RV-precision calculator for HISPEC/MODHIS-style
high-resolution spectrographs.

The usual entry point is simulate_from_config(), which reads a user-facing
.cfg plus its instrument YAML and returns a Simulate scene:

    from specsim import simulate_from_config
    sim = simulate_from_config('./configs/modhis_snr.cfg')
    observation = sim.snr()

The domain objects Simulate builds (Star, Bandpass, Atmosphere, AOSystem,
Spectrograph, TrackingCamera, Observation) are re-exported here so they can
be constructed directly when a config file isn't wanted.
"""

from specsim.analyze import Analyze, CCFSNRResult, ETCResult, RVPrecisionResult
from specsim.aosystem import AOSystem
from specsim.atmosphere import Atmosphere
from specsim.bandpass import Bandpass, YJHK
from specsim.config import load_config, load_instrument_yaml, simulate_from_config
from specsim.spectrograph import Spectrograph
from specsim.simulate import Simulate
from specsim.star import Star, StarParams
from specsim.trackingcamera import TrackingCamera

__all__ = [
    'simulate_from_config', 'load_config', 'load_instrument_yaml',
    'Simulate',
    'Star', 'StarParams', 'Bandpass', 'Atmosphere',
    'AOSystem', 'Spectrograph', 'TrackingCamera',
    'Analyze', 'ETCResult', 'RVPrecisionResult', 'CCFSNRResult',
    'YJHK',
]
