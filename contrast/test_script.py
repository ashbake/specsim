import os,sys
# sys.path.append('../')
# os.chdir('../')

specsim_path = os.path.join(os.environ['USERPROFILE'], 'Documents', 'GitHub', 'specsim')
sys.path.insert(0, specsim_path)

# import numpy as np
# import matplotlib.pylab as plt
# import pandas as pd
# from scipy import interpolate


# from specsim.objects import load_object
# from specsim.functions import *

print('Current working directory is', os.getcwd())
from specsim.load_inputs import fill_data

if 'CONDA_DEFAULT_ENV' in os.environ:
    print("Current conda environment:", os.environ['CONDA_DEFAULT_ENV'])
else:
    print("Not running in a conda environment.")

try:
    import matplotlib
    print("matplotlib is installed.")
except ImportError:
    print("matplotlib is not installed.")

print("Python interpreter path:", sys.executable)

print("Python version:", sys.version)
# import matplotlib.pylab as plt
