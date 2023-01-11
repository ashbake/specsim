# need to use tracking camera throughput arm
import sys
sys.path.append('../psisim')
from psisim import telescope,instrument,observation,spectrum,universe,plots
import time
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from importlib import reload

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

SMALL_SIZE = 32
MEDIUM_SIZE = 40
BIGGER_SIZE = 48

plt.rcParams['font.size'] = '14'
plt.rcParams['font.family'] = 'sans'
plt.rcParams['axes.linewidth'] = '1.3'
fontname = 'Arial Narrow'

#Setup the path to the peripheral files
datapath = '/Users/ashbake/Documents/Research/_Data/'
filters = spectrum.load_filters()

#Import the telescope and set some observing conditions
keck = telescope.Keck(path=path)
keck.airmass=1.0
keck.water_vapor=1.6
keck.seeing = keck.median_seeing

##### Some instrument setup
hispec = instrument.hispec(keck)
hispec.set_current_filter("TwoMASS-J")
#Get the set of wavelengths based on the current instrument setup
wavelengths = hispec.get_wavelength_range()

# Set the observing mode
hispec.set_observing_mode(3600,1,'TwoMASS-J', wavelengths) #Exposure time (per exposure), Number of Exposures,filter name, wavelength array


##################################
### Spectrum for the host star ###
##################################

# #The parameters that must be passed to get_stellar_spectrum
# #(the path, a filter name, the magnitude in that filter)
user_params = (path,'TwoMASS-J',5.0,filters,hispec.current_filter)

#Phoenix models need StarLogg, StarTeff, StarZ and some alpha parameter that Dimitri needs to explain to me
host_properties = {"StarLogg":4.50*u.dex(u.cm/ u.s**2),"StarTeff":3000*u.K,"StarZ":'-0.0',"StarAlpha":"0.0","StarRadialVelocity":100*u.km/u.s,
"StarVsini":10*u.km/u.s,"StarLimbDarkening":0.8}

print("Starting to generate a stellar spectrum (hi res)")
host_spectrum = spectrum.get_stellar_spectrum(host_properties,wavelengths,300000,model="Phoenix",user_params=user_params,
doppler_shift=True,broaden=True,delta_wv=hispec.current_dwvs)
print("Done generating a spectrum")

#Update the host properties to have the AO
# host_properties["StarAOmag"] = spectrum.get_obj_ABmag(wavelengths,host_spectrum,hispec.ao_filter[0],filters)
host_properties["StarAOmag"] = spectrum.get_model_ABmags(host_properties,[hispec.ao_filter], model='Phoenix',verbose=False,user_params = user_params)
host_properties["StarSpT"] = None


# Downgrade stellar spectrum after multiplying by tellurics


###############################
### Convert Host Spectrum to Photons ###
###############################
spec_phot = host_spectrum.spectrum * keck.collecting_area.to(u.cm**2)


###############################
### Noise for H2RG camera ###
###############################







###############################
### Spectrum for the object ###
###############################

# #The parameters that must be passed to get_stellar_spectrum
# #(the path, a filter name, the magnitude in that filter)
user_params = (path,'TwoMASS-K',20,filters,hispec.current_filter)

#Sonora models only need StarLogg and StarTeff
obj_properties = {"StarLogg":3.25*u.dex(u.cm/ u.s**2),"StarTeff":700*u.K,"StarRadialVelocity":20*u.km/u.s,"StarVsini":10*u.km/u.s,"StarLimbDarkening":0.9}

print("Starting to generate the object spectrum")
obj_spectrum = spectrum.get_stellar_spectrum(obj_properties,wavelengths,hispec.current_R,model="Sonora",user_params=user_params,
doppler_shift=True,broaden=True,delta_wv=hispec.current_dwvs)
print("Done generating the object spectrum")

### You could save stuff, here's an example
# tmp_host_fn = "/home/mblanchaer/hispec/host_spectrum.npy"
# print("Saving host to {} with units {}".format(tmp_host_fn,host_spectrum.unit))
# np.save(tmp_host_fn,host_spectrum.value)

# tmp_object_fn = "/home/mblanchaer/hispec/object_spectrum.npy"
# print("Saving object to {} with units {}".format(tmp_object_fn,obj_spectrum.unit))
# np.save(tmp_object_fn,np.vstack([wavelengths.value,obj_spectrum.value]))


# fig = plt.figure(figsize=(30,10)) 
# plt.semilogy(wavelengths,host_spectrum,linewidth=2) 
# plt.semilogy(wavelengths,obj_spectrum*host_spectrum) 
# plt.ylim(1e-17,3e0)
# plt.xlim(1.1,1.36)
# plt.show() 

host_properties['AngSep'] = 400*u.uarcsec # In microarcsecond
hispec.ao_mag = host_properties["StarAOmag"]
# #We want the object spectrum in contrast units
obj_spec_contrast = obj_spectrum.spectrum / host_spectrum.spectrum
obj_spec_contrast[np.where(np.isnan(obj_spec_contrast))] = 0


obj_spec,total_noise,stellar_spec,thermal_spectrum,noise_components = observation.simulate_observation(keck,hispec,host_properties,obj_spec_contrast,wavelengths,1e5,
    inject_noise=False,verbose=True,post_processing_gain = 10,return_noise_components=True,stellar_spec=host_spectrum.spectrum,
    apply_lsf=True,integrate_delta_wv=False)

### More Saving examples
# tmp_object_fn = "/home/mblanchaer/hispec/object_spectrum_final.npy"
# print("Saving object to {} with units {}".format(tmp_object_fn,obj_spec.unit))
# np.save(tmp_object_fn,np.vstack([wavelengths.value,obj_spec.value]))


fig = plt.figure(figsize=(20,10))
plt.semilogy(wavelengths,obj_spec,linewidth=2,label="Object Spectrum")
plt.plot(wavelengths,thermal_spectrum,label="Thermal Spectrum")
plt.ylabel('[{}]'.format(obj_spec.unit))
plt.xlim(1.10,1.36)
# plt.ylim(1e-6,1e2)
plt.show()
# # obj_spec = spectrum.downsample_spectrum(obj_spec,2e5,0.25e5)

wvs = wavelengths
th_total = keck.get_atmospheric_transmission(wvs)
th_total *= keck.get_telescope_throughput(wvs,hispec)
th_total *= hispec.get_inst_throughput(wvs)
th_total *= hispec.get_filter_transmission(wvs,hispec.current_filter)

# plt.plot(wavelengths,th_total,label="Total Thermal Flux")

plt.legend()

