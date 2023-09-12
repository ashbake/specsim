# specsim
Specsim is an SNR calculator developed for HISPEC/MODHIS, but it can be adapted to other instruments. It builds on a lot of functions from the PSISIM package. The main branch should be runnable after download - it does not include all the latest features and updates, but demonstrates the usage of the code.

## Installation
Clone the repo
```
> git clone https://github.com/ashbake/specsim.git
```
Move into that directory and run the following to pip install specsim and its dependencies:

**Depending on your python environment setup, this may not work. If not, install the packages listed in requirements.txt into your python 3 environment**

```
> pip install -e .
```

### Data Downloads & Setup
Many data files are needed to run the examples for MODHIS and HISPEC. A set of files are included in the repo in the data/ folder and are already linked to in the config file such that the only thing that needs to be done to run the example below is to unzip the telluric file provided. 

If you would like to download more files to run more stellar temperatures, magnitudes, and airmasses, read below. Otherwise skip to the *Running specsim* section at the bottom

#### AO Performance Files
AO files are needed to define the high order wavefront error and tip tilt residiuals as a function of the stellar magnitude. These WFE terms are used by the code to determine the fiber coupling performance. For HISPEC and MODHIS, we use AO simulations of the AO systems called HAKA and NFIRAOS, respectively, to generate the files provided in `data/wfe/`.

The MODHIS dynamic tip tilt file, for example, called `TTDYNAMIC_NFIRAOS.csv` contains columns of the magnitude, the flux (not sure what the flux is to, need to look into this) in that band in e-, and the tip tilt error in mas for the three main MODHIS AO modes: NGS, LGS_ON, and LGS_OFF. The header specifies that these magnitudes and flux values are defined in V band. In reality the MODHIS AO system receives a slightly more narrow range of wavelengths, so we should update this to some V_NFIRAOS label that specifies the specific wavelength range (this matters for red stars). Anywho, for now we can just use V band. 


#### Instrument (throughput) Files
Instrument throughput files follow a particular format that is currently hard coded to reflect the file structure from code developed for HISPEC/MODHIS. Luckily there is the option to bypass this by populating the `transmission_file` variable under `[inst]`. If this is filled with a filename that is not None, it will load the contents of that file as the total throughput.

Otherwise the code requires `transmission_path` to point to the folder that contains the subfolders named the following: ao, bspec, coupling, feiblue, feicam, feicom, feired,fibblue,fibred, rspec, and tel. All but the coupling/ folder should contain a file called '{x}_throughput.csv' where {x} is the folder name, e.g. ao/ should contain the file ao_throughput.csv. The file header is wavelength_um, throughput - the first column is the wavelength in microns and the second column is the fractional throughput.

The coupling folder should contain the output to fiber coupling simulations e.g. `couplingEff_atm1_adc1_PL0_defoc0nmRMS_LO0nmRMS_ttStatic1.5mas_ttDynamic5.5masRMS.csv`. The coupling depends on the wavefront error and also takes parameters specifying where atmospheric refraction and ADC corrections were assumed, and if the photonic lantern (PL) was used. These paramteres are defined in the config file as adc, atm, PLon, respectively.

In the future we will want the instrument files to include resolution as a function of wavelength. 

#### Tracking Camera Files
The file `HISPEC_ParaxialTel_OAP_TrackCamParax_SpotSizevsField.txt` lives in the WFE folder but is used to determine the off axis aberrations due to the tracking camera optics. This is only used in the tracking camera calculations to get the correct FWHM of the PSF as a function of field radius. This file is generated by Mitsuko using ZEMAX simulations for HISPEC and we can use it for MODHIS as well for now.

The tracking camera has its own transmission file variable (`transmission_file`) which is a static transmission profile unlike that of the spectrometer, which is split up. The tracking camera throughput file structure and units matches that of the individual instrument throughput files (microns, fractional transmission).

#### Filter Files
The filters used primarily here are 2MASS J/H/K and CFHT y band, similar to PSISIM. These are provided in the examples/data/ folder. Other filters can be used, but the code relies on the file `zeropoints.txt`, which contains zero point information for each filter. This file must be updated if a new filter is added. The filter band and the family is specified in the config file. This filter profile is primarily used to correctly scale the magnitude of the stellar model.

The [SVO service](http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=2MASS&asttype=) is a handy place to download filter profiles.

#### Telluric File
The telluric models loaded by specsim are assumed to be in the format of PSG models, which should be high resolution and can be created using the psg wrapper called run_psg located [here](https://github.com/ashbake/run_psg). 

A spectrum is zipped and provided in data/telluric/ that spans 800 to 2700nm. This file can be unzipped and linked to in the config file through the ```telluric_file``` variable.

#### Stellar Files

Phoenix Files: 

We recommend downloading specific Phoenix models [here](http://phoenix.astro.physik.uni-goettingen.de/?page_id=15), but if the full Phoenix HiRes Library is desired, it can be downloaded through FTP here: (ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/).


[Sonora](https://zenodo.org/record/1309035#.XbtLtpNKhMA) files: 

These should be unzipped into any directory, which should be specified as the variable ```sonora_folder``` under ```[spec]``` in the configuration files. 



# Running specsim

First (from the code directory) start a python session and import some key packages from specsim:
```
> from specsim import objects, load_inputs, plot_tools
```

Key parameters are stored in a configuration file e.g. "modhis_snr.cfg'. The function "load_object' loads the contents of this configuration file into a storage object "so'. The objects.py function is a useful reference for seeing what is contained in so, but it has class attributes like 'stel' for stellar properties and 'track' for tracking camera properties. For example, the stellar temperature defined in the config file will be loaded and stored in "so.stel.teff'.

The "fill_data" class takes the storage object and upon initiation, it fills the `so` object. As such, this process takes a little while - first it defines the high resolution wavelength grid, then does the dirty work of loading, reinterpolating files, and calculating things in the correct order.


```
> configfile = './configs/modhis_snr.cfg' # define our config file name and path
> so    = objects.load_object(configfile)     # Load the contents of the config file into the so "storage object"
> cload = load_inputs.fill_data(so)           # Initiate the fill_data class which runs an observation and stores the results in so
```


We can then use some plotting tools to plot the snr
```
> plot_tools.plot_snr(so,snrtype=0,savepath=savepath)
```

The instrument wavelength and SNR per pixel is stored in `so.obs.v` and  `so.obs.s`, respectively. 



