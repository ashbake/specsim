# specsim
SNR calculator developed for HISPEC/MODHIS tracking camera and main instrument. Can be adapted to other instruments. It utilizes PSISIM functions but with the single purpose to simulate signal and noise spectra for spectrographs and a tracking camera.

## Installation
Clone the repo
```
> git clone https://github.com/ashbake/specsim.git
```
Move into that directory and run the following to pip install psisim and its dependencies:

```
> pip install -e .
```

### Data Downloads & Setup
Many data files are needed to run the examples for MODHIS and HISPEC. 

#### AO Performance Files

#### Throughput Files
Instrument throughput files follow a particular format that is currently hard coded to reflect the file structure from code developed for HISPEC/MODHIS. Luckily there is the option to bypass this by populating the `transmission_file` variable under `[inst]`. If this is filled with a filename that is not None, it will load the contents of that file as the total throughput.

Otherwise the code requires the `transmission_path` to point to the folder that contains the subfolders named the following: ao, bspec, coupling, feiblue, feicam, feicom, feired,fibblue,fibred, rspec, and tel. All but the coupling/ folder should contain a file called '{x}_throughput.csv' where {x} is the folder name, e.g. ao/ should contain the file ao_throughput.csv. The file header is wavelength_um, throughput - the first column is the wavelength in microns and the second column is the fractional throughput.

The coupling folder should contain the output to fiber coupling simulations e.g. couplingEff_atm1_adc1_PL0_defoc0nmRMS_LO0nmRMS_ttStatic1.5mas_ttDynamic5.5masRMS.csv. The coupling depends on the wavefront error and also takes parameters specifying where atmospheric refraction and ADC corrections were assumed, and if the photonic lantern (PL) was used. These paramteres are defined in the config file as adc, atm, PLon, respectively.

#### Filter Files
The filters used primarily here 

#### Telluric File
The telluric models loaded by specsim are PSG models, which should be high resolution and can be created using the psg wrapper called run_psg located [here](https://github.com/ashbake/run_psg). 

A spectrum is zipped in the data/ folder in examples/ that runs from 800 to 2700nm that can be unzipped and linked to in the config file through the ```telluric_file``` variable.

#### Stellar Files

Phoenix Files: 

We recommend downloading specific Phoenix models [here](http://phoenix.astro.physik.uni-goettingen.de/?page_id=15), but if the full Phoenix HiRes Library is desired, it can be downloaded through FTP here: (ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/).


[Sonora](https://zenodo.org/record/1309035#.XbtLtpNKhMA) files: 

These should be unzipped into any directory, which should be specified as the variable ```sonora_folder``` under ```[spec]``` in the configuration files. 
