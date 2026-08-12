##############################################################
# Tools for loading spectra and scaling to correct magnitude
###############################################################

import numpy as np
from astropy.io import fits
from scipy import interpolate
import glob,os
from astropy.convolution import convolve

from specsim.functions import *

#all = {}



def load_phoenix(stelname,stelpath,wav_start=750,wav_end=780):
	"""
	Load a PHOENIX stellar spectrum fits file and return the requested
	wavelength subarray, converted to photon flux units.

	http://phoenix.astro.physik.uni-goettingen.de/?page_id=15

	Converts flux from ergs/s/cm2/cm to phot/cm2/s/nm using
	https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf
	then to phot/m2/s/nm.

	inputs:
	-------
	stelname - str
		filename of the PHOENIX spectrum fits file (flux only; the
		wavelength grid is loaded separately from
		'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits' in stelpath)

	stelpath - str
		directory path containing stelname and the PHOENIX wavelength file

	wav_start - float
		lower wavelength bound of the returned subarray [nm] (default 750)

	wav_end - float
		upper wavelength bound of the returned subarray [nm] (default 780)

	returns:
	--------
	wavelength - array
		wavelength subarray [nm]

	flux - array
		photon flux subarray [phot/m2/s/nm]
	"""
	# conversion factor

	f = fits.open(stelpath + stelname)
	spec = f[0].data / (1e8) # ergs/s/cm2/cm to ergs/s/cm2/Angstrom for conversion
	f.close()
	
	wave_file = os.path.join(stelpath + 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits') #assume wave in same folder
	f = fits.open(wave_file)
	lam = f[0].data # angstroms
	f.close()
	
	# Convert
	conversion_factor = 5.03*10**7 * lam #lam in angstrom here
	spec *= conversion_factor # phot/cm2/s/angstrom
	
	# Take subarray requested
	isub = np.where((lam > wav_start*10.0) & (lam < wav_end*10.0))[0]

	# Convert 
	return lam[isub]/10.0,spec[isub] * 10 * 100**2 #nm, phot/m2/s/nm

def load_filter(filter_path,family,band):
	"""
	Load a photometric filter transmission curve from a data file matching
	the given filter family and band.

	Searches filter_path for a file matching '*<family>*<band>.dat' and
	loads its two columns as wavelength [nm] and transmission. Filter
	files store wavelength in different units depending on family
	(e.g. Angstrom for Johnson, micron for 2MASS/cfht/decam), so the
	units are auto-detected the same way fill_data.filter() does: values
	with max > 5000 are assumed to be Angstrom (divided by 10 to get nm),
	values with max < 10 are assumed to be micron (multiplied by 1000 to
	get nm).

	inputs:
	-------
	filter_path - str
		directory to search for the filter file

	family - str
		filter family name (e.g. 'Johnson', 'SLOAN'), matched as a
		substring of the filter filename

	band - str
		filter band name (e.g. 'V', 'uprime_filter'), matched as a
		substring of the filter filename

	returns:
	--------
	xraw - array
		filter wavelength grid [nm]

	yraw - array
		filter transmission, unitless (out of 1)
	"""
	filter_file    = glob.glob(filter_path + '*' + family + '*' + band + '.dat')[0]
	xraw, yraw     = np.loadtxt(filter_file).T # units vary by file (Angstrom or micron)
	if np.max(xraw) > 5000: xraw = xraw / 10    # Angstrom -> nm
	if np.max(xraw) < 10: xraw = xraw * 1000    # micron -> nm
	return xraw, yraw

def load_sonora(stelname,wav_start=750,wav_end=780):
	"""
	Load a Sonora stellar/substellar atmosphere model file and return the
	requested wavelength subarray, converted to photon flux units.

	The file's wavelength column is loaded in microns, high to low, and is
	reversed and converted to Angstroms internally. Flux is converted from
	erg/cm2/s/Hz to erg/cm2/s/Angstrom (via c/lambda^2) and then to
	phot/cm2/s/Angstrom using
	https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf, before being
	returned as phot/m2/s/nm.

	inputs:
	-------
	stelname - str
		path to the Sonora model file (whitespace-delimited, 2 header rows,
		columns: wavelength [micron], flux [erg/cm2/s/Hz])

	wav_start - float
		lower wavelength bound of the returned subarray [nm] (default 750)

	wav_end - float
		upper wavelength bound of the returned subarray [nm] (default 780)

	returns:
	--------
	wavelength - array
		wavelength subarray [nm]

	flux - array
		photon flux subarray [phot/m2/s/nm]
	"""
	f = np.loadtxt(stelname,skiprows=2)

	lam  = 10000* f[:,0][::-1] #microns to angstroms, needed for conversiosn
	spec = f[:,1][::-1] # erg/cm2/s/Hz
	
	spec *= 3e18/(lam**2)# convert spec to erg/cm2/s/angstrom
	
	conversion_factor = 5.03*10**7 * lam #lam in angstrom here
	spec *= conversion_factor # phot/cm2/s/angstrom
	
	isub = np.where( (lam > wav_start*10.0) & (lam < wav_end*10.0))[0]

	return lam[isub]/10.0,spec[isub] * 10 * 100**2 #nm, phot/m2/s/nm (my fave)

def calc_nphot(dl_l, zp, mag):
	"""
	http://astroweb.case.edu/ssm/ASTR620/mags.html

	Values are all for a specific bandpass, can refer to table at link ^ for values
	for some bands. Function will return the photons per second per meter squared
	at the top of Earth atmosphere for an object of specified magnitude

	inputs:
	-------
	dl_l: float, delta lambda over lambda for the passband, unitless
	zp: float, flux at m=0 in Jansky (zero-point flux)
	mag: float, stellar magnitude in the passband

	outputs:
	--------
	photon flux: float, photons per second per square meter [phot/s/m2]
	at the top of Earth's atmosphere for an object of the given magnitude
	"""
	phot_per_s_m2_per_Jy = 1.51*10**7 # convert to phot/s/m2 from Jansky

	return dl_l * zp * 10**(-0.4*mag) * phot_per_s_m2_per_Jy

def scale_stellar(filt,stelv,stels,mag):
	"""
	Compute the scale factor needed to normalize a model stellar spectrum
	so that its integrated flux through a given filter matches a specified
	magnitude.

	Interpolates the filter transmission onto the stellar wavelength grid,
	integrates the filtered stellar spectrum, and compares it to the
	expected photon flux for the requested magnitude (via calc_nphot) to
	derive a single multiplicative scale factor. Raises a Warning if the
	stellar model does not fully cover the filter bandpass.

	inputs:
	-------
	filt - object
		so.filt object; must provide xraw/yraw (filter wavelength [nm] and
		transmission) as well as dl_l and zp attributes used by calc_nphot

	stelv - array
		stellar model wavelength grid [nm]

	stels - array
		stellar model flux density [phot/m2/s/nm] evaluated at stelv

	mag - float
		desired magnitude of the star in the filter bandpass

	returns:
	--------
	factor - float
		multiplicative scale factor to apply to stels so that its
		integrated flux through the filter matches mag
	"""
	if (np.min(filt.xraw) < np.min(stelv)) or (np.max(filt.xraw) > np.max(stelv)):
		raise Warning('Check that stellar model in scale_stellar extends past filter profile')
	
	filt_interp       =  interpolate.interp1d(filt.xraw,filt.yraw, bounds_error=False,fill_value=0)

	filtered_stellar   = stels * filt_interp(stelv)    # filter profile resampled to phoenix times phoenix flux density
	nphot_expected_0   = calc_nphot(filt.dl_l, filt.zp, mag)    # what's the integrated flux supposed to be in photons/m2/s?
	nphot_model        = integrate(stelv,filtered_stellar)            # what's the integrated flux now? in same units as ^
	
	return nphot_expected_0/nphot_model


def load_stellar_model(x,mag,teff,vsini,so,rv=0):
	"""
	Load a model stellar spectrum (Sonora or PHOENIX, chosen by effective
	temperature), scale it to the designated magnitude, broaden it by
	rotational velocity, optionally apply a radial-velocity Doppler shift,
	and interpolate the result onto the requested wavelength grid x.

	Sonora models are used for teff < 2300 K (assuming log g = log10(316*100)
	= 4.5), otherwise a PHOENIX model at so.stel.logg is used.

	inputs:
	-------
	x - array
		wavelength grid [nm] onto which the final spectrum is interpolated

	mag - float
		desired stellar magnitude in the filter so.filt (used to scale the
		model via scale_stellar)

	teff - float
		stellar effective temperature [K]; selects Sonora (teff < 2300) or
		PHOENIX (teff >= 2300) model grid

	vsini - float
		projected stellar rotational velocity [km/s]; if > 0, the spectrum
		is convolved with a rotational broadening kernel (see
		_lsf_rotate)

	so - object
		storage object; used only for paths and filter information
		(so.filt.xraw, so.stel.sonora_folder, so.stel.phoenix_folder,
		so.stel.logg)

	rv - float
		radial velocity offset to apply to the spectrum via a Doppler shift
		[km/s], used e.g. to offset the star from tellurics for CCF
		purposes (default 0)

	returns:
	--------
	shifted_spec - array
		final stellar spectrum [phot/s/m2/nm] interpolated onto x, scaled
		to mag, broadened by vsini, and Doppler-shifted by rv (negative
		values from interpolation artifacts are clipped to zero)

	vraw - array
		raw (unscaled, unbroadened) model wavelength grid [nm] as loaded
		from the Sonora/PHOENIX file

	sraw - array
		raw (unscaled, unbroadened) model flux density [phot/m2/s/nm] as
		loaded from the Sonora/PHOENIX file

	model - str
		which model grid was used, 'sonora' or 'phoenix'

	stel_file - str
		full path to the loaded model file (Sonora), or just the filename
		within so.stel.phoenix_folder (PHOENIX)

	factor_0 - float
		multiplicative scale factor applied to match mag (from
		scale_stellar)
	"""
	# wavelength bounds should incldue filter entirely
	l0,l1 = np.min((np.min(x),np.min(so.filt.xraw))),np.max((np.max(x),np.max(so.filt.xraw)))

	if teff < 2300: # sonora models arent sampled as well so use phoenix as low as can
		g    = '316' # mks units, np.log10(316 * 100)=4.5 to match what im holding for phoenix models.
		teff = str(int(teff))
		stel_file         = so.stel.sonora_folder + 'sp_t%sg%snc_m0.0' %(teff,g)
		vraw,sraw = load_sonora(stel_file,wav_start=l0,wav_end=l1)
		model             = 'sonora'
	else:
		teff = str(int(teff)).zfill(5)
		logg = '{:.2f}'.format(so.stel.logg)
		model             = 'phoenix' 
		stel_file         = 'lte%s-%s-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'%(teff,logg)
		vraw,sraw = load_phoenix(stel_file,so.stel.phoenix_folder,wav_start=l0, wav_end=l1) #phot/m2/s/nm
	

	# apply scaling factor to match filter zeropoint
	factor_0   = scale_stellar(so.filt,vraw,sraw,mag) # loads spectrum over selected filter and finds scaling to get correct magnitude
	
	# interpolate file onto x, apply factor
	tck_stel    = interpolate.splrep(vraw,sraw, k=2, s=0)
	s           = factor_0 * interpolate.splev(x,tck_stel,der=0,ext=1)
	
	#units = 'photons/s/m2/nm' # stellar spec is in photons/s/m2/nm

	# broaden star spectrum with rotation kernal
	SPEEDOFLIGHT   = 2.998e8 # m/s
	if vsini > 0:
		dwvl_mean = np.abs(np.nanmean(np.diff(x)))
		dvel_mean      = (dwvl_mean / np.nanmean(x)) * SPEEDOFLIGHT / 1e3 # average sampling in km/s
		vsini_kernel,_ = _lsf_rotate(dvel_mean,vsini,epsilon=0.6)
		flux_vsini     = convolve(s,vsini_kernel,normalize_kernel=True)  # photons / second / Ang
		s              = flux_vsini

	# Offset star by an RV (for CCF purposes to offset from tellurics)
	if rv!= 0:
		doppler_factor = (1.0 + ((rv * 1000) / SPEEDOFLIGHT)) # rv in km/s
		tck = interpolate.splrep(x*doppler_factor,s, k=3, s=0)
		shifted_spec = interpolate.splev(x,tck,der=0,ext=1)
	else: 
		shifted_spec=s.copy()

	# some negatives are created when interpolating, change these to zero
	ineg = np.where(shifted_spec<0)[0]
	shifted_spec[ineg] = 0
	
	return shifted_spec, vraw, sraw, model, stel_file, factor_0


def get_band_mag2(so,family,band,factor_0):
    """
    Compute the apparent magnitude of the currently loaded stellar model
    (scaled by factor_0) in a requested photometric filter band.

    Loads the filter transmission curve, reloads the stellar model over the
    filter's wavelength range if it extends beyond the instrument's
    wavelength range [so.inst.l0, so.inst.l1] (otherwise reuses the
    already-loaded so.stel.vraw/so.stel.sraw), integrates the scaled and
    filtered stellar spectrum to get flux in phot/m2/s, converts to Jansky,
    and compares to the filter's zero point to get magnitude.

    inputs:
    -------
    so - object
        storage object; uses so.filt.filter_path, so.filt.zp_file,
        so.inst.l0/so.inst.l1 [nm], and so.stel.model/so.stel.stel_file/
        so.stel.phoenix_folder/so.stel.vraw/so.stel.sraw as needed

    family - str
        filter family name (e.g. 'Johnson', 'SLOAN'), passed to load_filter

    band - str
        filter band name (e.g. 'V'), passed to load_filter

    factor_0 - float
        multiplicative scale factor applied to the stellar model flux
        (scaling model to photons; from scale_stellar/load_stellar_model)

    returns:
    --------
    mag - float
        apparent magnitude of the scaled stellar model in the requested
        filter band
    """
    x,y          = load_filter(so.filt.filter_path,family,band)
    filt_interp  = interpolate.interp1d(x, y, bounds_error=False,fill_value=0)
    dl_l         = np.mean(integrate(x,y)/x) # dlambda/lambda to account for spectral fraction
    
    # load stellar the multiply by scaling factor, factor_0, and filter. integrate
    if (np.min(x) < so.inst.l0) or (np.max(x) > so.inst.l1):
        if so.stel.model=='phoenix':
            vraw,sraw = load_phoenix(so.stel.stel_file,so.stel.phoenix_folder,wav_start=np.min(x), wav_end=np.max(x)) #phot/m2/s/nm
        elif so.stel.model=='sonora':
            vraw,sraw = load_sonora(so.stel.stel_file,wav_start=np.min(x), wav_end=np.max(x)) #phot/m2/s/nm
    else:
        vraw,sraw = so.stel.vraw, so.stel.sraw

    filtered_stel = factor_0 * sraw * filt_interp(vraw)
    flux = integrate(vraw,filtered_stel)    #phot/m2/s

    phot_per_s_m2_per_Jy = 1.51*10**7 # convert to phot/s/m2 from Jansky
    
    flux_Jy = flux/phot_per_s_m2_per_Jy/dl_l
    
    # get zps
    zps          = np.loadtxt(so.filt.zp_file,dtype=str).T
    izp          = np.where((zps[0]==family) & (zps[1]==band))[0]
    zp           = float(zps[2][izp])

    mag = -2.5*np.log10(flux_Jy/zp)

    return mag


def get_band_mag(so,vraw, sraw, model,stel_file,family,band,factor_0):
    """
    Compute the apparent magnitude of a given (already-loaded) stellar
    model spectrum, scaled by factor_0, in a requested photometric filter
    band.

    Like get_band_mag2, but takes the stellar model arrays/metadata
    explicitly rather than always pulling them from so.stel. Loads the
    filter transmission curve, reloads the stellar model (via load_phoenix
    or load_sonora) over the filter's wavelength range if it extends beyond
    the range of the passed-in vraw, integrates the scaled and filtered
    stellar spectrum to get flux in phot/m2/s, converts to Jansky, and
    compares to the filter's zero point to get magnitude.

    REDO TO NOT ASSUME SO!

    inputs:
    -------
    so - object
        storage object; uses so.filt.filter_path, so.filt.zp_file, and
        (if a reload is needed) so.stel.phoenix_folder

    vraw - array
        stellar model wavelength grid [nm]

    sraw - array
        stellar model flux density [phot/m2/s/nm] evaluated at vraw

    model - str
        which model grid vraw/sraw came from, 'phoenix' or 'sonora'; used
        to pick the reload function if the filter extends past vraw

    stel_file - str
        model filename/path used to reload the stellar model if needed
        (passed to load_phoenix or load_sonora)

    family - str
        filter family name (e.g. 'Johnson', 'SLOAN'), passed to load_filter

    band - str
        filter band name (e.g. 'V'), passed to load_filter

    factor_0 - float
        multiplicative scale factor applied to the stellar model flux
        (scaling model to photons; from scale_stellar/load_stellar_model)

    returns:
    --------
    mag - float
        apparent magnitude of the scaled stellar model in the requested
        filter band
    """
    xfilt,yfilt  = load_filter(so.filt.filter_path,family,band)
    filt_interp  = interpolate.interp1d(xfilt, yfilt, bounds_error=False,fill_value=0)
    dl_l         = np.mean(integrate(xfilt,yfilt)/xfilt) # dlambda/lambda to account for spectral fraction
    # load stellar the multiply by scaling factor, factor_0, and filter. integrate
    # reload if filter extends past currently loaded stellar model
    # if (np.min(xfilt) < np.min(vraw)) or (np.max(xfilt) > np.max(vraw)):
	# 	if model=='phoenix':
	#         vraw,sraw = load_phoenix(stel_file,so.stel.phoenix_folder,wav_start=np.min(xfilt), wav_end=np.max(xfilt)) #phot/m2/s/nm
	# 	elif model=='sonora':
	#         vraw,sraw = load_sonora(stel_file,wav_start=np.min(xfilt), wav_end=np.max(xfilt)) #phot/m2/s/nm
	#     print('Note had to reload stellar model for _get_band_mag')

    if (np.min(xfilt) < np.min(vraw)) or (np.max(xfilt) > np.max(vraw)):
        if model=='phoenix':
            vraw,sraw = load_phoenix(stel_file,so.stel.phoenix_folder,wav_start=np.min(xfilt), wav_end=np.max(xfilt)) #phot/m2/s/nm
            print('Note: had to reload Phoenix stellar model for _get_band_mag')
        elif model=='sonora':
            vraw,sraw = load_sonora(stel_file,wav_start=np.min(xfilt), wav_end=np.max(xfilt)) #phot/m2/s/nm
            print('Note: had to reload Sonora stellar model for _get_band_mag')

    filtered_stel = factor_0 * sraw * filt_interp(vraw)
    flux = integrate(vraw,filtered_stel)    #phot/m2/s

    phot_per_s_m2_per_Jy = 1.51*10**7 # convert to phot/s/m2 from Jansky
    
    flux_Jy = flux/phot_per_s_m2_per_Jy/dl_l
    
    # get zps
    zps          = np.loadtxt(so.filt.zp_file,dtype=str).T
    izp          = np.where((zps[0]==family) & (zps[1]==band))[0]
    zp           = float(zps[2][izp])
	
    mag = -2.5*np.log10(flux_Jy/zp)
	
    return mag



def _lsf_rotate(deltav,vsini,epsilon=0.6):
    '''
    Computes vsini rotation kernel.
    Based on the IDL routine LSF_ROTATE.PRO, which implements the
    analytic rotational broadening profile of Gray, D. F. 1992,
    "The Observation and Analysis of Stellar Photospheres".

    Parameters
    ----------
    deltav : float
        Velocity sampling for kernel (x-axis) [km/s]

    vsini : float
        Stellar vsini value [km/s]

    epsilon : float
        Limb darkening value (default is 0.6)

    Returns
    -------
    kernel : array
        Computed rotational broadening kernel profile, unitless, sampled
        on velgrid (suitable for use as a convolution kernel)

    velgrid : array
        x-values for kernel [km/s]

    '''

    # component calculations
    ep1 = 2.0*(1.0 - epsilon)
    ep2 = np.pi*epsilon/2.0
    ep3 = np.pi*(1.0 - epsilon/3.0)

    # make x-axis
    npts = np.ceil(2*vsini/deltav)
    if npts % 2 == 0:
        npts += 1
    nwid = np.floor(npts/2)
    x_vals = (np.arange(npts) - nwid) * deltav/vsini
    xvals_abs = np.abs(1.0 - x_vals**2)
    velgrid = xvals_abs*vsini

    # compute kernel
    kernel = (ep1*np.sqrt(xvals_abs) + ep2*xvals_abs)/ep3

    return kernel, velgrid
