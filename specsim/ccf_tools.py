##############################################################
# cross correlation function tools (RV, CCF SNR)
# taken from Sam Halverson and Arpita Roys code
###############################################################

import numpy as np
from scipy import signal
from scipy import signal, interpolate

all = {}
SPEEDOFLIGHT = 2.998e8 # m/s
GAUSSCONST = (2. * (2. * np.log(2))**0.5)

def spec_make(wvl, weights, line_wvls, fwhms):
	'''
	Generate fake (normalized) spectrum of gaussian 'absorption' lines.

	Inputs:
	-------
	wvl : array
		Input wavelength array [nm]
	weights : array
		Line depths of specified lines (fractional, 0-1)
	line_wvls : array
		Line centers of features to be added [nm], same units as wvl
	fwhms : array
		FWHMs of lines specified [nm], same units as wvl

	Outputs:
	-------
	spec_out: array
		 Final output absorption spectrum, normalized continuum at 1.0 with
		 gaussian dips at each line_wvl of the given depth and fwhm
	'''

	# initialize array
	spec_out = np.zeros_like(wvl)

	# for each line wavelength, add a gaussian at the specified depth
	for weight, line_wvl, fwhm in zip(weights, line_wvls, fwhms):
		spec_out += (weight * gaussian_fwhm(wvl, line_wvl, fwhm))
	return 1. - spec_out

def gaussian_fwhm(xarr, center, fwhm,A=1,B=0):
	'''
	Simple gaussian function, defined by center and FWHM

	Parameters
	----------
	xarr : array
		Input dependant variable array
	center : float
		Center of gaussian distribution
	fwhm : float
		FWHM of gaussian desired
	A : float
		Amplitude of gaussian
	B : float
		Vertical offset of gaussian

	Returns
	-------
	gauss : array
		Computed gaussian values for xarr
	'''
	# gaussian function parameterized with FWHM
	gauss = A*np.exp(-0.5 * (xarr - center) ** 2. / (fwhm / GAUSSCONST) ** 2.) + B
	return gauss

def spec_rv_noise_calc(wvl, spec, sigma_spec):
	'''
	Calculates photon-limited RV uncertainty of given spectrum in km/s

	Parameters
	----------
	wvl : array
		Input wavelength array of spectrum [nm]
	spec : array
		Flux values of spectrum -- assumes only photon noise
	sigma_spec : array
		1-sigma flux uncertainty of spec (e.g. sqrt(counts) photon noise),
		same length/units as spec. Zero-valued entries are reset in-place
		to a large number (1e5) to avoid division by zero.

	Returns
	-------
	sigma_rv : float
		Computed photon-limited RV uncertainty [m/s]
	'''

	# calculate pixel optimal weights, follows Murphy et al. 2007
	wvl_m_ord = wvl * 1e-9 # convert wavelength values to meters

	# calculate noise (photon only, assume root N)
	sigma_spec[np.where(sigma_spec==0)[0]] = 100000

	# calculate slopes of spectrum
	#slopes = np.gradient(spec, wvl_m_ord)
	flux_interp = interpolate.InterpolatedUnivariateSpline(wvl_m_ord,spec, k=1)
	dflux = flux_interp.derivative()
	slopes = dflux(wvl_m_ord)

	# calculate weighted slopes, ignoring the edge pixels (breaks derivative)
	top = (wvl_m_ord[1:slopes.size - 1]**2.) * (slopes[1:slopes.size - 1]**2.)
	bottom = (sigma_spec[1:slopes.size - 1]**2.)
	w_ord = top / bottom

	# combined weighted slopes
	return SPEEDOFLIGHT / ((np.nansum(w_ord[1:-1]))**0.5) # m/s


def get_order_bounds(filename):
	"""
	open order bounds file

	input
	-----
	filename - name of order file containing wavelength [nm], order width [nm] comma delimited

	output
	------
	cenlam - order center wavelength [nm]
	width  - order width [nm]
	"""
	f = np.loadtxt(filename,delimiter=',')
	cenlam, width = f.T[0],f.T[1]
	return cenlam, width



def make_telluric_mask(v,s,cutoff=0.01,velocity_cutoff=5):
	"""
	input
	-----
	v - array[nm]
		wavelength array of telluric spectrum
	s - array [transmission 0-1]
		spectrum of telluric spectrum
	cutoff - float
		cutoff (0-1) in what lines to mask. default is 0.01 (mask down to 1%)
	velocity_cutoff - float [km/s] 
		velocity around each telluric feature to mask out. Assumes 1 pix ~= km/s

	output
	------
	telluric_mask - array
		mask corresponding sampled at v array
	"""
	telluric_mask = np.ones_like(s)
	telluric_mask[np.where(s < (1-cutoff))[0]] = 0
	for iroll in range(velocity_cutoff): # assume one pixel is 1km/s approx
		telluric_mask[np.where(np.roll(s,iroll) < (1-cutoff))[0]] = 0
		telluric_mask[np.where(np.roll(s,-1*iroll) < (1-cutoff))[0]] = 0

	return telluric_mask


def get_rv_precision(v,s,n,order_cens,order_widths,noise_floor=0.5,mask=None):
	"""
	Compute the photon-limited RV precision achievable from a stellar
	spectrum, evaluated per spectral order and combined, following the
	spectral information-content method of Murphy et al. 2007 (weighting
	each pixel by the square of the local flux slope divided by its noise
	variance).

	inputs
	------
	v - array [nm]
		wavelength array
	s - array
		stellar spectrum (no other sources in it)
	n - array
		noise array (1-sigma uncertainty on s), same length/sampling as v and s
	order_cens - array [nm]
		center wavelength of each spectral order over which to compute RV
		precision
	order_widths - array [nm]
		full width of each spectral order, same length as order_cens (in
		the same order). Only the central 90% of each order's width is
		used when summing the RV information content, to avoid noisy
		order edges
	noise_floor - float, [m/s]
		RV noise floor added in quadrature to each order's photon-limited
		precision to represent additional non-photon error terms (e.g.
		wavelength calibration floor). Default is 0.5 m/s
	mask - array or None
		optional per-pixel weighting mask (same length/sampling as v, s,
		n), e.g. to zero out telluric-contaminated or otherwise unwanted
		pixels before computing RV information content. If None
		(default), no masking is applied (all pixels weighted equally)

	output
	------
	dv_tot -  array, [m/s]
		per order rv precision with rv floor added
	dv_spec - float, [m/s]
		combined order velocities, no floor added
	dv_vals - array [m/s]
		per order rv precision, no floor added
	"""
	# generate rv information content
	flux_interp = interpolate.InterpolatedUnivariateSpline(v,s, k=1)
	dflux = flux_interp.derivative()
	spec_deriv = dflux(v)
	sigma_ord = np.abs(n) #np.abs(s) ** 0.5 # np.abs(n)
	sigma_ord[np.where(sigma_ord ==0)] = 1e10
	all_w = (v ** 2.) * (spec_deriv ** 2.) / sigma_ord ** 2. # include read noise and dark here!!
	
	# make mask if none provided
	if np.any(mask==None):
		mask = np.ones_like(all_w)

	# go through each order
	dv_vals = np.zeros_like(order_cens)
	for i,lam_cen in enumerate(order_cens):
		order_ind   = np.where((v > lam_cen - 0.9*order_widths[i]/2) & (v < lam_cen + 0.9*order_widths[i]/2))[0]
		w_ord       = all_w[order_ind] * mask[order_ind]
		denom       = (np.nansum(w_ord[1:-1])**0.5) # m/s
		dv_order    = SPEEDOFLIGHT / (denom + 0.000001)
		dv_vals[i]  = dv_order
	
	dv_vals[np.where(dv_vals>1e4)[0]] = np.inf # where denom was 0 make inf

	dv_tot  = np.sqrt(dv_vals**2 + noise_floor**2)
	dv_spec  = 1. / (np.nansum(1./dv_vals**2.))**0.5
	dv_spec_floor  = 1. / (np.nansum(1./dv_tot**2.))**0.5

	return dv_tot,dv_spec,dv_vals


def doppler(v):
	"""
	Compute the (non-relativistic) Doppler shift factor for a given velocity.

	inputs
	------
	v - float or array, [m/s]
		velocity for the shift

	output
	------
	factor - float or array
		Doppler factor (1 + v/c); multiply a rest wavelength by this
		factor to get the Doppler-shifted wavelength
	"""
	return (1.0 + (v / SPEEDOFLIGHT))

