##############################################################
# cross correlation function tools (RV, CCF SNR)
# taken from Sam Halverson and Arpita Roys code
###############################################################

import numpy as np
from scipy.integrate import trapz
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
		Input wavelength array
	weights : array
		Line depths of specified lines
	line_wvls : array
		Line centers of features to be added
	fwhms : array
		FWHMs of lines specified

	Outputs:
	-------
	spec_out: array
		 Final output absorption spectrum
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

