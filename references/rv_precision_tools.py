'''
Package for computing photon-limited velocity uncertainties in high resolution spectra.
Adapted from original IDL package.
Written by Sam Halverson, JPL
08-2022
'''
import glob
import os
import math
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel, convolve
import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.io import readsav
from spectres import spectres

# sort out paths
LOCAL_PATH = os.path.dirname(os.path.realpath(__file__))
CONFIG_PATH = os.path.join(LOCAL_PATH, 'config_files')

# input config file
CONFIG_DEFAULT = os.path.join(CONFIG_PATH,'rv_calc_config_kpf.yaml')
MAG_ZERO_POINT_FILE = os.path.join(CONFIG_PATH,'vega_magnitude_zeropoints.yaml')
SPEEDOFLIGHT = 2.998e8 # m/s
HCONST = 6.6261e-27 # cm^2 g s^-1

#----------------------------------------
def _findel(num, arr):

    '''
    Finds index of nearest element in array to a given number

    Parameters
    ----------
    num : int/float
        number to search for nearest element in array

    arr : :obj:`ndarray` of :obj:`float`
        array to search for 'num'

    Returns
    -------
    idx : int
        index of array element with value closes to 'num'

    S Halverson - JPL - 29-Sep-2019
    '''
    arr = np.array(arr)
    idx = (np.abs(arr - num)).argmin()
    return idx
#----------------------------------------

# Loads PHOENIX model stellar spectrum from specified path at provded Teff
#----------------------------------------
def _spectrum_load(teff, spec_path, wvl_file, wvl=None):

    """
    Loads spectrum from library of PHOENIX models for specified stellar Teff
    Requires library of PHOENIX models to be in fits format, with 'TEFF' in header

    Parameters
    ----------
    teff : int/float
        Stellar effective temperature [K]

    spec_path : string
        Directory path for library of model spectra

    wvl_file : string
        File containing master wavelength array fits file for all spectra
        in 'spec_path'

    wvl : array
        Array of wavelengths to interpolate spectrum onto

    Returns
    -------
    output_dict : dictionary
        Output spectrum, referenced by {'flux','wvl','hdr','teff_fits'}
        'flux': output flux spectrum, in units of [Ergs/sec/cm^2/A]
        'wvl': output wavelength array of 'flux' [Ang]
        'teff_file': actual effective temperature of model (nearest to input Teff)

    S Halverson - JPL - 29-Sep-2019
    """

    # read ascii file containing spectra in given range
    spec_fits = glob.glob(os.path.join(spec_path,"*.fits"))

    # path to fits file containing wavelength solution
    wvl_model = fits.getdata(wvl_file) #Ang

    # get effective temperature for each file in directory
    teff_fits = []
    for num in spec_fits:
        #parse file name and grab effective temperature
        teff_fits_num = float(((num.split("lte"))[1]).split('-')[0])
        teff_fits.append(teff_fits_num)

    # find temperature of stored spectra closest to specified teff
    index_teff = _findel(teff, teff_fits)
    # index_teff = np.where(teff_fits==teff)

    # read relevant fits file
    flux_model = fits.getdata(spec_fits[index_teff])
    hdr = fits.getheader(spec_fits[index_teff])
    print(spec_fits[index_teff])

    # convert array units
    flux_model *= 1e-8	#convert from Ergs/sec/cm^2/cm to Ergs/sec/cm^2/A

    # restrict wavelength range to optical (cuts down on array sizes) (TBD)
    flux_model = flux_model[(wvl_model > 3000.) & (wvl_model < 10000.)] #Ergs/sec/cm^2/A
    wvl_model = wvl_model[(wvl_model > 3000.) & (wvl_model < 10000.)]   #Ang
    wvl_master = wvl_model

    # if wvl array is provided, interpolate spectrum onto specified array -- else use default
    if np.any(wvl):
        # interpolate elememnt transmission curve onto reference wavelength array
        flux_model = np.interp(wvl, wvl_model, flux_model)
        wvl_master = wvl

    output_dict = {'flux':flux_model, 'wvl':wvl_master,
                    'hdr':hdr, 'teff_file':teff_fits[index_teff]}
    return output_dict

# Scales provided spectrum to specified magnitude in photometric band
#-------------------------------------------------
def _spectrum_scale(wvl, spec, mag, filter_band, mag_zero_point_file=MAG_ZERO_POINT_FILE):

    """
    Performs scaling of input spectrum to specified magnitude,
    returns scaled stellar spectrum.

    Requires relevant photometric filter response curve to be saved as text file
    under 'astro_filters_Generic_Jonson.N.txt', where 'N' is the filter band.

    Parameters
    ----------
    wvl : :obj:`ndarray` of :obj:`float`
        Wavelength array of input spectrum [Ang]

    spec : :obj:`ndarray` of :obj:`float`
        Input spectrum [Ergs/sec/cm^2/Ang]

    mag : float
        Target magnitude to scale input spectrum to

    filter_band : string
        filter band of 'mag' - must have corresponding transmission file

    mag_zero_point_file : string
        YAML file conaining magnitude zero point offsets for Vega and
        pointers to generic filter response curve text files

    Returns
    -------
    flxu_scaled : :obj:`ndarray` of :obj:`float`
        Scaled spectrum in flux units [Ergs/sec/cm^2/Ang]

    S Halverson - JPL - 29-Sep-2019
    """
    # capitalize for convinience
    filter_band = filter_band.capitalize()

    # Vega zero point magnitude numbers for specified band
    with open(mag_zero_point_file, 'r') as stream:
        data_filters = yaml.safe_load(stream)

    # reference data
    # lambda_filter = data_filters[filter_band]['lambda0']  	#Ang, central wavelength of filter
    flux_filter_0 = data_filters[filter_band]['flux0'] 	#erg/sec/cm^2/A, Vega zeropoint
    lambda_wid_filter = data_filters[filter_band]['filter_wid']   	#A, effective filter width

    # load in filter data from relevant text tile
    filter_file = os.path.join(LOCAL_PATH, data_filters[filter_band]['filter_file'])
    filter_data = np.loadtxt(filter_file)
    wvl_filter = filter_data[:, 0]    #Ang
    trans_filter = filter_data[:, 1]    #filter transmission

    # interpolate filter transmission curve onto reference wavelength array
    filter_trans = np.interp(wvl, wvl_filter, trans_filter)
    filter_trans[(wvl > np.max(wvl_filter))] = 0.
    filter_trans[(wvl < np.min(wvl_filter))] = 0.

    # multiply filter profile by spectrum
    flux = spec * filter_trans  	#Ergs/sec/cm^2/Ang

    # integrate combined spectrum
    filter_flux_filter = np.trapz(flux, wvl)  #Ergs/sec/cm^2

    # get effective magnitude of model spectrum
    offset_filter = 2.5 * math.log10(flux_filter_0 * lambda_wid_filter)	#mag, Vega zero point offset
    mag_model = (-2.5) * math.log10(filter_flux_filter) + offset_filter	#mag

    # compare vmag from magnitude estimate routine to library version
    mag_diff = mag_model - mag	#mag
    multi_fac = 10. ** (0.4 * mag_diff)	#scaling factor

    # final output spectrum
    flux_scaled = multi_fac * spec #Ergs/sec/cm^2/A

    # independantly check if the output magnitude matches the desired input
    # mag_model_out = (-2.5) * math.log10(scipy.trapz(flux_scaled * filter_trans, wvl)) +
    #                  offset_filter
    return flux_scaled
#-------------------------------------------------

# extinction calculation
#-------------------------------------------------
def _atm_trans(zenith, wave, observatory='keck'):
    '''
    Computes atmospheric extinction profile as a function of wavelength
    in linear transmission.

    '''
    if observatory == 'keck':
        file = 'extinction_curves/MK_EXTINCTION.dat'
    elif observatory == 'kpno':
        file = 'extinction_curves/KPNO_EXTINCTION.dat'
    # load in extinction curve
    data = readsav(file)
    wvl_extinction = data['wvl_extinction'] * 10. # Ang
    extinction = data['extinction'] # mag

    # interpolate onto provided wavelength grid
    ext_lambda = np.interp(wave, wvl_extinction, extinction)

    # calculate extinction curve for specified zenith angle
    zrad = np.radians(zenith)

    # Young & Irvine (1967)
    ext = (1. / np.cos(zrad)) * (1. - 0.0012 * ((1. / np.cos(zrad))**2. - 1.))

    # final transmission calculation
    trans = np.exp(-1. * ext * ext_lambda)
    return trans
#-------------------------------------------------

# # calculate telluric line avoidance mask
# #-------------------------------------------------
# def _make_telluric_mask(template_fits, vel_wid, depth_thresh, stellar_velocity):
#     '''
#     Computes telluric mask for provided velocity width
#     and depth threshold

#     Parameters
#     ----------
#     template_fits : string
#         FITS file with template telluric spectrum

#     vel_wid : float
#         Velodity width of regions to avoid [km/s]

#     depth_thresh : float
#         Line depth threshold (normalized)

#     Returns
#     -------
#     tell_mask : array
#         Binary mask of 


#     '''
#     tell_mask = None

#     return tell_mask
# #-------------------------------------------------

# calculate wavelength span of echelle order
#-------------------------------------------------
def _order_calc(ordernum, blaze=75.96, gam=0.5, sigma=31.6):
    '''
    Computes wavelength span for user-specificed echelle order,
    given grating parameters

    Parameters
    ----------
    ordernum : int
        Echelle order number

    blaze : float
        Echelle blaze angle [degrees]

    gam : float
        Grating gamma angle [degrees]

    sigma : float
        Grating groove density [lines per mm]

    Returns
    -------
    minwvl : float
        Minimum wavelength of FSR for desired order

    maxwvl : float
        Maximum wavelength of FSR for desired order
    '''

    # line spacing
    d_spac = 1./sigma * 1e7 # Ang

    # angles
    alpha = blaze + gam # entrance angle
    beta = blaze - gam # exit angle

    # blaze center wavelength
    wvlcen = d_spac * (np.sin(np.radians(alpha)) + np.sin(np.radians(beta))) / ordernum

    # free spectral range
    fsr = wvlcen / ordernum # Ang

    # wavelength bounds
    maxwvl = wvlcen + fsr / 2. # Ang
    minwvl = wvlcen - fsr / 2. # Ang

    return minwvl, maxwvl, fsr
#-------------------------------------------------

# calculate rotational broadening profile
#-------------------------------------------------
def _lsf_rotate(deltav,vsini,epsilon=0.6):
    '''
    Computes vsini rotation kernel.
    Based on the IDL routine LSF_ROTATE.PRO

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
        Computed kernel profile

    velgrid : float
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
#-------------------------------------------------

# calculate photon-limited RV uncertainty for given observation
def photon_dv_calc(teff=5700., vmag=9., exptime=300.,
                    zenith=30., vsini=2., plot=False,
                    name=None,
                    config_file=CONFIG_DEFAULT):
    '''
    Parameters
    ----------
    teff : float
        Target stellar effective temperature

    vmag : float
        Target v magnitude

    exptime : float
        Exposure time [seconds]

    zenith : float
        Observation zenith angle [degrees]

    vsini : float
        Target vsini value [km/s]

    config_file : string
        Input config file (yaml) that contains instrument information
        and pointers to different required directories.

    Returns
    -------
    dv_photon : float
        Total photon-limited Doppler uncertainty for specified exposure [m/s]

    dv_ord : array
        Order-by-order Doppler uncertainties [m/s]

    wvl_mean_ord : array
        Mean order wavelengths for each order in dv_ord

    '''
    # load in input configuration file
    try:
        with open(config_file) as file:
            input_params = yaml.safe_load(file)
    except FileNotFoundError:
        raise IOError('input config file not found')

    # load in telescope information
    primd = input_params['primary_diameter'] # m
    secd = input_params['secondary_diameter'] # m
    collecting_area = 3.141592654 * ((primd/2.) ** 2. - (secd/2.) ** 2.) * 1e4 # cm^2

    # echelle order range to simulate
    order_arr = np.flip(np.arange(input_params['order_range'][0],input_params['order_range'][1],1))

    # read noise
    sigma_read_noise = input_params['rnpix'] * (input_params['sampling_xdisp']) ** 0.5

    # load input spectrum
    model_wvl_file = input_params['stellar_model_wvl_file']
    spec_dict = _spectrum_load(teff, input_params['stellar_model_dir'], model_wvl_file) # flux in Ergs/sec/cm^2/A
    
    # sort arrays by wavelength
    wvl_native = spec_dict['wvl'] # Ang
    flux_native = spec_dict['flux'] # Ergs/sec/cm^2/A
    inds_sort = np.argsort(wvl_native)
    wvl_native = np.asarray(wvl_native[inds_sort]) # Ang
    flux_native = np.asarray(flux_native[inds_sort]) # Ergs/sec/cm^2/A
    flux_native = flux_native[(wvl_native > 2000.) & (wvl_native < 20000.)] # shrink arrays
    wvl_native = wvl_native[(wvl_native > 2000.) & (wvl_native < 20000.)] # shrink arrays

    # scale spectrum to specified magnitude
    flux_scaled = _spectrum_scale(wvl_native, flux_native, vmag, 'v', MAG_ZERO_POINT_FILE)  # Ergs/sec/cm^2/A

    # get min an max wavelength bounds based on orders
    minord = np.nanmin(input_params['order_range'])
    maxord = np.nanmax(input_params['order_range'])
    min_wvl, _, _ = _order_calc(maxord, blaze=input_params['blaze_grating'], gam=input_params['gamma_angle']) # nm
    _, max_wvl, _ = _order_calc(minord, blaze=input_params['blaze_grating'], gam=input_params['gamma_angle']) # nm
    # print('Min wavelength [Ang]: ', min_wvl)
    # print('Max wavelength [Ang]: ', max_wvl)

    # interpolate wavelength and flux arrays onto uniform grid
    npts = 5e5 * (max_wvl - min_wvl) / (1500.) # number of supersampled pixels (empirically informed)
    wvl = np.linspace(min_wvl, max_wvl, num=int(npts)) # uniformly spaced wavelength array
    flux_interp_mod = interpolate.interp1d(wvl_native, flux_scaled, kind='cubic')

    # evaluate interpolation and scale with collecting area
    flux = flux_interp_mod(wvl) * collecting_area # starting point******, Ergs/sec/A

    # throughput corrections
    # ---------------------------------------
    # atmospheric transmission profile
    atm_transmission = _atm_trans(zenith, wvl, observatory=input_params['site'])
    flux_atm = flux * atm_transmission # Ergs/sec/A

    # instrumental throughput
    wvl_eff, eff = np.loadtxt(input_params['instrument_eff_file'], unpack=True, delimiter=',')
    eff_interp = interpolate.interp1d(wvl_eff, eff, kind='cubic',fill_value='extrapolate')
    flux_instrument = flux_atm * eff_interp(wvl)
    # ---------------------------------------
    
    # plot flux rate
    if plot:
        col_table = plt.get_cmap('Spectral_r') # color map for later
        plt.plot(wvl, flux,label='Top of atmosphere')
        plt.plot(wvl, flux_atm,label='Top of telescope')
        plt.plot(wvl, flux_instrument,label='At detector')
        plt.title(str(int(teff)) + ' K, v = ' + str(np.round(vmag, 1)))
        plt.xlabel('Wavelength [$\mathcal{\AA}$]')
        plt.ylabel('Flux [Erg sec$^\mathcal{-1}$ $\mathcal{\AA}^\mathcal{-1}$]')
        plt.legend(loc='best', handletextpad=0.3)
        plt.show()

    # convert units to photons / sec / Ang
    wvl_cm = wvl * 1e-8 # cm
    energy_arr = (HCONST * SPEEDOFLIGHT * 1e2 / wvl_cm) # erg / photon
    flux_instrument_counts = flux_instrument / energy_arr # photons / second / Ang

    # for each order number, perform relevant convolutions, resampling
    # calculate order-by-order RV uncertainty
    wvl_sampled = []
    flux_instrument_sampled = []
    snr_arr = []
    dv_ord = []
    wvl_mean_ord = []
    for ordi in order_arr:
        # compute order bounds
        min_wvl, max_wvl, _ = _order_calc(ordi, blaze=input_params['blaze_grating'], gam=input_params['gamma_angle']) # nm
        
        # get single order spectrum
        wvl_ord = wvl[(wvl > min_wvl) & (wvl < max_wvl)]
        flux_ord = flux_instrument_counts[(wvl > min_wvl) & (wvl < max_wvl)]
        
        # compute average dispersion, velocity scale
        dwvl_mean = np.abs(np.nanmean(np.diff(wvl_ord)))
        dvel_mean = (dwvl_mean / np.nanmean(wvl_ord)) * SPEEDOFLIGHT / 1e3 # average sampling in km/s
        if vsini > 0:
            vsini_kernel, _ = _lsf_rotate(dvel_mean, vsini)
            flux_vsini = convolve(flux_ord,vsini_kernel,normalize_kernel=True)  # photons / second / Ang
        else:
            flux_vsini = flux_ord

        # spectrometer resolution convolution
        # FWHM of resolution element, to be used in convolution kernel
        fwhm = (np.nanmean(wvl_ord) / input_params['resolution'])	# Ang
        fwhm_pix = fwhm / dwvl_mean	# pixels
        sigma_pix = fwhm_pix / 2.355 # pixels

        # generate kernel
        res_kernel = Gaussian1DKernel(sigma_pix)

        # convolve vsini kernel with spectrum
        spec_con_ord = convolve(flux_vsini,res_kernel,normalize_kernel=True)  # photons / second / Ang

        # resampling to native pixel pitch
        wvl_pitch_interp_ord = (np.nanmean(wvl_ord) / input_params['resolution']) / fwhm_pix		# Ang/pixel dispersion of interpolated spectrum (converting FROM this)
        wvl_pitch_kpf_ord = (np.nanmean(wvl_ord) / input_params['resolution']) / input_params['sampling']	#Ang/pixel dispersion at native sampling (converting TO this)
        # checked -----------------------------

        #total number of pixels spanning the full wvl range sampled at the KPF pixels
        n_pix_total_kpf_ord = len(spec_con_ord) * wvl_pitch_interp_ord / wvl_pitch_kpf_ord	#total number of pixels to bin down to
       
        # bin flux and wavelength to native pixel sampling
        wvl_sampled_ord = np.linspace(min_wvl,max_wvl,num=int(n_pix_total_kpf_ord)) # Ang
        flux_instrument_sampled_ord = spectres(wvl_sampled_ord, wvl_ord, spec_con_ord, fill=0.,verbose=False)	#photons/sec/A
 
        # convert to counts per second per pixel using average dispersion, photons/sec/Ang (flux) * Ang/pixel (dispersion at KPF sampling)
        flux_sampled_counts_ord_pixel = flux_instrument_sampled_ord * wvl_pitch_kpf_ord	# photons/sec/pix

        # convert to recorded counts per pixel
        flux_sampled_counts_ord_pixel *= exptime	#photons/pixel

        #store order spectra in master arrays
        wvl_sampled.append(wvl_sampled_ord)
        flux_instrument_sampled.append(flux_sampled_counts_ord_pixel)

        # calculate photon-limited velocity uncertainty
        # ------------------------------------------------
        wvl_m_ord = wvl_sampled_ord * 1e-10	#convert wavelength values to meters for weighting calc
        
        # interpolate spectrum for derivative
        flux_interp = interpolate.InterpolatedUnivariateSpline(wvl_m_ord,
                                                        flux_sampled_counts_ord_pixel, k=1)

        # calculate derivative (in final units)
        dflux = flux_interp.derivative()
        spec_deriv = dflux(wvl_m_ord)

        # calculate noise properties of sampled spectrum (photon only, assume root N)
        sigma_ord = flux_sampled_counts_ord_pixel ** 0.5

        # get approximate peak SNR (per pixel) value for the order
        snr_arr.append(np.nanmedian(sigma_ord))

        # calculate Doppler weights for each pixel
        w_ord = (wvl_m_ord ** 2.) * (spec_deriv ** 2.) / (sigma_ord ** 2. + sigma_read_noise ** 2.)

        # calculate sigma_rv -- clip edge pixels in case they were affected by convolutions/resampling
        dv_order = SPEEDOFLIGHT / (np.nansum(w_ord[1:-1])**0.5) # m/s
        dv_ord.append(dv_order)
        wvl_mean_ord.append(np.nanmean(wvl_sampled_ord))

        wvl_norm = (np.nanmean(wvl_sampled_ord) - 4200.) / (7200. - 4200.)
        if plot: 
            plt.plot(wvl_sampled_ord[1:-1], flux_sampled_counts_ord_pixel[1:-1],
                    color=col_table(wvl_norm))

    # convert to arrays for convinience
    dv_ord = np.asarray(dv_ord,dtype=object)
    wvl_sampled = np.asarray(wvl_sampled,dtype=object)
    flux_instrument_sampled = np.asarray(flux_instrument_sampled,dtype=object)

    # determine velocity error for total spectrum by combining all pixels
    dv_photon = 1. / (np.nansum(1./dv_ord**2.))**0.5	# calculate by combining individual orders
    
    if plot:
        if name is not None:
            title_final = name + ', ' + str(int(teff)) + ' K, v = ' + str(np.round(vmag, 1)) + \
            ', vsini = ' + str(np.round(vsini,1)) + ' km s$^\mathcal{-1}$' + ' , t$_\mathcal{exp.}$ = ' + str(int(exptime)) + ' seconds'
        else:
            title_final = str(int(teff)) + ' K, v = ' + str(np.round(vmag, 1)) + \
            ', vsini = ' + str(np.round(vsini,1)) + ' km s$^\mathcal{-1}$' + ' , t$_\mathcal{exp.}$ = ' + str(int(exptime)) + ' seconds'

        plt.title('Final spectrum - ' + title_final)
        plt.xlabel('Wavelength [$\mathcal{\AA}$]')
        plt.ylabel('Recorded counts')
        plt.show() 

        for indord, wav in enumerate(wvl_mean_ord):
            wvl_norm = (np.nanmean(wav) - 4200.) / (7200. - 4200.)
            plt.plot(wav, dv_ord[indord],'s',
                    markersize=12,markeredgecolor='k',markeredgewidth=2,
                    markerfacecolor=col_table(wvl_norm))
        
        plt.title('Expected order-by-order precision, ' + title_final)
        plt.xlabel('Wavelength [$\mathcal{\AA}$]')
        plt.ylabel('$\mathcal{\sigma_{RV}}$ [m s$^\mathcal{-1}$]')
        plt.show() 

    return dv_photon, dv_ord, wvl_mean_ord, wvl_sampled, flux_instrument_sampled