##############################################################
# General functions for calc_snr_max
###############################################################

import numpy as np
from scipy import signal, interpolate
import pandas as pd

import matplotlib.pylab as plt

from specsim.functions import integrate, degrade_spec, calc_strehl_marechal
from astropy.io import fits
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

all = {}

def pick_coupling_rounded(transmission_path,w,ho_wfe, tt_dynamic, lo_wfe=50, tt_static=0, defocus=0, atm=1,adc=1,pl_on=1,piaa_boost=1.3):
    """
    Look up fiber injection/coupling efficiency by rounding the requested
    wavefront-error and tip-tilt parameters to the nearest values available
    in the pre-computed coupling grid (a set of CSV files, one per parameter
    combination), then loading that single file and interpolating it onto
    the requested wavelength grid. High-order WFE is not rounded; instead
    it is converted analytically to a Strehl ratio (Marechal approximation)
    and multiplied onto the tabulated (rounded-grid) coupling.

    inputs
    ------
    transmission_path : string
        path to the directory containing the 'coupling/' subfolder with the
        couplingEff_atm%s_adc%s_PL%s_defoc%snmRMS_LO%snmRMS_ttStatic%smas_ttDynamic%smasRMS.csv
        grid files
    w : array
        wavelength array, in nm if max(w) >= 10, otherwise assumed to be in
        microns and converted to nm internally
    ho_wfe : float or array [nm]
        high-order wavefront error, used to compute the Strehl ratio applied
        multiplicatively to the tabulated coupling (not used to select the
        grid file)
    tt_dynamic : float [mas]
        dynamic tip-tilt RMS; rounded to the nearest 0.5 mas (grid sampling)
        to select the coupling file, and clipped to the 19.5 mas file if the
        rounded value is >= 20
    lo_wfe : float, optional [nm]
        low-order wavefront error RMS; rounded to the nearest 25 nm to select
        the coupling file. Default 50
    tt_static : float, optional [mas]
        static tip-tilt; rounded to the nearest 0.5 mas to select the
        coupling file. Default 0
    defocus : float, optional [nm]
        defocus term RMS; rounded to the nearest 25 nm to select the
        coupling file. Default 0
    atm : int, optional
        0 or 1, whether the atmosphere was included in the simulation grid
        used to select the coupling file. Default 1
    adc : int, optional
        0 or 1, whether the ADC (atmospheric dispersion corrector) was
        included in the simulation grid used to select the coupling file.
        Default 1
    pl_on : int, optional
        0 or 1, whether the photonic lantern is on. If on, the coupling
        efficiencies of the three PL output modes are summed; if off, only
        mode 1 (single-mode fiber) is used. Default 1
    piaa_boost : float, optional
        multiplicative coupling boost factor from the PIAA lens, applied on
        top of the tabulated coupling and Strehl. Default 1.3

    outputs
    -------
    coupling : array
        coupling efficiency vs wavelength (tabulated grid value, interpolated
        onto w, times ho_strehl times piaa_boost)
    ho_strehl : array
        Strehl ratio computed from ho_wfe via the Marechal approximation,
        same wavelength grid as w
    """
    if np.max(w) < 10: 
        wave=w.copy() * 1000
    else:
        wave=w.copy()
    
    filename_skeleton = 'coupling/couplingEff_atm%s_adc%s_PL%s_defoc%snmRMS_LO%snmRMS_ttStatic%smas_ttDynamic%smasRMS.csv'
    tt_dynamic_rounded = np.round(2 * tt_dynamic) / 2 # round to neared 0.5 because grid is sampled to 0.5mas
    lo_wfe_rounded = int(100*np.round(4*(lo_wfe/100))/4) # round to nearest 25
    tt_static_rounded = np.round(tt_static*2)/2
    if int(tt_static_rounded)==tt_static_rounded: tt_static_rounded  = int(tt_static_rounded)
    if int(tt_dynamic_rounded)==tt_dynamic_rounded: tt_dynamic_rounded  = int(tt_dynamic_rounded)
    defocus_rounded =  int(100*np.round(4*(defocus/100))/4)

    if tt_dynamic_rounded < 20:
        f = pd.read_csv(transmission_path+filename_skeleton%(int(atm),int(adc),int(pl_on),defocus_rounded,lo_wfe_rounded,tt_static_rounded,tt_dynamic_rounded)) # load file
    else:
        f = pd.read_csv(transmission_path+filename_skeleton%(int(atm),int(adc),int(pl_on),defocus_rounded,lo_wfe_rounded,tt_static_rounded,19.5)) # load file

    if pl_on:
        coupling_data_raw = f['coupling_eff_mode1'] + f['coupling_eff_mode2'] + f['coupling_eff_mode3']
    else:
        coupling_data_raw = f['coupling_eff_mode1']

    # interpolate onto self.x
    finterp = interpolate.interp1d(1000*f['wavelength_um'].values,coupling_data_raw,bounds_error=False,fill_value=0)
    coupling_data = finterp(wave)

    #piaa_boost = 1.3 # based on Gary's sims, but needs updating because will be less for when Photonic lantern is being used
    ho_strehl  = calc_strehl_marechal(ho_wfe,wave)
    coupling   = coupling_data  * ho_strehl * piaa_boost

    return coupling, ho_strehl

def pick_coupling_interpolate(w,dynwfe,ttStatic,ttDynamic,LO=50,PLon=0,piaa_boost=1.3,points=None,values=None):
    """
    Compute fiber injection/coupling efficiency by N-D interpolation (via
    scipy.interpolate.interpn) of the pre-computed coupling grid, rather
    than rounding to the nearest tabulated point as pick_coupling_rounded
    does. The grid ('points') and its tabulated values ('values') must be
    supplied by the caller, typically from grid_interp_coupling(). High-order
    WFE is applied analytically as a Strehl factor on top of the
    interpolated coupling, exactly as in pick_coupling_rounded.

    Note: docstring reflects current behavior; a TODO in the original code
    notes that full interpolation (vs. rounding) was still being implemented.

    inputs
    ------
    w : array
        wavelength array. If min(w) > 10 it is assumed to be in nm and is
        divided by 1000 to get microns (used to build the 'point' passed to
        interpn against the wavelength axis of 'points'); the working array
        is converted back to nm afterward (if still < 10) before computing
        the Strehl ratio, to match the nm units expected for dynwfe
    dynwfe : float or array [nm]
        high-order/dynamic wavefront error, used to compute ho_strehl via
        exp(-(2*pi*dynwfe/wave)^2) and applied multiplicatively to the
        interpolated coupling
    ttStatic : float [mas]
        static tip-tilt; must be in range 0-10 or a ValueError is raised
    ttDynamic : float [mas]
        dynamic tip-tilt; must be in range 0-20 or a ValueError is raised
    LO : float, optional [nm]
        low-order wavefront error RMS; must be in range 0-100 or a
        ValueError is raised. Default 50
    PLon : int, optional
        0 or 1, whether the photonic lantern is on; coerced to int and must
        be <= 1 or a ValueError is raised. If on, the three PL output-mode
        coupling efficiencies are interpolated separately and recombined
        (with an extra 0.95 recombination-loss factor applied below 1.4 um);
        if off, only the single-mode-fiber coupling (mode 1) is used.
        Default 0
    piaa_boost : float, optional
        multiplicative coupling boost factor from the PIAA lens. Default 1.3
    points : tuple of arrays, optional
        grid axis values (LO, ttStatic, ttDynamic, wavelength) defining the
        coupling table, as returned by grid_interp_coupling()
    values : tuple of arrays, optional
        tabulated coupling efficiency array(s) on the 'points' grid, as
        returned by grid_interp_coupling(); one array if PLon is off, three
        (mode1, mode2, mode3) if PLon is on

    outputs
    -------
    coupling : array
        interpolated coupling efficiency vs wavelength, times ho_strehl
        times piaa_boost
    ho_strehl : array
        Strehl ratio computed from dynwfe, same wavelength grid as w
    """
    PLon = int(PLon)

    waves = w.copy()
    if np.min(waves) > 10:
        waves/=1000 # convert nm to um

    # check range of each variable
    if ttStatic > 10 or ttStatic < 0:
        raise ValueError('ttStatic is out of range, 0-10')
    if ttDynamic > 20 or ttDynamic < 0:
        raise ValueError('ttDynamic is out of range, 0-10')
    if LO > 100 or LO < 0:
        raise ValueError('LO is out of range,0-100')
    if PLon >1:
        raise ValueError('PL is out of range')

    if PLon:
        values_1,values_2,values_3 = values
        point = (LO,ttStatic,ttDynamic,waves)
        mode1 = interpolate.interpn(points, values_1, point,bounds_error=False,fill_value=0) # see example https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html#scipy.interpolate.interpn
        mode2 = interpolate.interpn(points, values_2, point,bounds_error=False,fill_value=0) 
        mode3 = interpolate.interpn(points, values_3, point,bounds_error=False,fill_value=0) 

        #PLwav,PLdat = load_photonic_lantern() #transfer matrices input mode--> each SMF
        #mat = PLdat[10] # use middle one for now
        #test1 = mode1 * mat[0,0]  + mode2*mat[1,0] + mode3*mat[2,0]
        #test2 = mode1 * mat[0,1]  + mode2*mat[1,1] + mode3*mat[2,1]
        #test3 = mode1 * mat[2,2]  + mode2*mat[1,2] + mode3*mat[2,2]
        # apply only to YJ or make matrix diagonal for HK..map onto same wavelength grid somehow
        # get coupling
        losses = np.ones_like(mode1) # due to PL imperfection
        losses[np.where(waves< 1.400)[0]] = 0.95 # only apply to y band
        raw_coupling = losses*(mode1+mode2+mode3) # do dumb things for now #0.95 is a recombination loss term 
    else:
        values_1 = values[0]
        #points, values_1 = grid_interp_coupling(PLon)
        point = (LO,ttStatic,ttDynamic,waves)
        raw_coupling = interpolate.interpn(points, values_1, point,bounds_error=False,fill_value=0)

    if np.max(waves) < 10:
        waves*=1000 # nm to match dynwfe

    ho_strehl =  np.exp(-(2*np.pi*dynwfe/waves)**2) # computed per wavelength as grid
    coupling  = raw_coupling * piaa_boost * ho_strehl
    
    return coupling, ho_strehl

def grid_interp_coupling(PLon=1,path='./data/instrument/hispec/throughput/coupling/',atm=1,adc=1):
    """
    Build the N-D coupling-efficiency grid (axes: low-order WFE, static
    tip-tilt, dynamic tip-tilt, wavelength) used by pick_coupling_interpolate
    for true interpolation, as opposed to the nearest-grid-point rounding
    done in pick_coupling_rounded. Loops over every combination of LO
    (0-100 nm, step 25), ttStatic (0-10 mas, step 1), and ttDynamic
    (0-20 mas, step 0.5), reads the corresponding
    couplingEff_atm%s_adc%s_PL%s_defoc0nmRMS_LO%snmRMS_ttStatic%smas_ttDynamic%smasRMS.csv
    file (defocus fixed at 0), and stacks the per-mode coupling efficiency
    columns into 4-D arrays suitable for scipy.interpolate.interpn.

    inputs
    ------
    PLon : int, optional
        0 or 1, whether the photonic lantern is on. If on, the coupling
        efficiencies for all three PL output modes (mode1, mode2, mode3)
        are loaded into separate grids; if off, only mode1 (single-mode
        fiber) is loaded. Default 1
    path : string, optional
        directory containing the coupling grid CSV files
    atm : int, optional
        0 or 1, whether the atmosphere was included in the simulation grid
        being loaded (selects file name). Default 1
    adc : int, optional
        0 or 1, whether the ADC (atmospheric dispersion corrector) was
        included in the simulation grid being loaded (selects file name).
        Default 1

    outputs
    -------
    points : tuple of arrays
        grid axis values (LOs, ttStatics, ttDynamics, wavelength_um) defining
        the coordinates of the 'values' arrays, for use with interpn
    values_1 : array [len(LOs), len(ttStatics), len(ttDynamics), n_wave]
        tabulated coupling efficiency for PL output mode 1 (or the only
        mode, if PLon is 0)
    values_2 : array, only returned if PLon
        tabulated coupling efficiency for PL output mode 2
    values_3 : array, only returned if PLon
        tabulated coupling efficiency for PL output mode 3
    """
    LOs = np.arange(0,125,25)
    ttStatics = np.arange(11)
    ttDynamics = np.arange(0,20.5,0.5)
    
    filename_skeleton = 'couplingEff_atm%s_adc%s_PL%s_defoc0nmRMS_LO%snmRMS_ttStatic%smas_ttDynamic%smasRMS.csv'

    # to dfine values, must open up each file. not sure if can deal w/ wavelength
    values_1 = np.zeros((len(LOs),len(ttStatics),len(ttDynamics),100))
    values_2 = np.zeros((len(LOs),len(ttStatics),len(ttDynamics),100))
    values_3 = np.zeros((len(LOs),len(ttStatics),len(ttDynamics),100))  
    for i,LO in enumerate(LOs):
        for j,ttStatic in enumerate(ttStatics):
            for k,ttDynamic in enumerate(ttDynamics):
                if round(ttDynamic)==ttDynamic: ttDynamic=round(ttDynamic)
                f = pd.read_csv(path+filename_skeleton%(atm,adc,PLon,LO,ttStatic,ttDynamic))
                if PLon:
                    values_1[i,j,k,:]=f['coupling_eff_mode1'] #what to fill here?
                    values_2[i,j,k,:]=f['coupling_eff_mode2'] #what to fill here?
                    values_3[i,j,k,:]=f['coupling_eff_mode3'] #what to fill here?
                else:
                    values_1[i,j,k,:]=f['coupling_eff_mode1'] #what to fill here?

                #values_hk[i,j,k]=f['coupling_eff_mode1'][50] #what to fill here?
    
    points = (LOs, ttStatics, ttDynamics,f['wavelength_um'].values)

    if PLon:
        return points,values_1,values_2,values_3
    else:
        return points,values_1


def get_emissivity(wave,datapath = './data/throughput/hispec_subsystems_11032022/'):
    """
    Load and interpolate the per-surface emissivity curves for each optical
    element in the red and blue optical paths (excluding fiber coupling),
    onto the requested wavelength grid, along with the assumed physical
    temperature of each surface. Fiber contributions ('fib*') are doubled
    to account for the integrating-sphere measurement setup used to derive
    those emissivity files.

    inputs
    ------
    wave : array
        wavelength array to sample emissivity on; converted from nm to
        microns internally if min(wave) > 10
    datapath : string, optional
        path to the directory containing per-surface subfolders
        (tel/ao/feicom/feired/feiblue/fibred/fibblue/rspec), each with a
        '<surface>_emissivity.csv' file (columns: wavelength_um, emissivity)

    outputs
    -------
    em_red : list of arrays
        emissivity vs wave for each surface in the red path
        (['tel','ao','feicom','feired','fibred','rspec']), in that order
    em_blue : list of arrays
        emissivity vs wave for each surface in the blue path
        (['tel','ao','feicom','feiblue','fibblue','bspec']), in that order
    temps : list of floats [K]
        assumed physical temperature for each of the 6 surface slots,
        [276,276,276,276,276,77] (thermal background surfaces at ambient,
        detector/cold stage at 77 K)
    """
    x = wave.copy()
    if np.min(x) > 10:
        x/=1000 #convert nm to um

    red_include = ['tel', 'ao', 'feicom', 'feired','fibred','rspec']#,'coupling']
    blue_include = ['tel', 'ao', 'feicom', 'feiblue','fibblue','bspec']
    temps = [276,276,276,276,276,77]

    em_red, em_blue = [],[]
    for i in red_include:
        wtemp, stemp = np.loadtxt(datapath + i + '/%s_emissivity.csv'%i, delimiter=',',skiprows=1).T
        f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
        if i.startswith('fib'):
            em_red.append(2*f(x)) # count fib twice because of integrating sphere
        else:
            em_red.append(f(x))

    for i in blue_include:
        wtemp, stemp = np.loadtxt(datapath + i + '/%s_emissivity.csv'%i, delimiter=',',skiprows=1).T
        f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
        if i.startswith('fib'):
            em_blue.append(2*f(x)) # count fib twice bc of integrating sphere
        else:
            em_blue.append(f(x)) #

    return em_red,em_blue,temps

def get_emissivities(wave,surfaces=['tel'],datapath = './data/throughput/hispec_subsystems_11032022/'):
    """
    Derive per-surface emissivity as (1 - throughput) for an arbitrary list
    of named surfaces, by loading each surface's '<surface>_throughput.csv'
    file and interpolating it onto the requested wavelength grid. Unlike
    get_emissivity(), this does not use dedicated emissivity CSV files or
    apply the fiber integrating-sphere doubling factor, and the caller
    supplies the list of surfaces to include.

    inputs
    ------
    wave : array
        wavelength array to sample emissivity on; converted from nm to
        microns internally if min(wave) > 10
    surfaces : list of strings, optional
        names of the subfolders/surfaces to load, each expected to contain a
        '<surface>_throughput.csv' file (columns: wavelength_um, throughput).
        Default ['tel']
    datapath : string, optional
        path to the directory containing the per-surface subfolders

    outputs
    -------
    em : list of arrays
        1 - throughput vs wave, one array per entry in 'surfaces', in the
        same order
    """
    x = wave.copy()
    if np.min(x) > 10:
        x/=1000 #convert nm to um

    em= []
    for i in surfaces:
        wtemp, stemp = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
        f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
        em.append(1-f(x)) # 1 - interp throughput onto x

    return em

def get_base_throughput(wave,ploton=False,datapath = './data/throughput/hispec_subsystems_11032022/'):
    """
    Compute the total instrument throughput excluding fiber coupling, by
    multiplying together the per-surface throughput curves along the red
    path (['tel','ao','feicom','feired','fibred','rspec']) for wavelengths
    > 1.4 um and along the blue path
    (['tel','ao','feicom','feiblue','fibblue','bspec']) for wavelengths
    < 1.4 um, then concatenating the two bands into a single blue-to-red
    array on the input wavelength grid. Optionally plots and saves the
    per-band cumulative throughput curves.

    inputs
    ------
    wave - array
        wavelength array [nm] to sample throughput on (converted to microns
        internally if min(wave) > 10)
    ploton - Bool
        default is False, whether to plot throughput (blue and red curves
        vs wavelength) and save the figure to './base_throughput.png'
    datapath - string
        path to throughput files in special HISPEC/MODHIS structure, with
        one subfolder per surface each containing a '<surface>_throughput.csv'
        file (columns: wavelength_um, throughput)

    outputs:
    ---------
    s - array
        total base throughput, sampled on wave grid, blue band
        (wave < 1.4 um) followed by red band (wave > 1.4 um)
    data - dict
        nested dict {'red': {surface: throughput_array, ...},
        'blue': {surface: throughput_array, ...}} holding the individual
        per-surface throughput curves (each interpolated onto wave) used to
        build snew
    """
    # wavelength array to um
    x = wave.copy()
    if np.min(x) > 10:
        x/=1000 #convert nm to um

    data={}
    data['red']  = {}
    data['blue'] = {}
    #plt.figure()
    for spec in ['red','blue']:
        if spec=='red':
            include = ['tel', 'ao', 'feicom', 'feired','fibred','rspec']#,'coupling']
        if spec=='blue':
            include = ['tel', 'ao', 'feicom', 'feiblue','fibblue','bspec']#,'coupling']

        for i in include:
            if i==include[0]:
                wtemp, stemp = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
                s = f(x)
                #plt.plot(w,s,label=i)
            else:
                wtemp, stemp = np.loadtxt(datapath + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                # interpolate onto s
                f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
                s*=f(x)
                #plt.plot(w,s,label=i)
            # store throughput in dictionary
            data[spec][i] = f(x)

        if spec=='red':
            isub = np.where(x > 1.4) 
            wred = x[isub]
            specred = s[isub]
        if spec=='blue':
            isub = np.where(x<1.4)
            specblue = s[isub]
            wblue = x[isub]
    
    w = np.concatenate([wblue,wred])
    s = np.concatenate([specblue,specred])

    if ploton:
        plt.plot(wblue,specblue,label='blue')
        plt.plot(wred,specred,label='red')
        plt.grid(True)
        plt.xlabel('Wavelength (um)')
        plt.ylabel('Transmission')
        plt.title("Spectrograph Throughput Except Coupling")
        plt.savefig('base_throughput.png')

    return s, data

def load_photonic_lantern():
    """
    Load the photonic lantern's mode-transfer (unitary) matrices, which map
    input modes to output single-mode fibers, from a fixed .npy file, along
    with the wavelength grid they were computed on.

    inputs
    ------
    None

    outputs
    -------
    wavearr : array [nm]
        20-point wavelength grid spanning 970-1350 nm on which the unitary
        matrices are defined
    data : array
        unitary transfer matrices loaded from
        './data/throughput/photonic_lantern/unitary_matrices.npy', one
        matrix per wavelength in wavearr (shape depends on the saved file,
        e.g. [n_wave, n_mode, n_mode])
    """
    wavearr = np.linspace(970,1350,20)
    data = np.load('./data/throughput/photonic_lantern/unitary_matrices.npy')
    
    return wavearr,data
