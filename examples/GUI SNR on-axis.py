from PyQt5.QtWidgets import QApplication,QMainWindow,QPushButton,QPlainTextEdit, QTextBrowser,QLabel
from scipy.integrate import trapz
import numpy as np
import glob
from scipy import interpolate
from astropy.io import fits
import sys,glob,os
from scipy.ndimage.interpolation import shift
from astropy.convolution import Gaussian1DKernel, convolve
import astropy.units as u
import astropy.constants as consts
import pandas as pd
import matplotlib.pyplot as plt
from astropy.modeling.models import BlackBody
from scipy import interpolate
from scipy import signal







class obs_snr_on():
    def __init__(self):
        self.window = QMainWindow()
        self.window.resize(900, 400)
        self.window.move(300, 310)
        self.window.setWindowTitle('object spectrum')

        self.textEdit_1_teff = QPlainTextEdit(self.window)
        self.textEdit_1_teff.setPlaceholderText("Kelvin")
        self.textEdit_1_teff.move(140,25)
        self.textEdit_1_teff.resize(50,20)

        self.label_1_teff = QLabel(self.window)
        self.label_1_teff.setText('Temperature of Star:')
        self.label_1_teff.move(10,25)
        self.label_1_teff.resize(140,20)


        self.textEdit_1_mag = QPlainTextEdit(self.window)
        self.textEdit_1_mag.setPlaceholderText("Vega")
        self.textEdit_1_mag.move(140,50)
        self.textEdit_1_mag.resize(50,20)

        self.label_1_mag = QLabel(self.window)
        self.label_1_mag.setText('Magnitude of Star:')
        self.label_1_mag.move(10,50)
        self.label_1_mag.resize(140,20)

        self.textEdit_1_family = QPlainTextEdit(self.window)
        self.textEdit_1_family.setPlaceholderText("filtername")
        self.textEdit_1_family.move(140,75)
        self.textEdit_1_family.resize(50,20)

        self.label_1_family = QLabel(self.window)
        self.label_1_family.setText('Filter family:')
        self.label_1_family.move(10,75)
        self.label_1_family.resize(140,20)

        self.textEdit_1_band = QPlainTextEdit(self.window)
        self.textEdit_1_band.setPlaceholderText("filterband")
        self.textEdit_1_band.move(140,100)
        self.textEdit_1_band.resize(50,20)

        self.label_1_band = QLabel(self.window)
        self.label_1_band.setText('Filter band:')
        self.label_1_band.move(10,100)
        self.label_1_band.resize(140,20)

        self.textEdit_1_vsini = QPlainTextEdit(self.window)
        self.textEdit_1_vsini.setPlaceholderText("km/s")
        self.textEdit_1_vsini.move(140,125)
        self.textEdit_1_vsini.resize(50,20)

        self.label_1_vsini = QLabel(self.window)
        self.label_1_vsini.setText('Vsini of Star:')
        self.label_1_vsini.move(10,125)
        self.label_1_vsini.resize(140,20)

        self.textEdit_1_rv = QPlainTextEdit(self.window)
        self.textEdit_1_rv.setPlaceholderText("km/s")
        self.textEdit_1_rv.move(140,150)
        self.textEdit_1_rv.resize(50,20)

        self.label_1_rv = QLabel(self.window)
        self.label_1_rv.setText('RV of Star:')
        self.label_1_rv.move(10,150)
        self.label_1_rv.resize(140,20)

        self.textEdit_1_minwv = QPlainTextEdit(self.window)
        self.textEdit_1_minwv.setPlaceholderText("nm")
        self.textEdit_1_minwv.move(140,175)
        self.textEdit_1_minwv.resize(50,20)

        self.label_1_minwv = QLabel(self.window)
        self.label_1_minwv.setText('Wavelength, from:')
        self.label_1_minwv.move(10,175)
        self.label_1_minwv.resize(140,20)

        self.textEdit_1_maxwv = QPlainTextEdit(self.window)
        self.textEdit_1_maxwv.setPlaceholderText("nm")
        self.textEdit_1_maxwv.move(140,200)
        self.textEdit_1_maxwv.resize(50,20)

        self.label_1_maxwv = QLabel(self.window)
        self.label_1_maxwv.setText('Wavelength, to:')
        self.label_1_maxwv.move(10,200)
        self.label_1_maxwv.resize(140,20)

        self.textEdit_2_pl_on = QPlainTextEdit(self.window)
        self.textEdit_2_pl_on.setPlaceholderText("1on0off")
        self.textEdit_2_pl_on.move(340,25)
        self.textEdit_2_pl_on.resize(50,20)
#
        self.label_2_pl_on = QLabel(self.window)
        self.label_2_pl_on.setText('Photonic Lantern:')
        self.label_2_pl_on.move(210,25)
        self.label_2_pl_on.resize(140,20)

        self.textEdit_2_res = QPlainTextEdit(self.window)
        self.textEdit_2_res.setPlaceholderText("Resolution")
        self.textEdit_2_res.move(340,50)
        self.textEdit_2_res.resize(50,20)
#
        self.label_2_res = QLabel(self.window)
        self.label_2_res.setText('Resolution:')
        self.label_2_res.move(210,50)
        self.label_2_res.resize(140,20)
#
        self.textEdit_2_res_samp = QPlainTextEdit(self.window)
        self.textEdit_2_res_samp.setPlaceholderText("res samp")
        self.textEdit_2_res_samp.move(340,75)
        self.textEdit_2_res_samp.resize(50,20)
#
        self.label_2_res_samp = QLabel(self.window)
        self.label_2_res_samp.setText('res samp:')
        self.label_2_res_samp.move(210,75)
        self.label_2_res_samp.resize(140,20)
        

        self.textEdit_2_aomode = QPlainTextEdit(self.window)
        self.textEdit_2_aomode.setPlaceholderText("mode")
        self.textEdit_2_aomode.move(340,125)
        self.textEdit_2_aomode.resize(60,20)

        self.label_2_aomode = QLabel(self.window)
        self.label_2_aomode.setText('mode of AO:')
        self.label_2_aomode.move(210,125)
        self.label_2_aomode.resize(140,20)

        self.textEdit_2_instrument = QPlainTextEdit(self.window)
        self.textEdit_2_instrument.setPlaceholderText("MODHIS/HISPEC")
        self.textEdit_2_instrument.move(340,150)
        self.textEdit_2_instrument.resize(60,20)

        self.label_2_instrument = QLabel(self.window)
        self.label_2_instrument.setText('instrument:')
        self.label_2_instrument.move(210,150)
        self.label_2_instrument.resize(140,20)


#
        self.textEdit_3_pwv = QPlainTextEdit(self.window)
        self.textEdit_3_pwv.setPlaceholderText("pwv")
        self.textEdit_3_pwv.move(540,100)
        self.textEdit_3_pwv.resize(50,20)
#
        self.label_3_pwv = QLabel(self.window)
        self.label_3_pwv.setText('PWV:')
        self.label_3_pwv.move(410,100)
        self.label_3_pwv.resize(140,20)

        self.textEdit_3_airmass = QPlainTextEdit(self.window)
        self.textEdit_3_airmass.setPlaceholderText("airmass")
        self.textEdit_3_airmass.move(540,125)
        self.textEdit_3_airmass.resize(60,20)

        self.label_3_airmass = QLabel(self.window)
        self.label_3_airmass.setText('Air Mass:')
        self.label_3_airmass.move(410,125)
        self.label_3_airmass.resize(140,20)

        self.textEdit_4_expt = QPlainTextEdit(self.window)
        self.textEdit_4_expt.setPlaceholderText("s")
        self.textEdit_4_expt.move(740,50)
        self.textEdit_4_expt.resize(50,20)
#
        self.label_4_expt = QLabel(self.window)
        self.label_4_expt.setText('Exp time:')
        self.label_4_expt.move(610,50)
        self.label_4_expt.resize(140,20)



        
        self.button_1 = QPushButton('calculate', self.window)
        self.button_1.move(70,240)
        self.button_1.clicked.connect(self.parameter1)
        self.button_1.clicked.connect(self.integrate)
        self.button_1.clicked.connect(self.filter)
        self.button_1.clicked.connect(self.calc_nphot)
        self.button_1.clicked.connect(self.load_spec_model)
        self.button_1.clicked.connect(self.scale_stellar)
        self.button_1.clicked.connect(self._lsf_rotate)
        self.button_1.clicked.connect(self.stellar)

        self.button_2 = QPushButton('calculate', self.window)
        self.button_2.move(270,240)
        self.button_2.clicked.connect(self.parameter2)
        self.button_2.clicked.connect(self.tophat)
        self.button_2.clicked.connect(self.load_filter)
        self.button_2.clicked.connect(self.get_band_mag)
        self.button_2.clicked.connect(self.calc_strehl)
        self.button_2.clicked.connect(self.tt_to_strehl)
        self.button_2.clicked.connect(self.ao_spec)
        self.button_2.clicked.connect(self.get_base_throughput)
        self.button_2.clicked.connect(self.grid_interp_coupling)
        self.button_2.clicked.connect(self.pick_coupling)
        self.button_2.clicked.connect(self.instrument)

        self.button_3 = QPushButton('calculate', self.window)
        self.button_3.move(470,240)
        self.button_3.clicked.connect(self.parameter3)
        self.button_3.clicked.connect(self.get_emissivity)
        self.button_3.clicked.connect(self.get_inst_bg)
        self.button_3.clicked.connect(self.get_sky_bg)
        self.button_3.clicked.connect(self.telluric)


        self.button_4 = QPushButton('calculate', self.window)
        self.button_4.move(670,240)
        self.button_4.clicked.connect(self.parameter4)
        self.button_4.clicked.connect(self.define_lsf)
        self.button_4.clicked.connect(self.degrade_spec)
        self.button_4.clicked.connect(self.setup_band)
        self.button_4.clicked.connect(self.resample)
        self.button_4.clicked.connect(self.sum_total_noise)
        self.button_4.clicked.connect(self.observe)

    def parameter1(self):
        self.teff = float(self.textEdit_1_teff.toPlainText())
        self.mag = float(self.textEdit_1_mag.toPlainText())
        self.vsini = float(self.textEdit_1_vsini.toPlainText())
        self.rv = float(self.textEdit_1_rv.toPlainText())
        self.band = self.textEdit_1_band.toPlainText()        
        self.family = self.textEdit_1_family.toPlainText()
        self.inst_l0 = float(self.textEdit_1_minwv.toPlainText())
        self.inst_l1 = float(self.textEdit_1_maxwv.toPlainText())
        self.x = np.arange(self.inst_l0,self.inst_l1,0.0005)
        
    def parameter2(self):
        self.inst_res = float(self.textEdit_2_res.toPlainText())
        self.aomode = self.textEdit_2_aomode.toPlainText()  #new
        self.instrument = self.textEdit_2_instrument.toPlainText() #new
        self.inst_res_samp =   float(self.textEdit_2_res_samp.toPlainText())
        self.pl_on = float(self.textEdit_2_pl_on.toPlainText())  #new
        if self.instrument =='modhis':
            self.tel_area = 655.0
            self.tel_diam = 30.0
            self.pix_vert = 3.0
            self.tt_static = 0.0
            self.lo_wfe = 10.0
            self.adc = 0.0
            self.atm = 0.0
            self.read = 12.0
            self.dark = 0.01
            self.saturation = 100000.0
            self.nsamp = 16.0
        else:
            self.tel_area = 76.2
            self.tel_diam = 10.0
            self.pix_vert = 4.0
            self.tt_static = 0.0
            self.lo_wfe = 30.0
            self.adc = 1.0
            self.atm = 1.0
            self.read = 12.0
            self.dark = 0.01
            self.saturation = 100000.0
            self.nsamp = 16.0

    def parameter3(self):
        self.airmass = float(self.textEdit_3_airmass.toPlainText())  #new
        self.pwv = float(self.textEdit_3_pwv.toPlainText())


    def parameter4(self):
        self.exp_time = float(self.textEdit_4_expt.toPlainText())

    def integrate(self,x,y):
        """
        Integrate y wrt x
        """
        return trapz(y,x=x)
    
    def filter(self):
        """
        load band for scaling stellar spectrum
        """
        # read zeropoint file, get zp
        zps                     = np.loadtxt('/Users/huihaoz/specsim/_Data/filters/zeropoints.txt',dtype=str).T
        izp                     = np.where((zps[0]==self.family) & (zps[1]==self.band))[0]
        self.filt_zp              = float(zps[2][izp])

        # find filter file and load filter
        filter_file         = glob.glob('/Users/huihaoz/specsim/_Data/filters/' + '*' + self.family + '*' +self.band + '.dat')[0]
        self.filt_xraw, self.filt_yraw  = np.loadtxt(filter_file).T # nm, transmission out of 1
        if np.max(self.filt_xraw)>5000: self.filt_xraw /= 10
        if np.max(self.filt_xraw) < 10: self.filt_xraw *= 1000
        
        f                       = interpolate.interp1d(self.filt_xraw, self.filt_yraw, bounds_error=False,fill_value=0)
        self.filt_v, self.filt_s    = self.x, f(self.x)  #filter profile sampled at stellar

        self.filt_dl_l                 = np.mean(self.integrate(self.filt_xraw, self.filt_yraw)/self.filt_xraw) # dlambda/lambda
        self.filt_center_wavelength    = self.integrate(self.filt_xraw,self.filt_yraw*self.filt_xraw)/self.integrate(self.filt_xraw,self.filt_yraw)
    def calc_nphot(self,dl_l, zp, mag):
        """
        http://astroweb.case.edu/ssm/ASTR620/mags.html

        Values are all for a specific bandpass, can refer to table at link ^ for values
        for some bands. Function will return the photons per second per meter squared
        at the top of Earth atmosphere for an object of specified magnitude

        inputs:
        -------
        dl_l: float, delta lambda over lambda for the passband
        zp: float, flux at m=0 in Jansky
        mag: stellar magnitude

        outputs:
        --------
        photon flux
        """
        phot_per_s_m2_per_Jy = 1.51*10**7 # convert to phot/s/m2 from Jansky

        return dl_l * zp * 10**(-0.4*mag) * phot_per_s_m2_per_Jy
    
    def load_spec_model(self,wav_start=750,wav_end=780):
        if self.teff>2300:
            teff = str(int(self.teff)).zfill(5)
            self.stel_model             = 'phoenix' 
            self.stel_file         = '/Users/huihaoz/specsim/_Data/phoenix/' + 'lte%s-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'%(teff)
            f = fits.open(self.stel_file)
            spec = f[0].data / (1e8) # ergs/s/cm2/cm to ergs/s/cm2/Angstrom for conversion
            f.close()
            path = self.stel_file.split('/')
            wave_file = '/' + os.path.join(*self.stel_file.split('/')[0:-1]) + '/' + \
                            'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits' #assume wave in same folder
            f = fits.open(wave_file)
            lam = f[0].data # angstroms
            f.close()
            
            # Convert
            conversion_factor = 5.03*10**7 * lam #lam in angstrom here
            spec *= conversion_factor # phot/cm2/s/angstrom
            
            # Take subarray requested
            isub = np.where( (lam > wav_start*10.0) & (lam < wav_end*10.0))[0]

            # Convert 
            return lam[isub]/10.0,spec[isub] * 10 * 100**2 #nm, phot/m2/s/nm
        else:
            g    = '316' # mks units, np.log10(316 * 100)=4.5 to match what im holding for phoenix models.
            teff = str(int(self.teff))
            self.stel_file         = '/Users/huihaoz/specsim/_Data/sonora/' + 'sp_t%sg%snc_m0.0' %(teff,g)
            f = np.loadtxt(self.stel_file,skiprows=2)

            lam  = 10000* f[:,0][::-1] #microns to angstroms, needed for conversiosn
            spec = f[:,1][::-1] # erg/cm2/s/Hz
            
            spec *= 3e18/(lam**2)# convert spec to erg/cm2/s/angstrom
            
            conversion_factor = 5.03*10**7 * lam #lam in angstrom here
            spec *= conversion_factor # phot/cm2/s/angstrom
            
            isub = np.where( (lam > wav_start*10.0) & (lam < wav_end*10.0))[0]

            return lam[isub]/10.0,spec[isub] * 10 * 100**2 #nm, phot/m2/s/nm (my fave)
        
    def scale_stellar(self,mag):
        """
        scale spectrum by magnitude
        inputs: 
        so: object with all variables
        mag: magnitude in filter desired

        load new stellar to match bounds of filter since may not match working badnpass elsewhere
        """
        stelv,stels       =  self.load_spec_model(wav_start=np.min(self.filt_xraw), wav_end=np.max(self.filt_xraw)) #phot/m2/s/nm

        filt_interp       =  interpolate.interp1d(self.filt_xraw, self.filt_yraw, bounds_error=False,fill_value=0)

        filtered_stellar   = stels * filt_interp(stelv)    # filter profile resampled to phoenix times phoenix flux density
        nphot_expected_0   = self.calc_nphot(self.filt_dl_l, self.filt_zp, mag)    # what's the integrated flux supposed to be in photons/m2/s?
        nphot_phoenix      = self.integrate(stelv,filtered_stellar)            # what's the integrated flux now? in same units as ^
        
        return nphot_expected_0/nphot_phoenix
    
    def _lsf_rotate(self,deltav,vsini,epsilon=0.6):
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
    def stellar(self):
        """
        loads stellar spectrum
        returns spectrum scaled to input V band mag 

        everything in nm
        """
        # Part 1: load raw spectrum
        #
        self.stel_vraw,self.stel_sraw = self.load_spec_model(wav_start=self.inst_l0,wav_end=self.inst_l1)
        self.stel_v   = self.x
        tck_stel    = interpolate.splrep(self.stel_vraw,self.stel_sraw, k=2, s=0)
        self.stel_s   = interpolate.splev(self.x,tck_stel,der=0,ext=1)

        # apply scaling factor to match filter zeropoint
        self.stel_factor_0   = self.scale_stellar(self.mag) 
        self.stel_s   *= self.stel_factor_0
        self.stel_units = 'photons/s/m2/nm' # stellar spec is in photons/s/m2/nm

        # broaden star spectrum with rotation kernal
        if self.vsini > 0:
            dwvl_mean = np.abs(np.nanmean(np.diff(self.x)))
            SPEEDOFLIGHT = 2.998e8 # m/s
            dvel_mean = (dwvl_mean / np.nanmean(self.x)) * SPEEDOFLIGHT / 1e3 # average sampling in km/s
            vsini_kernel,_ = self._lsf_rotate(dvel_mean,self.vsini,epsilon=0.6)
            flux_vsini     = convolve(self.stel_s,vsini_kernel,normalize_kernel=True)  # photons / second / Ang
            self.stel_s      = flux_vsini
        if self.rv > 0:
            dvelocity =   (self.stel_v / 300000) * u.nm * consts.c / (self.stel_v * u.nm )
            rv_shift_resel = np.mean(self.rv * u.km / u.s / dvelocity) * 1000*u.m/u.km
            spec_shifted = shift(self.stel_s,rv_shift_resel.value)
            self.stel_s      = spec_shifted
        
        print('factor 0:',self.stel_factor_0)
        print('obj photon:',self.stel_s)
        print('obj wvs:',self.stel_v)
##############################

    def tophat(self,x,l0,lf,throughput):
        ion = np.where((x > l0) & (x<lf))[0]
        bandpass = np.zeros_like(x)
        bandpass[ion] = throughput
        return bandpass
        

    def load_filter(self,band_ao):
        """
        """
        if band_ao == False:
            return band_ao=='V'
        else:
            files = glob.glob('/Users/huihaoz/specsim/_Data/filters/' + '*' + 'Johnson' + '*'+ band_ao + '.dat')
            if not files:
                raise FileNotFoundError(f"No file matches the pattern {'/Users/huihaoz/specsim/_Data/filters/'}*{'Johnson'}*{band_ao}*.dat")
            filter_file = files[0]
        #	filter_file    = glob.glob(filter_path + '*' + family + '*' + band + '.dat')[0]
            xraw, yraw     = np.loadtxt(filter_file).T # nm, transmission out of 1
            return xraw/10, yraw 
        
    def get_band_mag(self,band_ao):
        """
        factor_0: scaling model to photons
        """
        factor_0=self.stel_factor_0
        if band_ao ==False:
            return band_ao == 'R'
        else:
            x,y          = self.load_filter(band_ao)
            filt_interp  = interpolate.interp1d(x, y, bounds_error=False,fill_value=0)
            dl_l         = np.mean(self.integrate(x,y)/x) # dlambda/lambda to account for spectral fraction
            
            # load stellar the multiply by scaling factor, factor_0, and filter. integrate
            if (np.min(x) < self.inst_l0) or (np.max(x) > self.inst_l1):
                vraw,sraw = self.load_spec_model(wav_start=np.min(x),wav_end=np.max(x)) #phot/m2/s/nm

            else:
                vraw,sraw = self.stel_vraw,self.stel_sraw #r

            filtered_stel = factor_0 * sraw * filt_interp(vraw)
            flux = self.integrate(vraw,filtered_stel)    #phot/m2/s

            phot_per_s_m2_per_Jy = 1.51*10**7 # convert to phot/s/m2 from Jansky
            
            flux_Jy = flux/phot_per_s_m2_per_Jy/dl_l
            
            # get zps
            zps                     = np.loadtxt('/Users/huihaoz/specsim/_Data/filters/zeropoints.txt',dtype=str).T
            izp                     = np.where((zps[0]=='Johnson') & (zps[1]==band_ao))[0]
            zp                      = float(zps[2][izp])

            mag = -2.5*np.log10(flux_Jy/zp)

            return mag,x,y
        
    def calc_strehl(self,wfe,wavelength):
        """
        wfe: nm
        wavelength: nm
        """
        strehl = np.exp(-(2*np.pi*wfe/wavelength)**2)

        return strehl
    def tt_to_strehl(self,tt,lam,D):
        """
        convert tip tilt residuals in mas to strehl according to Rich's equation
        
        equation 4.60 from Hardy 1998 (adaptive optics for astronomy) matches this

        lam: nm
            wavelength(s)
        D: m
            telescope diameter
        tt: mas
            tip tilt rms
        """
        tt_rad = tt * 1e-3/206265 # convert to radians from mas
        lam_m =lam * 1e-9
        bottom = 1 + np.pi**2/2*((tt_rad)/(lam_m/D))**2
        strehl_tt = 1/bottom

        #sig_D = 0.44* lam_m/D
        #1/(1 + tt_rad**2/sig_D**2) KAON1322 doc eq 5 method matches Richs eqn

        return strehl_tt

    def ao_spec(self):
        self.inst_y=[980,1100]
        self.inst_J=[1170,1327]
        self.inst_H=[1490,1780]
        self.inst_K=[1990,2460]
        aomode = self.aomode
        inst = self.instrument
        print(aomode)
        print(self.aomode)
        if aomode == 'auto':
            print('auto ao mode')

            if inst =='hispec':
                ho_wfe = []
                ao_ho_wfe_mag = []
                ao_ho_wfe_band = []
                tt_wfe_mag = []
                tt_wfe_band = []
                tt_wfe = []
                sr_tt = []
                sr_ho = []

                f_full = pd.read_csv('/Users/huihaoz/specsim/_Data/WFE/AO/HOwfe.csv',header=[0,1])
                mags             = f_full['mag'].values.T[0]
                f=f_full[['LGS_STRAP_45','SH','LGS_100J_45']]
                self.aomodes = f.columns
                for i in range(len(f.columns)):
                    wfes = f[f.columns[i][0]].values.T[0]
                    ho_wfe_band= f.columns[i][1]
                    ho_wfe_mag,x_test,y_test = self.get_band_mag(band_ao=ho_wfe_band)
                    f_howfe = interpolate.interp1d(mags,wfes, bounds_error=False,fill_value=10000)
                    ao_ho_wfe     = float(f_howfe(ho_wfe_mag))
                    strehl_ho = self.calc_strehl(ao_ho_wfe,self.filt_center_wavelength)
                    ho_wfe.append(ao_ho_wfe)
                    ao_ho_wfe_band.append(ho_wfe_band)
                    ao_ho_wfe_mag.append(ho_wfe_mag)
                    sr_ho.append(strehl_ho)

                f_full = pd.read_csv('/Users/huihaoz/specsim/_Data/WFE/AO/TT_dynamic.csv',header=[0,1])
                mags             = f_full['mag'].values.T[0]
                f=f_full[['LGS_STRAP_45','SH','LGS_100J_45']]
                self.aomodes = f.columns
                for i in range(len(f.columns)):
                    tts             = f[f.columns[i][0]].values.T[0]
                    ttdynamic_band=f.columns[i][1] # this is the mag band wfe is defined in, must be more readable way..			
                    ttdynamic_mag,x_test2,y_test2 = self.get_band_mag(band_ao=ttdynamic_band) # get magnitude of star in appropriate band
                    f_ttdynamic=  interpolate.interp1d(mags,tts, bounds_error=False,fill_value=10000)
                    tt_dynamic     = float(f_ttdynamic(ttdynamic_mag))
                    strehl_tt = self.tt_to_strehl(tt=tt_dynamic,lam=self.filt_center_wavelength,D=self.tel_diam)
                    tt_wfe.append(tt_dynamic)
                    tt_wfe_band.append(ttdynamic_band)
                    tt_wfe_mag.append(ttdynamic_mag)
                    sr_tt.append(strehl_tt)

                ind_auto_ao = np.where(np.array(sr_tt)*np.array(sr_ho) == np.max(np.array(sr_tt)*np.array(sr_ho)))[0][0]
                self.aomode=f.columns[ind_auto_ao][0]
                print("ao mode:", f.columns[ind_auto_ao][0], f.columns[ind_auto_ao][1])
                self.ao_tt_dynamic= tt_wfe[ind_auto_ao]
                self.ao_ho_wfe= ho_wfe[ind_auto_ao]
                self.ao_ho_wfe_mag=ao_ho_wfe_mag[ind_auto_ao]
                self.ao_ho_wfe_band=ao_ho_wfe_band[ind_auto_ao]
                self.ao_ttdynamic_mag=tt_wfe_mag[ind_auto_ao]
                self.ao_ttdynamic_band=tt_wfe_band[ind_auto_ao]
                print("tt:",self.ao_tt_dynamic)
                print("ho:",self.ao_ho_wfe)
                if self.aomode =='80J':
                    self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_J[0],self.inst_J[1],0.8)
                elif self.aomode =='80H':
                    self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_H[0],self.inst_H[1],0.8)
                elif self.aomode =='80JH':
                    self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_J[0],self.inst_H[1],0.8)
                elif self.aomode =='100JH':
                    self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_J[0],self.inst_H[1],1)
                elif self.aomode =='100K':
                    self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_K[0],self.inst_K[1],1)
                else:
                    self.ao_pywfs_dichroic = np.ones_like(self.x)
                                
            elif inst =='modhis':
                ho_wfe = []
                ao_ho_wfe_mag = []
                ao_ho_wfe_band = []
                tt_wfe_mag = []
                tt_wfe_band = []
                tt_wfe = []
                sr_tt = []
                sr_ho = []
                f = pd.read_csv('/Users/huihaoz/specsim/_Data/WFE/AO/HOWFE_NFIRAOS.csv',header=[0,1])
                mags             = f['mag'].values.T[0]
                self.aomodes = f.columns
                for i in range(len(f.columns[2:])):
                    wfes = f[f.columns[2:][i][0]].values.T[0]
                    ho_wfe_band= f.columns[2:][i][1]
                    ho_wfe_mag,x_test,y_test = self.get_band_mag(band_ao=ho_wfe_band)
                    f_howfe = interpolate.interp1d(mags,wfes, bounds_error=False,fill_value=10000)
                    ao_ho_wfe     = float(f_howfe(ho_wfe_mag))
                    strehl_ho = self.calc_strehl(ao_ho_wfe,self.filt_center_wavelength)
                    ho_wfe.append(ao_ho_wfe)
                    ao_ho_wfe_band.append(ho_wfe_band)
                    ao_ho_wfe_mag.append(ho_wfe_mag)
                    sr_ho.append(strehl_ho)

                f = pd.read_csv('/Users/huihaoz/specsim/_Data/WFE/AO/TTDYNAMIC_NFIRAOS.csv',header=[0,1])
                mags             = f['mag'].values.T[0]
                self.aomodes = f.columns
                for i in range(len(f.columns[2:])):
                    tts             = f[f.columns[2:][i][0]].values.T[0]
                    ttdynamic_band=f.columns[2:][i][1] # this is the mag band wfe is defined in, must be more readable way..			
                    ttdynamic_mag,x_test2,y_test2 = self.get_band_mag(band_ao=ttdynamic_band) # get magnitude of star in appropriate band
                    f_ttdynamic=  interpolate.interp1d(mags,tts, bounds_error=False,fill_value=10000)
                    tt_dynamic     = float(f_ttdynamic(ttdynamic_mag))
                    strehl_tt = self.tt_to_strehl(tt=tt_dynamic,lam=self.filt_center_wavelength,D=self.tel_diam)
                    tt_wfe.append(tt_dynamic)
                    tt_wfe_band.append(ttdynamic_band)
                    tt_wfe_mag.append(ttdynamic_mag)
                    sr_tt.append(strehl_tt)

                ind_auto_ao = np.where(np.array(sr_tt)*np.array(sr_ho) == np.max(np.array(sr_tt)*np.array(sr_ho)))[0][0]
                self.aomode=f.columns[2:][ind_auto_ao][0]
                print("ao mode:", f.columns[2:][ind_auto_ao][0], f.columns[2:][ind_auto_ao][1])
                self.ao_tt_dynamic= tt_wfe[ind_auto_ao]
                self.ao_ho_wfe= ho_wfe[ind_auto_ao]
                self.ao_ho_wfe_mag=ao_ho_wfe_mag[ind_auto_ao]
                self.ao_ho_wfe_band=ao_ho_wfe_band[ind_auto_ao]
                self.ao_ttdynamic_mag=tt_wfe_mag[ind_auto_ao]
                self.ao_ttdynamic_band=tt_wfe_band[ind_auto_ao]
                print("tt:",self.ao_tt_dynamic)
                print("ho:",self.ao_ho_wfe)
                if self.aomode =='80J':
                    self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_J[0],self.inst_J[1],0.8)
                elif self.aomode =='80H':
                    self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_H[0],self.inst_H[1],0.8)
                elif self.aomode =='80JH':
                    self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_J[0],self.inst_H[1],0.8)
                elif self.aomode =='100JH':
                    self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_J[0],self.inst_H[1],1)
                elif self.aomode =='100K':
                    self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_K[0],self.inst_K[1],1)
                else:
                    self.ao_pywfs_dichroic = np.ones_like(self.x)
            else:
                raise ValueError('instrument must be modhis or hispec')
        else:
            if inst =='hispec':

                    f = pd.read_csv('/Users/huihaoz/specsim/_Data/WFE/AO/HOwfe.csv',header=[0,1])
                    self.aomodes = f.columns
                    mags             = f['mag'].values.T[0]
                    wfes             = f[self.aomode].values.T[0]
                    self.ao_ho_wfe_band= f[self.aomode].columns[0] # this is the mag band wfe is defined in, must be more readable way..
                    self.ao_ho_wfe_mag,x_test3,y_test3 = self.get_band_mag(band_ao=self.ao_ho_wfe_band) # get magnitude of star in appropriate band
                    f_howfe          = interpolate.interp1d(mags,wfes, bounds_error=False,fill_value=10000)
                    self.ao_ho_wfe     = float(f_howfe(self.ao_ho_wfe_mag))
                    print('HO WFE %s mag is %s'%(self.ao_ho_wfe_band,self.ao_ho_wfe_mag))


                    f = pd.read_csv('/Users/huihaoz/specsim/_Data/WFE/AO/TT_dynamic.csv',header=[0,1])
                    self.aomodes_tt  = f.columns # should match howfe..
                    mags            = f['mag'].values.T[0]
                    tts             = f[self.aomode].values.T[0]
                    self.ao_ttdynamic_band=f[self.aomode].columns[0] # this is the mag band wfe is defined in, must be more readable way..			
                    self.ao_ttdynamic_mag,x_test3,y_test3 = self.get_band_mag(band_ao=self.ao_ttdynamic_band) # get magnitude of star in appropriate band
                    f_ttdynamic=  interpolate.interp1d(mags,tts, bounds_error=False,fill_value=10000)
                    self.ao_tt_dynamic     = float(f_ttdynamic(self.ao_ttdynamic_mag))
                    print('Tip Tilt %s mag is %s'%(self.ao_ttdynamic_band,self.ao_ttdynamic_mag))


                    print('AO mode: %s'%self.aomode)

                    #so.ao.ho_wfe = get_HO_WFE(so.ao.v_mag,so.ao.mode) #old
                    print('HO WFE is %s'%self.ao_ho_wfe)

                    #so.ao.tt_dynamic = get_tip_tilt_resid(so.ao.v_mag,so.ao.mode)
                    print('tt dynamic is %s'%self.ao_tt_dynamic)

                    # consider throughput impact of ao here
                    if self.aomode =='80J':
                        self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_J[0],self.inst_J[1],0.8)
                    elif self.aomode =='80H':
                        self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_H[0],self.inst_H[1],0.8)
                    elif self.aomode =='80JH':
                        self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_J[0],self.inst_H[1],0.8)
                    elif self.aomode =='100JH':
                        self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_J[0],self.inst_H[1],1)
                    elif self.aomode =='100K':
                        self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_K[0],self.inst_K[1],1)
                    else:
                        self.ao_pywfs_dichroic = np.ones_like(self.x)

            elif inst =='modhis':
                    f = pd.read_csv('/Users/huihaoz/specsim/_Data/WFE/AO/HOWFE_NFIRAOS.csv',header=[0,1])
                    self.aomodes = f.columns
                    mags             = f['mag'].values.T[0]
                    wfes             = f[self.aomode].values.T[0]
                    self.ao_ho_wfe_band= f[self.aomode].columns[0] # this is the mag band wfe is defined in, must be more readable way..
                    self.ao_ho_wfe_mag,x_test3,y_test3 = self.get_band_mag(band_ao=self.ao_ho_wfe_band) # get magnitude of star in appropriate band
                    f_howfe          = interpolate.interp1d(mags,wfes, bounds_error=False,fill_value=10000)
                    self.ao_ho_wfe     = float(f_howfe(self.ao_ho_wfe_mag))
                    #self.ao_ho_wfe     = float(f_howfe(self.ao_ho_wfe_mag))
                    print('HO WFE %s mag is %s'%(self.ao_ho_wfe_band,self.ao_ho_wfe_mag))


                    f = pd.read_csv('/Users/huihaoz/specsim/_Data/WFE/AO/TTDYNAMIC_NFIRAOS.csv',header=[0,1])
                    self.aomodes_tt  = f.columns # should match howfe..
                    mags            = f['mag'].values.T[0]
                    tts             = f[self.aomode].values.T[0]
                    self.ao_ttdynamic_band=f[self.aomode].columns[0] # this is the mag band wfe is defined in, must be more readable way..			
                    self.ao_ttdynamic_mag,x_test4,y_test4 = self.get_band_mag(band_ao=self.ao_ttdynamic_band) # get magnitude of star in appropriate band
                    f_ttdynamic=  interpolate.interp1d(mags,tts, bounds_error=False,fill_value=10000)
                    self.ao_tt_dynamic     = float(f_ttdynamic(self.ao_ttdynamic_mag))
                    print('Tip Tilt %s mag is %s'%(self.ao_ttdynamic_band,self.ao_ttdynamic_mag))


                    print('AO mode: %s'%self.aomode)

                    #so.ao.ho_wfe = get_HO_WFE(so.ao.v_mag,so.ao.mode) #old
                    print('HO WFE is %s'%self.ao_ho_wfe)

                    #so.ao.tt_dynamic = get_tip_tilt_resid(so.ao.v_mag,so.ao.mode)
                    print('tt dynamic is %s'%self.ao_tt_dynamic)

                    # consider throughput impact of ao here
                    if self.aomode =='80J':
                        self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_J[0],self.inst_J[1],0.8)
                    elif self.aomode =='80H':
                        self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_H[0],self.inst_H[1],0.8)
                    elif self.aomode =='80JH':
                        self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_J[0],self.inst_H[1],0.8)
                    elif self.aomode =='100JH':
                        self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_J[0],self.inst_H[1],1)
                    elif self.aomode =='100K':
                        self.ao_pywfs_dichroic = 1 - self.tophat(self.x,self.inst_K[0],self.inst_K[1],1)
                    else:
                        self.ao_pywfs_dichroic = np.ones_like(self.x)
            
            else:
                raise ValueError('instrument must be modhis or hispec')
            
    def get_base_throughput(self):
        """
        get throughput except leave out coupling

        also store emissivity
        """
        #plt.figure()
        for spec in ['red','blue']:
            if spec=='red':
                include = ['tel', 'ao', 'feicom', 'feired','fibred','rspec']#,'coupling']
            if spec=='blue':
                include = ['tel', 'ao', 'feicom', 'feiblue','fibblue','bspec']#,'coupling']

            for i in include:
                if i==include[0]:
                    w,s = np.loadtxt('/Users/huihaoz/specsim/_Data/Throughput/' + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                    #plt.plot(w,s,label=i)
                else:
                    wtemp, stemp = np.loadtxt('/Users/huihaoz/specsim/_Data/Throughput/' + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                    # interpolate onto s
                    f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
                    s*=f(w)
                    #plt.plot(w,s,label=i)

            if spec=='red':
                isub = np.where(w > 1.4) 
                wred = w[isub]
                specred = s[isub]
            if spec=='blue':
                isub = np.where(w<1.4)
                specblue = s[isub]
                wblue = w[isub]
        
        w = np.concatenate([wblue,wred])
        s = np.concatenate([specblue,specred])

        # reinterpolate 
        x = self.x.copy()
        if np.min(x) > 10:
            x/=1000 #convert nm to um

        tck    = interpolate.splrep(w,s, k=2, s=0)
        self.base_throughput   = interpolate.splev(x,tck,der=0,ext=1)

    def grid_interp_coupling(self,PLon,atm,adc):
        """
        interpolate coupling files over their various parameters
        PLon: 0 or 1, whether PL is on or not
        path: data path to coupling files
        atm: 0 or 1 - whether gary at atm turned on in sims
        adc: 0 or 1 - whether gary had adc included in sims
        """
        if (PLon ==False and atm == False and adc == False) :
            return print('PLon == 1')
        else:
            print(PLon)
            print(atm)
            print(adc)
            path='/Users/huihaoz/Downloads/psisim-kpic/psisim/data/coupling/'
            LOs = np.arange(0,125,25)
            ttStatics = np.arange(11)
            ttDynamics = np.arange(0,20.5,0.5)
            
            filename_skeleton = 'couplingEff_atm%s_adc%s_PL%s_defoc25nmRMS_LO%snmRMS_ttStatic%smas_ttDynamic%smasRMS.csv'

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
        
    def pick_coupling(self,w,dynwfe,ttStatic,ttDynamic,LO,PLon,points,values):
        """
        select correct coupling file
        to do:implement interpolation of coupling files instead of rounding variables
        """
        #if w == False & dynwfe == False & ttStatic == False & ttDynamic == False & LO==False  & points ==False & values ==False:
        #    return piaa_boost == 1
        #else:
        PLon = int(PLon)

        piaa_boost = 1.3
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
            values_1= values
            points, values_1 = self.grid_interp_coupling(PLon=1,adc=1,atm=1)
            point = (LO,ttStatic,ttDynamic,waves)
            raw_coupling = interpolate.interpn(points, values_1, point,bounds_error=False,fill_value=0)

        if np.max(waves) < 10:
            waves*=1000 # nm to match dynwfe

        ho_strehl =  np.exp(-(2*np.pi*dynwfe/waves)**2) # computed per wavelength as grid
        coupling  = raw_coupling * piaa_boost * ho_strehl

        return coupling, ho_strehl,raw_coupling
    
    def instrument(self):
		###########
		# load hispec transmission
		#xtemp, ytemp  = np.loadtxt(so.inst.transmission_file,delimiter=',').T #microns!
		#f = interp1d(xtemp*1000,ytemp,kind='linear', bounds_error=False, fill_value=0)
		
		#so.inst.xtransmit, so.inst.ytransmit = self.x, f(self.x) 

		# save dlambda

        sig = self.stel_v/self.inst_res/self.inst_res_samp # lambda/res = dlambda, nm per pixel
        self.inst_sig=sig
		# THROUGHPUT
        
        # interp grid
        try: self.inst_points
        except AttributeError: 
            out = self.grid_interp_coupling(PLon=int(self.pl_on),atm=int(self.atm),adc=int(self.adc))
            self.inst_grid_points, self.inst_grid_values = out[0],out[1:] #if PL, three values
        try:
            self.inst_coupling, self.inst_strehl,self.row_coup = self.pick_coupling(w=self.x,dynwfe=self.ao_ho_wfe,ttStatic=self.tt_static,ttDynamic=self.ao_tt_dynamic,LO=self.lo_wfe,PLon=self.pl_on,points=self.inst_grid_points, values=self.inst_grid_values)
        except ValueError:
            # hack here bc tt dynamic often is out of bounds
            self.inst_coupling, self.inst_strehl,self.row_coup = self.pick_coupling(w=self.x,dynwfe=self.ao_ho_wfe,ttStatic=self.tt_static,ttDynamic=20,LO=self.lo_wfe,PLon=self.pl_on,points=self.inst_grid_points, values=self.inst_grid_values)

        self.inst_xtransmit = self.x
        self.inst_ytransmit = self.base_throughput* self.inst_coupling * self.ao_pywfs_dichroic

##############################
    def get_emissivity(self,w):
        """
        get throughput except leave out couplingalso store emissivity
        """
        if type(w) == bool:
            return w =='1'
        else:
            x = w.copy()
            if np.min(x) > 10:
                x/=1000 #convert nm to um

            red_include = ['tel', 'ao', 'feicom', 'feired','fibred','rspec']#,'coupling']
            blue_include = ['tel', 'ao', 'feicom', 'feiblue','fibblue','bspec']
            temps = [276,276,276,276,276,77]

            em_red, em_blue = [],[]
            for i in red_include:
                wtemp, stemp = np.loadtxt('/Users/huihaoz/specsim/_Data/Throughput/' + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
                em_red.append(1-f(x)) # 1 - interp throughput onto x

            for i in blue_include:
                wtemp, stemp = np.loadtxt('/Users/huihaoz/specsim/_Data/Throughput/' + i + '/%s_throughput.csv'%i, delimiter=',',skiprows=1).T
                f = interpolate.interp1d(wtemp, stemp, bounds_error=False,fill_value=0)
                em_blue.append(1-f(x)) # 1 - interp throughput onto x
            return em_red,em_blue,temps

    

    def get_inst_bg(self,x):
        """
        generate sky background per reduced pixel, default to HIPSEC. Source: DMawet jup. notebook

        inputs:
        -------

        outputs:
        --------
        sky background (photons/s) already considering PSF sampling

        """
        # telescope
        if type(x)== bool:
            return x =='1'
 
        else:
            diam = self.tel_diam * u.m
            area = self.tel_area * u.m**2
            wave = x*u.nm
            R = self.inst_res
            npix = self.pix_vert
            em_red,em_blue, temps = self.get_emissivity(w=x)

            fwhm = ((wave  / diam) * u.radian).to(u.arcsec)

            solidangle = fwhm**2 * 1.13 #corrected for Gaussian beam (factor 1.13)
            pix_width_nm  = (wave/R/npix) #* u.nm 

            # step through temperatures and emissivities for red and blue
            # red
            for i,temp in enumerate(temps):
                bbtemp_fxn  = BlackBody(temp * u.K, scale=1.0 * u.erg / (u.micron * u.s * u.cm**2 * u.arcsec**2)) 
                bbtemp = bbtemp_fxn(wave) *  area.to(u.cm**2) * solidangle
                #bbtemp = blackbody_lambda(wave, temp).to(u.erg/(u.micron * u.s * u.cm**2 * u.arcsec**2)) * area.to(u.cm**2) * solidangle
                if i==0:
                    tel_thermal_red  = em_red[i] * bbtemp.to(u.photon/u.s/u.micron, equivalencies=u.spectral_density(wave)) * pix_width_nm
                    tel_thermal_blue = em_blue[i] * bbtemp.to(u.photon/u.s/u.micron, equivalencies=u.spectral_density(wave)) * pix_width_nm
                else:
                    therm_red_temp   = em_red[i] * bbtemp.to(u.photon/u.s/u.micron, equivalencies=u.spectral_density(wave)) * pix_width_nm
                    therm_blue_temp  = em_blue[i] * bbtemp.to(u.photon/u.s/u.micron, equivalencies=u.spectral_density(wave)) * pix_width_nm
                    tel_thermal_red+= therm_red_temp
                    tel_thermal_blue+= therm_blue_temp

            # interpolate and combine into one thermal spectrum
            isubred = np.where(wave > 1.4*u.um)[0]
            em_red_tot  = tel_thermal_red[isubred].decompose()
            isubblue = np.where(wave <1.4*u.um)[0]
            em_blue_tot  = tel_thermal_blue[isubblue].decompose()

            w = np.concatenate([x[isubblue],x[isubred]])
            s = np.concatenate([em_blue_tot,em_red_tot])

            tck        = interpolate.splrep(w,s.value, k=2, s=0)
            em_total   = interpolate.splev(x,tck,der=0,ext=1)
            return em_total

    
    def telluric(self):
        """
        load tapas telluric file
        """
        data      = fits.getdata('/Users/huihaoz/specsim/_Data/telluric/psg_out_2020.08.02_l0_800nm_l1_2700nm_res_0.001nm_lon_204.53_lat_19.82_pres_0.5826.fits')
        pwv0      = fits.getheader('/Users/huihaoz/specsim/_Data/telluric/psg_out_2020.08.02_l0_800nm_l1_2700nm_res_0.001nm_lon_204.53_lat_19.82_pres_0.5826.fits')['PWV']
        airmass0  = fits.getheader('/Users/huihaoz/specsim/_Data/telluric/psg_out_2020.08.02_l0_800nm_l1_2700nm_res_0.001nm_lon_204.53_lat_19.82_pres_0.5826.fits')['AIRMASS']
        
        _,ind     = np.unique(data['Wave/freq'],return_index=True)
        tck_tel   = interpolate.splrep(data['Wave/freq'][ind],data['Total'][ind]**(self.airmass/airmass0), k=2, s=0)
        self.tel_v, self.tel_s = self.x, interpolate.splev(self.x,tck_tel,der=0,ext=1)
        
        tck_tel    = interpolate.splrep(data['Wave/freq'][ind],data['H2O'][ind]**(self.pwv * self.airmass/pwv0/airmass0), k=2, s=0)
        self.tel_h2o = interpolate.splev(self.x,tck_tel,der=0,ext=1)

        tck_tel    = interpolate.splrep(data['Wave/freq'][ind],data['Rayleigh'][ind]**(self.airmass/airmass0), k=2, s=0)
        self.tel_rayleigh = interpolate.splev(self.x,tck_tel,der=0,ext=1)

        tck_tel    = interpolate.splrep(data['Wave/freq'][ind],data['O3'][ind]**(self.airmass/airmass0), k=2, s=0)
        self.tel_o3 = interpolate.splev(self.x,tck_tel,der=0,ext=1)

    def get_sky_bg(self,x):
        if type(x) == bool:
            return x=='1'
        else:
            """
            generate sky background per reduced pixel, default to HIPSEC. Source: DMawet jup. notebook

            inputs:
            -------

            outputs:
            --------
            sky background (ph/s)
            """
            diam = 10 * u.m
            area = 76 * u.m**2
            wave=x*u.nm
            R=self.inst_res
            npix = self.pix_vert

            fwhm = ((wave  / diam) * u.radian).to(u.arcsec)

            solidangle = fwhm**2 * 1.13 #corrected for Gaussian beam (factor 1.13)

            sky_background_MK_tmp  = np.genfromtxt('/Users/huihaoz/specsim/_Data/sky/'+'mk_skybg_zm_'+str(self.pwv)+'_'+str(self.airmass)+'_ph.dat', skip_header=0)
            sky_background_MK      = sky_background_MK_tmp[:,1]
            sky_background_MK_wave = sky_background_MK_tmp[:,0] #* u.nm

            pix_width_nm  = (wave/R/npix) #* u.nm 
            sky_background_interp=np.interp(wave.value, sky_background_MK_wave, sky_background_MK) * u.photon/(u.s*u.arcsec**2*u.nm*u.m**2) * area * solidangle * pix_width_nm 
            sky_background_interp=sky_background_interp

            return  sky_background_interp.value
        
######################
    def define_lsf(self,v,res):
        """
        define gaussian in pixel elements to convolve resolved spectrum with to get rightish resolution
        """
        dlam  = np.median(v)/res
        fwhm  = dlam/np.mean(np.diff(v)) # desired lambda spacing over current lambda spacing resolved to give sigma in array elements
        sigma = fwhm/2.634 # FWHM is dl/l but feed sigma    
        x = np.arange(sigma*10)
        gaussian = (1./sigma/np.sqrt(2*np.pi)) * np.exp(-0.5*( (x - 0.5*len(x))/sigma)**2 )

        return gaussian

    def degrade_spec(self,x,y,res):
        """
        given wavelength, flux array, and resolving power R, return  spectrum at that R
        """
        lsf      = self.define_lsf(x,res=res)
        y_lowres = np.convolve(y,lsf,mode='same')

        return y_lowres
    def setup_band(self,x, x0=0, sig=0.3, eta=1):
        if type(x) ==bool:
            return x=='1'
        else:
            """
            give step function

            inputs:
            ------
            x0
            sig
            eta
            """
            y = np.zeros_like(x)

            ifill = np.where((x > x0-sig/2) & (x < x0 + sig/2))[0]
            y[ifill] = eta

            return y

    def resample(self,x,y,sig=0.3, dx=0, eta=1,mode='variable'):
        """
        resample using convolution

        x: wavelength array in nm
        y_in/y_out: two y arrays (evaluated at x) to resample, units in spectral density (e.g. photons/nm)

        sig in nanometers - width of bin, default 0.3nm
        dx - offset for taking first bin, defaul 0
        eta 0-1 for efficiency (amplitude of bin) default 1
        
        modes: slow, fast
        slow more accurate (maybe?), fast uses fft

        slow method uses trapz so slightly more accurate, i think? both return similar flux values

        """
        if mode=='fast':
            dlam    = np.median(np.diff(x)) # nm per pixel, most accurate if x is uniformly sampled in wavelength
            if sig <= dlam: raise ValueError('Sigma value is smaller than the sampling of the provided wavelength array')
            nsamp   = int(sig / dlam)     # width of tophat
            tophat  = eta * np.ones(nsamp) # do i need to pad this?

            int_spec_oversample    = dlam * signal.fftconvolve(y,tophat,mode='same') # dlam integrating factor
            
            int_lam  = x[int(nsamp/2 + dx/dlam):][::nsamp] # shift over by dx/dlam (npoints) before taking every nsamp point
            int_spec =  int_spec_oversample[int(nsamp/2 + dx/dlam):][::nsamp]

        if mode=='variable':
            # mode to take variable res element
            dlam    = np.median(np.diff(x)) # nm per pixel, most accurate if x is uniformly sampled in wavelength
            if np.min(sig) <= dlam: raise ValueError('Sigma value is smaller than the sampling of the provided wavelength array')
            nsamp   = sig // dlam    # width of tophat
            nsamp = nsamp.astype('int')
            
            nsamp_unique = np.unique(nsamp)
            int_lam=np.array([])
            int_spec=np.array([])
            for n in nsamp_unique:
                isub = np.where(nsamp==n)[0]
                tophat  = eta * np.ones(n) # do i need to pad this?
                int_spec_oversample    = dlam * signal.fftconvolve(y[isub],tophat,mode='same') # dlam integrating factor
                xnew = x[isub][::n]
                ynew = int_spec_oversample[::n]
                int_lam   = np.concatenate((int_lam,xnew))
                int_spec  = np.concatenate((int_spec,ynew))

        elif mode=='slow':
            i=0
            int_lam, int_spec  = [], []
            # step through and integrate each segment
            while i*sig/2 + dx< np.max(x)-sig/2 - np.min(x): # check
                xcent    = np.min(x) + dx + i*sig/2
                tophat   = self.setup_band(x, x0=xcent, sig=sig, eta=eta) # eta throughput of whole system
                int_spec.append(self.integrate(x,tophat * y))
                int_lam.append(xcent)
                i += 1

        if mode=='pixels':
            """
            reample by binning pixels, sig in pixels now
            """
            nsamp = int(sig)
            tophat  = eta * np.ones(int(nsamp)) # do i need to pad this?

            int_spec_oversample    = signal.fftconvolve(y,tophat,mode='same') # dlam integrating factor
            
            int_lam  = x[int(nsamp//2):][::nsamp] # shift over by dx/dlam (npoints) before taking every nsamp point
            int_spec =  int_spec_oversample[int(nsamp//2):][::nsamp]

        return int_lam, int_spec
    
    def sum_total_noise(self,flux,texp, nsamp, inst_bg, sky_bg,darknoise,readnoise,npix,noisecap=None):
        """
        noise in 1 exposure

        inputs:
        --------
        flux - array [e-] 
            spectrum of star in units of electrons
        texp - float [seconds]
            exposure time, (0s,900s] (for one frame)
        nsamp - int
            number of samples in a ramp which will reduce read noise [1,inf] - 16 max for kpic
        inst_bg - array or float [e-/s]
            instrument background, if array should match sampling of flux
        sky_bg - array or float [e-/s]
            sky background, if array should match sampling of flux
        darknoise - float [e-/s/pix]
            dark noise of detector
        readnoise - float [e-/s]
            read noise of detector
        npix - float [pixels]
            number of pixels in cross dispersion of spectrum being combined into one 1D spectrum
        noisecap - float or None (default: None)
            noise cap to be applied. Defined relative to flux such that 1/noisecap is the max SNR allowed
        
        outputs:
        -------
        noise: array [e-]
            total noise sampled on flux grid
        """
        # shot noise - array w/ wavelength or integrated over band
        sig_flux = flux**(1/2)

        # background (instrument and sky) - array w/ wavelength matching flux array sampling or integrated over band
        total_bg = (inst_bg + sky_bg) * texp # per reduced pixel already so dont need to include vertical pixel extent
        sig_bg   = total_bg**(1/2) 

        # read noise  - reduces by number of ramps, limit to 6 at best
        sig_read = np.max((3,(readnoise/((nsamp)**(1/2)))))
        
        # dark current - times time and pixels
        sig_dark = (darknoise * npix * texp)**(1/2) #* get dark noise every sample
        
        noise = (flux + total_bg + npix * sig_read**2 + (darknoise * npix * texp))**(1/2)

        # cap the noise if a number is provided
        if noisecap is not None:
            noise[np.where(noise < noisecap)] = noisecap * flux # noisecap is fraction of flux, 1/noisecap gives max SNR

        return noise
    
    def observe(self):
            """
            """
            flux_per_sec_nm = self.stel_s  * self.tel_area * self.inst_ytransmit * np.abs(self.tel_s)

            
            max_ph_per_s  =  np.max(flux_per_sec_nm * self.inst_sig)
            if self.exp_time < 900: 
                self.exp_frame = np.min((self.exp_time,self.saturation/max_ph_per_s))
            else:
                self.exp_frame = np.min((900,self.saturation/max_ph_per_s))
            print('Texp per frame set to %s'%self.exp_frame)
            self.nframes = int(np.ceil(self.exp_time/self.exp_frame))
            print('Nframes set to %s'%self.nframes)
            

            # degrade to instrument resolution
            self.flux_per_nm = flux_per_sec_nm * self.exp_frame
            s_ccd_lores = self.degrade_spec(self.stel_v, self.flux_per_nm, self.inst_res)

            # resample onto res element grid
            self.obs_v, self.obs_s_frame = self.resample(self.stel_v,s_ccd_lores,sig=self.inst_sig, dx=0, eta=1,mode='variable')
            self.obs_s_frame *= 0.925 # extraction fraction, reduce photons
            self.obs_s =  self.obs_s_frame * self.nframes

            # resample throughput for applying to sky background
            base_throughput_interp= interpolate.interp1d(self.inst_xtransmit,self.inst_ytransmit)

            # load background spectrum - sky is top of telescope and will be reduced by inst BASE throughput. Coupling already accounted for in solid angle of fiber. Does inst bkg need throughput applied?
            self.obs_sky_bg_ph    = base_throughput_interp(self.obs_v) * self.get_sky_bg(x=self.obs_v)
            self.obs_inst_bg_ph   = self.get_inst_bg(x=self.obs_v)
           #  calc noise
            if self.pl_on == 1: # 3 port lantern hack
                noise_frame_yJ  = np.sqrt(3) * self.sum_total_noise(self.obs_s_frame/3,self.exp_frame, self.nsamp,self.obs_inst_bg_ph, self.obs_sky_bg_ph, self.dark,self.read,self.pix_vert)
                noise_frame     = self.sum_total_noise(self.obs_s_frame,self.exp_frame, self.nsamp,self.obs_inst_bg_ph,self.obs_sky_bg_ph,self.dark,self.read,self.pix_vert)
                yJ_sub          = np.where(self.obs_v < 1400)[0]
                noise_frame[yJ_sub] = noise_frame_yJ[yJ_sub] # fill in yj with sqrt(3) times noise in PL case
            else:
                noise_frame = self.sum_total_noise(self.obs_s_frame,self.exp_frame,self.nsamp,self.obs_inst_bg_ph,self.obs_sky_bg_ph,self.dark,self.read,self.pix_vert)
            noise_frame[np.where(np.isnan(noise_frame))] = np.inf
            noise_frame[np.where(noise_frame==0)] = np.inf

            self.obs_noise_frame = noise_frame
            self.obs_noise = np.sqrt(self.nframes)*noise_frame

            self.obs_snr = self.obs_s/self.obs_noise
            print((noise_frame))
        #    print(np.mean(sig_flux))
        #    print(np.mean(sig_bg))
        #    print(np.mean(npix))
        #    print(np.mean(sig_read))
        #    print(np.mean(sig_dark))
        #    print(np.mean(total_bg))
        #    print(np.mean(inst_bg))
        #    print(np.mean(sky_bg))
        #    print(np.mean(darknoise))
        #    print(np.mean(texp))
        #    print(np.mean(flux))

        #    print(np.max(self.obs_s))
        #    print(np.mean(self.obs_s))
        #    print(np.max(self.obs_sky_bg_ph))
        #    print(np.mean(self.obs_sky_bg_ph))
        #    print(np.max(self.obs_inst_bg_ph))
        #    print(np.mean(self.obs_inst_bg_ph))
        #    print(np.max(self.obs_noise))
        #    print(np.mean(self.obs_noise))
        #    
        #    print(np.max(self.obs_s_frame))
        #    print(np.mean(self.obs_s_frame))
        #    print(np.max(self.exp_frame))
        #    print(np.mean(self.exp_frame))
        #    print(np.max(self.nsamp))
        #    print(np.mean(self.nsamp))
        #    print(np.max(self.obs_inst_bg_ph))
        #    print(np.mean(self.obs_inst_bg_ph))
        #    print(np.max(self.obs_sky_bg_ph))
        #    print(np.mean(self.obs_sky_bg_ph))
        #    print(np.max(self.dark))
        #    print(np.mean(self.dark))
        #    print(np.max(self.read))
        #    print(np.mean(self.read))
        #    print(np.max(self.pix_vert))
        #    print(np.mean(self.pix_vert))

app = QApplication([])
spect = obs_snr_on()
spect.window.show()
app.exec_()