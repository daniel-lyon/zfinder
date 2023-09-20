"""
Doc string
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt

from zfinder.fits2flux import Fits2flux
from zfinder.template import template_zfind, find_lines, gaussf, calc_template_params
from zfinder.fft import fft_zfind, double_damped_sinusoid, calc_fft_params, fft

warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in divide", category=RuntimeWarning)

class zfinder():
    """
    Doc string
    """

    def __init__(self, fitsfile, ra, dec, aperture_radius, transition, bkg_radius=(50,50), beam_tolerance=1):
        self._filename = fitsfile
        self._ra = ra
        self._dec = dec
        self._aperture_radius = aperture_radius
        self._transition = transition
        self._bkg_radius = bkg_radius
        self._beam_tolerance = beam_tolerance

        # Ignore warnings
        warnings.filterwarnings("ignore", module='astropy.wcs.wcs')
        warnings.filterwarnings("ignore", message='Metadata was averaged for keywords CHAN,POL', category=UserWarning)
    
    def _plot_sslf_lines(self):
        """ Helper function to plot sslf lines in the found flux """
        peaks, snrs, scales = find_lines(self._flux)
        text_offset_high = max(self._flux)/20
        text_offset_low = 0.4*text_offset_high
        
        for i, line in enumerate(peaks):
            x = self._frequency[line]
            y = self._flux[line]
            plt.plot(x, y, 'bo')
            plt.text(x, y+text_offset_high, f'snr={snrs[i]}', color='blue')
            plt.text(x, y+text_offset_low, f'scale={scales[i]}', color='blue')
    
    def _plot_template_chi2(self):
        min_chi2 = min(self._template_chi2)
        self._template_best_z = self._template_z[np.argmin(self._template_chi2)]
        plt.figure(figsize=(15,7))
        plt.plot(self._template_z, self._template_chi2, color='black')
        plt.plot(self._template_best_z, min_chi2, 'bo', markersize=5)
        plt.title(f'Template $\chi^2_r$ = {round(min_chi2, 2)} @ z={self._template_best_z}', fontsize=15)
        plt.xlabel('Redshift', fontsize=15)
        plt.ylabel('$\chi^2_r$', x=0.01, fontsize=15)
        plt.yscale('log')
        plt.savefig('template_chi2.png')
        plt.show()

    def _plot_template_flux(self):
        x0 = self._transition/(1+self._template_best_z)
        params = calc_template_params(self._frequency, self._flux, x0)
        plt.figure(figsize=(15,7))
        plt.plot(self._frequency, np.zeros(len(self._frequency)), color='black', linestyle=(0, (5, 5)))
        plt.plot(self._frequency, self._flux, color='black', drawstyle='steps-mid')
        plt.plot(self._frequency, gaussf(self._frequency, *params, x0), color='red')
        self._plot_sslf_lines()
        plt.margins(x=0)
        plt.fill_between(self._frequency, self._flux, 0, where=(np.array(self._flux) > 0), color='gold', alpha=0.75)
        plt.title(f'Template Fit z={self._template_best_z}', fontsize=15)
        plt.xlabel(f'Frequency $({_unit_prefixes[self._freq_exp]}Hz)$', fontsize=15)
        plt.ylabel(f'Flux $({_unit_prefixes[self._flux_exp]}Jy)$', fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig('template_flux.png')
        plt.show()

    def _plot_fft_chi2(self):
        min_chi2 = min(self._fft_chi2)
        self._fft_best_z = self._fft_z[np.argmin(self._fft_chi2)]
        plt.figure(figsize=(15,7))
        plt.plot(self._fft_z, self._fft_chi2, color='black')
        plt.plot(self._fft_best_z, min_chi2, 'bo', markersize=5)
        plt.title(f'FFT $\chi^2_r$ = {round(min_chi2, 2)} @ z={self._fft_best_z}', fontsize=15)
        plt.xlabel('Redshift', fontsize=15)
        plt.ylabel('$\chi^2_r$', x=0.01, fontsize=15)
        plt.yscale('log')
        plt.savefig('fft_chi2.png')
        plt.show()
    
    def _plot_fft_flux(self):
        params = calc_fft_params(self._transition, self._ffreq, 
            self._fflux, self._fft_best_z, self._frequency[0])
        plt.figure(figsize=(15,7))
        plt.plot(self._ffreq, self._fflux, color='black', drawstyle='steps-mid')
        plt.plot(self._ffreq, np.zeros(len(self._fflux)), color='black', linestyle=(0, (5, 5)))
        plt.plot(self._ffreq, double_damped_sinusoid(self._ffreq, *params, 
            self._fft_best_z, self._frequency[0], self._transition), color='red')
        plt.margins(x=0)
        plt.title(f'FFT Fit z={self._fft_best_z}', fontsize=15)
        plt.xlabel('Scale', fontsize=15)
        plt.ylabel('Amplitude', fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig('fft_flux.png')
        plt.show()

    def _calc_freq_flux(self):
        """ Calculate the frequency and flux """
        _f2f_instance = Fits2flux(self._filename, self._ra, self._dec, self._aperture_radius)
        self._frequency = _f2f_instance.get_freq()
        self._flux, self._flux_uncert = _f2f_instance.get_flux(self._bkg_radius, self._beam_tolerance)
        self._freq_exp, self._flux_exp = _f2f_instance.get_exponents()

    def template(self, z_start=0, dz=0.01, z_end=10):
        """ Doc string here """

        # Calculate the frequency and flux
        if not hasattr(self, '_frequency'):
            self._calc_freq_flux()

        # Calculate the template chi2
        self._template_z, self._template_chi2 = template_zfind(self._transition, self._frequency, self._flux, self._flux_uncert, z_start, dz, z_end)

        # Plot the template chi2 and flux
        self._plot_template_chi2()
        self._plot_template_flux()

    def fft(self, z_start=0, dz=0.01, z_end=10):
        """ Doc string here """

        # Calculate the frequency and flux
        if not hasattr(self, '_frequency'):
            self._calc_freq_flux()
        
        # Calculate the fft frequency and flux
        self._ffreq, self._fflux = fft(self._frequency, self._flux)

        # Calculate the fft chi2
        self._fft_z, self._fft_chi2 = fft_zfind(self._transition, self._frequency, self._flux, z_start, dz, z_end)

        # Plot the fft chi2 and flux
        self._plot_fft_chi2()
        self._plot_fft_flux()

# Get the prefix of a unit from an exponent
_unit_prefixes = {
    -24 : 'y', 
    -21 : 'z',
    -18 : 'a',
    -15 : 'f',
    -12 : 'p',
    -9 : 'n',
    -6 : '\u03BC',
    -3 : 'm',
    0 : '', 
    3 : 'k',
    6 : 'M',
    9 : 'G',
    12 : 'T',
    15 : 'P',
    18 : 'E', 
    21 : 'Z',
    24 : 'Y'}

def main():
    fitsfile = 'zfinder/0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = '08:56:14.8'
    dec = '02:24:00.6'
    aperture_radius = 2
    transition = 115.2712

    source = zfinder(fitsfile, ra, dec, aperture_radius, transition)
    source.template()
    source.fft()
    
if __name__ == '__main__':
    main()