"""
Doc string
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt

from zfinder.fits2flux import Fits2flux
from zfinder.zfind_template import template_zfind

warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide", category=RuntimeWarning)

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
    
    def _plot_template_chi2(self):
        min_chi2 = min(self._template_chi2)
        min_z = self._template_z[np.argmin(self._template_chi2)]
        plt.figure(figsize=(15,7))
        plt.plot(self._template_z, self._template_chi2, color='black')
        plt.plot(min_z, min_chi2, 'bo', markersize=5)
        plt.title(f'Template $\chi^2_r$ = {round(min_chi2, 2)} @ z={min_z}', fontsize=15)
        plt.xlabel('Redshift', fontsize=15)
        plt.ylabel('$\chi^2_r$', x=0.01, fontsize=15)
        plt.yscale('log')
        plt.show()

    def _plot_template_flux(self):
        plt.figure(figsize=(15,7))
        plt.plot(self._frequency, np.zeros(len(self._frequency)), color='black', linestyle=(0, (5, 5)))
        plt.plot(self._frequency, self._flux, color='black', drawstyle='steps-mid')
        plt.margins(x=0)
        plt.fill_between(self._frequency, self._flux, 0, where=(np.array(self._flux) > 0), color='gold', alpha=0.75)
        plt.title(f'Template Fit z=', fontsize=15)
        plt.xlabel(f'Frequency $({_unit_prefixes[self._f2f_instance._freq_exponent]}Hz)$', fontsize=15)
        plt.ylabel(f'Flux $({_unit_prefixes[self._f2f_instance._flux_exponent]}Jy)$', fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
    
    def template(self, z_start=0, dz=0.01, z_end=10):

        # Calculate the flux and frequency
        self._f2f_instance = Fits2flux(self._filename, self._ra, self._dec, self._aperture_radius)
        self._frequency = self._f2f_instance.get_freq()
        self._flux, self._flux_uncert = self._f2f_instance.get_flux(self._bkg_radius, self._beam_tolerance)

        # Calculate the template chi2
        self._template_z, self._template_chi2 = template_zfind(self._transition, self._frequency, self._flux, self._flux_uncert, z_start, dz, z_end)

        # Plot the template chi2 and flux
        self._plot_template_chi2()
        self._plot_template_flux()


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
    aperture_radius = 3
    transition = 115.2712

    source = zfinder(fitsfile, ra, dec, aperture_radius, transition)
    source.template(dz=0.01)
    
if __name__ == '__main__':
    main()