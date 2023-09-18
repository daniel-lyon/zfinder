"""
Doc string
"""

import numpy as np
import matplotlib.pyplot as plt

from fits2flux import Fits2flux
from zfind_template import zfind_template

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

class zfinder(Fits2flux):
    """
    Doc string
    """

    def __init__(self, fitsfile, ra, dec, aperture_radius, transition, bkg_radius=(50,50), beam_tolerance=1):
        super().__init__(fitsfile, ra, dec, aperture_radius)
        self._transition = transition
        self._bkg_radius = bkg_radius
        self._beam_tolerance = beam_tolerance

        # self.zfind_template = lambda z_start=0, dz=0.01, z_end=10: zfind_template(self._transition, self._frequency, self._flux, self._flux_uncert, z_start, dz, z_end)
    
    def _get_freq(self):
        return super().get_freq()
    
    def _get_flux(self):
        return super().get_flux(self._bkg_radius, self._beam_tolerance)
    
    def _calc_freq_flux(self):
        self._frequency = self._get_freq()
        self._flux, self._flux_uncert = self._get_flux()
    
    def plot_template_flux(self):
        if not hasattr(zfinder, '_frequency'):
            self._calc_freq_flux()
        plt.figure(figsize=(15,7))
        plt.plot(self._frequency, np.zeros(len(self._frequency)), color='black', linestyle=(0, (5, 5)))
        plt.plot(self._frequency, self._flux, color='black', drawstyle='steps-mid')
        plt.margins(x=0)
        plt.fill_between(self._frequency, self._flux, 0, where=(np.array(self._flux) > 0), color='gold', alpha=0.75)
        plt.title(f'Template Fit z=', fontsize=15)
        plt.xlabel(f'Frequency $({_unit_prefixes[self._freq_exponent]}Hz)$', fontsize=15)
        plt.ylabel(f'Flux $({_unit_prefixes[self._flux_exponent]}Jy)$', fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

def main():
    fitsfile = 'zfinder/0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = '08:56:14.8'
    dec = '02:24:00.6'
    aperture_radius = 3
    transition = 115.2712

    source = zfinder(fitsfile, ra, dec, aperture_radius, transition)
    source.plot_template_flux()
    # source.zfind_template(z_end=2)
    
if __name__ == '__main__':
    main()