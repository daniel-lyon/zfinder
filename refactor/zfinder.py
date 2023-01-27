import numpy as np
import matplotlib.pyplot as plt

from zfft import zfft
from zflux import zflux
from fits2flux import fits2flux
from uncertainty import uncertainty
from line_statistics import line_statistics

class zfinder():
    def __init__(self, image, ra, dec, transition, aperture_radius, bvalue):
        self.image = image
        self.ra = ra
        self.dec = dec
        self.transition = transition
        self.aperture_radius = aperture_radius
        self.bvalue = bvalue

    @staticmethod
    def get_unit(exponent):

        # A dictionary of exponent and unit prefix pairs
        prefix = {-24 : 'y', -21 : 'z', -18 : 'a',
                  -15 : 'f', -12 : 'p',  -9 : 'n',
                   -6 : 'mu', -3 : 'm',   0 : '', 
                    3 : 'k',   6 : 'M',   9 : 'G',
                   12 : 'T',  15 : 'P',  18 : 'E', 
                   21 : 'Z', 24 : 'Y'}

        return prefix[exponent]
    
    def plot_flux(self):
        symbol = self.get_unit(self.source.exponent)
        plt.plot(self.frequency, np.zeros(len(self.frequency)), color='black', linestyle=(0, (5, 5)))
        plt.plot(self.frequency, self.flux, color='black', drawstyle='steps-mid')
        plt.margins(x=0)
        plt.fill_between(self.frequency, self.flux, 0, where=(np.array(self.flux) > 0), color='gold', alpha=0.75)
        plt.xlabel(f'Frequency $({symbol}Hz)$')
        plt.ylabel('Flux $(mJy)$')
        plt.show()

    def fits2flux(self):
        self.source = fits2flux(self.image, self.ra, self.dec, self.aperture_radius, self.bvalue)
        self.frequency = self.source.get_freq()
        self.flux, uncert = self.source.get_flux()
        self.plot_flux()

def main():
    image = '0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = [8, 56, 14.8]
    dec = [2, 24, 0.6, 1]
    transition = 115.2712
    aperture_radius = 3
    bvalue = 3

    gleam_0856 = zfinder(image, ra, dec, transition, aperture_radius, bvalue)
    gleam_0856.fits2flux()

if __name__ == '__main__':
    main()