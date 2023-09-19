import numpy as np
import matplotlib.pyplot as plt

from zfinder.fits2flux import Fits2flux

def test_fits2flux():
    fitsfile = 'zfinder/0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = '08:56:14.8'
    dec = '02:24:00.6'
    aperture_radius = 3

    source = Fits2flux(fitsfile, ra, dec, aperture_radius)
    freq = source.get_freq()
    flux, flux_uncert = source.get_flux()

    plt.figure(figsize=(15,7))
    plt.plot(freq, np.zeros(len(freq)), color='black', linestyle=(0, (5, 5)))
    plt.plot(freq, flux, color='black', drawstyle='steps-mid')
    plt.margins(x=0)
    plt.fill_between(freq, flux, 0, where=(np.array(flux) > 0), color='gold', alpha=0.75)
    plt.title(f'Template Fit', fontsize=15)
    plt.xlabel('Frequency $(Hz)$', fontsize=15)
    plt.ylabel('Flux $(Jy)$', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

if __name__ == '__main__':
    test_fits2flux()