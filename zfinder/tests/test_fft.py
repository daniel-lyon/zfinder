import pytest
    
import numpy as np
import matplotlib.pyplot as plt

from zfinder.fits2flux import Fits2flux
from zfinder.fft import fft_zfind

def test_fft():
    fitsfile = 'zfinder/0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = '08:56:14.8'
    dec = '02:24:00.6'
    aperture_radius = 3
    transition = 115.2712

    source = Fits2flux(fitsfile, ra, dec, aperture_radius)
    freq = source.get_freq()
    flux, flux_uncert = source.get_flux()

    z, chi2 = fft_zfind(transition, freq, flux)

    lowest_index = np.argmin(chi2)
    best_fit_redshift = z[lowest_index]
    print(best_fit_redshift)

    assert best_fit_redshift == pytest.approx(5.55)

    plt.plot(z, chi2)
    plt.show()

if __name__ == '__main__':
    test_fft()