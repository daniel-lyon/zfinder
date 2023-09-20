import pytest

import numpy as np

from zfinder.fits2flux import Fits2flux
from zfinder.template import template_zfind

def test_template():
    """ Test the template method """

    # Load in the data
    fitsfile = 'zfinder/0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = '08:56:14.8'
    dec = '02:24:00.6'
    aperture_radius = 3
    transition = 115.2712

    # Calculate the frequency and flux
    source = Fits2flux(fitsfile, ra, dec, aperture_radius)
    freq = source.get_freq()
    flux, flux_uncert = source.get_flux()

    # Calculate the template chi2
    z, chi2 = template_zfind(transition, freq, flux, flux_uncert)

    # Find the best fit redshift
    lowest_index = np.argmin(chi2)
    best_fit_redshift = z[lowest_index]

    assert best_fit_redshift == pytest.approx(5.55)

if __name__ == '__main__':
    test_template()