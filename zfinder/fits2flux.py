""" 
Define a class to extract flux and frequency information from a .fits file
"""

import warnings
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
from radio_beam import Beam
from astropy.io import fits
from photutils.background import Background2D
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry

from utils import wcs2pix, get_eng_exponent, average_zeroes

class Fits2flux():
    """
    Easily calculate and extract the frequency and flux data based on 
    specified coordinates and aperture radius.

    Parameters
    ----------
    fitsfile : .fits
        fits image filename

    ra : str, list
        Right Ascension of the target.

    dec : str, list
        Declination of the target

    aperture_radius : float
        Radius of the aperture to use over the source in pixels

    Methods
    -------
    get_freq:
        Caclulate the frequency axis (Hz)

    get_flux:
        Caclulate the flux axis and uncertainty (Jy)
        
    Example
    -------
    ```python
    fitsfile = '0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = '08:56:14.8'
    dec = '02:24:00.6'
    aperture_radius = 3

    image = Fits2flux(fitsfile, ra, dec, aperture_radius)
    freq = image.get_freq()
    flux, flux_uncert = image.get_flux()
    ```
    """

    def __init__(self, fitsfile, ra, dec, aperture_radius):
        self._hdr = fits.getheader(fitsfile)
        self._data = fits.getdata(fitsfile)[0]
        self._bin_hdu = fits.open(fitsfile)[1]
        self._ra = ra
        self._dec = dec
        self._aperture_radius = aperture_radius

        # Ignore warnings
        warnings.filterwarnings("ignore", module='astropy.wcs.wcs')
        warnings.filterwarnings("ignore", category=UserWarning)

    @staticmethod
    def _calc_beam_area(bin_hdu, tolerance=1):
        """ Caclulate the corrected beam area (from Jy/beam to Jy) """
        beam = Beam.from_fits_bintable(bin_hdu, tolerance)
        bmaj = beam.major.value / 3600  # in degrees
        bmin = beam.minor.value / 3600  # in degrees
        beam_area = 1.1331 * bmaj * bmin
        return beam_area
    
    def _mp_flux_jobs(self, aperture, annulus, bkg_radius, pix2deg, beam_area, verbose):
        """ Process the flux arrays using multiprocessing """
        with Pool() as pool:
            jobs = [pool.apply_async(_process_channel_data, 
                    (channel, aperture, annulus, bkg_radius, pix2deg, beam_area)) 
                    for channel in self._data]
            results = [res.get() for res in tqdm(jobs, disable=not verbose)]
        flux, flux_uncert = zip(*results)
        return np.array(flux), np.array(flux_uncert)
    
    def _serial_flux_jobs(self, aperture, annulus, bkg_radius, pix2deg, beam_area, verbose):
        """ Process the flux arrays serially """
        flux, flux_uncert = [], []
        for channel in tqdm(self._data, disable=not verbose):
            f, u = _process_channel_data(channel, aperture, annulus, bkg_radius, pix2deg, beam_area)
            flux.append(f)
            flux_uncert.append(u)
        flux = np.array(flux)
        flux_uncert = np.array(flux_uncert)
        return flux, flux_uncert

    def get_freq(self):
        """ 
        Caclulate the frequency axis list (x-axis) of the flux
        """
        # Get frequency axis
        start = self._hdr['CRVAL3']
        increment = self._hdr['CDELT3']
        length = self._hdr['NAXIS3']
        end = start + length * increment

        # Create frequency axis
        frequency = np.linspace(start, end, length)

        # Normalise to engineering notation
        self._freq_exponent = get_eng_exponent(frequency[0])
        frequency = frequency / 10**self._freq_exponent
        return frequency

    def get_flux(self, bkg_radius=(50, 50), beam_tolerance=1, verbose=True, parallel=True):
        """ 
        For every frequency channel, find the flux and associated uncertainty at a position

        Paramters
        ---------
        bkg_radius : tuple, optional
            The radius of which to find the background flux. Default=(50,50).

        beam_tolerance : int, optional
            The tolerance of the differences between multiple beams. Default=1.

        Returns
        -------
        flux : list
            A list of flux values from each frequency channel

        f_uncert : list
            A list of flux uncertainty values for each flux measurement        
        """

        # Calculate area of the beam
        beam_area = self._calc_beam_area(self._bin_hdu, beam_tolerance)
        pix2deg = self._hdr['CDELT2']  # Pixel to degree conversion factor

        # The position to find the flux at
        position = wcs2pix(self._ra, self._dec, self._hdr)

        # Setup the apertures
        inner_radius = 2*self._aperture_radius
        outter_radius = 3*self._aperture_radius
        aperture = CircularAperture(position, self._aperture_radius)
        annulus = CircularAnnulus(position, inner_radius, outter_radius)

        # Process the flux arrays
        if verbose:
            print('Calculating flux values...')
        if parallel:
            flux, flux_uncert = self._mp_flux_jobs(aperture, annulus, bkg_radius, pix2deg, beam_area, verbose)
        else:
            flux, flux_uncert = self._serial_flux_jobs(aperture, annulus, bkg_radius, pix2deg, beam_area, verbose)

        # Average zeroes so there isn't div by zero error later
        flux_uncert = average_zeroes(flux_uncert)

        # Normalise to engineering notation
        self._flux_exponent = get_eng_exponent(np.max(flux))
        flux = flux / 10**self._flux_exponent
        flux_uncert = flux_uncert / 10**self._flux_exponent
        return flux, flux_uncert
    
    def get_exponents(self):
        """ Return the exponents of the flux and frequency """
        return self._freq_exponent, self._flux_exponent
    
def _process_channel_data(channel, aperture, annulus, bkg_radius, pix2deg, barea):
    """ Function for processing channels in get_flux() """

    # Ignore warnings
    warnings.filterwarnings("ignore", module='photutils.background')

    # Get flux uncertainty
    aperstats = ApertureStats(channel, annulus)
    flux_uncert = aperstats.std

    # Get background flux
    bkg = Background2D(channel, bkg_radius).background

    # Calculate the sum of pixels in the aperture
    apphot = aperture_photometry(channel - bkg, aperture)
    apsum = apphot['aperture_sum'][0]

    # Calculate corrected flux
    flux = apsum*(pix2deg**2)/barea
    return flux, flux_uncert