""" 
Define a class to extract flux and frequency information from a .fits file
"""

import warnings
from multiprocessing import Pool

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle
from radio_beam import Beam
from photutils.background import Background2D
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry

def wcs2pix(ra, dec, hdr):
    """ Convert RA, DEC to x, y pixel coordinates """
    # Drop stokes and frequency axis.
    wcs = WCS(hdr)  # Get the world coordinate system
    if hdr['NAXIS'] > 2:
        wcs = wcs.dropaxis(3)  # stokes
        wcs = wcs.dropaxis(2)  # frequency

    # Get the RA & DEC in degrees
    c = SkyCoord(ra, dec, unit=(u.hourangle, u.degree))
    ra = Angle(c.ra).degree
    dec = Angle(c.dec).degree

    # Convert RA & DEC to pixel world coordinates
    x, y = wcs.all_world2pix(ra, dec, 1)
    return [x, y]

def get_sci_exponent(number):
    """ Find the scientific exponent of a number """
    abs_num = np.abs(number)
    base = np.log10(abs_num)  # Log rules to find exponent
    exponent = int(np.floor(base))  # convert to floor integer
    return exponent

def get_eng_exponent(number):
    """ 
    Find the nearest power of 3 (lower). In engineering format,
    exponents are multiples of 3.
    """
    exponent = get_sci_exponent(number)  # Get scientific exponent
    for i in range(3):
        if exponent > 0:
            unit = exponent-i
        else:
            unit = exponent+i
        if unit % 3 == 0:  # If multiple of 3, return it
            return unit

def average_zeroes(array):
    """ Average zeroes with left & right values in a list """
    for i, val in enumerate(array):
        if val == 0:
            try:
                array[i] = (array[i-1] + array[i+1])/2
            except IndexError:
                array[i] = (array[i-2] + array[i-1])/2
    return array

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
        self._freq_exponent = None
        self._flux_exponent = None

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

    @staticmethod
    def _process_channel_data(channel, aperture, annulus, bkg_radius, pix2deg, barea):
        """ Function for processing channels in get_flux() """

        # Ignore warnings
        warnings.filterwarnings("ignore", module='photutils.background')

        # Get flux uncertainty
        aperstats = ApertureStats(channel, annulus)
        rms = aperstats.std

        # Get background flux
        bkg = Background2D(channel, bkg_radius).background

        # Calculate the sum of pixels in the aperture
        apphot = aperture_photometry(channel - bkg, aperture)
        apsum = apphot['aperture_sum'][0]

        # Calculate corrected flux
        total_flux = apsum*(pix2deg**2)/barea
        return total_flux, rms

    def get_freq(self):
        """ 
        Caclulate the frequency axis (x-axis) of the flux

        Returns
        -------
        frequency : list
            A list of frequencies corresponding to individual channels of a .fits image
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

    def get_flux(self, bkg_radius=(50, 50), beam_tolerance=1):
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

        # Parallelise slow loop to execute much faster (why background2D?!)
        inputs = [(channel, aperture, annulus, bkg_radius, pix2deg, beam_area)
                  for channel in self._data]
        with Pool() as p:
            results = p.starmap(self._process_channel_data, inputs)
        flux, flux_uncert = zip(*results)

        # Convert to np arrays
        flux = np.array(flux)
        flux_uncert = np.array(flux_uncert)

        # Average zeroes so there isn't div by zero error later
        flux_uncert = average_zeroes(flux_uncert)

        # Normalise to engineering notation
        self._flux_exponent = get_eng_exponent(np.max(flux))
        flux = flux / 10**self._flux_exponent
        flux_uncert = flux_uncert / 10**self._flux_exponent
        return flux, flux_uncert