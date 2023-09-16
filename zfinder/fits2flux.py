# Import packages
import numpy as np
import warnings

from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle
from radio_beam import Beam

from photutils.background import Background2D
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry

from multiprocessing import Pool

def wcs2pix(ra, dec, hdr):
    """ Convert RA, DEC to x, y pixel coordinates """
    
    # Drop stokes and frequency axis.
    w = WCS(hdr) # Get the world coordinate system
    if hdr['NAXIS'] > 2:
        w = w.dropaxis(3) # stokes
        w = w.dropaxis(2) # frequency

    # Get the RA & DEC in degrees
    c = SkyCoord(ra, dec, unit=(u.hourangle, u.degree))
    ra = Angle(c.ra).degree
    dec = Angle(c.dec).degree
    
    # Convert RA & DEC to pixel world coordinates
    x, y = w.all_world2pix(ra, dec, 1)
    return [x, y]

class Fits2flux():
    """
    Easily calculate and extract the frequency and flux data based on specified coordinates and aperture radius.
    
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
        bmaj = beam.major.value / 3600 # in degrees
        bmin = beam.minor.value / 3600 # in degrees
        beam_area = 1.1331 * bmaj * bmin
        return beam_area
    
    @staticmethod
    def _process_channel_data(channel, aperture, annulus, bkg_radius, pix2deg, barea):
        """ Function for processing channels in get_flux() """

        # Filter the hundreds of photutils.background.background_2d warnings
        warnings.filterwarnings("ignore", module='photutils.background')

        # Uncertainty
        aperstats = ApertureStats(channel, annulus) 
        rms  = aperstats.std 

        # Background
        bkg = Background2D(channel, bkg_radius).background 

        # Aperture sum of the fits image minus the background
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
        
        # Required values
        start = self._hdr['CRVAL3']
        increment = self._hdr['CDELT3']
        length = self._hdr['NAXIS3']

        # Calculate end point
        end = start + length * increment

        # Create axis list
        frequency = np.linspace(start, end, length)
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

        # Initialise array of fluxes and uncertainties to be returned
        flux = []
        f_uncert = []
        
        # Calculate area of the beam
        barea = self._calc_beam_area(self._bin_hdu, beam_tolerance)
        pix2deg = self._hdr['CDELT2'] # Pixel to degree conversion factor

        # The position to find the flux at
        position = wcs2pix(self._ra, self._dec, self._hdr)

        # Setup the apertures 
        aperture = CircularAperture(position, self._aperture_radius)
        annulus = CircularAnnulus(position, r_in=2*self._aperture_radius, r_out=3*self._aperture_radius)      

        # Parallelise slow loop to execute much faster (why background2D?!)
        inputs = [(channel, aperture, annulus, bkg_radius, pix2deg, barea) for channel in self._data]
        with Pool() as p:
            results = p.starmap(self._process_channel_data, inputs)
        flux, f_uncert = zip(*results)
        
        flux = np.array(flux)
        f_uncert = np.array(f_uncert)
        return flux, f_uncert

def main():
    fitsfile = 'zfinder/0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = '08:56:14.8'
    dec = '02:24:00.6'
    aperture_radius = 3

    image = Fits2flux(fitsfile, ra, dec, aperture_radius)
    freq = image.get_freq()
    flux, flux_uncert = image.get_flux()

    print(freq)
    print()
    print(flux)
    print()
    print(flux_uncert)


if __name__ == '__main__':
    main()