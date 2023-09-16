# Import packages
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from radio_beam import Beam
from astropy import units as u
from warnings import filterwarnings
from photutils.background import Background2D
from astropy.coordinates import SkyCoord, Angle
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry

# Ignore warnings (there are hundreds)
filterwarnings("ignore", module='photutils.background')
filterwarnings("ignore", module='astropy.wcs.wcs')

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
    def __init__(self, fitsfile, ra, dec, aperture_radius):
        self._hdr = fits.getheader(fitsfile)
        self._data = fits.getdata(fitsfile)[0]
        self._bin_hdu = fits.open(fitsfile)[1]
        self._ra = ra
        self._dec = dec
        self._aperture_radius = aperture_radius


    @staticmethod
    def __calc_beam_area(bin_hdu, tolerance=1):
        """ Caclulate the corrected beam area (from Jy/beam to Jy) """

        # Get 
        beam = Beam.from_fits_bintable(bin_hdu, tolerance)
        bmaj = beam.major.value / 3600
        bmin = beam.minor.value / 3600

        # bmaj = bvalue/3600
        barea = 1.1331 * bmaj * bmin
        return barea

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
    
    def get_flux(self, bkg_radius=(50, 50)):
        """ 
        For every frequency channel, find the flux and associated uncertainty at a position
        
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
        barea = self.__calc_beam_area(self._bin_hdu)
        pix2deg = self._hdr['CDELT2'] # Pixel to degree conversion factor

        # The position to find the flux at
        position = wcs2pix(self._ra, self._dec, self._hdr)

        # Setup the apertures 
        aperture = CircularAperture(position, self._aperture_radius)
        annulus = CircularAnnulus(position, r_in=2*self._aperture_radius, r_out=3*self._aperture_radius)
    
        # For every page of the 3D data matrix, find the flux around a point (aperture)
        for channel in self._data:

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

            # Save values
            flux.append(total_flux)
            f_uncert.append(rms)
        
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