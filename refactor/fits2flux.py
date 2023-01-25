import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PyAstronomy import pyasl
from warnings import filterwarnings
from photutils.background import Background2D
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry

class fits2flux():
    def __init__(self, image, ra, dec, aperture_radius, bvalue):
        """ Calculate the flux over a frequency range of a fits image
        
        Parameters
        ----------
        image : .fits
            A .fits image file
        
        ra : list
           Right ascension of the target [h, m, s]
        
        dec : list
            Declination of the target [d, m, s, esign]
        
        aperture_radius : float
            Radius of the aperture to use over the source in pixels.

        bvalue : float
            The value of BMAJ and BMIN
        """

        self.image = fits.open(image)
        self.hdr = self.image[0].header
        self.data = self.image[0].data[0]
        self.ra = ra
        self.dec = dec
        self.aperture_radius = aperture_radius
        self.bvalue = bvalue
        filterwarnings("ignore", module='photutils.background')
        filterwarnings("ignore", module='astropy.wcs.wcs')

    @staticmethod
    def get_eng_exponent(number: float):
        """ In engineering format, exponents are multiples of 3 """

        # Find scientific exponent
        number_abs = np.abs(number)
        base = np.log10(number_abs) # Log rules to find exponent
        exponent = int(np.floor(base)) # convert to floor integer
        
        # If the exponent is a multiple of 3, return it
        for i in range(3):
            eng_exponent = exponent-i
            if eng_exponent % 3 == 0:
                return eng_exponent
    
    @staticmethod
    def wcs2pix(ra, dec, hdr):
        """ Convert right ascension and declination to x, y positional world coordinates """

        w = WCS(hdr) # Get the world coordinate system
    
        # If there are more than 2 axis, drop them
        if hdr['NAXIS'] > 2:
            w = w.dropaxis(3) # stokes
            w = w.dropaxis(2) # frequency

        # Convert to decimal degrees
        ra = pyasl.hmsToDeg(ra[0], ra[1], ra[2])
        dec = pyasl.dmsToDeg(dec[0], dec[1], dec[2], esign=dec[3])

        # Convert world coordinates to pixel
        x, y = w.all_world2pix(ra, dec, 1)

        # Round to nearest integer
        x = int(np.round(x))
        y = int(np.round(y))
        return [x, y]
    
    @staticmethod
    def average_zeroes(array):
        for i, val in enumerate(array):
            if val == 0:
                array[i] = (array[i-1] + array[i+1])/2
        return array

    def calc_freq_axis(self):
        
        # Start and increment values
        start = self.hdr['CRVAL3']
        increment = self.hdr['CDELT3']
        length = self.hdr['NAXIS3']

        # Get the exponent
        exponent = self.get_eng_exponent(start)

        # Normalise start and increment to the exponent
        norm_factor = 10**exponent
        start /= norm_factor
        increment /= norm_factor

        # Calculate end point
        end = start + length * increment

        # Create axis list
        freq_axis = np.linspace(start, end, length)
        return freq_axis

    def fits_flux(self):
        """ For every frequency channel, find the flux and associated uncertainty at a position. """

        # Initialise array of fluxes and uncertainties to be returned
        fluxes = []
        uncertainties = []

        # Pixel to degree conversion factor
        pix2deg = self.hdr['CDELT2']

        # The area of the beam
        bmaj = self.bvalue/3600
        bmin = bmaj
        barea = 1.1331 * bmaj * bmin

        # The position to find the flux at
        position = self.wcs2pix(self.ra, self.dec, self.hdr)
        
        # For every page of the 3D data matrix, find the flux around a point (aperture)
        for page in self.data:

            # Setup the apertures 
            aperture = CircularAperture(position, self.aperture_radius)
            annulus = CircularAnnulus(position, r_in=2*self.aperture_radius, r_out=3*self.aperture_radius)

            # Uncertainty
            aperstats = ApertureStats(page, annulus) 
            rms  = aperstats.std 

            # Background
            bkg = Background2D(page, (50, 50)).background 

            # Aperture sum of the fits image minus the background
            apphot = aperture_photometry(page - bkg, aperture)
            apsum = apphot['aperture_sum'][0]

            # Calculate corrected flux
            total_flux = apsum*(pix2deg**2)/barea

            # Convert from uJy to mJy
            total_flux *= 1000
            rms *= 1000

            fluxes.append(total_flux)
            uncertainties.append(rms)
        
        # Average 0's from values left & right
        uncertainties = self.average_zeroes(uncertainties) 

        return fluxes, uncertainties
    
    def get_flux_axis(self):
        freq = self.calc_freq_axis()
        flux, uncert = self.fits_flux()
        return freq, flux, uncert

def main():
    import matplotlib.pyplot as plt

    image = '0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = [8, 56, 14.8]
    dec = [2, 24, 0.6, 1]
    aperture_radius = 3
    bvalue = 3

    gleam_0856 = fits2flux(image, ra, dec, aperture_radius, bvalue)
    freq, flux, uncert = gleam_0856.get_flux_axis()

    plt.plot(freq, flux)
    plt.show()

    a = None
    b = [1,2,3]
    c = []
    print(bool(a))
    print(bool(b)) 
    print(bool(c))

if __name__ == '__main__':
    main()