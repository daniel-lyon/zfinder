import datetime
import numpy as np

from time import time
from random import random
from astropy.io import fits
from astropy.wcs import WCS
from flux_zfind import find_lines
from warnings import filterwarnings
from scipy.spatial.distance import cdist
from photutils.background import Background2D
from PyAstronomy.pyasl import hmsToDeg, dmsToDeg, degToHMS, degToDMS
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry

def wcs2pix(ra, dec, hdr):
    """ Convert right ascension and declination to x, y positional world coordinates """

    w = WCS(hdr) # Get the world coordinate system

    # If there are more than 2 axis, drop them
    if hdr['NAXIS'] > 2:
        w = w.dropaxis(3) # stokes
        w = w.dropaxis(2) # frequency

    # Convert to decimal degrees
    ra = hmsToDeg(ra[0], ra[1], ra[2])
    dec = dmsToDeg(dec[0], dec[1], dec[2], esign=dec[3])

    # Convert world coordinates to pixel
    x, y = w.all_world2pix(ra, dec, 1)

    # Round to nearest integer
    x = int(np.round(x))
    y = int(np.round(y))
    return [x, y]

def pix2wcs(x, y, hdr):
    """ Convert x,y positional coordinates to world ra, dec"""
    # Drop redundant axis
    w = WCS(hdr)
    if hdr['NAXIS'] > 2:
        w = w.dropaxis(3) # stokes
        w = w.dropaxis(2) # frequency

    # Convert x,y to degrees
    ra, dec = w.all_pix2world(x, y, 1)

    # Convert degrees to ra, dec
    ra = degToHMS(ra)
    dec = degToDMS(dec)
    return ra, dec

def get_sci_exponent(number):
    """ Find the scientific exponent of a number """
    abs_num = np.abs(number)
    base = np.log10(abs_num) # Log rules to find exponent
    exponent = int(np.floor(base)) # convert to floor integer
    return exponent

def get_eng_exponent(number):
    """ 
    Find the nearest power of 3 (lower). In engineering format,
    exponents are multiples of 3.
    """
    
    # First get the scientific exponent
    exponent = get_sci_exponent(number)
    
    # If the exponent is a multiple of 3, return it
    for i in range(3):
        
        if exponent > 0:
            unit = exponent-i
        else:
            unit = exponent+i
            
        if unit % 3 == 0:
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

def _calc_beam_area(bvalue):
    """ Caclulate the corrected beam area (from Jy/beam to Jy) """
    bmaj = bvalue/3600
    bmin = bmaj
    barea = 1.1331 * bmaj * bmin
    return barea

def spaced_circle_points(num_points, circle_radius, centre_coords, minimum_spread_distance):
    """ Calculate points in a circle with a minimum spread """
    
    points = [centre_coords] # centre_coords = [x,y] -> points = [[x,y]]
    
    # Iterate through the number of points.
    for _ in range(num_points-1):
        
        # Keep generating the current point until it is at least the minimum distance away from all 
        while True:
            theta = 2 * np.pi * random() # choose a random direction
            r = circle_radius * random() # choose a random radius

            # Convert coordinates to cartesian
            x = r * np.cos(theta) + centre_coords[0]
            y = r * np.sin(theta) + centre_coords[1]

            # Find the distance between all the placed points
            distances = cdist([[x,y]], points, 'euclidean')
            min_distance = min(distances[0])
            
            # If the minimum distance is satisfied for all points, go to next point
            if min_distance >= minimum_spread_distance or len(points) == 1:
                points.append([x,y])
                break
    return points

def calc_pixels(spec_peaks, flux):
    """ 
    Calculate the number of pixels above 0 Jy left & right of all lines found 
    by the sslf line finder.
    """
    pixels = []
    for peak in spec_peaks:
        num_pix = 0

        # left side
        i = 1
        try:
            while flux[peak-i] > 0:
                num_pix += 1
                i += 1
        except IndexError:
            pass

        # right side
        i = 0
        try:
            while flux[peak+i] > 0:
                num_pix += 1
                i += 1
        except IndexError:
            pass

        pixels.append(num_pix)
    return pixels

class fits2flux(object):
    def __init__(self, image, ra, dec, aperture_radius, bvalue):
        """ 
        Calculate the flux over the frequency range of a .fits image
        
        Parameters
        ----------
        image : .fits
            A .fits image file
        
        ra : list
           Right ascension of the target [h, m, s]
        
        dec : list
            Declination of the target [d, m, s, esign]
        
        aperture_radius : float
            Radius of the aperture to use over the source (pixels)

        bvalue : float
            The value of BMAJ and BMIN (arcseconds)
        """
        image = fits.open(image)
        self.hdr = image[0].header
        self.data = image[0].data[0]
        self.ra = ra
        self.dec = dec
        self.aperture_radius = aperture_radius
        self.bvalue = bvalue

        # Ignore warnings (there are hundreds)
        filterwarnings("ignore", module='photutils.background')
        filterwarnings("ignore", module='astropy.wcs.wcs')

    def get_freq(self):
        """ 
        Caclulate the frequency axis (x-axis) of the flux

        Returns
        -------
        frequency : list
            A list of frequencies corresponding to individual channels of a .fits image
        """
        
        # Required values
        start = self.hdr['CRVAL3']
        increment = self.hdr['CDELT3']
        length = self.hdr['NAXIS3']

        # Get the exponent
        self.xexponent = get_eng_exponent(start)

        # Normalise start and increment to the exponent
        norm_factor = 10**self.xexponent
        start /= norm_factor
        increment /= norm_factor

        # Calculate end point
        end = start + length * increment

        # Create axis list
        frequency = np.linspace(start, end, length)
        return frequency

    def get_flux(self):
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
        barea = _calc_beam_area(self.bvalue)
        pix2deg = self.hdr['CDELT2'] # Pixel to degree conversion factor

        # The position to find the flux at
        position = wcs2pix(self.ra, self.dec, self.hdr)
        
        # For every page of the 3D data matrix, find the flux around a point (aperture)
        for channel in self.data:

            # Setup the apertures 
            aperture = CircularAperture(position, self.aperture_radius)
            annulus = CircularAnnulus(position, r_in=2*self.aperture_radius, r_out=3*self.aperture_radius)

            # Uncertainty
            aperstats = ApertureStats(channel, annulus) 
            rms  = aperstats.std 

            # Background
            bkg = Background2D(channel, (50, 50)).background 

            # Aperture sum of the fits image minus the background
            apphot = aperture_photometry(channel - bkg, aperture)
            apsum = apphot['aperture_sum'][0]

            # Calculate corrected flux
            total_flux = apsum*(pix2deg**2)/barea

            # Save values
            flux.append(total_flux)
            f_uncert.append(rms)

        # Average 0's from values left & right
        f_uncert = average_zeroes(f_uncert) 
        
        # Normalising factor
        self.yexponent = get_eng_exponent(np.max(flux))
        norm_factor = 10**-self.yexponent

        flux = np.array(flux)*norm_factor
        f_uncert = np.array(f_uncert)*norm_factor
        return flux, f_uncert
    
    def random_analysis(self, num_points=100, radius=50, min_spread=1):
        """ 
        Iterate through n randomly generated coordinates to find the 
        signal-to-noise ratio, number of pixels, and channel peaks for each.
        
        Parameters
        ----------
        num_points : int, optional
            The number of points to find statistics for. Default = 100
        
        radius : float, optional
            The radius of the image to find statistics (in pixels) centred on
            the given ra and dec. Default = 50
        
        min_spread : float, optional
            The minimum spread of random points (in pixels). Default = 1
        
        Returns
        -------
        snrs : list
            A list of signifcant point signal-to-noise ratios
            
        pixels : list
            The list of pixels for each random position
        
        peaks : list
            The channel location of all peaks in every position
        """
        
        # Initialise return arrays
        all_snrs = []
        all_pixels = []
        all_spec_peaks = []

        # Find x,y coordinates of the target
        x, y = wcs2pix(self.ra, self.dec, self.hdr)

        # Calculate the coordinates for the number of points
        self.coordinates = spaced_circle_points(num_points, radius, [x,y], min_spread)

        # Use sslf to find lines on all points
        for i, (x, y) in enumerate(self.coordinates):
            start = time()
            self.ra, self.dec = pix2wcs(x,y, self.hdr) # convert x, y to ra,dec

            flux, uncert = self.get_flux()
            
            spec_peaks, snrs, scales = find_lines(flux) # use flux axis to find lines on
            
            pixels = calc_pixels(spec_peaks, flux)

            all_snrs.append(snrs)
            all_pixels.append(pixels)
            all_spec_peaks.append(spec_peaks)
            
            end = time()
            elapsed = end - start 
            remaining = datetime.timedelta(seconds=round(elapsed*(num_points-(i+1))))
            
            print(f'{i+1}/{len(self.coordinates)}, took {round(elapsed,2)} seconds, approx {remaining} remaining')

        return all_snrs, all_pixels, all_spec_peaks
    
    def get_exponents(self):
        """ get the x and y factors that were used to normalise the data. 
            -9 = nano, -6 = micro, -3 = milli, 3 = kilo, 6 = mega, 9 = giga,
            etc, etc, etc.
        
        Returns
        -------
        x, y : int
            The x and y normalisation factors
        """
        return self.xexponent, self.yexponent