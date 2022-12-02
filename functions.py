# Required functions
import numpy as np
from astropy.wcs import WCS
from PyAstronomy import pyasl
from photutils.background import Background2D
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry

def wcs2pix(ra, dec, header):
    '''Converts the right ascension and declination to x & y pixel coordinates in a fits image

    Parameters
    ----------
    ra : list 
        The right ascension of the target

    dec : list
        The declination of the target

    header : `astropy.io.fits.header.Header`
        The header of the .fits image

    Returns
    -------
    x : int
        X-axis pixel coordinate that corresponds to ra  

    y : int
        Y-axis pixel coordinate that corresponds to dec

    '''
    w = WCS(header) # Get the world coordinate system
    
    # If there are more than 2 axis, drop them
    if header['NAXIS'] > 2:
        w = w.dropaxis(3) # stokes
        w = w.dropaxis(2) # frequency

    # Convert to decimal degrees
    ra = pyasl.hmsToDeg(ra[0], ra[1], ra[2])
    dec = pyasl.dmsToDeg(dec[0], dec[1], dec[2])

    # Convert world coordinates to pixel
    x, y = w.all_world2pix(ra, dec, 1)

    # Round to nearest integer
    x = int(np.round(x))
    y = int(np.round(y))

    return x, y

def fits_flux(image, position, radius, bvalue):
    ''' Calculates the flux around a chosen x & y position and radius

    Parameters
    ----------
    image : 'astropy.io.fits.hdu.hdulist.HDUList'
        The saved image that was opened with fits.open()
    
    position : tuple
        Pixel coordinares x & y
    
    radius : float
        The radius of the aperture around the position

    bvalue : float
        The BMAJ and BMIN values (major and minor gaussian axes) in arcseconds

    Returns
    -------
    fluxes : list
        A list of flux values at each page
    
    uncertainties : list
        A list of uncertainty values at each page

    '''
    # Get the data
    hdr = image[0].header 
    data = image[0].data[0]

    # Initialise array of fluxes and uncertainties to be returned
    fluxes = []
    uncertainties = []

    # Conversion of pixel to degrees
    pix2deg = hdr['CDELT2'] # unit conversion
    bmaj = bvalue/3600
    bmin = bmaj
    barea = 1.1331 * bmaj * bmin

    # For every page of the 3D data matrix, find the flux around a point (aperture)
    for page in data:

        # Setup the apertures 
        aperture = CircularAperture(position, radius)
        annulus = CircularAnnulus(position, r_in=2*radius, r_out=3*radius)

        # Uncertainty
        anullusstats = ApertureStats(page, annulus)
        bkg_mean = anullusstats.mean
        aperture_area = aperture.area_overlap(page)
        total_bkg = bkg_mean * aperture_area

        # Background data
        bkg = Background2D(page, (50, 50), exclude_percentile=10).background

        # Aperture sum of the fits image minus the background
        aphot = aperture_photometry(page - bkg, aperture)
        apsum = aphot['aperture_sum'][0]

        # Calculate corrected flux
        total_flux = apsum*(pix2deg**2)/barea
        fluxes.append(total_flux)
        uncertainties.append(total_bkg)

    return np.array(fluxes), np.array(uncertainties)

def gaussf(x, a, x0, y0=0):
    ''' Function to calculate a gaussian distriubtion

    Parameters
    ----------
    x : 'numpy.ndarray'
        List of x values for corresponding y values

    a : float
        The peak amplitude

    x0 : float
        The x-axis offset
    
    s : float
        The standard deviation of x
    
    y0 : float
        The y-intercept constant. Default = 0.

    Returns
    -------
    y : 'numpy.ndarray'
        List of y-values forming the distribution
    
    '''
    y = y0
    for i in range(1,11):
        y += (a * np.exp(-((x-i*x0)**2) / (1/10)*1**2)) # i = 1,2,3 ... 9, 10
    
    return y

def arrayfix(array):
    for i, val in enumerate(array):
        if val == 0:
            array[i] = (array[i-1] + array[i+1])/2
    return array