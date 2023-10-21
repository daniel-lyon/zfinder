"""
Utility functions for zfinder
"""

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle

def wcs2pix(ra, dec, hdr):
    """ Convert RA, DEC to x, y pixel coordinates """
    # Get the RA & DEC in degrees
    c = SkyCoord(ra, dec, unit=(u.hourangle, u.degree))
    ra = Angle(c.ra).degree
    dec = Angle(c.dec).degree

    # Convert RA & DEC to pixel world coordinates
    wcs = WCS(hdr, naxis=2)
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

def radec2str(ra, dec):
    """ Convert RA and DEC to a string """
    ra = Angle(ra, unit=u.degree).to_string(unit=u.hour, sep=':', precision=4)
    dec = Angle(dec, unit=u.degree).to_string(unit=u.degree, sep=':', precision=4)
    return ra, dec

def generate_square_pix_coords(size, target_x, target_y, aperture_radius=0.5):
    """ Generate a list of coordinates for a square of pixel coordinates around a target x,y pixel """
    matrix = np.arange(size) - size//2
    matrix = matrix * aperture_radius * 2
    x, y = np.meshgrid(matrix + target_x, matrix + target_y)
    x, y = x.ravel(), y.ravel()
    return x, y

def generate_square_world_coords(fitsfile, ra, dec, size, aperture_radius):
    """ Generate a list of coordinates for a square of ra & dec around a target ra & dec """
    # Generate x, y pix coordinates around target ra & dec
    hdr = fits.getheader(fitsfile)
    target_pix_ra_dec = wcs2pix(ra, dec, hdr)
    x, y = generate_square_pix_coords(size, *target_pix_ra_dec, aperture_radius)

    # Convert x, y pix coordinates to world ra and dec
    wcs = WCS(hdr, naxis=2)
    ra, dec = wcs.all_pix2world(x, y, 1)
    ra, dec = radec2str(ra, dec)
    return ra, dec