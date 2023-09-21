
import numpy as np

from zfinder.fits2flux import Fits2flux
from zfinder.fft import fft_zfind

def generate_coords(size, centre_coords):
    """ Generate a list of coordinates for a square of size 'size' centred on 'centre_coords' """
    matrix = np.arange(size) - size//2
    x_coords, y_coords = np.meshgrid(matrix + centre_coords[0], matrix + centre_coords[1])
    coordinates = np.column_stack((x_coords.ravel(), y_coords.ravel()))
    return coordinates

def fft_per_pixel(transition, frequency, flux):
    z, chi2 = fft_zfind(transition, frequency, flux)
    best_z = z[np.argmin(chi2)]

def fft_per_pixel(size, z_start=0, dz=0.01, z_end=10):
    """ 
    Performs the FFT redshift finding method in a square around the 
    target right ascension and declination. Automatically saves and
    shows a heatmap of the distribution of the redshifts.
    
    Parameters
    ----------
    size : int
        size to perform the fft method around the centre. size=3 performs
        a 3x3, size=5 a 5x5, etc. 
            
    z_start : int, optional
        The first value in the redshift list. Default = 0
        
    dz : float, optional
        The change in redshift. Default = 0.01
    
    z_end : int, optional
        The final value in the redshift list. Default = 10
    """
    
    # Get the redshift
    z, chi2 = fft_zfind(z_start, dz, z_end)
    best_z = z[np.argmin(chi2)]
    centre_coords = wcs2pix(ra, dec, hdr)
    
    for y in reversed(y_coords):
        for x in x_coords:
    
            ra, dec = pix2wcs(x, y, hdr)
            
            gleam_0856 = Fits2flux(image, ra, dec, aperture_radius, bvalue)
            freq = gleam_0856.get_freq()
            flux, uncert = gleam_0856.get_flux()

            z, chi2 = fft_zfind(transition, freq, flux)

            lowest_z = z[np.argmin(chi2)]
            lowest_z = round(lowest_z, d)

            z_fft_pp.append(lowest_z)

    z_fft_pp = np.array_split(z_fft_pp, size)
    return z_fft_p