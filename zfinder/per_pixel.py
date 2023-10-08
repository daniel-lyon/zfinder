import warnings

import numpy as np
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import Angle
from multiprocessing import Pool

from zfinder.fits2flux import Fits2flux, wcs2pix
from zfinder.fft import fft_zfind
from zfinder.template import template_zfind

warnings.filterwarnings("ignore", module='astropy.wcs.wcs')

# TODO: merge get_all_flux to Fits2flux class --> update get_flux to take in a list of ra and dec

def radec2str(ra, dec):
    """ Convert RA and DEC to a string """
    ra = Angle(ra, unit=u.degree).to_string(unit=u.hour, sep=':', precision=4)
    dec = Angle(dec, unit=u.degree).to_string(unit=u.degree, sep=':', precision=4)
    return ra, dec

def generate_square_pix_coords(size, target_x, target_y, aperture_radius):
    """ Generate a list of coordinates for a square of pixel coordinates around a target pixels """
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

def _mp_all_flux(fitsfile, ra, dec, aperture_radius):
    """ Get the flux values for all ra and dec coordinates """
    flux, flux_uncert = Fits2flux(fitsfile, ra, dec, aperture_radius).get_flux(verbose=False, parallel=False)
    return flux, flux_uncert

def get_all_flux(fitsfile, ra, dec, aperture_radius):
    """ Get the flux values for all ra and dec coordinates """
    
    print('Calculating all flux values...')
    pool = Pool()
    jobs = [pool.apply_async(_mp_all_flux, (fitsfile, r, d, aperture_radius)) for r, d in zip(ra, dec)]
    pool.close()
    
    all_flux = []
    all_uncert = []
    for res in tqdm(jobs):
        flux, flux_uncert = res.get()
        all_flux.append(flux)
        all_uncert.append(flux_uncert)
    return all_flux, all_uncert

def fft_per_pixel(transition, frequency, all_flux, z_start=0, dz=0.01, z_end=10, size=3):
    """ Doc string here """
    # Calculate the chi-squared values
    print('Calculating all FFT fit chi-squared values...')
    verbose, parallel = False, False
    pool = Pool()
    jobs = [pool.apply_async(fft_zfind, (transition, frequency, flux, z_start, dz, z_end, verbose, parallel)) for flux in all_flux]
    pool.close()
    
    # Parse results
    all_z = []
    for res in tqdm(jobs):
        z, chi2 = res.get()
        all_z.append(z[np.argmin(chi2)])

    # Reshape the array
    z = np.reshape(all_z, (size, size))
    return z 

def template_per_pixel(transition, frequency, all_flux, all_flux_uncertainty, z_start=0, dz=0.01, z_end=10, size=3):
    """ Doc string here """
    # Calculate the chi-squared values
    print('Calculating all Template fit chi-squared values...')
    verbose, parallel = False, False
    pool = Pool()
    jobs = [pool.apply_async(template_zfind, (transition, frequency, flux, flux_uncert, z_start, dz, z_end, verbose, parallel)) 
        for flux, flux_uncert in zip(all_flux, all_flux_uncertainty)]
    pool.close()
    
    # Parse results
    all_z = []
    for res in tqdm(jobs):
        z, chi2 = res.get()
        all_z.append(z[np.argmin(chi2)])

    # Reshape the array
    z = np.reshape(all_z, (size, size))
    return z     