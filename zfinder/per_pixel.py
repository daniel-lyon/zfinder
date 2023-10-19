import warnings

import numpy as np
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from multiprocessing import Pool

from fits2flux import Fits2flux
from fft import fft_zfind
from template import template_zfind
from utils import wcs2pix, radec2str, generate_square_pix_coords

warnings.filterwarnings("ignore", module='astropy.wcs.wcs')

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
    """ Get the flux values for a single ra and dec coordinate """
    flux, flux_uncert = Fits2flux(fitsfile, ra, dec, aperture_radius).get_flux(verbose=False, parallel=False)
    return flux, flux_uncert

def get_all_flux(fitsfile, all_ra, all_dec, aperture_radius):
    """ Get the flux values for all ra and dec coordinates """
    print('Calculating all flux values...')       
    with Pool() as pool:
        jobs = [pool.apply_async(_mp_all_flux, (fitsfile, r, d, aperture_radius)) for r, d in zip(all_ra, all_dec)]
        results = [res.get() for res in tqdm(jobs)]
    all_flux, all_uncert = zip(*results)
    return all_flux, all_uncert

def _process_per_pixel_results(results, size):
    """ Process multiprocessing results """
    all_z = [z[np.argmin(chi2)] for z, chi2 in results]
    z = np.reshape(all_z, (size, size))
    return z

def fft_per_pixel(transition, frequency, all_flux, z_start=0, dz=0.01, z_end=10, size=3):
    """ Perform the FFT fitting method on all flux values """
    print('Calculating all FFT fit chi-squared values...')
    verbose, parallel = False, False
    with Pool() as pool:
        jobs = [pool.apply_async(fft_zfind, (transition, frequency, flux, z_start, dz, z_end, verbose, parallel)) for flux in all_flux]
        results = [res.get() for res in tqdm(jobs)]
    z = _process_per_pixel_results(results, size)
    return z

def template_per_pixel(transition, frequency, all_flux, all_flux_uncertainty, z_start=0, dz=0.01, z_end=10, size=3):
    """ Perform the Template fitting method on all flux values """
    print('Calculating all Template fit chi-squared values...')
    verbose, parallel = False, False
    with Pool() as pool:
        jobs = [pool.apply_async(template_zfind, (transition, frequency, flux, flux_uncert, z_start, dz, z_end, verbose, parallel)) 
            for flux, flux_uncert in zip(all_flux, all_flux_uncertainty)]
        results = [res.get() for res in tqdm(jobs)]
    z = _process_per_pixel_results(results, size)
    return z