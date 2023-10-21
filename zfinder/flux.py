""" 
Helper functions for calculating fluxes from FITS files
"""

import warnings
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
from radio_beam import Beam
from photutils.background import Background2D
from photutils.aperture import ApertureStats, aperture_photometry

def calc_beam_area(bin_hdu, tolerance=1):
    """ Caclulate the corrected beam area (from Jy/beam to Jy) """
    beam = Beam.from_fits_bintable(bin_hdu, tolerance)
    bmaj = beam.major.value / 3600  # in degrees
    bmin = beam.minor.value / 3600  # in degrees
    beam_area = 1.1331 * bmaj * bmin
    return beam_area

def mp_flux_jobs(data, aperture, annulus, bkg_radius, pix2deg, beam_area, verbose):
    """ Process the flux arrays using multiprocessing """
    with Pool() as pool:
        jobs = [pool.apply_async(process_channel_data, 
                (channel, aperture, annulus, bkg_radius, pix2deg, beam_area)) 
                for channel in data]
        results = [res.get() for res in tqdm(jobs, disable=not verbose)]
    flux, flux_uncert = zip(*results)
    return np.array(flux), np.array(flux_uncert)

def serial_flux_jobs(data, aperture, annulus, bkg_radius, pix2deg, beam_area, verbose):
    """ Process the flux arrays serially """
    flux, flux_uncert = [], []
    for channel in tqdm(data, disable=not verbose):
        f, u = process_channel_data(channel, aperture, annulus, bkg_radius, pix2deg, beam_area)
        flux.append(f)
        flux_uncert.append(u)
    flux = np.array(flux)
    flux_uncert = np.array(flux_uncert)
    return flux, flux_uncert
    
def process_channel_data(channel, aperture, annulus, bkg_radius, pix2deg, barea):
    """ Function for processing channels in get_flux() """

    # Ignore warnings
    warnings.filterwarnings("ignore", module='photutils.background')

    # Get flux uncertainty
    aperstats = ApertureStats(channel, annulus)
    flux_uncert = aperstats.std

    # Get background flux
    bkg = Background2D(channel, bkg_radius).background

    # Calculate the sum of pixels in the aperture
    apphot = aperture_photometry(channel - bkg, aperture)
    apsum = apphot['aperture_sum'][0]

    # Calculate corrected flux
    flux = apsum*(pix2deg**2)/barea
    return flux, flux_uncert