"""
Doc string
"""

import warnings
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.aperture import CircularAperture, CircularAnnulus
from tqdm import tqdm

from flux import calc_beam_area, mp_flux_jobs, serial_flux_jobs
from template import template_zfind, template_per_pixel
from fft import fft_zfind, fft, fft_per_pixel
from plotter import Plotter
from utils import get_eng_exponent, average_zeroes, wcs2pix, generate_square_world_coords

warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in divide", category=RuntimeWarning)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Cambria'
plt.rcParams['mathtext.it'] = 'Cambria:italic'
plt.rcParams['mathtext.bf'] = 'Cambria:bold'
plt.rcParams['axes.formatter.use_mathtext'] = True

# TODO: on init, calculate flux and frequency?
# TODO: remove unnecessary self attributes

class zfinder():
    """
    Doc string
    """

    def __init__(self, fitsfile, ra, dec, aperture_radius, transition, bkg_radius=50, beam_tolerance=1, showfig=True, export=True):
        self._fitsfile = fitsfile
        self._hdr = fits.getheader(fitsfile)
        self._data = fits.getdata(fitsfile)[0]
        self._bin_hdu = fits.open(fitsfile)[1]
        self._ra = ra
        self._dec = dec
        self._aperture_radius = aperture_radius
        self._transition = transition
        self._bkg_radius = bkg_radius
        self._beam_tolerance = beam_tolerance
        self._export = export
        self._plotter = Plotter(showfig=showfig)

        # Ignore warnings
        warnings.filterwarnings("ignore", module='astropy.wcs.wcs')
        warnings.filterwarnings("ignore", message='Metadata was averaged for keywords CHAN,POL', category=UserWarning)
    
    def get_freq(self):
        """ 
        Caclulate the frequency axis list (x-axis) of the flux
        """
        # Get frequency axis
        start = self._hdr['CRVAL3']
        increment = self._hdr['CDELT3']
        length = self._hdr['NAXIS3']
        end = start + length * increment

        # Create frequency axis
        frequency = np.linspace(start, end, length)

        # Normalise to engineering notation
        self._freq_exponent = get_eng_exponent(frequency[0])
        frequency = frequency / 10**self._freq_exponent
        return frequency
    
    def get_flux(self, verbose=True, parallel=True):
        """ 
        For every frequency channel, find the flux and associated uncertainty at a position

        Paramters
        ---------
        verbose : bool, optional
            If True, print progress. Default=True
        
        parallel : bool, optional
            If True, use multiprocessing. Default=True

        Returns
        -------
        flux : list
            A list of flux values from each frequency channel

        f_uncert : list
            A list of flux uncertainty values for each flux measurement        
        """

        # Calculate area of the beam
        beam_area = calc_beam_area(self._bin_hdu, self._beam_tolerance)
        pix2deg = self._hdr['CDELT2']  # Pixel to degree conversion factor

        # The position to find the flux at
        position = wcs2pix(self._ra, self._dec, self._hdr)

        # Setup the apertures
        inner_radius = 2*self._aperture_radius
        outter_radius = 3*self._aperture_radius
        aperture = CircularAperture(position, self._aperture_radius)
        annulus = CircularAnnulus(position, inner_radius, outter_radius)

        # Process the flux arrays
        if verbose:
            print('Calculating flux values...')
        if parallel:
            flux, flux_uncert = mp_flux_jobs(self._data, aperture, annulus, (self._bkg_radius, self._bkg_radius), pix2deg, beam_area, verbose)
        else:
            flux, flux_uncert = serial_flux_jobs(self._data, aperture, annulus, (self._bkg_radius, self._bkg_radius), pix2deg, beam_area, verbose)

        # Average zeroes so there isn't div by zero error later
        flux_uncert = average_zeroes(flux_uncert)

        # Normalise to engineering notation
        self._flux_exponent = get_eng_exponent(np.max(flux))
        flux = flux / 10**self._flux_exponent
        flux_uncert = flux_uncert / 10**self._flux_exponent
        return flux, flux_uncert

    def get_all_flux(self, all_ra, all_dec, aperture_radius_pp):
        """ Get the flux values for all ra and dec coordinates """
        print('Calculating all flux values...')       
        with Pool() as pool:
            jobs = [pool.apply_async(_mp_all_flux, (self._fitsfile, r, d, aperture_radius_pp, \
                self._transition, self._bkg_radius, self._beam_tolerance)) for r, d in zip(all_ra, all_dec)]
            results = [res.get() for res in tqdm(jobs)]
        all_flux, all_uncert = zip(*results)
        return all_flux, all_uncert

    def _calc_freq_flux(self, verbose, parallel):
        """ Calculate the frequency and flux """
        frequency = self.get_freq()
        flux, flux_uncert = self.get_flux(verbose, parallel)
        return frequency, flux, flux_uncert

    def template(self, z_start=0, dz=0.01, z_end=10, sigma=1, verbose=True, parallel=True):
        """ Doc string here """
        # Calculate the frequency and flux
        if not hasattr(self, '_frequency'):
            self._frequency, self._flux, self._flux_uncert = self._calc_freq_flux(verbose, parallel)

        # Calculate the template chi2
        z, chi2 = template_zfind(self._transition, self._frequency, self._flux, self._flux_uncert, z_start, dz, z_end, verbose, parallel)

        # Plot the template chi2 and flux
        self._plotter.plot_chi2(z, dz, chi2, title='Template')
        self._plotter.plot_template_flux(self._transition, self._frequency, self._freq_exponent, self._flux, self._flux_exponent)
        if self._export:
            self._plotter.export_template_data(filename='template.csv', sigma=sigma, flux_uncert=self._flux_uncert)

    def template_pp(self, size, z_start=0, dz=0.01, z_end=10, aperture_radius_pp=0.5, flux_limit=0.001):
        """ 
        Performs the template redshift finding method in a square around the target ra and dec
        """
        # If the other pp method has not been run, calculate all fluxes and uncertainties
        if not hasattr(self, '_all_flux'):
            all_ra, all_dec = generate_square_world_coords(self._fitsfile, self._ra, self._dec, size, aperture_radius_pp)
            self._all_flux, self._flux_uncertainty = self.get_all_flux(self._fitsfile, all_ra, all_dec, aperture_radius_pp)

        # Calculate the template chi2 for each pixel
        frequency = self.get_freq()
        z = template_per_pixel(self._transition, frequency, self._all_flux, self._flux_uncertainty, z_start, dz, z_end, size)
        
        # Plot the template pp heatmap
        if self._export:
            self._plotter.export_heatmap_data('template_per_pixel.csv', 'w', z) # export redshifts to csv
        self._plotter.plot_heatmap(ra=self._ra, dec=self._dec, hdr=self._hdr, data=self._data, size=size, \
            z=z, title='Template', aperture_radius=aperture_radius_pp, flux_limit=flux_limit, export=self._export) # Plot the template pp heatmap

    def fft(self, z_start=0, dz=0.01, z_end=10, sigma=1, verbose=True, parallel=True):
        """ Doc string here """
        # Calculate the frequency and flux
        if not hasattr(self, '_frequency'):
            self._frequency, self._flux, self._flux_uncert = self._calc_freq_flux(verbose, parallel)
        
        # Calculate the fft frequency and flux
        self._ffreq, self._fflux = fft(self._frequency, self._flux)

        # Calculate the fft chi2
        z, chi2 = fft_zfind(self._transition, self._frequency, self._flux, z_start, dz, z_end, verbose, parallel)

        # Plot the fft chi2 and flux
        self._plotter.plot_chi2(z, dz, chi2, title='FFT')
        self._plotter.plot_fft_flux(self._transition, self._frequency, self._ffreq, self._fflux)
        if self._export:
            self._plotter.export_fft_data(filename='fft.csv', sigma=sigma)

    def fft_pp(self, size, z_start=0, dz=0.01, z_end=10, aperture_radius_pp=0.5, flux_limit=0.001):
        """ 
        Performs the fft redshift finding method in a square around the target ra and dec
        """      
        # If the other pp method has not been run, calculate all fluxes and uncertainties
        if not hasattr(self, '_all_flux'):
            all_ra, all_dec = generate_square_world_coords(self._fitsfile, self._ra, self._dec, size, aperture_radius_pp)
            self._all_flux, self._flux_uncertainty = self.get_all_flux(all_ra, all_dec, aperture_radius_pp)
        
        # Calculate the fft chi2 for each pixel
        frequency = self.get_freq()
        z = fft_per_pixel(self._transition, frequency, self._all_flux, z_start, dz, z_end, size)

        # Plot the fft pp heatmap
        if self._export:
            self._plotter.export_heatmap_data('fft_per_pixel.csv', 'w', z) # export redshifts to csv
        self._plotter.plot_heatmap(ra=self._ra, dec=self._dec, hdr=self._hdr, data=self._data, size=size, \
            z=z, title='FFT', aperture_radius=aperture_radius_pp, flux_limit=flux_limit, export=self._export) # Plot the template pp heatmap
        
def _mp_all_flux(fitsfile, ra, dec, aperture_radius, transition, bkg_radius, beam_tolerance):
    """ Get the flux values for a single ra and dec coordinate """
    flux, flux_uncert = zfinder(fitsfile, ra, dec, aperture_radius, transition, bkg_radius, beam_tolerance).get_flux(verbose=False, parallel=False)
    return flux, flux_uncert