"""
Doc string
"""

import warnings

import matplotlib.pyplot as plt
from astropy.io import fits

from fits2flux import Fits2flux
from template import template_zfind
from fft import fft_zfind, fft
from per_pixel import fft_per_pixel, template_per_pixel, generate_square_world_coords, get_all_flux
from plotter import Plotter

warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in divide", category=RuntimeWarning)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Cambria'
plt.rcParams['mathtext.it'] = 'Cambria:italic'
plt.rcParams['mathtext.bf'] = 'Cambria:bold'
plt.rcParams['axes.formatter.use_mathtext'] = True

# TODO: merge fits2flux into zfinder
# TODO: on init, calculate flux and frequency?
# TODO: merge get_all_flux to Fits2flux class --> update get_flux to take in a list of ra and dec
# TODO: change bkg_radius to a single value
# TODO: remove unnecessary self attributes

class zfinder():
    """
    Doc string
    """

    def __init__(self, fitsfile, ra, dec, aperture_radius, transition, bkg_radius=(50,50), beam_tolerance=1, flux_parallel=True, showfig=True, export=True):
        self._fitsfile = fitsfile
        self._hdr = fits.getheader(fitsfile)
        self._data = fits.getdata(fitsfile)[0]
        self._ra = ra
        self._dec = dec
        self._aperture_radius = aperture_radius
        self._transition = transition
        self._bkg_radius = bkg_radius
        self._beam_tolerance = beam_tolerance
        self._flux_parallel = flux_parallel
        self._export = export
        self._plotter = Plotter(showfig=showfig)

        # Ignore warnings
        warnings.filterwarnings("ignore", module='astropy.wcs.wcs')
        warnings.filterwarnings("ignore", message='Metadata was averaged for keywords CHAN,POL', category=UserWarning)

    def _calc_freq_flux(self):
        """ Calculate the frequency and flux """
        _f2f_instance = Fits2flux(self._fitsfile, self._ra, self._dec, self._aperture_radius)
        frequency = _f2f_instance.get_freq()
        flux, flux_uncert = _f2f_instance.get_flux(self._bkg_radius, self._beam_tolerance, parallel=self._flux_parallel)
        freq_exp, flux_exp = _f2f_instance.get_exponents()
        return frequency, flux, flux_uncert, freq_exp, flux_exp

    def template(self, z_start=0, dz=0.01, z_end=10, sigma=1, verbose=True, parallel=True):
        """ Doc string here """
        # Calculate the frequency and flux
        if not hasattr(self, '_frequency'):
            self._frequency, self._flux, self._flux_uncert, \
                self._freq_exp, self._flux_exp = self._calc_freq_flux()

        # Calculate the template chi2
        z, chi2 = template_zfind(self._transition, 
            self._frequency, self._flux, self._flux_uncert, z_start, dz, z_end, verbose, parallel)

        # Plot the template chi2 and flux
        self._plotter.plot_chi2(z, dz, chi2, title='Template')
        self._plotter.plot_template_flux(self._transition, self._frequency, self._freq_exp, self._flux, self._flux_exp)
        if self._export:
            self._plotter.export_template_data(filename='template.csv', sigma=sigma, flux_uncert=self._flux_uncert)

    def template_pp(self, size, z_start=0, dz=0.01, z_end=10, aperture_radius_pp=0.5, flux_limit=0.001):
        """ 
        Performs the template redshift finding method in a square around the target ra and dec
        """
        # If the other pp method has not been run, calculate all fluxes and uncertainties
        if not hasattr(self, '_all_flux'):
            all_ra, all_dec = generate_square_world_coords(self._fitsfile, self._ra, self._dec, size, aperture_radius_pp)
            self._all_flux, self._flux_uncertainty = get_all_flux(self._fitsfile, all_ra, all_dec, aperture_radius_pp)

        # Calculate the template chi2 for each pixel
        frequency = Fits2flux(self._fitsfile, self._ra, self._dec, aperture_radius_pp).get_freq()
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
            self._frequency, self._flux, self._flux_uncert, \
                self._freq_exp, self._flux_exp = self._calc_freq_flux()
        
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
            self._all_flux, self._flux_uncertainty = get_all_flux(self._fitsfile, all_ra, all_dec, aperture_radius_pp)
        
        # Calculate the fft chi2 for each pixel
        frequency = Fits2flux(self._fitsfile, self._ra, self._dec, aperture_radius_pp).get_freq()
        z = fft_per_pixel(self._transition, frequency, self._all_flux, z_start, dz, z_end, size)
        
        # Plot the fft pp heatmap
        if self._export:
            self._plotter.export_heatmap_data('fft_per_pixel.csv', 'w', z) # export redshifts to csv
        self._plotter.plot_heatmap(ra=self._ra, dec=self._dec, hdr=self._hdr, data=self._data, size=size, \
            z=z, title='FFT', aperture_radius=aperture_radius_pp, flux_limit=flux_limit, export=self._export) # Plot the template pp heatmap