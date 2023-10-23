"""
Class for finding the redshift of a source in a FITS file. Exports data to csv files and plots figures.
"""

import warnings
from multiprocessing import Pool
from itertools import zip_longest

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, CircularAnnulus

from zfinder.flux import calc_beam_area, mp_flux_jobs, serial_flux_jobs
from zfinder.template import template_zfind, template_per_pixel, find_lines
from zfinder.fft import fft_zfind, fft, fft_per_pixel
from zfinder.plotter import Plotter
from zfinder.utils import get_eng_exponent, average_zeroes, wcs2pix, generate_square_world_coords, radec2str, gen_random_coords
from zfinder.uncertainty import z_uncert

warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in divide", category=RuntimeWarning)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Cambria'
plt.rcParams['mathtext.it'] = 'Cambria:italic'
plt.rcParams['mathtext.bf'] = 'Cambria:bold'
plt.rcParams['axes.formatter.use_mathtext'] = True

class zfinder():
    """ 
    zfinder is used to find the redshift of a fits image via two different methods: 
    
    `template` fits gaussian functions to the data to find redshift and
    
    `zfft` performs the fast fourier transform on the flux data to find redshift. 
    
    These methods will create and save a series of plots and csv files with raw data
    by default. Can be changed with the `showfig` and `export` parameters.
    
    If running in a `.py` file, you may need to add the following to your code.
    Otherwise, jupyter notebooks should work fine.
    
    ```python
    if __name__ == '__main__':
        source = zfinder(fitsfile, ra, dec, aperture_radius, transition)
        ...
    ```

    Parameters
    ----------
    fitsfile : `.fits`
        A .fits image file
    
    ra : list
        Right ascension of the target [h, m, s]
    
    dec : list
        Declination of the target [d, m, s, esign]
    
    aperture_radius : float
        Radius of the aperture to use over the source (pixels)

    transition : float
        The first transition frequency of the target element or molecule (GHz)
        
    bkg_radius : float, optional
        Radius of the background annulus (pixels). Default=50
    
    beam_tolerance : float, optional
        The tolerance of the beam area (arcsec). Default=1
    
    showfig : bool, optional
        If True, show the figures. Default=True
    
    savefig : bool, optional
        If True, save the figures. Default=True
    
    export : bool, optional
        If True, export the data to csv files. Default=True
        
    Methods
    -------
    get_freq()
        Caclulate the frequency axis
    
    get_flux()
        For every frequency channel, find the flux and associated uncertainty at a position
    
    get_all_flux()
        Get the flux values for all ra and dec coordinates
    
    template()
        Performs the template redshift finding method
    
    template_pp()
        Performs the template redshift finding method in a square around the target ra and dec
    
    fft()
        Performs the fft redshift finding method
    
    fft_pp()
        Performs the fft redshift finding method in a square around the target ra and dec
    
    fft_uncert()
        Calculates the uncertainty on the fft flux. 
    
    Examples
    --------
    >>> from zfinder import zfinder
    >>>
    >>> fitsfile = '0856_cube_c0.4_nat_80MHz_taper3.fits'
    >>> ra = '08:56:14.8'
    >>> dec = '02:24:00.6'
    >>> aperture_radius = 3
    >>> transition = 115.2712
    >>> z_start = 0
    >>> dz = 0.001
    >>> z_end = 10
    >>> 
    >>> source = zfinder(fitsfile, ra, dec, aperture_radius, transition)
    >>> source.template(z_start, dz, z_end)
    >>> source.fft(z_start, dz, z_end)
    >>>
    >>> # Once redshift is found, narrow down the redshift range and run per pixel methods
    >>> aperture_radius_pp = 0.5
    >>> size = 15
    >>> z_start = 5.4
    >>> dz = 0.001
    >>> z_end = 5.7
    >>> source.fft_pp(size, z_start, dz, z_end, aperture_radius_pp)
    >>> source.template_pp(size, z_start, dz, z_end, aperture_radius_pp)
    """
    
    def __init__(self, fitsfile, ra, dec, aperture_radius, transition, bkg_radius=50, beam_tolerance=1, showfig=True, savefig=True, export=True):
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
        self._plotter = Plotter(showfig=showfig, savefig=savefig)

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

    def get_all_flux(self, all_ra, all_dec, aperture_radius=None):
        """ 
        Get the flux values for all ra and dec coordinates
        
        Paramters
        ---------
        all_ra : list
            A list of all ra coordinates
            
        all_dec : list
            A list of all dec coordinates
        
        aperture_radius : float
            Radius of the aperture to use over the source (pixels). If None, use the aperture_radius
        """
        if aperture_radius is None:
            aperture_radius = self._aperture_radius
        print('Calculating all flux values...')       
        with Pool() as pool:
            jobs = [pool.apply_async(_mp_all_flux, (self._fitsfile, r, d, aperture_radius, \
                self._transition, self._bkg_radius, self._beam_tolerance)) for r, d in zip(all_ra, all_dec)]
            results = [res.get() for res in tqdm(jobs)]
        all_flux, all_uncert = zip(*results)
        return all_flux, all_uncert

    def _calc_freq_flux(self, verbose, parallel):
        """ Calculate the frequency and flux """
        frequency = self.get_freq()
        flux, flux_uncert = self.get_flux(verbose, parallel)
        return frequency, flux, flux_uncert
    
    def _z_uncert(self, z, chi2, sigma):
        """ Caclulate the uncertainty on the best fitting redshift """
        peaks, _, _ = find_lines(self._flux)
        reduced_sigma = sigma**2 / (len(self._flux) - 2*len(peaks) - 1)
        neg, pos = z_uncert(z, chi2, reduced_sigma)
        return neg, pos

    def template(self, z_start=0, dz=0.001, z_end=10, sigma=1, verbose=True, parallel=True):
        """ 
        Perform the template redshift finding method
        
        Parameters
        ----------
        z_start : float, optional
            The starting redshift to search from. Default=0
            
        dz : float, optional
            The step size of the redshift search. Default=0.01
        
        z_end : float, optional
            The ending redshift to search to. Default=10
        
        sigma : float, optional
            The significance level to calculate the redshift uncertainty. Default=1
        
        verbose : bool, optional
            If True, print progress. Default=True
        
        parallel : bool, optional
            If True, use multiprocessing. Default=True
        
        Returns
        -------
        z : float
            Best fit redshift 
        
        z_uncert : tuple(float, float)
            Lower and upper uncertainty on the best fit redshift
        """
        # Calculate the frequency and flux
        if not hasattr(self, '_frequency'):
            self._frequency, self._flux, self._flux_uncert = self._calc_freq_flux(verbose, parallel)

        # Calculate the template chi2
        z, chi2 = template_zfind(self._transition, self._frequency, self._flux, self._flux_uncert, z_start, dz, z_end, verbose, parallel)

        # Plot the template chi2 and flux
        self._plotter.plot_chi2(z, dz, chi2, title='Template')
        self._plotter.plot_template_flux(self._transition, self._frequency, self._freq_exponent, self._flux, self._flux_exponent)
        if self._export:
            self._plotter._export_template_data(self._fitsfile, self._ra, self._dec, self._aperture_radius, self._transition, 
                filename='template.csv', sigma=sigma, flux_uncert=self._flux_uncert)
        neg, pos = self._z_uncert(z, chi2, sigma)
        return z[np.argmin(chi2)], (neg, pos)

    def template_pp(self, size, z_start=0, dz=0.001, z_end=10, aperture_radius_pp=0.5, flux_limit=0.001, contfile=None):
        """ 
        Performs the template redshift finding method in a square around the target ra and dec.
        
        Not recommended for very large redshift search ranges. Remember to narrow down the redshift
        range before using this method. Recommended to have 100-300 redshifts to check per pixel.
        
        z_start = 5.4, dz = 0.001, z_end = 5.7 is a good range for GLEAM J0856.
        
        z_start = 4.28, dz = 0.0001, z_end = 4.31 is a good range for SPT 0345-47.
        
        Parameters
        ----------
        size : int
            The size of a square (centred on ra and dec) to calculate redshifts in (pixels). size=15 is 15x15 pixels
        
        z_start : float, optional
            The starting redshift to search from. Default=0
        
        dz : float, optional
            The step size of the redshift search. Default=0.01
        
        z_end : float, optional
            The ending redshift to search to. Default=10
        
        aperture_radius_pp : float, optional
            Radius of the aperture to use over the source (pixels). Default=0.5
        
        flux_limit : float, optional
            The minimum flux value to calculate redshifts for. Default=0.001
            If contfile is specified, this is the minimum continuum flux.
        
        contfile : str, optional
            The fits file containing the continuum data. Default=None
        """
        # If the other pp method has not been run, calculate all fluxes and uncertainties
        if not hasattr(self, '_all_flux'):
            all_ra, all_dec = generate_square_world_coords(self._fitsfile, self._ra, self._dec, size, aperture_radius_pp)
            self._all_flux, self._flux_uncertainty = self.get_all_flux(all_ra, all_dec, aperture_radius_pp)            

        # Calculate the template chi2 for each pixel
        frequency = self.get_freq()
        z = template_per_pixel(self._transition, frequency, self._all_flux, self._flux_uncertainty, z_start, dz, z_end, size)
        
        # Plot the template pp heatmap
        if self._export:
            header = ['fitsfile', 'ra', 'dec', 'aperture_radius', 'transition', 'size', 'flux_limit', 'contfile']
            data = [[self._fitsfile, self._ra, self._dec, aperture_radius_pp, self._transition, size, flux_limit, contfile], ['']]
            self._plotter._write_method_to_csv('template_per_pixel.csv', header, data)
            self._plotter._export_heatmap_data('template_per_pixel.csv', 'a', z) # export redshifts to csv
        self._plotter.plot_heatmap(ra=self._ra, dec=self._dec, hdr=self._hdr, data=self._data, size=size, \
            z=z, title='Template', aperture_radius=aperture_radius_pp, flux_limit=flux_limit, export=self._export, contfile=contfile) # Plot the template pp heatmap

    def fft(self, z_start=0, dz=0.001, z_end=10, sigma=1, verbose=True, parallel=True, uncertainty=False):
        """ 
        Perform the fft redshift finding method
        
        Parameters
        ----------
        z_start : float, optional
            The starting redshift to search from. Default=0
            
        dz : float, optional
            The step size of the redshift search. Default=0.001
        
        z_end : float, optional
            The ending redshift to search to. Default=10
        
        sigma : float, optional
            The significance level to calculate the redshift uncertainty. Default=1
            
        verbose : bool, optional
            If True, print progress. Default=True
        
        parallel : bool, optional
            If True, use multiprocessing. Default=True
        
        uncertainty : list, bool, optional
            A list of the uncertainty values calculated by `zfinder.fft_uncert`. 
            If True, read the uncertainty values from the csv file.
            If False, no uncertainty values will be used for redshift calculation
        
        Returns
        -------
        z : float
            Best fit redshift
        
        z_uncert : tuple(float, float)
            Lower and upper uncertainty on the best fit redshift
        """
        # Calculate the frequency and flux
        if not hasattr(self, '_frequency'):
            self._frequency, self._flux, self._flux_uncert = self._calc_freq_flux(verbose, parallel)
        
        # Calculate the fft frequency and flux
        self._ffreq, self._fflux = fft(self._frequency, self._flux)
        
        if uncertainty is False:
            uncertainty = 1
        if uncertainty is True:
            uncertainty = self.read_fft_uncert()

        # Calculate the fft chi2
        z, chi2 = fft_zfind(self._transition, self._frequency, self._flux, uncertainty, z_start, dz, z_end, verbose, parallel)

        # Plot the fft chi2 and flux
        self._plotter.plot_chi2(z, dz, chi2, title='FFT')
        self._plotter.plot_fft_flux(self._transition, self._frequency, self._ffreq, self._fflux)
        if self._export:
            self._plotter._export_fft_data(self._fitsfile, self._ra, self._dec, self._aperture_radius, self._transition, 
                filename='fft.csv', sigma=sigma, frequency=self._frequency, flux=self._flux, flux_uncert=uncertainty)
        neg, pos = self._z_uncert(z, chi2, sigma)
        return z[np.argmin(chi2)], (neg, pos)

    def fft_pp(self, size, z_start=0, dz=0.01, z_end=10, aperture_radius_pp=0.5, flux_limit=0.001, contfile=None):
        """ 
        Performs the fft redshift finding method in a square around the target ra and dec
        
        Not recommended for very large redshift search ranges. Remember to narrow down the redshift
        range before using this method. Recommended to have 100-300 redshifts to check per pixel.
        
        z_start = 5.4, dz = 0.001, z_end = 5.7 is a good range for GLEAM J0856.
        
        z_start = 4.28, dz = 0.0001, z_end = 4.31 is a good range for SPT 0345-47.
        
        Parameters
        ----------
        size : int
            The size of a square (centred on ra and dec) to calculate redshifts in (pixels). size=15 is 15x15 pixels
            
        z_start : float, optional
            The starting redshift to search from. Default=0
        
        dz : float, optional
            The step size of the redshift search. Default=0.01
        
        z_end : float, optional
            The ending redshift to search to. Default=10
        
        aperture_radius_pp : float, optional
            Radius of the aperture to use over the source (pixels). Default=0.5
        
        flux_limit : float, optional
            The minimum flux value to calculate redshifts for. Default=0.001
            If contfile is specified, this is the minimum continuum flux.
        
        contfile : str, optional
            The fits file containing the continuum data. Default=None
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
            headings = ['fitsfile', 'ra', 'dec', 'aperture_radius', 'transition', 'size', 'flux_limit', 'contfile']
            data = [[self._fitsfile, self._ra, self._dec, aperture_radius_pp, self._transition, size, flux_limit, contfile], ['']]
            self._plotter._write_method_to_csv('fft_per_pixel.csv', headings, data)
            self._plotter._export_heatmap_data('fft_per_pixel.csv', 'a', z) # export redshifts to csv
        self._plotter.plot_heatmap(ra=self._ra, dec=self._dec, hdr=self._hdr, data=self._data, size=size, \
            z=z, title='FFT', aperture_radius=aperture_radius_pp, flux_limit=flux_limit, export=self._export, contfile=contfile) # Plot the template pp heatmap
    
    def fft_uncert(self, n=100, radius=50, min_spread=1):
        """ Doc string here """
        # Find x,y coordinates of the target
        wcs = WCS(self._hdr, naxis=2)
        center_x, center_y = wcs2pix(self._ra, self._dec, self._hdr)
        x, y = gen_random_coords(n, radius, [center_x, center_y], min_spread)
        # self._x_coords, self._y_coords = x, y
            
        # Convert x, y pix coordinates to world ra and dec
        ra, dec = wcs.all_pix2world(x, y, 1)
        all_ra, all_dec = radec2str(ra, dec)

        # Get the flux values for all ra and dec coordinates
        all_flux, _ = self.get_all_flux(all_ra, all_dec, self._aperture_radius)
        freq = self.get_freq()
        
        # Calculate the FFT for each flux
        all_fflux = np.transpose([fft(freq, flux)[1] for flux in all_flux])
        
        # Calculate the standard deviation of all the fflux channels
        all_std = np.std(all_fflux, axis=1)
        
        # column headers
        field_names = ['std', 'x_centre', 'y_centre', 'x', 'y', 'ra', 'dec', 'n', 'radius', 'min_spread', 'fitsfile']
        dtype = [(name, object) for name in field_names]
        data = np.array(list(zip_longest(
            all_std, [center_x], [center_y], x, y, all_ra, all_dec, [n], [radius], 
            [min_spread], [self._fitsfile], fillvalue='')), dtype=dtype)
            
        # Write the data to the csv
        if self._export:
            np.savetxt('.csv', data, delimiter=',', fmt='%s', header=','.join(data.dtype.names))
        self._plotter.plot_coords(center_x, center_y, x, y, radius, fitsfile=self._fitsfile)
        return all_std
    
    @staticmethod
    def read_fft_uncert(filename='fft_uncertainty.csv'):
        """ Read the fft uncertainty csv file """
        std = np.genfromtxt(filename, delimiter=',', usecols=(0)).T
        return std
        
def _mp_all_flux(fitsfile, ra, dec, aperture_radius, transition, bkg_radius, beam_tolerance):
    """ Get the flux values for a single ra and dec coordinate """
    flux, flux_uncert = zfinder(fitsfile, ra, dec, aperture_radius, transition, bkg_radius, beam_tolerance).get_flux(verbose=False, parallel=False)
    return flux, flux_uncert