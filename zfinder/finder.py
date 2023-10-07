"""
Doc string
"""

import csv
import warnings
from itertools import zip_longest

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

from fits2flux import Fits2flux, wcs2pix
from template import template_zfind, find_lines, gaussf, calc_template_params
from fft import fft_zfind, double_damped_sinusoid, calc_fft_params, fft
from per_pixel import fft_per_pixel, template_per_pixel, \
    generate_square_pix_coords, generate_square_world_coords, get_all_flux
from uncertainty import z_uncert

warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in divide", category=RuntimeWarning)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Cambria'
plt.rcParams['mathtext.it'] = 'Cambria:italic'
plt.rcParams['mathtext.bf'] = 'Cambria:bold'
plt.rcParams['axes.formatter.use_mathtext'] = True

# TODO: Add uncertainty for fft

class zfinder():
    """
    Doc string
    """

    def __init__(self, fitsfile, ra, dec, aperture_radius, transition, bkg_radius=(50,50), beam_tolerance=1):
        self._fitsfile = fitsfile
        self._ra = ra
        self._dec = dec
        self._aperture_radius = aperture_radius
        self._transition = transition
        self._bkg_radius = bkg_radius
        self._beam_tolerance = beam_tolerance

        # Ignore warnings
        warnings.filterwarnings("ignore", module='astropy.wcs.wcs')
        warnings.filterwarnings("ignore", message='Metadata was averaged for keywords CHAN,POL', category=UserWarning)
    
    def _plot_sslf_lines(self):
        """ Helper function to plot sslf lines in the found flux """
        peaks, snrs, scales = find_lines(self._flux)
        text_offset_high = max(self._flux)/20
        text_offset_low = 0.4*text_offset_high
        
        for i, line in enumerate(peaks):
            x = self._frequency[line]
            y = self._flux[line]
            plt.plot(x, y, 'bo')
            plt.text(x, y+text_offset_high, f'snr={snrs[i]}', color='blue')
            plt.text(x, y+text_offset_low, f'scale={scales[i]}', color='blue')
    
    def _plot_chi2(self, title):
        """ Plot the chi-sqaured vs redshift """
        min_chi2 = min(self._chi2)
        self._best_z = self._z[np.argmin(self._chi2)]
        self._round_to = len(str(self._dz).split('.')[1])
        plt.figure(figsize=(15,7))
        plt.plot(self._z, self._chi2, color='black')
        plt.plot(self._best_z, min_chi2, 'bo', markersize=5)
        plt.title(f'{title} $\chi^2_r$ = {round(min_chi2, 2)} @ z={round(self._best_z, self._round_to)}', fontsize=15)
        plt.xlabel('Redshift', fontsize=15)
        plt.ylabel('$\chi^2_r$', x=0.01, fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.yscale('log')
        plt.savefig(f'{title.lower()}_chi2.png')
        plt.show()

    def _plot_template_flux(self):
        """ Plot the template flux """
        x0 = self._transition/(1+self._best_z)
        self._params, covars = calc_template_params(self._frequency, self._flux, x0)
        self._p_err = np.sqrt(np.diag(covars)) # calculate the error on the parameters
        plt.figure(figsize=(15,7))
        plt.plot(self._frequency, np.zeros(len(self._frequency)), color='black', linestyle=(0, (5, 5)))
        plt.plot(self._frequency, self._flux, color='black', drawstyle='steps-mid')
        plt.plot(self._frequency, gaussf(self._frequency, *self._params, x0), color='red')
        self._plot_sslf_lines()
        plt.margins(x=0)
        plt.fill_between(self._frequency, self._flux, 0, where=(np.array(self._flux) > 0), color='gold', alpha=0.75)
        plt.title(f'Template Fit z={round(self._best_z, self._round_to)}', fontsize=15)
        plt.xlabel(f'Frequency $({_unit_prefixes[self._freq_exp]}Hz)$', fontsize=15)
        plt.ylabel(f'Flux $({_unit_prefixes[self._flux_exp]}Jy)$', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig('template_flux.png')
        plt.show()
    
    def _plot_fft_flux(self):
        """ Plot the fft flux """
        self._params, covars = calc_fft_params(self._transition, self._ffreq, 
            self._fflux, self._best_z, self._frequency[0])
        self._p_err = np.sqrt(np.diag(covars)) # calculate the error on the parameters
        plt.figure(figsize=(15,7))
        plt.plot(self._ffreq, self._fflux, color='black', drawstyle='steps-mid')
        plt.plot(self._ffreq, np.zeros(len(self._fflux)), color='black', linestyle=(0, (5, 5)))
        plt.plot(self._ffreq, double_damped_sinusoid(self._ffreq, *self._params, 
            self._best_z, self._frequency[0], self._transition), color='red')
        plt.margins(x=0)
        plt.title(f'FFT Fit z={round(self._best_z, self._round_to)}', fontsize=15)
        plt.xlabel('Scale', fontsize=15)
        plt.ylabel('Amplitude', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig('fft_flux.png')
        plt.show()   

    def _plot_heatmap(self, z, title, aperture_radius):
        # Calculate the velocities
        target_z = np.take(z, z.size // 2) # redshift of the target ra and dec
        velocities = 3*10**5*((((1 + target_z)**2 - 1) / ((1 + target_z)**2 + 1)) - (((1 + z)**2 - 1) / ((1 + z)**2 + 1))) # km/s
        scale_velo = np.max(np.abs(velocities))
        
        # Need to get x and y coordinates to plot the heatmap with bounds for correct ra and dec
        hdr = fits.getheader(self._fitsfile)
        target_pix_ra_dec = wcs2pix(self._ra, self._dec, hdr)
        x, y = generate_square_pix_coords(self._size, *target_pix_ra_dec, aperture_radius)

        # velocities = np.flipud(velocities)
        w = WCS(hdr, naxis=2)
        plt.subplot(projection=w)
        hm = plt.imshow(velocities, cmap='bwr', interpolation='nearest', vmin=-scale_velo, vmax=scale_velo,
                extent=[x[0], x[-1], y[0], y[-1]],  
                origin='lower')
        cbar = plt.colorbar(hm)
        cbar.ax.set_ylabel('km/s', fontsize=15)
        cbar.ax.tick_params(labelsize=15)
        plt.title(f'{title} Per Pixel', fontsize=15)
        plt.xlabel('RA', fontsize=15)      
        plt.ylabel('DEC', fontsize=15)   
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15) 
        plt.savefig(f'{title.lower()}_per_pixel.png')
        plt.show()         

    def _calc_freq_flux(self):
        """ Calculate the frequency and flux """
        _f2f_instance = Fits2flux(self._fitsfile, self._ra, self._dec, self._aperture_radius)
        frequency = _f2f_instance.get_freq()
        flux, flux_uncert = _f2f_instance.get_flux(self._bkg_radius, self._beam_tolerance)
        freq_exp, flux_exp = _f2f_instance.get_exponents()
        return frequency, flux, flux_uncert, freq_exp, flux_exp

    @staticmethod
    def _write_csv_rows(filename, mode, data):
        """ Doc string here """
        with open(filename, mode, newline='') as f:
            wr = csv.writer(f)
            wr.writerows(data)
            wr.writerow('')
    
    def _export_method_data(self, filename, sigma):
        """ Export template/fft csv data"""
        # Get the results
        z_low_err, z_up_err = self._z_uncert(self._z, self._chi2, sigma)
        results = [[round(z_low_err, self._round_to)], [round(self._best_z, self._round_to)], 
            [round(z_up_err, self._round_to)], [self._params[0]], [self._p_err[0]], [self._params[1]], [self._p_err[1]]]
        exponents = [[self._freq_exp], [self._flux_exp]]   
        
        # Get the headings and data for the template
        if filename == 'template.csv':
            headings = ['z_low_err', 'z', 'z_up_err', 'amp', 'amp_err', 'std_dev', 
                'std_dev_err', 'dz', 'chi2_r', 'freq', 'flux', 'flux_uncert', 'freq_exp', 'flux_exp']
            data = [*zip_longest(*results, self._z, self._chi2, self._frequency, 
                self._flux, self._flux_uncert, *exponents, fillvalue='')]
        
        # Get the headings and data for the fft
        elif filename == 'fft.csv':
            headings = ['z_low_err', 'z', 'z_up_err', 'amp', 'amp_err', 'std_dev',
                'std_dev_err', 'dz', 'chi2_r', 'ffreq', 'fflux']
            data = [*zip_longest(*results, self._z, self._chi2, self._ffreq, self._fflux, fillvalue='')]
        
        # Write the data to the csv
        with open(filename, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(headings)
            wr.writerows(data)
            
    def _z_uncert(self, z, chi2, sigma):
        """ Caclulate the uncertainty on the best fitting redshift """
        peaks, _, _ = find_lines(self._flux)
        reduced_sigma = sigma**2 / (len(self._flux) - 2*len(peaks) - 1)
        neg, pos = z_uncert(z, chi2, reduced_sigma)
        return neg, pos

    def template(self, z_start=0, dz=0.01, z_end=10, sigma=1):
        """ Doc string here """
        self._dz = dz

        # Calculate the frequency and flux
        if not hasattr(self, '_frequency'):
            self._frequency, self._flux, self._flux_uncert, \
                self._freq_exp, self._flux_exp = self._calc_freq_flux()

        # Calculate the template chi2
        self._z, self._chi2 = template_zfind(self._transition, 
            self._frequency, self._flux, self._flux_uncert, z_start, dz, z_end)

        # Plot the template chi2 and flux
        self._plot_chi2(title='Template')
        self._plot_template_flux()
        self._export_method_data('template.csv', sigma)

    def template_pp(self, size, z_start=0, dz=0.01, z_end=10, aperture_radius_pp=0.5):
        """ 
        Performs the template redshift finding method in a square around the target ra and dec
        """
        self._size = size

        # If the other pp method has not been run, calculate all fluxes and uncertainties
        if not hasattr(self, '_all_flux'):
            all_ra, all_dec = generate_square_world_coords(self._fitsfile, self._ra, self._dec, size, aperture_radius_pp)
            self._all_flux, self._flux_uncertainty = get_all_flux(self._fitsfile, all_ra, all_dec, aperture_radius_pp)

        frequency = Fits2flux(self._fitsfile, self._ra, self._dec, aperture_radius_pp).get_freq()
        z = template_per_pixel(self._transition, frequency, self._all_flux, self._flux_uncertainty, z_start, dz, z_end, size)
        
        self._write_csv_rows('template_per_pixel.csv', 'w', z) # export redshifts to csv
        self._plot_heatmap(z, title='Template', aperture_radius=aperture_radius_pp) # Plot the template pp heatmap

    def fft(self, z_start=0, dz=0.01, z_end=10, sigma=1):
        """ Doc string here """
        self._dz = dz

        # Calculate the frequency and flux
        if not hasattr(self, '_frequency'):
            self._frequency, self._flux, self._flux_uncert, \
                self._freq_exp, self._flux_exp = self._calc_freq_flux()
        
        # Calculate the fft frequency and flux
        self._ffreq, self._fflux = fft(self._frequency, self._flux)

        # Calculate the fft chi2
        self._z, self._chi2 = fft_zfind(self._transition, self._frequency, self._flux, z_start, dz, z_end)

        # Plot the fft chi2 and flux
        self._plot_chi2(title='FFT')
        self._plot_fft_flux()
        self._export_method_data('fft.csv', sigma) 

    def fft_pp(self, size, z_start=0, dz=0.01, z_end=10, aperture_radius_pp=0.5):
        """ 
        Performs the fft redshift finding method in a square around the target ra and dec
        """
        self._size = size
        
        # If the other pp method has not been run, calculate all fluxes and uncertainties
        if not hasattr(self, '_all_flux'):
            all_ra, all_dec = generate_square_world_coords(self._fitsfile, self._ra, self._dec, size, aperture_radius_pp)
            self._all_flux, self._flux_uncertainty = get_all_flux(self._fitsfile, all_ra, all_dec, aperture_radius_pp)
        
        frequency = Fits2flux(self._fitsfile, self._ra, self._dec, aperture_radius_pp).get_freq()
        z = fft_per_pixel(self._transition, frequency, self._all_flux, z_start, dz, z_end, size)
        
        self._write_csv_rows('fft_per_pixel.csv', 'w', z) # export redshifts to csv
        self._plot_heatmap(z, title='FFT', aperture_radius=aperture_radius_pp) # Plot the fft pp heatmap

# Get the prefix of a unit from an exponent
_unit_prefixes = {
    -24 : 'y', 
    -21 : 'z',
    -18 : 'a',
    -15 : 'f',
    -12 : 'p',
    -9 : 'n',
    -6 : '\u03BC',
    -3 : 'm',
    0 : '', 
    3 : 'k',
    6 : 'M',
    9 : 'G',
    12 : 'T',
    15 : 'P',
    18 : 'E', 
    21 : 'Z',
    24 : 'Y'}