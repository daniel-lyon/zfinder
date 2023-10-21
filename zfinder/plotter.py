""" Class to plot the results of the zfinder """

import csv
from itertools import zip_longest

import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS

from template import calc_template_params, gaussf, find_lines
from fft import calc_fft_params, double_damped_sinusoid
from uncertainty import z_uncert
from utils import wcs2pix, generate_square_pix_coords

class Plotter():
    
    unit_prefixes = {
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
    
    def __init__(self, showfig=True):
        self._showfig = showfig
        self._template_best_z = None
        self._fft_best_z = None
        self._round_to = 2
        self._freq_exp = None
        self._flux_exp = None
    
    def calc_best_z(self, z, chi2, title):
        if title == 'Template':
            self._template_best_z = z[np.argmin(chi2)]
            best_z = self._template_best_z
        elif title == 'FFT':
            self._fft_best_z = z[np.argmin(chi2)]
            best_z = self._fft_best_z
        return best_z
            
    def plot_chi2(self, z, dz, chi2, title):
        """ Plot the chi-sqaured vs redshift """
        self._z = z
        self._chi2 = chi2
        min_chi2 = min(chi2)
        best_z = self.calc_best_z(z, chi2, title)
        self._round_to = len(str(dz).split('.')[1])
        plt.figure(figsize=(15,7))
        plt.plot(z, chi2, color='black', label='$\chi^2_r$')
        plt.plot(best_z, min_chi2, 'bo', markersize=5, label='Best Fit')
        plt.title(f'{title} $\chi^2_r$ = {round(min_chi2, 2)} @ z={round(best_z, self._round_to)}', fontsize=15)
        plt.xlabel('Redshift', fontsize=15)
        plt.ylabel('$\chi^2_r$', x=0.01, fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'{title.lower()}_chi2.png')
        if self._showfig:
            plt.show()
    
    @staticmethod
    def _plot_sslf_lines(frequency, flux):
        """ Helper function to plot sslf lines in the found flux """
        peaks, snrs, scales = find_lines(flux)
        text_offset_high = max(flux)/20
        text_offset_low = 0.4*text_offset_high
        for i, line in enumerate(peaks):
            x = frequency[line]
            y = flux[line]
            plot, = plt.plot(x, y, 'bo')
            plt.text(x, y+text_offset_high, f'snr={snrs[i]}', color='blue')
            plt.text(x, y+text_offset_low, f'scale={scales[i]}', color='blue')
            if i == 0:
                plot.set_label('Lines')
        
    def plot_template_flux(self, transition, frequency, freq_exp, flux, flux_exp):
        """ Plot the template flux """
        self._frequency = frequency
        self._flux = flux
        self._freq_exp = freq_exp
        self._flux_exp = flux_exp
        plt.figure(figsize=(15,7))
        plt.plot(frequency, np.zeros(len(frequency)), color='black', linestyle=(0, (5, 5)))
        plt.plot(frequency, flux, color='black', drawstyle='steps-mid')
        if self._template_best_z is None:
            raise ValueError("No best redshift found. Run plotter.plot_chi2() first.")
        x0 = transition/(1+self._template_best_z)
        self._params, covars = calc_template_params(frequency, flux, x0)
        self._p_err = np.sqrt(np.diag(covars)) # calculate the error on the parameters
        plt.plot(frequency, gaussf(frequency, *self._params, x0), color='red', label='Template Fit')
        self._plot_sslf_lines(frequency, flux)
        plt.margins(x=0)
        plt.fill_between(frequency, flux, 0, where=(np.array(flux) > 0), color='gold', alpha=0.75, label='Aperture Flux')
        plt.title(f'Template Fit z={round(self._template_best_z, self._round_to)}', fontsize=15)
        plt.xlabel(f'Frequency $({self.unit_prefixes[freq_exp]}Hz)$', fontsize=15)
        plt.ylabel(f'Flux $({self.unit_prefixes[flux_exp]}Jy)$', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend()
        plt.savefig('template_flux.png')
        if self._showfig:
            plt.show()
        
    def plot_fft_flux(self, transition, frequency, ffreq, fflux):
        """ Plot the fft flux """
        self._ffreq = ffreq
        self._fflux = fflux
        plt.figure(figsize=(15,7))
        plt.plot(ffreq, fflux, color='black', drawstyle='steps-mid', label='FFT Flux')
        plt.plot(ffreq, np.zeros(len(fflux)), color='black', linestyle=(0, (5, 5)))
        if self._fft_best_z is None:
            raise ValueError("No best redshift found. Run plotter.plot_chi2() first.")
        self._params, covars = calc_fft_params(transition, ffreq, fflux, self._fft_best_z, frequency[0])
        self._p_err = np.sqrt(np.diag(covars))
        plt.plot(ffreq, double_damped_sinusoid(ffreq, *self._params, 
            self._fft_best_z, frequency[0], transition), color='red', label='FFT Fit')
        plt.margins(x=0)
        plt.title(f'FFT Fit z={round(self._fft_best_z, self._round_to)}', fontsize=15)
        plt.xlabel('Scale', fontsize=15)
        plt.ylabel('Amplitude', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend()
        plt.savefig('fft_flux.png')
        if self._showfig:
            plt.show()
            
    @staticmethod
    def export_heatmap_data(filename, mode, data):
        """ Doc string here """
        with open(filename, mode, newline='') as f:
            wr = csv.writer(f)
            wr.writerows(data)
            wr.writerow('')
    
    def plot_heatmap(self, ra, dec, hdr, data, size, z, title, aperture_radius, flux_limit, export=True):
        """ Plot a heatmap of the redshifts """
        # Calculate the velocities
        target_z = np.take(z, z.size // 2) # redshift of the target ra and dec
        velocities = 3*10**5*((((1 + target_z)**2 - 1) / ((1 + target_z)**2 + 1)) - (((1 + z)**2 - 1) / ((1 + z)**2 + 1))) # km/s
        scale_velo = np.max(np.abs(velocities))
        
        # Need to get x and y coordinates to plot the heatmap with bounds for correct ra and dec
        target_pix_ra_dec = wcs2pix(ra, dec, hdr)
        x, y = generate_square_pix_coords(size, *target_pix_ra_dec, aperture_radius)
        
        # Mask velocities lower than the flux limit
        data_summed = np.sum(np.maximum(data, 0), axis=0)
        uy = np.round(np.unique(y)).astype(int)
        ux = np.round(np.unique(x)).astype(int)
        fluxes = data_summed[uy][:, ux]
        mask = fluxes < flux_limit
        velocities[mask] = np.nan
        
        if export:
            self.export_heatmap_data(f'{title.lower()}_per_pixel.csv', 'a', velocities) # export redshifts to csv
            self.export_heatmap_data(f'{title.lower()}_per_pixel.csv', 'a', fluxes) # export redshifts to csv

        cmap = plt.cm.seismic
        cmap.set_bad('black')
        
        # velocities = np.flipud(velocities)
        w = WCS(hdr, naxis=2)
        plt.figure(figsize=(7,5))
        plt.subplot(projection=w)
        hm = plt.imshow(np.flipud(velocities), cmap=cmap, interpolation='nearest', vmin=-scale_velo, vmax=scale_velo,
                extent=[x[0], x[-1], y[0], y[-1]], origin='lower')
        cbar = plt.colorbar(hm)
        cbar.ax.set_ylabel('km/s', fontsize=15)
        cbar.ax.tick_params(labelsize=15)
        plt.title(f'{title} Per Pixel', fontsize=15)
        plt.xlabel('RA', fontsize=15)      
        plt.ylabel('DEC', fontsize=15)   
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15) 
        plt.savefig(f'{title.lower()}_per_pixel.png')
        if self._showfig:
            plt.show()

    @staticmethod
    def _write_method_to_csv(filename, headings, data):
        with open(filename, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(headings)
            wr.writerows(data)       
        
    def _z_uncert(self, sigma):
        """ Caclulate the uncertainty on the best fitting redshift """
        peaks, _, _ = find_lines(self._flux)
        reduced_sigma = sigma**2 / (len(self._flux) - 2*len(peaks) - 1)
        neg, pos = z_uncert(self._z, self._chi2, reduced_sigma)
        return neg, pos
            
    def _calculate_results(self, sigma, best_z):
        z_low_err, z_up_err = self._z_uncert(sigma)
        results = [[z_low_err], [round(best_z, self._round_to)], 
                [z_up_err], [self._params[0]], [self._p_err[0]], [self._params[1]], [self._p_err[1]]]
        if self._freq_exp is None:
            return results, None
        exponents = [[self._freq_exp], [self._flux_exp]]   
        return results, exponents

    def export_template_data(self, filename='template.csv', sigma=1, flux_uncert=1):
        """ Export the Template fit data to a csv file """
        results, exponents = self._calculate_results(sigma, self._template_best_z)
        headings = ['z_low_err', 'z', 'z_up_err', 'amp', 'amp_err', 'std_dev', 
                    'std_dev_err', 'dz', 'chi2_r', 'freq', 'flux', 'flux_uncert', 'freq_exp', 'flux_exp']
        data = [*zip_longest(*results, self._z, self._chi2, self._frequency, self._flux, flux_uncert, *exponents, fillvalue='')]
        self._write_method_to_csv(filename, headings, data)

    def export_fft_data(self, filename='fft.csv', sigma=1):
        """ Export the FFT data to a csv file """
        results, _ = self._calculate_results(sigma, self._fft_best_z)
        headings = ['z_low_err', 'z', 'z_up_err', 'amp', 'amp_err', 'std_dev',
                    'std_dev_err', 'dz', 'chi2_r', 'ffreq', 'fflux']
        data = [*zip_longest(*results, self._z, self._chi2, self._ffreq, self._fflux, fillvalue='')]
        self._write_method_to_csv(filename, headings, data)