""" Class to plot the results of the zfinder """

import csv
import warnings
from itertools import zip_longest

import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits

from zfinder.template import calc_template_params, gaussf, find_lines
from zfinder.fft import calc_fft_params, double_damped_sinusoid
from zfinder.uncertainty import z_uncert
from zfinder.utils import wcs2pix, generate_square_pix_coords, longest_decimal

warnings.filterwarnings(action="ignore", message='Some errors were detected !')
warnings.filterwarnings("ignore", module='astropy.wcs.wcs')

class Plotter():
    """ 
    Class to plot the results of zfinder. Includes methods of plotting exported csv data
    
    Parameters
    ----------
    showfig : bool, optional
        Whether to show the figure after plotting. Default is True.
    
    savefig : bool, optional
        Whether to save the figure after plotting. Default is True.
    
    Attributes
    ----------
    unit_prefixes : dict
        Dictionary of unit prefixes for plotting
    
    Methods
    -------
    calc_best_z()
        Calculate the best redshift
    
    plot_chi2()
        Plot the chi-squared vs redshift
    
    plot_template_flux()
        Plot the template flux
    
    plot_fft_flux()
        Plot the fft flux
        
    plot_heatmap()
        Plot a heatmap of the redshifts
    
    export_template_data()
        Export the Template fit data to a csv file
    
    export_fft_data()
        Export the FFT data to a csv file 
        
    plot_chi2_fromcsv()
        Plot the chi-squared vs redshift from a csv file
    
    plot_template_flux_fromcsv()
        Plot the template flux from a csv file
    
    plot_fft_flux_fromcsv()
        Plot the fft flux from a csv file
    
    plot_heatmap_fromcsv()
        Plot a heatmap of the redshifts from a csv file
    
    Examples
    --------
    >>> # After csv files exported with zfinder
    >>> source = Plotter()
    >>> source.plot_chi2_fromcsv('template.csv')
    >>> source.plot_template_flux_fromcsv()
    >>> 
    >>> source.plot_chi2_fromcsv('fft.csv')
    >>> source.plot_fft_flux_fromcsv()
    >>> 
    >>> source.plot_heatmap_fromcsv('template_per_pixel.csv')
    >>> source.plot_heatmap_fromcsv('fft_per_pixel.csv')
    """
    
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
    
    def __init__(self, showfig=True, savefig=False):
        self._showfig = showfig
        self._savefig = savefig
        self._template_best_z = None
        self._fft_best_z = None
        self._round_to = 2
        self._freq_exp = None
        self._flux_exp = None
    
    def calc_best_z(self, z, chi2, title=None):
        """ 
        Calculate the best redshift
        
        Parameters
        ----------
        z : list
            Array of redshifts
        
        chi2 : list
            Array of chi-squared values
        
        Returns
        -------
        best_z : float
            The best redshift
        """
        best_z = z[np.argmin(chi2)]
        if title == 'Template':
            self._template_best_z = best_z
        elif title.upper() == 'FFT':
            self._fft_best_z = best_z
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
        if self._savefig:
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
        """ 
        Plot the template flux 
        
        Parameters
        ----------
        transition : float
            The transition frequency
        
        frequency : list
            Array of frequencies
        
        freq_exp : int
            The exponent of the frequency 
        
        flux : list
            Array of fluxes
        
        flux_exp : int
            The exponent of the flux
        """
        self._frequency = frequency
        self._flux = flux
        self._freq_exp = freq_exp
        self._flux_exp = flux_exp
        plt.figure(figsize=(15,7))
        plt.plot(frequency, np.zeros(len(frequency)), color='black', linestyle=(0, (5, 5)))
        plt.plot(frequency, flux, color='black', drawstyle='steps-mid')
        if self._template_best_z is None:
            raise ValueError("No best redshift found. Run Plotter.plot_chi2() first.")
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
        if self._savefig:
            plt.savefig('template_flux.png')
        if self._showfig:
            plt.show()
        
    def plot_fft_flux(self, transition, frequency, ffreq, fflux):
        """ 
        Plot the fft flux
        
        Parameters
        ----------
        transition : float
            The transition frequency
        
        frequency : list
            Array of frequencies
        
        ffreq : list
            Array of fft frequencies
        
        fflux : list
            Array of fft fluxes
        """
        self._ffreq = ffreq
        self._fflux = fflux
        plt.figure(figsize=(15,7))
        plt.plot(ffreq, fflux, color='black', drawstyle='steps-mid', label='FFT Flux')
        plt.plot(ffreq, np.zeros(len(fflux)), color='black', linestyle=(0, (5, 5)))
        if self._fft_best_z is None:
            raise ValueError("No best redshift found. Run Plotter.plot_chi2() first.")
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
        if self._savefig:
            plt.savefig('fft_flux.png')
        if self._showfig:
            plt.show()
            
    @staticmethod
    def _export_heatmap_data(filename, mode, data):
        """ Export the redshifts, velocities, and fluxes to a csv file """
        with open(filename, mode, newline='') as f:
            wr = csv.writer(f)
            wr.writerows(data)
            wr.writerow('')
    
    def plot_heatmap(self, ra, dec, hdr, data, size, z, title, aperture_radius, flux_limit, export=False):
        """ 
        Plot a heatmap of the redshifts
        
        Parameters
        ----------
        ra : list[string]
            Target right ascension
        
        dec : list[string]
            Target declination
        
        hdr : astropy.io.fits.header.Header
            The header of the fits file
        
        data : astropy.io.fits.hdu.image.PrimaryHDU
            The data of the fits file
        
        size : int
            The size of the square in pixels
            
        z : list
            Array of redshifts
        
        title : string
            The title method of the plot. Either 'Template' or 'FFT'
        
        aperture_radius : float
            The radius of the aperture in pixels
        
        flux_limit : float
            The flux limit to mask the velocities
        
        export : bool, optional
            Whether to export the redshifts to a csv file. Default is True.  
        """
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
            self._export_heatmap_data(f'{title.lower()}_per_pixel.csv', 'a', velocities) # export redshifts to csv
            self._export_heatmap_data(f'{title.lower()}_per_pixel.csv', 'a', fluxes) # export redshifts to csv

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
        if self._savefig:
            plt.savefig(f'{title.lower()}_per_pixel.png')
        if self._showfig:
            plt.show()
    
    def plot_coords(self, x_centre, y_centre, x_coords, y_coords, radius, fitsfile=None):
        if fitsfile is not None: 
            hdr = fits.getheader(fitsfile)
            w = WCS(hdr, naxis=2)
            fig, ax = plt.subplots(subplot_kw={'projection': w})
        else:
            fig, ax = plt.subplots()
        circ = plt.Circle((x_centre, y_centre), radius, fill=False, color='blue', label='_nolegend_')
        fig.set_figwidth(7)
        fig.set_figheight(7)
        plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
        ax.add_patch(circ)
        plt.scatter(x_coords, y_coords, color='blue', label='Random')
        plt.scatter(x_centre, y_centre, color='black', label='Target')
        plt.title(f'{len(x_coords)} random points')
        plt.xlim(-radius-1+x_centre, radius+1+x_centre)
        plt.ylim(-radius-1+y_centre, radius+1+y_centre)
        plt.xlabel('RA', fontsize=15)
        plt.ylabel('DEC', fontsize=15)
        plt.legend(loc='upper left')
        if self._savefig:
            plt.savefig('Point Distribution.png', dpi=200)
        if self._showfig:
            plt.show()

    @staticmethod
    def _write_method_to_csv(filename, headings, data):
        with open(filename, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(headings)
            wr.writerows(data)       
        
    def _z_uncert(self, sigma, flux):
        """ Caclulate the uncertainty on the best fitting redshift """
        peaks, _, _ = find_lines(flux)
        reduced_sigma = sigma**2 / (len(flux) - 2*len(peaks) - 1)
        neg, pos = z_uncert(self._z, self._chi2, reduced_sigma)
        return neg, pos
            
    def _calculate_results(self, sigma, best_z, flux):
        z_low_err, z_up_err = self._z_uncert(sigma, flux)
        results = [[z_low_err], [round(best_z, self._round_to)], 
                [z_up_err], [self._params[0]], [self._p_err[0]], [self._params[1]], [self._p_err[1]]]
        if self._freq_exp is None:
            return results, None
        exponents = [[self._freq_exp], [self._flux_exp]]   
        return results, exponents

    def _export_template_data(self, fitsfile, ra, dec, aperture_radius, transition, filename='template.csv', sigma=1, flux_uncert=1):
        """ 
        Export the Template fit data to a csv file 
        
        Ensure that Plotter.plot_template_flux() and Plotter.plot_chi2() have been run first.
        
        Parameters
        ----------
        filename : string, optional
            The filename of the csv file. Default is 'template.csv'.
        
        sigma : float, optional
            The sigma value for the uncertainty on the redshift. Default is 1.
        
        flux_uncert : float, optional
            The uncertainty on the flux. Default is 1.
        """
        results, exponents = self._calculate_results(sigma, self._template_best_z, self._flux)
        headings = ['z_low_err', 'z', 'z_up_err', 'amp', 'amp_err', 'std_dev', 
                    'std_dev_err', 'fitsfile', 'ra', 'dec', 'aperture_radius', 'transition', 
                    'dz', 'chi2_r', 'freq', 'flux', 'flux_uncert', 'freq_exp', 'flux_exp']
        common = [[fitsfile], [ra], [dec], [aperture_radius], [transition]]
        data = [*zip_longest(*results, *common, self._z, self._chi2, self._frequency, self._flux, flux_uncert, *exponents, fillvalue='')]
        self._write_method_to_csv(filename, headings, data)

    def _export_fft_data(self, fitsfile, ra, dec, aperture_radius, transition, filename='fft.csv', sigma=1, frequency=None, flux=None, flux_uncert=1):
        """ 
        Export the FFT data to a csv file 
        
        Ensure that Plotter.plot_fft_flux() and Plotter.plot_chi2() have been run first.
        
        Parameters
        ----------
        filename : string, optional
            The filename of the csv file. Default is 'fft.csv'.
        
        sigma : float, optional
            The sigma value for the uncertainty on the redshift. Default is 1.
        
        flux_uncert : float, optional
        """
        results, _ = self._calculate_results(sigma, self._fft_best_z, flux)
        headings = ['z_low_err', 'z', 'z_up_err', 'amp', 'amp_err', 'std_dev',
                    'std_dev_err', 'fitsfile', 'ra', 'dec', 'aperture_radius', 'transition',
                    'dz', 'chi2_r', 'frequency', 'flux', 'ffreq', 'fflux', 'fflux_uncert']
        if type(flux_uncert) == int:
            flux_uncert = [flux_uncert]
        common = [[fitsfile], [ra], [dec], [aperture_radius], [transition]]
        data = [*zip_longest(*results, *common, self._z, self._chi2, frequency, flux, self._ffreq, self._fflux, flux_uncert, fillvalue='')]
        self._write_method_to_csv(filename, headings, data)
    
    def plot_chi2_fromcsv(self, filename):
        """ Plot the chi-squared vs redshift from a csv file """
        z, chi2 = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=(12,13)).T
        z = z[~np.isnan(z)]
        chi2 = chi2[~np.isnan(chi2)]
        dz = longest_decimal(z)
        self.plot_chi2(z, dz, chi2, filename.split('.')[0].capitalize())
    
    def plot_template_flux_fromcsv(self, filename='template.csv'):
        """ Plot the template flux from a csv file """
        transition, frequency, flux, freq_exp, flux_exp = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=(11, 14, 15, 17, 18)).T
        transition = transition[0]
        frequency = frequency[~np.isnan(frequency)]
        flux = flux[~np.isnan(flux)]
        freq_exp = int(freq_exp[0])
        flux_exp = int(flux_exp[0])
        self.plot_template_flux(transition, frequency, freq_exp, flux, flux_exp)
    
    def plot_fft_flux_fromcsv(self, filename='fft.csv'):
        """ Plot the fft flux from a csv file """
        transition, frequency, ffreq, fflux = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=(11, 14, 16, 17)).T
        transition = transition[0]
        frequency = frequency[~np.isnan(frequency)]
        ffreq = ffreq[~np.isnan(ffreq)]
        fflux = fflux[~np.isnan(fflux)]
        self.plot_fft_flux(transition, frequency, ffreq, fflux)
    
    def plot_heatmap_fromcsv(self, filename):
        """ Plot a heatmap of the redshifts from a csv file """
        dtypes = [('fitsfile', 'U100'), ('ra', 'U100'), ('dec', 'U100'), ('aperture_radius', 'f8'),
          ('transition', 'f8'), ('size', 'i4'), ('flux_limit', 'f8')]
        fitsfile, ra, dec, aperture_radius, _, size, flux_limit = \
            np.genfromtxt(filename, delimiter=',', skip_header=1, dtype=dtypes, max_rows=2, invalid_raise=False).tolist()
        z = np.genfromtxt(filename, delimiter=',', skip_header=3, skip_footer=size*2)
        hdr = fits.getheader(fitsfile)
        data = fits.getdata(fitsfile)[0]
        self.plot_heatmap(ra, dec, hdr, data, size, z, filename.split('_')[0].capitalize(), aperture_radius, flux_limit)
    
    def plot_coords_fromcsv(self, filename='fft_uncertainty.csv'):
        """ Plot the distribution of random points from a csv file """
        x_centre, y_centre, x_coords, y_coords, radius = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=(1,2,3,4,8)).T
        fitsfile = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype='U100', usecols=(10)).T
        x_centre = x_centre[0]
        y_centre = y_centre[0]
        x_coords = x_coords[~np.isnan(x_coords)]
        y_coords = y_coords[~np.isnan(y_coords)]
        radius = radius[0]
        fitsfile = fitsfile[0]
        self.plot_coords(x_centre, y_centre, x_coords, y_coords, radius, fitsfile)
        
def main():
    source = Plotter()
    source.plot_chi2_fromcsv('template.csv')
    source.plot_template_flux_fromcsv()
    
    source.plot_chi2_fromcsv('fft.csv')
    source.plot_fft_flux_fromcsv()
    
    source.plot_heatmap_fromcsv('template_per_pixel.csv')
    source.plot_heatmap_fromcsv('fft_per_pixel.csv')
    
    source.plot_coords_fromcsv()

if __name__ == '__main__':
    main()