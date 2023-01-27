import csv
import numpy as np
import matplotlib.pyplot as plt

from zflux import zflux
from zfft import zfft, fft
from astropy.io import fits
from decimal import Decimal
from fits2flux import fits2flux
from uncertainty import uncertainty
from line_statistics import line_statistics, line_stats, flatten_list, pix2wcs

def count_decimals(number: float):
    d = Decimal(str(number))
    d = abs(d.as_tuple().exponent)
    return d

def get_unit(exponent):
    # A dictionary of exponent and unit prefix pairs
    prefix = {-24 : 'y', -21 : 'z', -18 : 'a',
                -15 : 'f', -12 : 'p',  -9 : 'n',
                -6 : 'mu', -3 : 'm',   0 : '', 
                3 : 'k',   6 : 'M',   9 : 'G',
                12 : 'T',  15 : 'P',  18 : 'E', 
                21 : 'Z', 24 : 'Y'}
    return prefix[exponent]

class zfinder():
    def __init__(self, image, ra, dec, transition, aperture_radius, bvalue):
        """
        Wrapper on zflux, zfft, fits2flux, and line_statistics to create nice plots. If you
        need to manually create these plots, use the aforementioned functions together.

        Parameters
        ----------
        image : .fits
            A .fits image file
        
        ra : list
           Right ascension of the target [h, m, s]
        
        dec : list
            Declination of the target [d, m, s, esign]
        
        aperture_radius : float
            Radius of the aperture to use over the source (pixels)

        bvalue : float
            The value of BMAJ and BMIN (arcseconds)
        """

        self.image = image
        self.ra = ra
        self.dec = dec
        self.transition = transition
        self.aperture_radius = aperture_radius
        self.bvalue = bvalue
    
    @staticmethod
    def _plot_snr_scales(snrs, scales, redshifts):
        """ Plot two histograms, one of snrs and one of scales """
        snrs = flatten_list(snrs[1:])
        scales = flatten_list(scales[1:])

        # Setup the figure and axes
        fig, (ax_snr, ax_scale, ax_redshift) = plt.subplots(1, 3, sharey=True)
        fig.supylabel('Count (#)')

        # Plot the snrs histogram(s)
        ax_snr.hist(snrs, 20)
        ax_snr.set_title('SNR histogram')
        ax_snr.set_xlabel('SNR')

        # Plot the scales histogram
        ax_scale.hist(scales, [8,10,12,14,16,18,20])
        ax_scale.set_title('Scales Histogram')
        ax_scale.set_xlabel('Scale')

        # Plot the redshift histogram
        z_origin = redshifts[0]
        zs = [round(i-z_origin, 2) for i in redshifts[1:]]
        ax_redshift.hist(zs)
        ax_redshift.set_title('Redshift Histogram')
        ax_redshift.set_xlabel('$\Delta z$ ')
        plt.show()

    @staticmethod
    def _plot_circle_points(coords, radius):
        circle_points = np.transpose(coords)
        points_x = circle_points[0, :] # all x coordinates except the first which is the original
        points_y = circle_points[1, :] # all y coordinates except the first which is the original
        centre_x = points_x[0]
        centre_y = points_y[0]
        circ = plt.Circle((centre_x, centre_y), radius, fill=False, color='blue')
        fig, ax = plt.subplots()
        fig.set_figwidth(7)
        fig.set_figheight(7)
        ax.add_patch(circ)
        plt.scatter(points_x[0], points_y[0], color='black')
        plt.scatter(points_x[1:], points_y[1:], color='blue')
        plt.title('Distribution of spaced random points')
        plt.xlim(-radius-1+centre_x, radius+1+centre_x)
        plt.ylim(-radius-1+centre_y, radius+1+centre_y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('Point Distribution.png')
        plt.show() # Show the plot
    
    def _plot_blind_lines(self):
        """ Helper function to plot blind lines in the found flux """
        snrs, scales = line_stats(self.flux)
        lines = self.zf.blind_lines
        text_offset_high = max(self.flux)/20
        text_offset_low = 0.4*text_offset_high

        for i, line in enumerate(lines):
            x = self.frequency[line]
            y = self.flux[line]
            plt.plot(x, y, 'bo')
            plt.text(x, y+text_offset_high, f'snr={snrs[i]}', color='blue')
            plt.text(x, y+text_offset_low, f'scale={scales[i]}', color='blue')

    def _run_fits2flux(self):
        """ Run if flux data is needed """
        self.source = fits2flux(self.image, self.ra, self.dec, self.aperture_radius, self.bvalue)
        self.frequency = self.source.get_freq()
        self.flux, self.uncert = self.source.get_flux()
    
    def _plot_flux(self):
        """ Plot the flux with best fitting redshift"""
        symbol = get_unit(self.source.exponent)
        self.lowest_flux_params = self.zf.all_params[self.lowest_index]
        x0 = self.transition/(1+self.min_z)
        plt.figure(figsize=(20,9))
        plt.plot(self.frequency, np.zeros(len(self.frequency)), color='black', linestyle=(0, (5, 5)))
        plt.plot(self.frequency, self.flux, color='black', drawstyle='steps-mid')
        plt.plot(self.frequency, zflux._gaussf(self.frequency, *self.lowest_flux_params, x0), color='red')
        self._plot_blind_lines()
        plt.margins(x=0)
        plt.fill_between(self.frequency, self.flux, 0, where=(np.array(self.flux) > 0), color='gold', alpha=0.75)
        plt.title(f'Flux z={round(self.min_z, self.d)}')
        plt.xlabel(f'Frequency $({symbol}Hz)$')
        plt.ylabel('Flux $(mJy)$')
        plt.savefig('Flux Best Fit.png')
        plt.show()
    
    def _plot_chi2(self, z, chi2, title):
        """ Plot the chi-squared vs redshift """
        self.d = count_decimals(self.dz)
        plt.figure(figsize=(20,9))
        plt.plot(z, chi2, color='black')
        plt.plot(self.min_z, self.min_chi2, 'bo', markersize=5)
        plt.title(f'{title} $\chi^2_r$ = {round(self.min_chi2, 2)} @ z={round(self.min_z, self.d)}')
        plt.xlabel('Redshift')
        plt.ylabel('$\chi^2_r$', x=0.01)
        plt.yscale('log')
        plt.savefig(f'{title} Chi2.png', dpi=200)
        plt.show()
    
    def _plot_fft_flux(self):
        """ Plot the FFT with best fitting redshift"""
        self.lowest_fft_params = self.fft_instance.fft_params[self.lowest_index]
        plt.figure(figsize=(20,9))
        plt.plot(self.ffreq, self.fflux, color='black', drawstyle='steps-mid')
        plt.plot(self.ffreq, np.zeros(len(self.fflux)), color='black', linestyle=(0, (5, 5)))
        plt.plot(self.ffreq, zfft._double_damped_sinusoid(self.ffreq, *self.lowest_fft_params, 
            self.dz, self.frequency[0], self.transition), color='red')
        plt.fill_between(self.ffreq, self.fflux, 0, where=(np.array(self.fflux) > 0), color='gold', alpha=0.75)
        plt.title(f'FFT z={round(self.min_z, self.d)}')
        plt.xlabel('Scale')
        plt.ylabel('Amplitude')
        plt.margins(x=0)
        plt.savefig('FFT Best Fit.png', dpi=200)
        plt.show()
    
    def _export_flux_csv(self):
        """ Export all the flux data to a csv file """

        # Write chi2 vs redshift and uncertainties
        with open('flux_data.csv', 'w') as f:
            wr = csv.writer(f)
            # Export Uncertainty of the Flux data
            wr.writerow(['Flux Redshift and uncertainties'])
            wr.writerow(['-', 'z', '+'])
            wr.writerow([self.flux_z_uncert[0], self.min_z, self.flux_z_uncert[1]])
            wr.writerow([])

            wr.writerow(['Best Fitting Parameters for the Flux Plot'])
            wr.writerow(['Parameter', 'Value', 'Uncertainty'])
            rows = zip(['amplitude', 'standard deviation'], self.lowest_flux_params, self.flux_perr)
            for row in rows:
                wr.writerow(row)
            wr.writerow([])

            rows = zip(self.flux_z, self.flux_chi2)
            wr.writerow(['Redshift', 'Chi-squared'])
            for row in rows:
                wr.writerow(row)
    
    def _export_fft_csv(self):
        """ Export all the fft data to a csv file"""

        # Write chi2 vs redshift and uncertainties
        with open('fft_data.csv', 'w') as f:
            wr = csv.writer(f)
            # Export Uncertainty of the Flux data
            wr.writerow(['FFT Redshift and uncertainties'])
            wr.writerow(['-', 'z', '+'])
            wr.writerow([self.ffz_uncert[0], self.min_z, self.ffz_uncert[1]])
            wr.writerow([])

            wr.writerow(['Best Fitting Parameters for the FFT Plot'])
            wr.writerow(['Parameter', 'Value', 'Uncertainty'])
            rows = zip(['amplitude', 'standard deviation'], self.lowest_fft_params, self.flux_perr)
            for row in rows:
                wr.writerow(row)
            wr.writerow([])

            rows = zip(self.fft_z, self.fft_chi2)
            wr.writerow(['Redshift', 'Chi-squared'])
            for row in rows:
                wr.writerow(row)
        
    def zfind_flux(self, z_start=0, dz=0.01, z_end=10, sigma=1):
        """ Create plots of chi-squared vs redshift and best fitting flux """

        # Get the frequency and flux
        if 'frequency' not in globals():
            self._run_fits2flux()
            self.dz = dz
            self.sigma = sigma
        
        # Find the best fitting chi-squared
        self.zf = zflux(self.transition, self.frequency, self.flux, self.uncert, z_start, dz, z_end)
        self.flux_z, self.flux_chi2, self.flux_z_uncert, self.flux_perr = self.zf.zfind(self.sigma)
        
        self.lowest_index = np.argmin(self.flux_chi2)
        self.min_z = self.flux_z[self.lowest_index]
        self.min_chi2 = min(self.flux_chi2)
        
        # Plot chi-squared vs redshift and flux vs frequency
        self._plot_chi2(self.flux_z, self.flux_chi2, 'Flux')
        self._plot_flux()
        self._export_flux_csv()
    
    def zfind_fft(self, z_start=0, dz=0.01, z_end=10, sigma=1):
        """ Create plots of chi-squared vs redshift and best fitting fft """
        if 'frequency' not in globals():
            self._run_fits2flux()
            self.dz = dz
            self.sigma = sigma

        # Get the fft flux data
        self.ffreq, self.fflux = fft(self.frequency, self.flux)

        # Get the fft chi-squared vs redshift data
        self.fft_instance = zfft(self.transition, self.frequency, self.flux, z_start, dz, z_end)
        self.fft_z, self.fft_chi2, self.ffz_uncert, self.fft_perr = self.fft_instance.zfind() 

        self.lowest_index = np.argmin(self.fft_chi2)
        self.min_z = self.fft_z[self.lowest_index]
        self.min_chi2 = min(self.fft_chi2)

        self._plot_chi2(self.fft_z, self.fft_chi2, 'FFT')
        self._plot_fft_flux()
        self._export_fft_csv()
    
    def line_stats(self, num_points, radius=50, min_spread=1):
        """ Measure flux at random points and get statistics on blind lines and redshift """
        stats = line_statistics(self.image, self.ra, self.dec, 
            self.aperture_radius, self.bvalue, self.transition)
        
        snrs, scales, lowest_z = stats.perform_analysis(num_points, radius, min_spread)

        self._plot_circle_points(stats.coordinates, radius)
        self._plot_snr_scales(snrs, scales, lowest_z)

    def fft_per_pixel(self, side_length, centre_coords):

        z_fft_pp = []
        
        self.opened_image = fits.open(self.image)
        self.hdr = self.opened_image[0].header

        if side_length %2 == 0:
            side_length += 1
        
        matrix = np.zeros(side_length**2)

        for i in range(len(matrix)):
            matrix[i] = i-len(matrix)//2
        x_coords = matrix + centre_coords[0]
        y_coords = matrix + centre_coords[1]

        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            ra, dec = pix2wcs(x, y, self.hdr)
            self._run_fits2flux()

            zf = zfft(self.transition, self.frequency, self.flux)
            z, chi2, z_uncert, perr = zf.zfind()

            lowest_z = z[np.argmin(chi2)]

            z_fft_pp.append(lowest_z)
            print(f'{i+1}/{len(x_coords)} ')

        return z_fft_pp
            
        # calculate square of coordinates in a "side length" radius
        # run fft and get
        
def main():
    image = '0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = [8, 56, 14.8]
    dec = [2, 24, 0.6, 1]
    transition = 115.2712
    aperture_radius = 2
    bvalue = 3

    gleam_0856 = zfinder(image, ra, dec, transition, aperture_radius, bvalue)
    gleam_0856.zfind_flux()
    gleam_0856.zfind_fft()
    # gleam_0856.line_stats(num_points=10, radius=5, min_spread=1)
    # z = gleam_0856.fft_per_pixel(side_length=7, centre_coords=[144, 155])
    # print(z)

if __name__ == '__main__':
    main()