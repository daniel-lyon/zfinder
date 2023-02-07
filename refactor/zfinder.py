import csv
import numpy as np
import matplotlib.pyplot as plt

from zflux import zflux
from zfft import zfft, fft
from astropy.io import fits
from decimal import Decimal
from uncertainty import z_uncert
from warnings import filterwarnings
from fits2flux import fits2flux, wcs2pix, get_eng_exponent
from line_statistics import line_statistics, find_lines, flatten_list, pix2wcs

filterwarnings("ignore", module='astropy.wcs.wcs')

def count_decimals(number: float):
    d = Decimal(str(number))
    d = abs(d.as_tuple().exponent)
    return d

def get_unit(exponent):
    """ Get the prexif of a unit from an exponent """
    # A dictionary of exponent and unit prefix pairs
    prefix = {-24 : 'y', -21 : 'z', -18 : 'a',
              -15 : 'f', -12 : 'p',  -9 : 'n',
               -6 : 'mu', -3 : 'm',   0 : '', 
                3 : 'k',   6 : 'M',   9 : 'G',
               12 : 'T',  15 : 'P',  18 : 'E', 
               21 : 'Z',  24 : 'Y'}
    return prefix[exponent]

class zfinder():
    def __init__(self, image, ra, dec, transition, aperture_radius, bvalue):
        """
        Create plots of flux, chi-squared, redshift velocit. Use zfind_flux for flux redshift finding,
        zfind_fft for fft redshing finding, line_stats for statistics at random points,
        and fft_per_pixel to calculate the redshift with the fft method in a grid of
        pixels.

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
    def _plot_scale_snr_scatter(snr, scale):
        snr = flatten_list(snr)
        scale = flatten_list(scale)
        plt.scatter(snr, scale)
        plt.xlabel('SNR')
        plt.ylabel('Scale')
        plt.savefig('Scale vs SNR.png')
        plt.show()

    @staticmethod
    def _plot_circle_points(coords, radius):
        """ Plot the corodinates of the random points """
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
        plt.savefig('Point Distribution.png', dpi=200)
        plt.show()
    
    @staticmethod
    def _plot_pix_snr_scatter(pixels, snrs):
        pixels = flatten_list(pixels)
        snrs = flatten_list(snrs)
        plt.scatter(pixels, snrs)
        plt.xlabel('No. of Pixels')
        plt.ylabel('SNR')
        plt.yticks([0,1,2,3,4,5,6,7,8,9,10])
        plt.savefig('#Pix vs SNR.png')
        plt.show()

    @staticmethod
    def _export_heatmap_csv(delta_z):
        """ Export matrix of delta redshifts"""
        with open(f'heatmap_data.csv', 'w') as f:
            wr = csv.writer(f)
            for row in delta_z:
                wr.writerow(row)
            wr.writerow([])
    
    def _plot_blind_lines(self):
        """ Helper function to plot blind lines in the found flux """
        peaks, snrs, scales = find_lines(self.flux)
        lines = self.zf.blind_lines
        text_offset_high = max(self.flux)/20
        text_offset_low = 0.4*text_offset_high

        for i, line in enumerate(lines):
            x = self.frequency[line]
            y = self.flux[line]
            plt.plot(x, y, 'bo')
            plt.text(x, y+text_offset_high, f'snr={snrs[i]}', color='blue')
            plt.text(x, y+text_offset_low, f'scale={scales[i]}', color='blue')

    def _heatmap(self):
        """ Grid heatmap of the redshift found per pixel """
        velocities = [3*10**5*(1/(1+self.best_z) - 1/(1+i)) for i in self.z_fft_pp]
        coord_array = np.zeros(len(self.z_fft_pp), dtype=np.int64)
        for i in range(len(coord_array)):
            coord_array[i] = i-len(coord_array)//2
        x_coords = coord_array + self.centre_coords[0]
        y_coords = np.abs(coord_array - self.centre_coords[1])
        total_coords = []
        for x, y in zip(x_coords, y_coords):
            ra, dec = pix2wcs(x,y, self.hdr)
            total_coords.append([ra, dec])
        total_coords = np.array(total_coords, dtype=object)
        x_coords = total_coords[:,0]
        y_coords = total_coords[:,1]
        x_coords = [f"{ra[0]}$^h${ra[1]}$^m${round(ra[2],2)}$^s$" for ra in x_coords]
        y_coords = [f"{dec[0]}\u00b0{dec[1]}\"{round(dec[2],2)}\'" for dec in y_coords]
        self.z_fft_pp = np.array(self.z_fft_pp).tolist()
        d = count_decimals(self.best_z)

        # Show a heatmap of the redshifts
        # hm_r = plt.imshow(velocities, cmap='bwr_r', interpolation='nearest', norm=[1,0])
        hm = plt.imshow(velocities, cmap='bwr', interpolation='nearest', vmin=-300, vmax=300)
        plt.xticks(np.arange(len(x_coords)), labels=x_coords, rotation=90, fontsize=5)
        plt.yticks(np.arange(len(y_coords)), labels=y_coords, fontsize=5)
        plt.colorbar(hm)

        all_deltas = []
        for i in range(len(self.z_fft_pp)):
            deltas = []
            for j in range(len(self.z_fft_pp)):
                dat = float(self.z_fft_pp[i][j])
                delta_z = dat - self.best_z
                delta_z = round(delta_z, d)
                deltas.append(delta_z)
            all_deltas.append(deltas)
        
        self._export_heatmap_csv(all_deltas)        
        plt.savefig('FFT PP Heatmap.png')
        plt.show()
    
    def _plot_flux(self):
        """ Plot the flux with best fitting redshift """
        exponent = get_eng_exponent(self.frequency[0])
        symbol = get_unit(exponent)
        self.lowest_params = self.zf.all_params[self.lowest_index]
        self.lowest_perr = self.zf.all_perrs[self.lowest_index]
        x0 = self.transition/(1+self.min_z)
        plt.figure(figsize=(20,9))
        plt.plot(self.frequency, np.zeros(len(self.frequency)), color='black', linestyle=(0, (5, 5)))
        plt.plot(self.frequency, self.flux, color='black', drawstyle='steps-mid')
        plt.plot(self.frequency, zflux._gaussf(self.frequency, *self.lowest_params, x0), color='red')
        self._plot_blind_lines()
        plt.margins(x=0)
        plt.fill_between(self.frequency, self.flux, 0, where=(np.array(self.flux) > 0), color='gold', alpha=0.75)
        plt.title(f'Flux z={round(self.min_z, self.d)}', fontsize=15)
        plt.xlabel(f'Frequency $({symbol}Hz)$', fontsize=15)
        plt.ylabel('Flux $(mJy)$', fontsize=15)
        plt.savefig('Flux Best Fit.png')
        plt.show()
    
    def _plot_chi2(self, z, chi2, title):
        """ Plot the chi-squared vs redshift """
        self.d = count_decimals(self.dz)
        plt.figure(figsize=(20,9))
        plt.plot(z, chi2, color='black')
        plt.plot(self.min_z, self.min_chi2, 'bo', markersize=5)
        plt.title(f'{title} $\chi^2_r$ = {round(self.min_chi2, 2)} @ z={round(self.min_z, self.d)}', fontsize=15)
        plt.xlabel('Redshift', fontsize=15)
        plt.ylabel('$\chi^2_r$', x=0.01, fontsize=15)
        plt.yscale('log')
        plt.savefig(f'{title} Chi2.png', dpi=200)
        plt.show()
    
    def _plot_fft_flux(self):
        """ Plot the FFT with best fitting redshift"""
        self.lowest_params = self.fft_instance.fft_params[self.lowest_index]
        self.lowest_perr = self.fft_instance.fft_perrs[self.lowest_index]
        plt.figure(figsize=(20,9))
        plt.plot(self.ffreq, self.fflux, color='black', drawstyle='steps-mid')
        plt.plot(self.ffreq, np.zeros(len(self.fflux)), color='black', linestyle=(0, (5, 5)))
        plt.plot(self.ffreq, zfft._double_damped_sinusoid(self.ffreq, *self.lowest_params, 
            self.dz, self.frequency[0], self.transition), color='red')
        plt.fill_between(self.ffreq, self.fflux, 0, where=(np.array(self.fflux) > 0), color='gold', alpha=0.75)
        plt.title(f'FFT z={round(self.min_z, self.d)}', fontsize=15)
        plt.xlabel('Scale', fontsize=15)
        plt.ylabel('Amplitude', fontsize=15)
        plt.margins(x=0)
        plt.savefig('FFT Best Fit.png', dpi=200)
        plt.show()
    
    def _run_fits2flux(self):
        """ Run if flux data is needed """
        self.source = fits2flux(self.image, self.ra, self.dec, self.aperture_radius, self.bvalue)
        self.frequency = self.source.get_freq()
        self.flux, self.uncert = self.source.get_flux()
    
    def _z_uncert(self):
        """ Caclulate the uncertainty on the best fitting redshift """
        lowest_index = np.argmin(self.all_chi2)
        num_peaks = self.all_num_peaks[lowest_index]
        reduced_sigma = self.sigma**2 / (len(self.flux) - 2*num_peaks - 1)
        self.neg, self.pos = z_uncert(self.z, self.all_chi2, reduced_sigma)
        return self.neg, self.pos
            
    def _export_csv(self, title):
        """ Export the data to a csv file """

        # Write chi2 vs redshift and uncertainties
        with open(f'{title}_data.csv', 'w') as f:
            wr = csv.writer(f)
            # Export Uncertainty of the Flux data
            wr.writerow([f'{title} Redshift and uncertainties'])
            wr.writerow(['-', 'z', '+'])
            wr.writerow([self.neg, self.min_z, self.pos])
            wr.writerow([])

            wr.writerow([f'Best Fitting Parameters for the {title} Plot'])
            wr.writerow(['Parameter', 'Value', 'Uncertainty'])
            rows = zip(['amplitude', 'standard deviation'], self.lowest_params, self.lowest_perr)
            for row in rows:
                wr.writerow(row)
            wr.writerow([])

            rows = zip(self.z, self.all_chi2)
            wr.writerow(['Redshift', 'Chi-squared'])
            for row in rows:
                wr.writerow(row)

    def zfind_flux(self, z_start=0, dz=0.01, z_end=10, sigma=1):
        
        # Get the frequency and flux
        if 'frequency' not in globals():
            self._run_fits2flux()
        self.dz = dz
        self.sigma = sigma
        
        # Find the best fitting chi-squared
        self.zf = zflux(self.transition, self.frequency, self.flux, self.uncert)
        self.z, self.all_chi2 = self.zf.zfind(z_start, dz, z_end, self.sigma)
        self.all_num_peaks = self.zf.all_num_peaks

        self._z_uncert()
        
        self.lowest_index = np.argmin(self.all_chi2)
        self.min_z = self.z[self.lowest_index]
        self.min_chi2 = min(self.all_chi2)
        
        # Plot chi-squared vs redshift and flux vs frequency
        self._plot_chi2(self.z, self.all_chi2, 'Flux')
        self._plot_flux()
        self._export_csv('Flux')
    
    def zfind_fft(self, z_start=0, dz=0.01, z_end=10, sigma=1):
        """
        Iterate through small changes in redshift and caclulate the chi-squared at each dz.
        Automatically saves and shows a plot of the chi-squared and result of the best 
        fitting parameters at the lowest chi-squared.
        
        Parameters
        ----------
        z_start : int, optional
            The beginning of the redshift list. Default = 0
        
        dz : float, optional
            The change in redshift. Default = 0.01
        
        z_end : int, optional
            The final value of the redshift list. Default = 10
        
        sigma : float
            The significance level of the uncertainty in the redshift 
            found at the minimum chi-squared
        
        Returns
        -------
        z : list
            The list of redshifts that was used to calculate the chi-squared
        
        chi2 : list
            A list of calculated chi-squared values
        """
        if 'frequency' not in globals():
            self._run_fits2flux()
        self.dz = dz
        self.sigma = sigma

        # Get the fft flux data
        self.ffreq, self.fflux = fft(self.frequency, self.flux)

        # Get the fft chi-squared vs redshift data
        self.fft_instance = zfft(self.transition, self.frequency, self.flux)
        self.z, self.all_chi2 = self.fft_instance.zfind(z_start, dz, z_end, sigma) 
        self.all_num_peaks = self.fft_instance.all_num_peaks

        self._z_uncert()

        self.lowest_index = np.argmin(self.all_chi2)
        self.min_z = self.z[self.lowest_index]
        self.min_chi2 = min(self.all_chi2)

        self._plot_chi2(self.z, self.all_chi2, 'FFT')
        self._plot_fft_flux()
        self._export_csv('FFT')
    
    def line_stats(self, num_points, radius=50, min_spread=1):
        """ 
        Find statistics of: signal-to-noise ratio, scale, and delta-z of random points.
        Automatically saves and shows a plot of the distribution of random points and
        three side-by-side histograms of the snr, scale, and delta-z.
        
        Parameters
        ----------
        num_points : int
            The number of points to find statistics for
        
        radius : float, optional
            The radius of the image to find statistics (in pixels). Default = 50
        
        min_spread : float, optional
            The minimum spread of random points (in pixels). Default = 1
        
        Returns
        -------
        snrs : list
            A list of signifcant point signal-to-noise ratios
        
        scales : list
            The scale of the significant points
        
        z : list
            A list of the redshifts corresponding to the minimum chi-squared
        """
        stats = line_statistics(self.image, self.ra, self.dec, 
            self.aperture_radius, self.bvalue, self.transition)
        
        snrs, scales, lowest_z, pixels = stats.perform_analysis(num_points, radius, min_spread)

        self._plot_circle_points(stats.coordinates, radius)
        self._plot_snr_scales(snrs, scales, lowest_z)
        self._plot_scale_snr_scatter(snrs, scales)
        self._plot_pix_snr_scatter(pixels, snrs)

    def fft_per_pixel(self, length, z_start=0, dz=0.001, z_end=10):
        """
        Performs the FFT redshift finding method in a square around the 
        target right ascension and declination. Automatically saves and
        shows a heatmap of the distribution of the redshifts.

        Parameters
        ----------
        length : integer
            The length of the cube in pixels. Example: length=3 will compute
            a 3x3 grid with the center being the target. Length=5 is a 5x5
            grid with the target in the center, etc.
        """

        z_fft_pp = []
        d = count_decimals(dz)
        
        self.opened_image = fits.open(self.image)
        self.hdr = self.opened_image[0].header
        self.centre_coords = wcs2pix(self.ra, self.dec, self.hdr)

        source = fits2flux(self.image, self.ra, self.dec, self.aperture_radius, self.bvalue)
        freq = source.get_freq()
        flux, uncert = source.get_flux()
        fft_instance = zfft(self.transition, freq, flux)
        z, chi2 = fft_instance.zfind(z_start, dz, z_end)
        self.best_z = z[np.argmin(chi2)]
        self.best_z = round(self.best_z, d)
        if length %2 == 0:
            length += 1
        
        matrix = np.zeros(length)
        for i in range(len(matrix)):
            matrix[i] = i-len(matrix)//2
        x_coords = matrix + self.centre_coords[0]
        y_coords = matrix + self.centre_coords[1]
        from time import time

        i = 0
        for y in reversed(y_coords):
            for x in x_coords:
                start = time()
        
                ra, dec = pix2wcs(x, y, self.hdr)
                
                gleam_0856 = fits2flux(self.image, ra, dec, self.aperture_radius, self.bvalue)
                freq = gleam_0856.get_freq()
                flux, uncert = gleam_0856.get_flux()

                zf = zfft(self.transition, freq, flux)
                z, chi2 = zf.zfind(z_start, dz, z_end)

                lowest_z = z[np.argmin(chi2)]
                lowest_z = round(lowest_z, d)

                z_fft_pp.append(lowest_z)

                end = time()
                elapsed = end - start
                print(f'{i+1}/{length**2}, {round(elapsed,2)} seconds')
                i+=1

        self.z_fft_pp = np.array_split(z_fft_pp, length)
        self._heatmap()
        return z_fft_pp
        
def main():
    image = '0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = [8, 56, 14.8]
    dec = [2, 24, 0.6, 1]
    transition = 115.2712
    aperture_radius = 3
    bvalue = 3
    z_start = 0
    dz = 0.01
    z_end = 10

    gleam_0856 = zfinder(image, ra, dec, transition, aperture_radius, bvalue)
    gleam_0856.zfind_flux(z_start, dz, z_end)
    gleam_0856.zfind_fft(z_start, dz, z_end)
    # gleam_0856.line_stats(num_points=5, radius=5, min_spread=1)
    # gleam_0856.fft_per_pixel(length=3, z_start=z_start, dz=dz, z_end=z_end)

if __name__ == '__main__':
    main()