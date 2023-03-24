import csv
import numpy as np
import matplotlib.pyplot as plt

from time import time
from astropy.wcs import WCS
from decimal import Decimal
from uncertainty import z_uncert
from flux_zfind import flux_zfind, gaussf, find_lines
from fft_zfind import fft_zfind, fft, double_damped_sinusoid
from fits2flux import fits2flux, wcs2pix, pix2wcs, get_eng_exponent

# Get the prefix of a unit from an exponent
prefix = {-24 : 'y', -21 : 'z', -18 : 'a',
        -15 : 'f', -12 : 'p',  -9 : 'n',
        -6 : '\u03BC', -3 : 'm',   0 : '', 
        3 : 'k',   6 : 'M',   9 : 'G',
        12 : 'T',  15 : 'P',  18 : 'E', 
        21 : 'Z',  24 : 'Y'}

def count_decimals(number: float):
    """ Count the number of decimals in a float """
    d = Decimal(str(number))
    d = abs(d.as_tuple().exponent)
    return d

def flatten_list(input_list: list[list]):
    """ Turns lists of lists into a single list """
    flattened_list = []
    for array in input_list:
        for x in array:
            flattened_list.append(x)
    return flattened_list

class zfinder(flux_zfind):
    def __init__(self, image, ra, dec, transition, aperture_radius, bvalue):
        """
        zfinder is a wrapper on individual function files fit2flux, flux_zfind, and fft_zfind.
        use zfinder to find the redshift of a fits image via two different methods: zflux
        fits gaussian functions to the data to find redshift; and zfft performs the fast
        fourier transform on the flux data to find redshift. These methods will create and save
        a series of plots and csv files with raw data.

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
        
        self.source = fits2flux(image, ra, dec, aperture_radius, bvalue)
        self.frequency = self.source.get_freq()
        self.flux, self.uncertainty = self.source.get_flux()
    
    @staticmethod
    def _plot_chi2(z, chi2, dz, title):
        min_chi2 = min(chi2)
        min_z = z[np.argmin(chi2)]
        d = count_decimals(dz)
        plt.figure(figsize=(20,9))
        plt.plot(z, chi2, color='black')
        plt.plot(min_z, min_chi2, 'bo', markersize=5)
        plt.title(f'{title} $\chi^2_r$ = {round(min_chi2, 2)} @ z={round(min_z, d)}', fontsize=15)
        plt.xlabel('Redshift', fontsize=15)
        plt.ylabel('$\chi^2_r$', x=0.01, fontsize=15)
        plt.yscale('log')
        plt.savefig(f'{title} Chi2.png', dpi=200)
        plt.show()

    @staticmethod
    def _export_csv(title, neg, min_z, pos, params, perr, z, chi2):
        """ Export the data to a csv file """

        # Write chi2 vs redshift and uncertainties
        with open(f'{title}_data.csv', 'w', newline='') as f:
            wr = csv.writer(f)
            
            # Export redshift and uncertainties
            wr.writerow([f'{title} Redshift and uncertainties'])
            wr.writerow(['-', 'z', '+'])
            wr.writerow([neg, min_z, pos])
            wr.writerow([])

            # Export function parameters and uncertainty
            wr.writerow([f'Best Fitting Parameters for the {title} Plot'])
            wr.writerow(['Parameter', 'Value', 'Uncertainty'])
            rows = zip(['amplitude', 'standard deviation'], params, perr)
            for row in rows:
                wr.writerow(row)
            wr.writerow([])

            # Export raw arrays of chi2 for each redshift
            rows = zip(z, chi2)
            wr.writerow(['Redshift', 'Chi-squared'])
            for row in rows:
                wr.writerow(row)
    
    @staticmethod
    def _plot_pixels(snrs, pixels, peaks):
        """ Plot the snr vs # pixels """
        
        # Initialise arrays
        blue_points_x = [pixels[0]]
        blue_points_y = [snrs[0]]
        green_points_x = []
        green_points_y = []
        orange_points_x = []
        orange_points_y = []
        red_points_x = []
        red_points_y = []

        # Sort points
        for snr, pix, pk in zip(snrs[1:], pixels[1:], peaks[1:]):
            num_pks = len(pk)
            if num_pks < 2 or num_pks > 4: # bad - no redshift or extremely high redshift (z>15)
                green_points_x.append(pix)
                green_points_y.append(snr)
                continue
            
            if num_pks == 2:
                pk_diffs = np.abs(np.diff(pk)) # needs to be at least 200 channels  
                if pk_diffs > 200:
                    red_points_x.append(pix)
                    red_points_y.append(snr)
                else:
                    orange_points_x.append(pix)
                    orange_points_y.append(snr)
            else:
                pk_diffs = np.abs(np.diff(pk))            
                pk_diffs = np.abs(np.diff(pk_diffs))
                pk_diffs = np.average(pk_diffs)
                if pk_diffs < 15:
                    red_points_x.append(pix)
                    red_points_y.append(snr)
                else:
                    orange_points_x.append(pix)
                    orange_points_y.append(snr)

        # Flatten lists of lists to one big list
        blue_points_x = flatten_list(blue_points_x)
        blue_points_y = flatten_list(blue_points_y)
        green_points_x = flatten_list(green_points_x)
        green_points_y = flatten_list(green_points_y)
        orange_points_x = flatten_list(orange_points_x)
        orange_points_y = flatten_list(orange_points_y)
        red_points_x = flatten_list(red_points_x)
        red_points_y = flatten_list(red_points_y)

        # Make the plot
        plt.scatter(blue_points_x, blue_points_y, s=60, marker='*', color='blue')
        plt.scatter(green_points_x, green_points_y, s=60, marker='X', color='green')
        plt.scatter(orange_points_x, orange_points_y, s=60, marker='D', color='darkorange')
        plt.scatter(red_points_x, red_points_y, s=60, marker='s', color='red')
        # plt.title('No. Random Points = 5', fontsize=20)
        plt.xlabel('No. of Pixels', fontsize=20)
        plt.ylabel('SNR', fontsize=20)
        plt.legend(['Target', 'No significance', 'Low significance', 'High significance'])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig('SNR vs Pix.png')
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
    def _plot_scale_snr_scatter(snr, scale):
        snr = flatten_list(snr)
        scale = flatten_list(scale)
        plt.scatter(snr, scale)
        plt.xlabel('SNR')
        plt.ylabel('Scale')
        plt.savefig('Scale vs SNR.png')
        plt.show()
        
    @staticmethod
    def _export_heatmap_csv(delta_z):
        """ Export matrix of delta redshifts"""
        with open(f'heatmap_data.csv', 'w') as f:
            wr = csv.writer(f)
            for row in delta_z:
                wr.writerow(row)
            wr.writerow([])
        
    def _heatmap(self):
        """ Grid heatmap of the redshift found per pixel """
        
        # Velocity of the gas
        velocities = [3*10**5*((((1+self.best_z)**2 - 1) / ((1+self.best_z)**2 + 1)) - (((1+i)**2 - 1) / ((1+i)**2 + 1))) for i in self.z_fft_pp]
        self.z_fft_pp = np.array(self.z_fft_pp).tolist()
        d = count_decimals(self.best_z)
        
        # If there are more than 2 axis, drop them
        w = WCS(self.source.hdr) # Get the world coordinate system
        if self.source.hdr['NAXIS'] > 2:
            w = w.dropaxis(3) # stokes
            w = w.dropaxis(2) # frequency
        
        # Show a heatmap of the redshifts
        plt.subplot(projection=w)
        hm = plt.imshow(velocities, cmap='bwr', interpolation='nearest', vmin=-300, vmax=300)
        plt.xlim(left=self.x_coords[0], right=self.x_coords[-1])
        plt.ylim(bottom=self.y_coords[0], top=self.y_coords[-1])
        plt.colorbar(hm)

        # Calculate the difference between the redshifts
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

    def _z_uncert(self, z, chi2, sigma):
        """ Caclulate the uncertainty on the best fitting redshift """
        reduced_sigma = sigma**2 / (len(self.flux) - 2*len(self.peaks) - 1)
        neg, pos = z_uncert(z, chi2, reduced_sigma)
        return neg, pos
    
    def _plot_blind_lines(self):
        """ Helper function to plot sslf lines in the found flux """
        peaks, snrs, scales = find_lines(self.flux)
        text_offset_high = max(self.flux)/20
        text_offset_low = 0.4*text_offset_high

        for i, line in enumerate(peaks):
            x = self.frequency[line]
            y = self.flux[line]
            plt.plot(x, y, 'bo')
            plt.text(x, y+text_offset_high, f'snr={snrs[i]}', color='blue')
            plt.text(x, y+text_offset_low, f'scale={scales[i]}', color='blue')
    
    def _plot_flux(self, params):
        """ Plot the flux with best fitting redshift """
        # Normalising factor
        self.yexponent = get_eng_exponent(np.average(self.flux))
        # # print(np.average(flux))
        # # print(self.yexponent)
        # norm_factor = 10**-self.yexponent
        # # print(norm_factor)
        # # norm_factor = 1
        d = count_decimals(self.dz)
        x0 = self.transition/(1+self.min_z)
        plt.figure(figsize=(20,9))
        plt.plot(self.frequency, np.zeros(len(self.frequency)), color='black', linestyle=(0, (5, 5)))
        plt.plot(self.frequency, self.flux, color='black', drawstyle='steps-mid')
        plt.plot(self.frequency, gaussf(self.frequency, *params, x0), color='red')
        self._plot_blind_lines()
        plt.margins(x=0)
        plt.fill_between(self.frequency, self.flux, 0, where=(np.array(self.flux) > 0), color='gold', alpha=0.75)
        plt.title(f'Flux z={round(self.min_z, d)}', fontsize=15)
        plt.xlabel(f'Frequency $({prefix[self.source.xexponent]}Hz)$', fontsize=15)
        plt.ylabel(f'Flux $({prefix[self.yexponent]}Jy)$', fontsize=15)
        plt.savefig('Flux Best Fit.png')
        plt.show()
    
    def _plot_fft_flux(self, params):
        """ Plot the FFT with best fitting redshift"""
        d = count_decimals(self.dz)
        plt.figure(figsize=(20,9))
        plt.plot(self.ffreq, self.fflux, color='black', drawstyle='steps-mid')
        plt.plot(self.ffreq, np.zeros(len(self.fflux)), color='black', linestyle=(0, (5, 5)))
        plt.plot(self.ffreq, double_damped_sinusoid(self.ffreq, *params, 
            self.dz, self.frequency[0], self.transition), color='red')
        plt.title(f'FFT z={round(self.min_z, d)}', fontsize=15)
        plt.xlabel('Scale', fontsize=15)
        plt.ylabel('Amplitude', fontsize=15)
        plt.margins(x=0)
        plt.savefig('FFT Best Fit.png', dpi=200)
        plt.show()
    
    def zflux(self, z_start=0, dz=0.01, z_end=10, sigma=1, penalise=True):
        """ 
        Finds the best redshift by fitting gaussian functions overlayed on flux data. The
        chi-squared is calculated at every redshift by iterating through delta-z. The most 
        likely redshift corresponds to the minimum chi-squared. Saves and shows the chi2 
        vs z plot, flux plot, and csv file with additional information
        
        Parameters
        ----------
        z_start : int, optional
            The first value in the redshift list. Default = 0
        
        dz : float, optional
            The change in redshift. Default = 0.01
        
        z_end : int, optional
            The final value in the redshift list. Default = 10
        
        sigma : float, optional
            The significance level of the uncertainty in the redshift 
            found at the minimum chi-squared. Default = 1
            
        penalise : bool, optional
            If True, perform chi-squared penalisation with sslf. Default = True
        """
        
        self.dz = dz
        
        # Find the redshift 
        source = flux_zfind(self.transition, self.frequency, self.flux, self.uncertainty)
        z, chi2 = source.gauss_zfind(z_start, dz, z_end, sigma, penalise)
        params, perr = source.gauss_params()
        self.peaks, self.snrs, self.scales = source.gauss_sslf()
        self.min_z = z[np.argmin(chi2)]
        
        # Save the data
        self._plot_chi2(z, chi2, dz, 'Flux')
        self._plot_flux(params)
        neg, pos = self._z_uncert(z, chi2, sigma)
        self._export_csv('flux', neg, self.min_z, pos, params, perr, z, chi2)
        
    def zfft(self, z_start=0, dz=0.01, z_end=10, sigma=1):
        """ 
        Finds the best redshift by performing the fast fourier transform on the flux data. The
        chi-squared is caclulated at every redshift by iterating through delta-z. The most 
        likely redshift corresponds to the minimum chi-squared. Saves and shows the chi2 vs z 
        plot, flux plot, and csv file with additional information.
        
        Parameters
        ----------        
        z_start : int, optional
            The first value in the redshift list. Default = 0
            
        dz : float, optional
            The change in redshift. Default = 0.01
        
        z_end : int, optional
            The final value in the redshift list. Default = 10
        
        sigma : float, optional
            The significance level of the uncertainty in the redshift 
            found at the minimum chi-squared. Default = 1
        """
        
        self.dz = dz
        
        # Get the fft flux data
        self.ffreq, self.fflux = fft(self.frequency, self.flux)
        
        # Get the redshift
        fft_source = fft_zfind(self.transition, self.frequency, self.flux)
        z, chi2 = fft_source.fft_zfind(z_start, dz, z_end, sigma)
        params, perr = fft_source.fft_params()
        self.all_num_peaks = fft_source.peaks
        self.min_z = z[np.argmin(chi2)]
        
        # Save the data
        self._plot_chi2(z, chi2, dz, 'FFT')
        self._plot_fft_flux(params)
        neg, pos = self._z_uncert(z, chi2, sigma)
        self._export_csv('FFT', neg, self.min_z, pos, params, perr, z, chi2)
    
    def random_stats(self, n=100, radius=50, spread=1):
        """ 
        Iterate through n-1 randomly generated coordinates to find the 
        signal-to-noise ratio, number of pixels, and channel peaks for each.
        The first position is always the target, so if n=100, only 99 are
        random.
        
        Parameters
        ----------
        n : int, optional
            The number of radnom points to find statistics for. Default = 100
        
        radius : float, optional
            The radius of the image to find statistics (in pixels) centred on
            the given ra and dec. Default = 50
        
        spread : float, optional
            The minimum spread of random points (in pixels). Default = 1
        
        Returns
        -------
        snrs : list
            A list of signifcant point signal-to-noise ratios
            
        pixels : list
            The list of pixels for each random position
        
        peaks : list
            The channel location of all peaks in every position
        """
        
        snrs, pixels, peaks = self.source.random_analysis(n, radius, spread)
        self._plot_circle_points(self.source.coordinates, radius)        
        self._plot_pixels(snrs, pixels, peaks)
    
    def fft_per_pixel(self, size, z_start=0, dz=0.01, z_end=10, sigma=1):
        """ 
        Performs the FFT redshift finding method in a square around the 
        target right ascension and declination. Automatically saves and
        shows a heatmap of the distribution of the redshifts.
        
        Parameters
        ----------
        size : int
            size to perform the fft method around the centre. size=3 performs
            a 3x3, size=5 a 5x5, etc. 
                
        z_start : int, optional
            The first value in the redshift list. Default = 0
            
        dz : float, optional
            The change in redshift. Default = 0.01
        
        z_end : int, optional
            The final value in the redshift list. Default = 10
        
        sigma : float, optional
            The significance level of the uncertainty in the redshift 
            found at the minimum chi-squared. Default = 1
        """
        
        # Get the redshift
        fft_source = fft_zfind(self.transition, self.frequency, self.flux)
        z, chi2 = fft_source.fft_zfind(z_start, dz, z_end, sigma)
        self.best_z = z[np.argmin(chi2)]
        self.centre_coords = wcs2pix(self.ra, self.dec, self.source.hdr)
        
        # If size is even, make it odd
        if size %2 == 0:
            size += 1
        
        # Generate the x and y coordinates to iterate through
        matrix = np.zeros(size)
        for i in range(len(matrix)):
            matrix[i] = i-len(matrix)//2
        self.x_coords = matrix + self.centre_coords[0]
        self.y_coords = matrix + self.centre_coords[1]
        
        i = 0
        z_fft_pp = []
        d = count_decimals(dz)
        for y in reversed(self.y_coords):
            for x in self.x_coords:
                start = time()
        
                ra, dec = pix2wcs(x, y, self.source.hdr)
                
                gleam_0856 = fits2flux(self.image, ra, dec, self.aperture_radius, self.bvalue)
                freq = gleam_0856.get_freq()
                flux, uncert = gleam_0856.get_flux()

                zf = fft_zfind(self.transition, freq, flux)
                z, chi2 = zf.fft_zfind(z_start, dz, z_end)

                lowest_z = z[np.argmin(chi2)]
                lowest_z = round(lowest_z, d)

                z_fft_pp.append(lowest_z)

                end = time()
                elapsed = end - start
                print(f'{i+1}/{size**2}, {round(elapsed,2)} seconds')
                i+=1

        self.z_fft_pp = np.array_split(z_fft_pp, size)
        self._heatmap()
        return z_fft_pp

def main():
    image = '0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = [8, 56, 14.8]
    dec = [2, 24, 0.6, 1]
    
    # image = 'SPT_0345-47.contsub.clean.taper.image.fits'
    # ra = [3, 45, 10.77]
    # dec = [-47, 25, 39.5, -1]
    
    transition = 115.2712
    aperture_radius = 3
    bvalue = 3
    
    gleam_0856 = zfinder(image, ra, dec, transition, aperture_radius, bvalue)
    # gleam_0856.zflux(penalise=True)
    gleam_0856.zfft()
    # gleam_0856.random_stats(n=5)
    # gleam_0856.fft_per_pixel(size=11, z_start=4.28,  dz=0.0001, z_end=4.31)

if __name__ == '__main__':
    main()