import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt

from time import time
from astropy.wcs import WCS
from decimal import Decimal
from .uncertainty import z_uncert
from .flux_zfind import Template, gaussf
from .fits2flux import Fits2flux, wcs2pix, pix2wcs
from .fft_zfind import Fourier, double_damped_sinusoid

# Get the prefix of a unit from an exponent
_prefix = {-24 : 'y', -21 : 'z', -18 : 'a',
        -15 : 'f', -12 : 'p',  -9 : 'n',
        -6 : '\u03BC', -3 : 'm',   0 : '', 
        3 : 'k',   6 : 'M',   9 : 'G',
        12 : 'T',  15 : 'P',  18 : 'E', 
        21 : 'Z',  24 : 'Y'}

def _count_decimals(number: float):
    """ Count the number of decimals in a float """
    d = Decimal(str(number))
    d = abs(d.as_tuple().exponent)
    return d

def _flatten_list(input_list):
    """ Turns lists of lists into a single list """
    flattened_list = []
    for array in input_list:
        for x in array:
            flattened_list.append(x)
    return flattened_list

class zfinder():
    def __init__(self, image, ra, dec, transition, aperture_radius, bvalue):
        """
        zfinder is a wrapper on individual function files fit2flux, Template, and Fourier.
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
        self._image = image
        self._ra = ra
        self._dec = dec
        self._transition = transition
        self._aperture_radius = aperture_radius
        self._bvalue = bvalue
        
        self._source = Fits2flux(image, ra, dec, aperture_radius, bvalue)
        self._frequency = self._source.get_freq()
        self._flux, self._uncertainty = self._source.get_flux()
    
    @staticmethod
    def __plot_chi2(z, chi2, dz, title):
        min_chi2 = min(chi2)
        min_z = z[np.argmin(chi2)]
        d = _count_decimals(dz)
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
    def __export_csv(title, neg, min_z, pos, params, perr, z, chi2):
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
    def __plot_pixels(snrs, pixels, peaks):
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
            
            # If there are no lines
            if len(snr) == 0:
                continue
            
            # bad - no redshift or extreme redshift (z>15)
            if num_pks < 2 or num_pks > 4: 
                green_points_x.append(pix)
                green_points_y.append(snr)
                continue
            
            # Two peaks is a good sign its real
            if num_pks == 2:
                pk_diffs = np.abs(np.diff(pk)) # needs to be at least 200 channels  
                if pk_diffs > 200:
                    red_points_x.append(pix)
                    red_points_y.append(snr)
                else:
                    orange_points_x.append(pix)
                    orange_points_y.append(snr)
            
            # Check 3 or 4 peaks
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
        
        # The number of points irrespective of how many sslf lines found
        blue_points = len(blue_points_x)
        green_points = len(green_points_x)
        orange_points = len(orange_points_x)
        red_points = len(red_points_x)

        # Flatten lists of lists to one big list
        blue_points_x = _flatten_list(blue_points_x)
        blue_points_y = _flatten_list(blue_points_y)
        green_points_x = _flatten_list(green_points_x)
        green_points_y = _flatten_list(green_points_y)
        orange_points_x = _flatten_list(orange_points_x)
        orange_points_y = _flatten_list(orange_points_y)
        red_points_x = _flatten_list(red_points_x)
        red_points_y = _flatten_list(red_points_y)
        
        # Random points
        with open(f'snr_vs_pix.csv', 'w', newline='') as f:
            wr = csv.writer(f)
            rows = zip(['Blue', 'Green', 'Yellow', 'Red'], 
                       [blue_points, green_points, orange_points, red_points])
            for row in rows:
                wr.writerow(row)
            wr.writerow([])

        # Make the plot
        plt.figure(figsize=(20,9))
        plt.scatter(blue_points_x, blue_points_y, s=60, marker='*', color='blue')
        plt.scatter(green_points_x, green_points_y, s=60, marker='X', color='green')
        plt.scatter(orange_points_x, orange_points_y, s=60, marker='D', color='darkorange')
        plt.scatter(red_points_x, red_points_y, s=60, marker='s', color='red')
        plt.title(f'No. Random Points = {len(snrs)-1}', fontsize=20)
        plt.xlabel('No. of Pixels', fontsize=20)
        plt.ylabel('SNR', fontsize=20)
        plt.legend(['Target', 'No significance', 'Low significance', 'High significance'])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig('SNR vs Pix.png')
        plt.show()
        
    @staticmethod
    def __plot_circle_points(coords, radius, header):
        """ Plot the corodinates of the random points """
        # If there are more than 2 axis, drop them
        w = WCS(header) # Get the world coordinate system
        if header['NAXIS'] > 2:
            w = w.dropaxis(3) # stokes
            w = w.dropaxis(2) # frequency
        
        circle_points = np.transpose(coords)
        points_x = circle_points[0, :] # all x coordinates except the first which is the original
        points_y = circle_points[1, :] # all y coordinates except the first which is the original
        centre_x = points_x[0]
        centre_y = points_y[0]
        circ = plt.Circle((centre_x, centre_y), radius, fill=False, color='blue', label='_nolegend_')
        fig, ax = plt.subplots(subplot_kw={'projection': w})
        fig.set_figwidth(7)
        fig.set_figheight(7)
        ax.add_patch(circ)
        plt.scatter(points_x[0], points_y[0], color='black')
        plt.scatter(points_x[1:], points_y[1:], color='blue')
        plt.title(f'Distribution of {len(coords)-1} spaced random points')
        plt.xlim(-radius-1+centre_x, radius+1+centre_x)
        plt.ylim(-radius-1+centre_y, radius+1+centre_y)
        plt.xlabel('RA', fontsize=15)
        plt.ylabel('DEC', fontsize=15)
        plt.legend(['Target', 'Random'])
        plt.savefig('Point Distribution.png', dpi=200)
        plt.show()
        
    @staticmethod
    def __export_heatmap_csv(delta_z):
        """ Export matrix of delta redshifts"""
        with open(f'heatmap_data.csv', 'w', newline='') as f:
            wr = csv.writer(f)
            for row in delta_z:
                wr.writerow(row)
            wr.writerow([])
        
    def __heatmap(self):
        """ Grid heatmap of the redshift found per pixel """
    
        # Velocity of the gas
        velocities = [3*10**5*((((1+self._best_z)**2 - 1) / ((1+self._best_z)**2 + 1)) - (((1+i)**2 - 1) / ((1+i)**2 + 1))) for i in self._z_fft_pp] # km/s
        self._z_fft_pp = np.array(self._z_fft_pp).tolist()
        d = _count_decimals(self._best_z)

        # Calculate the difference between the redshifts
        all_deltas = []
        for i in range(len(self._z_fft_pp)):
            deltas = []
            for j in range(len(self._z_fft_pp)):
                dat = float(self._z_fft_pp[i][j])
                delta_z = dat - self._best_z
                delta_z = round(delta_z, d)
                deltas.append(delta_z)
            all_deltas.append(deltas)
        
        # Export the redshifts
        self.__export_heatmap_csv(all_deltas)
        
        # If there are more than 2 axis, drop them
        w = WCS(self._source._hdr) # Get the world coordinate system
        if self._source._hdr['NAXIS'] > 2:
            w = w.dropaxis(3) # stokes
            w = w.dropaxis(2) # frequency
        
        # Show a heatmap of the redshifts
        velocities = np.flipud(velocities)
        plt.subplot(projection=w)
        hm = plt.imshow(velocities, cmap='bwr', interpolation='nearest', vmin=-300, vmax=300, 
                extent=[self._x_coords[0], self._x_coords[-1], self._y_coords[0], self._y_coords[-1]], 
                origin='lower')
        plt.colorbar(hm)
        plt.xlabel('RA', fontsize=15)      
        plt.ylabel('DEC', fontsize=15)      
        plt.savefig('FFT PP Heatmap.png')
        plt.show()

    def _z_uncert(self, z, chi2, sigma, reduction):
        """ Caclulate the uncertainty on the best fitting redshift """
        reduced_sigma = sigma**2 / (len(self._flux) - 2*len(self._peaks) - 1)
        neg, pos = z_uncert(z, chi2, reduced_sigma, reduction)
        return neg, pos
    
    def _plot_blind_lines(self):
        """ Helper function to plot sslf lines in the found flux """
        text_offset_high = max(self._flux)/20
        text_offset_low = 0.4*text_offset_high
        
        for i, line in enumerate(self._peaks):
            x = self._frequency[line]
            y = self._flux[line]
            plt.plot(x, y, 'bo')
            plt.text(x, y+text_offset_high, f'snr={self._snrs[i]}', color='blue')
            plt.text(x, y+text_offset_low, f'scale={self._scales[i]}', color='blue')
    
    def __plot_flux(self, params):
        """ Plot the flux with best fitting redshift """
        x_unit, y_unit = self._source.get_exponents()
        d = _count_decimals(self._dz)
        x0 = self._transition/(1+self._min_z)
        plt.figure(figsize=(20,9))
        plt.plot(self._frequency, np.zeros(len(self._frequency)), color='black', linestyle=(0, (5, 5)))
        plt.plot(self._frequency, self._flux, color='black', drawstyle='steps-mid')
        plt.plot(self._frequency, gaussf(self._frequency, *params, x0), color='red')
        self._plot_blind_lines()
        plt.margins(x=0)
        plt.fill_between(self._frequency, self._flux, 0, where=(np.array(self._flux) > 0), color='gold', alpha=0.75)
        plt.title(f'Flux z={round(self._min_z, d)}', fontsize=15)
        plt.xlabel(f'Frequency $({_prefix[x_unit]}Hz)$', fontsize=15)
        plt.ylabel(f'Flux $({_prefix[y_unit]}Jy)$', fontsize=15)
        plt.savefig('Flux Best Fit.png')
        plt.show()
    
    def __plot_fft_flux(self, params):
        """ Plot the FFT with best fitting redshift"""
        d = _count_decimals(self._dz)
        plt.figure(figsize=(20,9))
        plt.plot(self._ffreq, self._fflux, color='black', drawstyle='steps-mid')
        plt.plot(self._ffreq, np.zeros(len(self._fflux)), color='black', linestyle=(0, (5, 5)))
        plt.plot(self._ffreq, double_damped_sinusoid(self._ffreq, *params, 
            self._dz, self._frequency[0], self._transition), color='red')
        plt.title(f'FFT z={round(self._min_z, d)}', fontsize=15)
        plt.xlabel('Scale', fontsize=15)
        plt.ylabel('Amplitude', fontsize=15)
        plt.margins(x=0)
        plt.savefig('FFT Best Fit.png', dpi=200)
        plt.show()
    
    def zflux(self, z_start=0, dz=0.01, z_end=10, sigma=1, penalise=True, reduction=False):
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
        
        reduction : bool, optional
            Can greatly reduce uncertainty. Choose to reduce all chi2 values
            such that the minimum chi2 is 0. Default = False
        """
        self._dz = dz
        
        # Find the redshift 
        self._flux_source = Template(self._transition, self._frequency, self._flux, self._uncertainty)
        z, chi2 = self._flux_source.zfind(z_start, dz, z_end, sigma, penalise)
        if reduction:
            min_chi2 = min(chi2)
            chi2 = [i - min_chi2 for i in chi2]
        params, perr = self._flux_source.parameters()
        self._peaks, self._snrs, self._scales = self._flux_source.sslf()
        self._min_z = z[np.argmin(chi2)]
        
        # Save the data
        self.__plot_chi2(z, chi2, dz, 'Flux')
        self.__plot_flux(params)
        neg, pos = self._z_uncert(z, chi2, sigma, reduction=reduction)
        self.__export_csv('flux', neg, self._min_z, pos, params, perr, z, chi2)
        
    def zfft(self, z_start=0, dz=0.01, z_end=10, sigma=1, reduction=False):
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
        
        reduction : bool, optional
            Can greatly reduce uncertainty. Choose to reduce all chi2 values
            such that the minimum chi2 is 0. Default = False
        """
        
        self._dz = dz
        
        # Get the fft flux data
        self._ffreq, self._fflux = Fourier.fft(self._frequency, self._flux)
        
        # Get the redshift
        fft_source = Fourier(self._transition, self._frequency, self._flux)
        z, chi2 = fft_source.zfind(z_start, dz, z_end, sigma)
        if reduction:
            min_chi2 = min(chi2)
            chi2 = [i - min_chi2 for i in chi2]
        params, perr = fft_source.fft_params()
        self._all_num_peaks = fft_source._peaks
        self._min_z = z[np.argmin(chi2)]
        
        # Save the data
        self.__plot_chi2(z, chi2, dz, 'FFT')
        self.__plot_fft_flux(params)
        neg, pos = self._z_uncert(z, chi2, sigma, reduction=reduction)
        self.__export_csv('FFT', neg, self._min_z, pos, params, perr, z, chi2)
    
    def random_stats(self, n=100, radius=50, spread=1):
        """ 
        Iterate through n-1 randomly generated coordinates to find the 
        signal-to-noise ratio, number of pixels, and channel peaks for each.
        The first position is always the target: if n=100, only 99 are
        random, with the first being the target.
        
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
        
        snrs, pixels, peaks = self._source.random_analysis(n, radius, spread)
        self.__plot_circle_points(self._source._coordinates, radius, self._source._hdr)        
        self.__plot_pixels(snrs, pixels, peaks)
    
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
        fft_source = Fourier(self._transition, self._frequency, self._flux)
        z, chi2 = fft_source.zfind(z_start, dz, z_end, sigma)
        self._best_z = z[np.argmin(chi2)]
        self._centre_coords = wcs2pix(self._ra, self._dec, self._source._hdr)
        
        # If size is even, make it odd
        if size %2 == 0:
            size += 1
        
        # Generate the x and y coordinates to iterate through
        matrix = np.zeros(size)
        for i in range(len(matrix)):
            matrix[i] = i-len(matrix)//2
        self._x_coords = matrix + self._centre_coords[0]
        self._y_coords = matrix + self._centre_coords[1]
        
        i = 0
        z_fft_pp = []
        d = _count_decimals(dz)
        for y in reversed(self._y_coords):
            for x in self._x_coords:
                start = time()
        
                ra, dec = pix2wcs(x, y, self._source._hdr)
                
                gleam_0856 = Fits2flux(self._image, ra, dec, self._aperture_radius, self._bvalue)
                freq = gleam_0856.get_freq()
                flux, uncert = gleam_0856.get_flux()

                zf = Fourier(self._transition, freq, flux)
                z, chi2 = zf.zfind(z_start, dz, z_end)

                lowest_z = z[np.argmin(chi2)]
                lowest_z = round(lowest_z, d)

                z_fft_pp.append(lowest_z)

                end = time()
                elapsed = end - start
                
                remaining = datetime.timedelta(seconds=round((size**2-(i+1))*elapsed))
                
                print(f'{i+1}/{size**2}, took {round(elapsed,2)} seconds, approx {remaining} remaining')
                i+=1

        self._z_fft_pp = np.array_split(z_fft_pp, size)
        self.__heatmap()
        return z_fft_pp