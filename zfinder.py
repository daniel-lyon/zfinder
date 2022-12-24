# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Import functions
from time import time
from random import random
from decimal import Decimal
from astropy.wcs import WCS
from astropy.io import fits
from PyAstronomy import pyasl
from sslf.sslf import Spectrum
from warnings import filterwarnings
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from scipy.stats import binned_statistic
from photutils.background import Background2D
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry

# TODO: Move outside functions to their own util/functions file ?
def average_zeroes(data: list):
    ''' Take the average of adjacent points (left and right) if the value is zero.

    Parameters
    ----------
    data : `list`
        A list of data on which to operate

    Returns
    -------
    data : `list`
        An updated list of data with zeroes averaged
    '''
    for i, val in enumerate(data):
        if val == 0:
            data[i] = (data[i-1] + data[i+1])/2
    return data

def count_decimals(number: float):
    ''' count the amount of numbers in the decimal places

    Parameters
    ----------
    number : `float`
        The given float to count the number of decimal places

    Returns
    -------
    d : int
        The count of decimal places
    '''
    d = Decimal(str(number))
    d = abs(d.as_tuple().exponent)
    return d

def flatten_list(input_list: list):
    ''' Turns lists of lists into a single list

    Parameters
    ----------
    input_list : `list`
        A list of lists to flatten

    Returns
    -------
    flattened_list: `list`
        A flattened list
    '''
    flattened_list = []
    for array in input_list:
        for x in array:
            flattened_list.append(x)
    return flattened_list

class RedshiftFinder(object):
    def __init__(self, image: str, right_ascension: list, declination: list, aperture_radius: float,
            bvalue: float, num_plots=1, minimum_point_distance=1.0, circle_radius=50.0, warnings=False):

        # TODO: Make circle_radius automatic ?
        # TODO: Make use of *args and **kwargs ?

        '''
        `RedshiftFinder` looks at transition lines and attempts to find the best fitting red shift.
        This operates by plotting gaussian functions over the data and calculating the chi-squared
        at small changes in red shift. By taking the minimised chi-squared result, the most likely 
        red shift result is returned. Unrealistic and/or bad fits penalise the chi2 to be higher.

        Parameters
        ----------
        image : `str`
            An image of the `.fits` file type. Must be a three demensional image with axes Ra, Dec, 
            & Freq. 
        
        right_ascension : `list`
            The right ascension of the target object. Input as [h, m, s]
        
        declination : `list`
            The declination of the target object. input as [d, m, s, esign]. Esign is -1 or 1
            depending on if the decination is positive or negative.
        
        aperture_radius : `float`
            The radius of the aperture which the image used in pixels. This is converted to
            degrees when calculating the flux

        bvalue : 'float`
            The value of the BMAJ and BMIN vaues

        num_plots : `int`, optional
            The number  of random points to work with and plot. Default=1
        
        minimum_point_distance : `float`, optional
            The distance between random points in pixels. Defaul=1.0
        
        circle_radius : `float`, optional
            The smallest radius of the .fits image. Default=50.0

        warnings : `bool`, optional
            Optional setting to display warnings or not. If True, warnings are displayed.
            Default=False
        '''

        # Main data
        self.image = fits.open(image)
        self.hdr = self.image[0].header
        self.data = self.image[0].data[0]
        self.ra = right_ascension
        self.dec = declination
        self.aperture_radius = aperture_radius
        self.minimum_point_distance = minimum_point_distance
        self.num_plots = num_plots
        self.circle_radius = circle_radius

        # The area of the beam
        bmaj = bvalue/3600
        bmin = bmaj
        self.barea = 1.1331 * bmaj * bmin

        # Conversion of pixels to degrees for calculating flux
        self.pix2deg = self.hdr['CDELT2'] # unit conversion 

        # There are many, many warnings
        if not warnings:
            filterwarnings("ignore", module='photutils.background')
            filterwarnings("ignore", module='astropy.wcs.wcs')
            filterwarnings("ignore", module='scipy.optimize')

    @staticmethod
    def wcs2pix(ra: list, dec: list, hdr):

        # TODO: rounded x and y values are different to non rounded versions?

        ''' Convert right ascension and declination to x, y positional world coordinates

        Parameters
        ----------
        ra : `list`
            Right ascension coordinate given [h, m, s]
        
        dec : `list`
            Declination coordinate given as [d, m, s, esign]

        hdr : `astropy.io.fits.header.Header`
            The image header
        
        Returns
        -------
        x : `int`
            The transformed ra to world coordinate
        
        y : `int`
            The transformed dec to world coordinate
        '''

        w = WCS(hdr) # Get the world coordinate system
    
        # If there are more than 2 axis, drop them
        if hdr['NAXIS'] > 2:
            w = w.dropaxis(3) # stokes
            w = w.dropaxis(2) # frequency

        # Convert to decimal degrees
        ra = pyasl.hmsToDeg(ra[0], ra[1], ra[2])
        dec = pyasl.dmsToDeg(dec[0], dec[1], dec[2], esign=dec[3])

        # Convert world coordinates to pixel
        x, y = w.all_world2pix(ra, dec, 1)

        # Round to nearest integer
        x = int(np.round(x))
        y = int(np.round(y))

        return x, y
    
    @staticmethod
    def spaced_circle_points(num_points, circle_radius, centre_coords, minimum_spread_distance):
        ''' Generate points in a circle that are a minimum distance a part

        Parameters
        ----------
        num_points : `int`
            The number of points to plot. Defaults to 1
        
        circle_radius : `float`
            The radius in which points can be plotted around the centre. Defaults to 50
        
        centre_coords : `list`
            The centre of the circle.
        
        minimum_spread_distance : `float`
            The minimum distance between points.
        
        Returns
        -------
        points: `list`
            A list of points containing x, y coordinates.
        '''

        # centre_coords = [x,y] -> points = [[x,y]]
        points = [centre_coords]
        
        # Iterate through the number of points.
        for _ in range(num_points-1):

            # Keep generating the current point until it is at least the minimum distance away from all 
            while True:
                theta = 2 * np.pi * random() # choose a random direction
                r = circle_radius * random() # choose a random radius

                # Convert coordinates to cartesian
                x = r * np.cos(theta) + centre_coords[0]
                y = r * np.sin(theta) + centre_coords[1]

                # Find the distance between all the placed points
                distances = cdist([[x,y]], points, 'euclidean')
                min_distance = min(distances[0])
                
                # If the minimum distance is satisfied for all points, go to next point
                if min_distance >= minimum_spread_distance or len(points) == 1:
                    points.append([x,y])
                    break
        return points
    
    @staticmethod
    def gaussf(x, a, s, x0):

        # TODO: *args for number of gaussian functions?

        ''' Gaussian function used to fit to a data set

        Parameters
        ----------
        x : `list`
            the x-axis list
        
        a : `float`
            the amplitude of the gaussians
        
        s : `float`
            the standard deviation and width of the gaussians
        
        x0 : `float`
            the x-axis offset
        
        Returns
        -------
        y : `list`
            the corresponding y-axis 
        '''

        y = 0
        for i in range(1,12):
            y += (a * np.exp(-((x-i*x0) / s)**2)) # i = 1,2,3 ... 9, 10
        return y
    
    def fits_flux(self, position):
        ''' For every frequency channel, find the flux and associated uncertainty.

        Parameters
        ----------
        position : `list`
            an x,y coodinate to measure the flux at.

        Returns
        -------
        fluxes : 'numpy.ndarray'
            A numpy array of fluxes at every frequency
        
        uncertainties : 'numpy.ndarray'
            A numpy array of uncertainties at every frequency
        '''

        # Initialise array of fluxes and uncertainties to be returned
        fluxes = np.array([])
        uncertainties = np.array([])
        
        # For every page of the 3D data matrix, find the flux around a point (aperture)
        for page in self.data:

            # Setup the apertures 
            aperture = CircularAperture(position, self.aperture_radius)
            annulus = CircularAnnulus(position, r_in=2*self.aperture_radius, r_out=3*self.aperture_radius)

            # Uncertainty
            aperstats = ApertureStats(page, annulus) 
            rms  = aperstats.std 

            # Background
            bkg = Background2D(page, (50, 50)).background 

            # Aperture sum of the fits image minus the background
            apphot = aperture_photometry(page - bkg, aperture)
            apsum = apphot['aperture_sum'][0]

            # Calculate corrected flux
            total_flux = apsum*(self.pix2deg**2)/self.barea
            fluxes = np.append(fluxes, total_flux)
            uncertainties = np.append(uncertainties, rms)

        return fluxes, uncertainties      

    def zfind(self, ftransition, z_start=0, dz=0.01, z_end=10):

        # TODO: change 'x-axis to GHz' to detect the unit prefix ?

        ''' For every point in coordinates, find the flux and uncertainty. Then find the significant
        lines with the line finder. For each point, iterate through all redshift values and calculate
        the chi-squared that corresponds to that redshift by fitting gaussians to overlay the flux. If
        The points found by the line finder do not match within 4 frequency channels of the gaussian
        peaks, penalise the chi-squared at that redshift by a factor of 1.2.

        Parameters
        ----------
        ftransition : `float`
            The first transition frequency (in GHz) of the target element/molecule/etc
        
        z_start : `float`, optional
            The starting value of redshift value. Default = 0
        
        dz : `float`, optional
            The change in redshift to iterature through. Default = 0.01
        
        z_end : `float`, optional
            The final redshift  value. Default = 10
        
        Returns
        -------
        self.all_lowest_z : list
            A list of the lowest measured redshift values with length equal to the number of points.
        '''
        
        # Object values
        self.dz = dz
        self.ftransition = ftransition 
        self.all_chi2 = []
        self.all_flux = []
        self.all_params = []
        self.all_snrs = []
        self.all_scales = []
        self.all_lowest_z = []
        self.plot_colours = []

        # Setup for spaced random points
        self.centre_x, self.centre_y = self.wcs2pix(self.ra, self.dec, self.hdr) 

        # Generate the random coordinates for statistical analysis
        self.coordinates = self.spaced_circle_points(self.num_plots, self.circle_radius, 
            centre_coords=[self.centre_x, self.centre_y], minimum_spread_distance=self.minimum_point_distance)
        
        # Convert the x-axis to GHz
        freq_start = self.hdr['CRVAL3']/10**9 # GHz
        freq_incr = self.hdr['CDELT3']/10**9 # GHz
        freq_len = np.shape(self.data)[0] # length
        freq_end = freq_start + freq_len * freq_incr # where to stop
        self.xAxisFlux = np.linspace(freq_start, freq_end, freq_len) # axis to plot

        # Create the redshift values to iterate through
        self.z = np.arange(z_start, z_end+dz, dz) # number of redshifts to iterate through

        start = time() # Measure how long it takes to execute 

        # For every coodinate point, find the associated flux and uncertainty 
        for index, coord in enumerate(self.coordinates):

            # Initialise arrays for each coordinate
            chi2_array = [] 
            param_array = []

            # Get fluxes and uncertainties at each point
            y_flux, uncert = self.fits_flux(coord)
            self.all_flux.append(y_flux)

            uncert = average_zeroes(uncert) # average 0's from values left & right
            y_flux *= 1000; uncert *= 1000 # convert from uJy to mJy
            
            # Create a line finder to find significant points
            s = Spectrum(y_flux)
            s.find_cwt_peaks(scales=np.arange(4,10), snr=3)
            spec_peaks = s.channel_peaks
            spec_peaks = np.sort(spec_peaks) # sort the peaks from left to right instead of right to left
            num_spec_peaks = len(spec_peaks)

            # Calculate the ratio of the snrs and scales
            snrs = [round(i,2) for i in s.peak_snrs] # the snrs of the peaks
            scales = [i[1]-i[0] for i in s.channel_edges] # the scales of the peaks

            # For every redshift, calculate the corresponding chi squared value
            for ddz in self.z:
                loc = ftransition/(1+ddz) # location of the gaussian peaks
                
                # Determine the best fitting parameters
                try:
                    params, covariance = curve_fit(lambda x, a, s: self.gaussf(x, a, s, x0=loc), 
                        self.xAxisFlux, y_flux, bounds=[[0, (1/8)], [max(y_flux), (2/3)]], absolute_sigma=True) # best fit
                except RuntimeError:
                    chi2_array.append(max(chi2_array)) # if no returned parameters, set the chi-squared for this redshift to the maximum
                    continue
                
                # Using the best fit parameters, calculate the chi2 corresponding to this redshift {ddz}
                f_exp = self.gaussf(self.xAxisFlux, a=params[0], s=params[1], x0=loc) # expected function
                chi2 = sum(((y_flux - f_exp) / uncert)**2)

                # Find the location of the expected gaussian peaks
                if num_spec_peaks != 0:
                    exp_peak = np.argsort(f_exp)[-num_spec_peaks:] # the index of the gaussian peaks
                    exp_peak = np.sort(exp_peak) # sort ascending
                else:
                    exp_peak = []

                # Calculate the peak_distance beween the spectrum and expected peaks
                delta_peaks = []
                for p1, p2 in zip(spec_peaks, exp_peak):
                    delta_peaks.append(abs(p1-p2))
                peak_distance = sum(delta_peaks)

                # If the peak_distance is greater than the number of spectrum peaks multiplied by 3 channels,
                # or if there are no peaks, penalise the chi2 by multiplying it my 1.2
                if peak_distance > num_spec_peaks*3 or num_spec_peaks < 2:
                    chi2 *= 1.2

                # Append parameters for use later
                chi2_array.append(chi2)
                param_array.append(params)

            # Find the colours to map to each chi2
            min_plot_chi2 = min(chi2_array)
            if index == 0:
                self.plot_colours.append('black') # the original
                target_chi2 = min_plot_chi2
            elif min_plot_chi2 <= target_chi2:
                self.plot_colours.append('red') # if chi2 lower than original
            elif min_plot_chi2 > target_chi2 and min_plot_chi2 <= 1.05*target_chi2:
                self.plot_colours.append('gold') # if chi2 within 5% above the original
            else:
                self.plot_colours.append('green') # if chi2 more than 5% above the original

            # Find the lowest redshift of each source point
            lowest_index = np.argmin(chi2_array)
            lowest_redshift = self.z[lowest_index]
            
            # Append parameters for use later
            self.all_chi2.append(chi2_array)
            self.all_params.append(param_array)
            self.all_snrs.append(snrs)
            self.all_scales.append(scales)
            self.all_lowest_z.append(lowest_redshift)

            print(f'{index+1}/{len(self.coordinates)} completed..')

        # Return an array with the lowest redshift from each source
        end = time()
        print(f'Data processed in {round((end-start)/60, 3)} minutes')
        return self.all_lowest_z

class zf_plotter(RedshiftFinder):
    def __init__(self, obj, plots_per_page=25):

        # TODO: Add number of rows (or maybe columns instead?) to automatically calculate plots per page
        # TODO: Use np.ceil instead of checking if pages == 0
        # TODO: Change saving to work for multiple pages

        ''' `zf_plotter` takes a `RedshiftFinder` object as an input to easily compute plots
        used for statistical analysis.
        '''

        # Values
        self.obj = obj

        # The number of pages of data to plot
        self.pages = self.obj.num_plots // plots_per_page
        if self.pages == 0:
            self.pages = 1
        
        # The number of rows and columns used for each page
        if self.obj.num_plots >= 5:
            self.cols = 5
            self.rows = self.obj.num_plots // (self.cols * self.pages)
            self.squeeze = True
        else:
            self.cols = 1
            self.rows = 1
            self.squeeze = False
        
    @staticmethod
    def plot_peaks(y_axis, x_axis, plot_type):
        ''' Plot the found peaks of a line finder on top of another plot.

        Parameters
        ----------
        y_axis : `list`
            The y-axis with which to find the significant peaks

        x_axis : `list`
            The x-axis with which to plot the significant peaks

        plot_type : 'matplotlib.axes._subplots.AxesSubplot'
            : The figure to plot the peaks on 
        '''

        s = Spectrum(y_axis)
        s.find_cwt_peaks(scales=np.arange(4,10), snr=3)
        peaks = s.channel_peaks

        scales = [i[1]-i[0] for i in s.channel_edges]
        snrs = [round(i,2) for i in s.peak_snrs]
        snr_text = [0.35 if i%2==0 else -0.35 for i in range(len(s.peak_snrs))]
        scale_text = [0.40 if i%2==0 else -0.40 for i in range(len(scales))]

        for i, snr, s_text, sc_text, sc in zip(peaks, snrs, snr_text, scale_text, scales):
            plot_type.plot(x_axis[i], y_axis[i], marker='o', color='blue')
            plot_type.text(x_axis[i], s_text, s=f'snr={snr}', color='blue')
            plot_type.text(x_axis[i], sc_text, s=f'scale={sc}', color='blue')

    def plot_points(self, savefile=None):
        ''' Plot the distribution of coordinates

        Parameters
        ----------
        savefile : `str`, None, optional
            The filename of the saved figure. Default = None
        '''

        circle_points = np.transpose(self.obj.coordinates)
        points_x = circle_points[0, :] # all x coordinates except the first which is the original
        points_y = circle_points[1, :] # all y coordinates except the first which is the original
        circ = plt.Circle((self.obj.centre_x, self.obj.centre_y), self.obj.circle_radius, fill=False, color='blue')
        fig, ax = plt.subplots()
        fig.set_figwidth(7)
        fig.set_figheight(7)
        ax.add_patch(circ)
        plt.title('Distribution of spaced random points')
        plt.scatter(points_x, points_y, color=self.obj.plot_colours)
        plt.xlim(-self.obj.circle_radius-1+self.obj.centre_x, self.obj.circle_radius+1+self.obj.centre_x)
        plt.ylim(-self.obj.circle_radius-1+self.obj.centre_y, self.obj.circle_radius+1+self.obj.centre_y)
        plt.xlabel('x')
        plt.ylabel('y')
        if savefile != None:
            plt.savefig(f'{savefile}', dpi=200)
        plt.show()
    
    def plot_chi2(self, savefile=None):
        ''' Plot the chi-squared vs redshift at every coordinate

        Parameters
        ----------
        savefile : `str`, None, optional
            The filename of the saved figure. Default = None
        '''

        all_chi2 = np.array_split(self.obj.all_chi2, self.pages)
        AllColours = np.array_split(self.obj.plot_colours, self.pages)
        AllCoords = np.array_split(self.obj.coordinates, self.pages)
        
        # Plot the reduced chi-squared histogram(s) across multiple pages (if more than one)
        for chi2, colours, coordinates in zip(all_chi2, AllColours, AllCoords):

            # Setup the figure and axes
            fig, axs = plt.subplots(self.rows, self.cols, tight_layout=True, sharex=True, squeeze=self.squeeze)
            fig.supxlabel('Redshift')
            fig.supylabel('$\chi^2$', x=0.01)
            axs = axs.flatten()

            # Plot the chi-squared(s) and redshift
            for index, (c2, color, coordinate) in enumerate(zip(chi2, colours, coordinates)):
                lowest_redshift = self.obj.z[np.argmin(c2)]
                axs[index].plot(self.obj.z, c2, color=color)
                axs[index].plot(lowest_redshift, min(c2), 'bo', markersize=5)
                coord = np.round(coordinate, 2)
                axs[index].set_title(f'x,y = {coord}. Min Chi2 = {round(min(c2), 2)}')
                axs[index].set_yscale('log')

            # Save the file and show
            if savefile != None:
                fig.savefig(f'{savefile}', dpi=200)
            plt.show()

    def plot_flux(self, savefile=None):

        # TODO: Change outside boarder colour to use all_colours

        ''' Plot the flux vs frequency at every coordinate

        Parameters
        ----------
        savefile : `str`, None, optional
            The filename of the saved figure. Default = None
        '''

        # Split data into pages
        all_chi2 = np.array_split(self.obj.all_chi2, self.pages)
        all_flux = np.array_split(self.obj.all_flux, self.pages)
        all_params = np.array_split(self.obj.all_params, self.pages)
        d = count_decimals(self.obj.dz) # decimal places to round to

        # Plot the reduced chi-squared histogram(s) across multiple pages (if more than one)
        for fluxes, chi2, params in zip(all_flux, all_chi2, all_params):

            # Setup the figure and axes
            fig, axs = plt.subplots(self.rows, self.cols, tight_layout=True, 
                sharex=True, sharey=True, squeeze=self.squeeze)
            fig.supxlabel('Frequency $(GHz)$')
            fig.supylabel('Flux $(mJy)$')
            axs = axs.flatten()

            # Plot the flux(s) and best fit gaussians
            for index, (flux, c2, param) in enumerate(zip(fluxes, chi2, params)):
                lowest_index = np.argmin(c2)
                lowest_redshift = self.obj.z[lowest_index]
                axs[index].plot(self.obj.xAxisFlux, flux, color='black', drawstyle='steps-mid')
                axs[index].plot(self.obj.xAxisFlux, self.obj.gaussf(self.obj.xAxisFlux, *param[lowest_index], 
                    x0=self.obj.ftransition/(1+lowest_redshift)), color='red')
                axs[index].margins(x=0)
                axs[index].fill_between(self.obj.xAxisFlux, flux, 0, where=(flux > 0), color='gold', alpha=0.75)
                axs[index].set_title(f'z={round(lowest_redshift, d)}')
                self.plot_peaks(flux, self.obj.xAxisFlux, axs[index])
            
            # Save the file and show
            if savefile != None:
                fig.savefig(f'{savefile}', dpi=200)
            plt.show()
    
    def plot_hist_chi2(self, savefile=None):
        ''' Plot a histogram of the chi-squared at every coordinate

        Parameters
        ----------
        savefile : `str`, None, optional
            The filename of the saved figure. Default = None
        '''

        # Initialise return arrays
        all_std = []
        all_mean = []

        # Split data into pages
        all_chi2 = np.array_split(self.obj.all_chi2, self.pages)
        colours = np.array_split(self.obj.plot_colours, self.pages)
        
        # Plot the reduced chi-squared histogram(s) across multiple pages (if more than one)
        for page, color in zip(all_chi2, colours):

            # Setup the figure and axes
            fig, axs = plt.subplots(self.rows, self.cols, tight_layout=True, sharey=True, squeeze=self.squeeze)
            fig.supxlabel('$\chi^2$') 
            fig.supylabel('Count (#)')
            axs = axs.flatten()

            for index, (chi2, c) in enumerate(zip(page, color)):
                reduced_chi2 = np.array(chi2)/(len(chi2)-1)
                axs[index].hist(reduced_chi2, 20, color=c)
                axs[index].set_yscale('log')

                # Calculate the mean and standard deviation of each bin
                mean = binned_statistic(reduced_chi2, reduced_chi2, 'mean', 20)
                std = binned_statistic(reduced_chi2, reduced_chi2, 'std', 20)

                # Flip the arrays to be read left to right and append to the return arrays
                all_mean.append(np.flip(mean[0]))
                all_std.append(np.flip(std[0]))

            # Save the file and show
            if savefile != None:
                fig.savefig(f'{savefile}', dpi=200)
            plt.show()

        return all_mean, all_std

    def plot_snr_scales(self, savefile=None):
        ''' Plot a hsitogram of the snr and scale

        Parameters
        ----------
        savefile : `str`, None, optional
            The filename of the saved figure. Default = None
        '''

        snrs = flatten_list(self.obj.all_snrs)
        scales = flatten_list(self.obj.all_scales)

        # Setup the figure and axes
        fig, (ax_snr, ax_scale) = plt.subplots(1, 2, sharey=True)
        fig.supylabel('Count (#)')

        # Plot the snrs histogram(s)
        ax_snr.hist(snrs, 20)
        ax_snr.set_title('SNR histogram')
        ax_snr.set_xlabel('SNR')

        # Plot the scales histogram
        ax_scale.hist(scales, [8,10,12,14,16,18,20])
        ax_scale.set_title('Scales Histogram')
        ax_scale.set_xlabel('Scale')

        # Save the file and show
        if savefile != None:
            fig.savefig(f'{savefile}', dpi=200)
        plt.show()

if __name__ == '__main__':
    
    image = '0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = [8, 56, 14.8] # Right Ascenion (h, m, s)
    dec = [2, 24, 0.6, 1] # Declination (d, m, s, sign)
    aperture_radius = 3 # Aperture Radius (pixels)
    bvalue = 3 # BMAJ & BMIN (arcseconds)
    num_plots = 1 # Number of plots to make (must be a multiple of 5 or 1)
    min_sep = 1 # Minimum separation between points (pixels)
    circle_radius = 87 # Radius of the largest frequency (pixels)
    ftransition = 115.2712 # the first transition in GHz
    z_start = 0 # initial redshift
    dz = 0.01 # change in redshift
    z_end = 10 # final redshift

    zfind1 = RedshiftFinder(image, ra, dec, aperture_radius, bvalue, num_plots, min_sep, circle_radius)
    zfind1.zfind(ftransition, z_start, dz, z_end)

    zf1 = zf_plotter(zfind1)
    zf1.plot_points()
    zf1.plot_flux()
    zf1.plot_chi2()
    zf1.plot_hist_chi2()
    zf1.plot_snr_scales()