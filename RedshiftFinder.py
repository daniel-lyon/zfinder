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

def count_decimals(number):
    d = Decimal(str(number))
    d = abs(d.as_tuple().exponent)
    return d

def list_div_list(numerator_list, denominator_list):
    return_list = []
    for num, den in zip(numerator_list, denominator_list):
        return_list.append(num/den)
    return return_list

def flatten_list(input_array):
    return_array = []
    for array in input_array:
        for x in array:
            return_array.append(x)
    return return_array

class RedshiftFinder(object):
    '''
    `RedshiftFinder` looks at transition lines and attempts to find the best fitting red shift.
    This operates by plotting gaussian functions over the data and calculating the chi-squared
    at small changes in red shift. By taking the minimised chi-squared result, the most likely 
    red shift result is returned
    '''

    def __init__(self, image, rightAscension, declination, apertureRadius, bvalue, warnings=False):
        self.image = fits.open(image)
        self.hdr = self.image[0].header
        self.data = self.image[0].data[0]
        self.ra = rightAscension
        self.dec = declination
        self.apertureRadius = apertureRadius
        self.bvalue = bvalue

        # There are many, many warnings
        if not warnings:
            filterwarnings("ignore", module='photutils.background')
            filterwarnings("ignore", module='astropy.wcs.wcs')
            filterwarnings("ignore", module='scipy.optimize')

    @staticmethod
    def wcs2pix(ra, dec, hdr):
        '''
        Convert right ascension and declination to x, y positional world coordinates
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
        xcor = np.round(x)
        ycor = np.round(y)

        return xcor, ycor
    
    @staticmethod
    def spaced_circle_points(num_points, circle_radius, centre_coords, minimum_spread_distance):
        '''
        Generates points in a circle that are a minimum distance a part
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
    def arrayfix(array):
        '''
        Dividing by an array with zeroes will error. Take the average of adjacent points. 
        '''
        for i, val in enumerate(array):
            if val == 0:
                array[i] = (array[i-1] + array[i+1])/2
        return array
    
    @staticmethod
    def gaussf(x, a, s, x0):
        '''
        Gaussian function used to fit the data
        '''
        y = 0
        for i in range(1,12):
            y += (a * np.exp(-((x-i*x0) / s)**2)) # i = 1,2,3 ... 9, 10
        return y
    
    def fits_flux(self, position):
        '''
        For every frequency channel, find the flux and associated uncertainty.
        '''

        # Initialise array of fluxes and uncertainties to be returned
        fluxes = []
        uncertainties = []

        # Conversion of pixel to degrees
        pix2deg = self.hdr['CDELT2'] # unit conversion
        bmaj = self.bvalue/3600
        bmin = bmaj
        barea = 1.1331 * bmaj * bmin

        # For every page of the 3D data matrix, find the flux around a point (aperture)
        for page in self.data:

            # Setup the apertures 
            aperture = CircularAperture(position, self.apertureRadius)
            annulus = CircularAnnulus(position, r_in=2*self.apertureRadius, r_out=3*self.apertureRadius)

            # Uncertainty
            aperstats = ApertureStats(page, annulus) 
            rms  = aperstats.std 

            # Background
            bkg = Background2D(page, (50, 50)).background 

            # Aperture sum of the fits image minus the background
            apphot = aperture_photometry(page - bkg, aperture)
            apsum = apphot['aperture_sum'][0]

            # Calculate corrected flux
            total_flux = apsum*(pix2deg**2)/barea
            fluxes.append(total_flux)
            uncertainties.append(rms)

        return np.array(fluxes), np.array(uncertainties)

    def circle_point_vars(self, minimum_point_distance, num_plots, circle_radius):
        '''
        Values for the user to specify about the circular points
        '''
        
        self.centre_x, self.centre_y = self.wcs2pix(self.ra, self.dec, self.hdr)
        self.plotColours = []
        self.minimum_point_distance = minimum_point_distance
        self.num_plots = num_plots
        if self.num_plots < 4 and self.num_plots > 1:
            raise ValueError('num_plots must be >= 5 or num_plots == 1')
        self.circle_radius = circle_radius
        self.points = [self.centre_x, self.centre_y]

    def zfind(self, ftransition, zStart=0, dz=0.01, zEnd=10):
        
        # Object values
        self.dz = dz
        self.ftransition = ftransition 
        self.allChi2 = []
        self.allFlux = []
        self.allParams = []
        self.allSNR_scales = []

        # Generate the random coordinates for statistical analysis
        self.coordinates = self.spaced_circle_points(self.num_plots, self.circle_radius, 
            centre_coords=self.points, minimum_spread_distance=self.minimum_point_distance)
        
        # Convert the x-axis to GHz
        freq_start = self.hdr['CRVAL3']/10**9 # GHz
        freq_incr = self.hdr['CDELT3']/10**9 # GHz
        freq_len = np.shape(self.data)[0] # length
        freq_end = freq_start + freq_len * freq_incr # where to stop
        self.xAxisFlux = np.linspace(freq_start, freq_end, freq_len) # axis to plot

        # Create the redshift values to iterate through
        self.z = np.arange(zStart, zEnd+dz, dz) # number of redshifts to iterate through

        start = time() # Measure how long it takes to execute 

        # For every coodinate point, find the associated flux and uncertainty 
        for index, coord in enumerate(self.coordinates):

            # Initialise arrays for each coordinate
            chi2_array = [] 
            param_array = []

            # Get fluxes and uncertainties at each point
            y_flux, uncert = self.fits_flux(coord)
            self.allFlux.append(y_flux)

            uncert = self.arrayfix(uncert) # average 0's from values left & right
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
            snr_div_scale = list_div_list(snrs, scales)

            # For every redshift, calculate the corresponding chi squared value
            for ddz in self.z:
                loc = ftransition/(1+ddz) # location of the gaussian peaks
                
                try:
                    params, covariance = curve_fit(lambda x, a, s: self.gaussf(x, a, s, x0=loc), 
                        self.xAxisFlux, y_flux, bounds=[[0, (1/8)], [max(y_flux), (2/3)]], absolute_sigma=True) # best fit
                except RuntimeError:
                    chi2_array.append(max(chi2_array))
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
                self.plotColours.append('black') # the original
                target_chi2 = min_plot_chi2
            elif min_plot_chi2 <= target_chi2:
                self.plotColours.append('red') # if chi2 lower than original
            elif min_plot_chi2 > target_chi2 and min_plot_chi2 <= 1.05*target_chi2:
                self.plotColours.append('gold') # if chi2 within 5% above the original
            else:
                self.plotColours.append('green') # if chi2 more than 5% above the original
            
            # Append parameters for use later
            self.allChi2.append(chi2_array)
            self.allParams.append(param_array)
            self.allSNR_scales.append(snr_div_scale)

            print(f'{index+1}/{len(self.coordinates)} completed..')

        end = time()
        print(f'Data processed in {round((end-start)/60, 3)} minutes')

class zf_plotter(RedshiftFinder):

    def __init__(self, obj):
        self.obj = obj
        
        num_plots = self.obj.num_plots
        if num_plots >= 5:
            self.cols = 5
            self.rows = num_plots // self.cols
            self.squeeze = True
        else:
            self.cols = 1
            self.rows = 1
            self.squeeze = False

    @staticmethod
    def plot_peaks(y_axis, x_axis, plot_type):
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
        circle_points = np.transpose(self.obj.coordinates)
        points_x = circle_points[0, :] # all x coordinates except the first which is the original
        points_y = circle_points[1, :] # all y coordinates except the first which is the original
        circ = plt.Circle((self.obj.centre_x, self.obj.centre_y), self.obj.circle_radius, fill=False, color='blue')
        fig, ax = plt.subplots()
        fig.set_figwidth(7)
        fig.set_figheight(7)
        ax.add_patch(circ)
        plt.title('Distribution of spaced random points')
        plt.scatter(points_x, points_y, color=self.obj.plotColours)
        plt.xlim(-self.obj.circle_radius-1+self.obj.centre_x, self.obj.circle_radius+1+self.obj.centre_x)
        plt.ylim(-self.obj.circle_radius-1+self.obj.centre_y, self.obj.circle_radius+1+self.obj.centre_y)
        plt.xlabel('x')
        plt.ylabel('y')

        if savefile != None:
            plt.savefig(f'{savefile}', dpi=200)
        plt.show()
    
    def plot_chi2(self, savefile=None):

        # Setup the figure and axes
        fig, axs = plt.subplots(self.rows, self.cols, tight_layout=True, sharex=True, squeeze=self.squeeze)
        fig.supxlabel('Redshift')
        fig.supylabel('$\chi^2$', x=0.01)
        axs = axs.flatten()

        # Plot the chi-squared(s) and redshift
        for index, chi2 in enumerate(self.obj.allChi2):
            lowest_redshift = self.obj.z[np.argmin(self.obj.allChi2[index])]
            axs[index].plot(self.obj.z, chi2, color=self.obj.plotColours[index])
            axs[index].plot(lowest_redshift, min(chi2), 'bo', markersize=5)
            coord = np.round(self.obj.coordinates[index], 2)
            axs[index].set_title(f'x,y = {coord}. Min Chi2 = {round(min(chi2), 2)}')
            axs[index].set_yscale('log')

        # Save the file and show
        if savefile != None:
            fig.savefig(f'{savefile}', dpi=200)
        plt.show()

    def plot_flux(self, savefile=None):
        
        # Setup the figure and axes
        d = count_decimals(self.obj.dz)
        fig, axs = plt.subplots(self.rows, self.cols, tight_layout=True, 
            sharex=True, sharey=True, squeeze=self.squeeze)
        fig.supxlabel('Frequency $(GHz)$')
        fig.supylabel('Flux $(mJy)$')
        axs = axs.flatten()

        # Plot the flux(s) and best fit gaussians
        for index, flux in enumerate(self.obj.allFlux):
            lowest_index = np.argmin(self.obj.allChi2[index])
            lowest_redshift = self.obj.z[lowest_index]
            axs[index].plot(self.obj.xAxisFlux, flux, color='black', drawstyle='steps-mid')
            axs[index].plot(self.obj.xAxisFlux, self.obj.gaussf(self.obj.xAxisFlux, *self.obj.allParams[index][lowest_index], 
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
        
        # Setup the figure and axes
        fig, axs = plt.subplots(self.rows, self.cols, tight_layout=True, sharey=True, squeeze=self.squeeze)
        fig.supxlabel('$\chi^2$') 
        fig.supylabel('Count (#)')
        axs = axs.flatten()

        # Initialise return arrays
        all_std = []
        all_mean = []
        
        # Plot the reduced chi-squared histogram(s)
        for index, chi2 in enumerate(self.obj.allChi2):
            reduced_chi2 = np.array(chi2)/(len(chi2)-1)
            axs[index].hist(reduced_chi2, 20, color=self.obj.plotColours[index])
            axs[index].invert_xaxis()
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

    def plot_hist_snr(self, savefile=None):

        snrs_scales = flatten_list(self.obj.allSNR_scales)

        # Setup the figure and axes
        fig, axs = plt.subplots()
        fig.supxlabel('SNR/Scale Ratio') 
        fig.supylabel('Count (#)')
        print(snrs_scales)

        # Plot the reduced chi-squared histogram(s)
        axs.hist(snrs_scales, 20)
        axs.invert_xaxis()
        
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
    min_sep = 5 # Minimum separation between points (pixels)
    num_plots = 1 # Number of plots to make (must be a multiple of 5 or 1)
    circle_radius = 87 # Radius of the largest frequency (pixels)
    ftransition = 115.2712 # the first transition in GHz
    z_start = 0 # initial redshift
    dz = 0.01 # change in redshift
    z_end = 10 # final redshift

    zfind1 = RedshiftFinder(image, ra, dec, aperture_radius, bvalue)
    zfind1.circle_point_vars(min_sep, num_plots, circle_radius)
    zfind1.zfind(ftransition, z_start, dz, z_end)

    zf1 = zf_plotter(zfind1)
    zf1.plot_points()
    zf1.plot_flux() 
    zf1.plot_chi2()
    zf1.plot_hist_chi2()