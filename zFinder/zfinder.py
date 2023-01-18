# Import libraries
import numpy as np

from time import time
from zFFT import zFFT
from zPlot import zPlot
from random import random
from astropy.wcs import WCS
from astropy.io import fits
from PyAstronomy import pyasl
from zAnimate import zAnimate
from sslf.sslf import Spectrum
from warnings import filterwarnings
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from photutils.background import Background2D
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry

def average_zeroes(data: list[float]):
    ''' Take the average of adjacent points (left and right) if the value is zero'''
    for i, val in enumerate(data):
        if val == 0:
            data[i] = (data[i-1] + data[i+1])/2
    return data

def get_eng_exponent(number: float):
    ''' Get the exponent of a number in engineering format. In eng
        format, exponents are multiples of 3. E+0, E+3, E+6, etc.
        Also returns the unit prefix symbol for the exponent from
        -24 to +24 '''

    # A dictionary of exponent and unit prefix pairs
    prefix = {-24 : 'y', -21 : 'z', -18 : 'a',-15 : 'f', -12 : 'p',
        -9 : 'n', -6 : 'mu', -3 : 'm', 0 : '', 3 : 'k', 6 : 'M',
        9 : 'G', 12 : 'T', 15 : 'P', 18 : 'E', 21 : 'Z', 24 : 'Y'}

    base = np.log10(np.abs(number)) # Log rules to find exponent
    exponent = int(np.floor(base)) # convert to floor integer
    
    # Check if the the exponent is a multiple of 3
    for i in range(3):
        if (exponent-i) % 3 == 0:
            
            # Return the exponent and associated unit prefix
            symbol = prefix[exponent-i]
            return exponent-i, symbol

class zFinder(object):
    def __init__(self, image: str, right_ascension: list, declination: list, aperture_radius: float,
            bvalue: float, num_plots=1, minimum_point_distance=1.0, warnings=False):
        '''
        `zFinder` looks at transition lines and attempts to find the best fitting red shift.
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
            The number  of random points to work with and plot. Default = 1
        
        minimum_point_distance : `float`, optional
            The distance between random points in pixels. Default = 1.0

        warnings : `bool`, optional
            Optional setting to display warnings or not. If True, warnings are displayed.
            Default = False
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
        self.circle_radius = self.fits_circle_radius(self.data[-1])

        # Initialise instance lists
        self.all_chi2 = []
        self.all_flux = []
        self.all_params = []
        self.all_snrs = []
        self.all_scales = []
        self.all_lowest_z = []
        self.plot_colours = []

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
    def fits_circle_radius(data: np.ndarray[np.ndarray]):
        ''' 
        With the fits image, find the radius of the smallest image. This radius is used in
        the `spaced_circle_points` function as the radius.

        Parameters
        ----------
        data : `list`
            The data from the header that corresponds to the smallest sized image in the frequency
            of a .fits image. AKA: the image with the most nans (usually the last image)
        
        Returns
        -------
        largest_radius : `int`
            The radius of the given image
        '''

        # Assuming the image is a cube with a circle of non-nan values
        data_len = len(data[0]) # The total length
        target_row = data[(data_len//2) - 1] # the middle row
        
        # The true radius is the total length minus the number of nans
        nan_count = sum(np.isnan(x) for x in target_row) 
        diameter = data_len - nan_count
        radius = (diameter // 2) - 7 # minus 7 as a little buffer
        return radius

    @staticmethod
    def wcs2pix(ra: list, dec: list, hdr):

        # TODO: rounded x and y values are different to non rounded versions?

        ''' 
        Convert right ascension and declination to x, y positional world coordinates

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
    def spaced_circle_points(num_points: int, circle_radius: float, centre_coords: list[float], minimum_spread_distance: float):
        ''' 
        Generate points in a circle that are a minimum distance apart.

        Parameters
        ----------
        num_points : `int`
            The number of points to plot.
        
        circle_radius : `float`
            The radius in which points can be plotted around the centre.
        
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
        
        # TODO: add unfixed variable for y0?

        ''' 
        Gaussian function used to fit to a data set

        Parameters
        ----------
        x : `list`
            The x-axis list
        
        a : `float`
            The amplitude of the gaussians
        
        s : `float`
            The standard deviation and width of the gaussians
        
        x0 : `float`
            The x-axis offset
        
        Returns
        -------
        y : `list`
            A y-axis list of guassians functions
        '''

        y = 0
        for i in range(1,11):
            y += (a * np.exp(-((x-i*x0) / s)**2)) # i = 1,2,3 ... 9, 10
        return y
    
    def fits_flux(self, position):
        ''' For every frequency channel, find the flux and associated uncertainty at a position.

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

    def plot_plots(self):
        zp = zPlot(self)
        if self.num_plots != 1:
            zp.plot_points()

        zp.plot_chi2()
        zp.plot_flux()

        zfft = zFFT(self)
        zfft.plot_fft()

        zp.plot_hist_chi2()
        zp.plot_snr_scales()
    
    def animate_animations(self):
        za = zAnimate(self)
        za.animate_flux()
        za.animate_chi2()
        za.animate_redshift()

    def zfind(self, ftransition, z_start=0, dz=0.01, z_end=10, plots=True, animations=False):
        ''' 
        For every point in coordinates, find the flux and uncertainty, then find the significant
        lines with the line finder. Then iterate through all redshift values and calculate
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

        # Setup for spaced random points
        self.centre_x, self.centre_y = self.wcs2pix(self.ra, self.dec, self.hdr) 

        # Generate the random coordinates for statistical analysis
        self.coordinates = self.spaced_circle_points(self.num_plots, self.circle_radius, 
            centre_coords=[self.centre_x, self.centre_y], minimum_spread_distance=self.minimum_point_distance)
        
        # Convert the x-axis to GHz
        exponent, self.symbol = get_eng_exponent(self.hdr['CRVAL3'])
        self.nu = self.hdr['CRVAL3']/10**exponent # {symbol}Hz --> frequency start
        freq_incr = self.hdr['CDELT3']/10**exponent # {symbol}Hz
        freq_len = np.shape(self.data)[0] # length
        freq_end = self.nu + freq_len * freq_incr # where to stop
        self.x_axis_flux = np.linspace(self.nu, freq_end, freq_len) # axis to plot

        # Create the redshift values to iterate through
        self.z = np.arange(z_start, z_end+dz, dz)

        start = time() # Measure how long it takes to execute 

        # For every coodinate point, find the associated flux and uncertainty 
        for index, coord in enumerate(self.coordinates):

            # Initialise arrays for each coordinate
            chi2_array = [] 
            param_array = []

            # Get fluxes and uncertainties at each point
            y_flux, uncert = self.fits_flux(coord)
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
                        self.x_axis_flux, y_flux, bounds=[[0, (1/8)], [max(y_flux), (2/3)]], absolute_sigma=True) # best fit
                except RuntimeError:
                    chi2_array.append(max(chi2_array)) # if no returned parameters, set the chi-squared for this redshift to the maximum
                    continue
                
                # Using the best fit parameters, calculate the chi2 corresponding to this redshift {ddz}
                f_exp = self.gaussf(self.x_axis_flux, a=params[0], s=params[1], x0=loc) # expected function
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
            self.all_flux.append(y_flux)
            self.all_chi2.append(chi2_array)
            self.all_params.append(param_array)
            self.all_snrs.append(snrs)
            self.all_scales.append(scales)
            self.all_lowest_z.append(lowest_redshift)

            print(f'{index+1}/{len(self.coordinates)} completed..')

        # Return an array with the lowest redshift from each source
        end = time()
        print(f'Data processed in {round((end-start)/60, 3)} minutes')

        if plots:
            self.plot_plots()
        
        if animations:
            self.animate_animations()

        return self.all_lowest_z 

def main():
    image = '0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = [8, 56, 14.8] # Right Ascenion (h, m, s)
    dec = [2, 24, 0.6, 1] # Declination (d, m, s, sign)
    aperture_radius = 3 # Aperture Radius (pixels)
    bvalue = 3 # BMAJ & BMIN (arcseconds)
    num_plots = 1 # Number of plots to make (must be a multiple of 5 or 1)
    min_sep = 1 # Minimum separation between points (pixels)
    ftransition = 115.2712 # the first transition in GHz
    z_start = 0 # initial redshift
    dz = 0.01 # change in redshift
    z_end = 10 # final redshift

    # Find the redshift of source(s)
    zf = zFinder(image, ra, dec, aperture_radius, bvalue, num_plots, min_sep)
    zf.zfind(ftransition, z_start, dz, z_end)

if __name__ == '__main__':
    main()