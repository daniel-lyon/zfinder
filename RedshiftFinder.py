import numpy as np
import matplotlib.pyplot as plt

from time import time
from random import random
from decimal import Decimal
from astropy.wcs import WCS
from astropy.io import fits
from PyAstronomy import pyasl
from warnings import filterwarnings
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from photutils.background import Background2D
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry

class RedshiftFinder:

    def __init__(self, image, rightAscension, declination, apertureRadius, bvalue, warnings=False):
        self.image = fits.open(image)
        self.hdr = self.image[0].header
        self.data = self.image[0].data[0]
        self.ra = rightAscension
        self.dec = declination
        self.apertureRadius = apertureRadius
        self.bvalue = bvalue

        if warnings == False:
            filterwarnings("ignore", module='photutils.background')
            filterwarnings("ignore", module='astropy.wcs.wcs')
            filterwarnings("ignore", module='scipy.optimize')

    @staticmethod
    def wcs2pix(ra, dec, hdr):
        w = WCS(hdr) # Get the world coordinate system
    
        # If there are more than 2 axis, drop them
        if hdr['NAXIS'] > 2:
            w = w.dropaxis(3) # stokes
            w = w.dropaxis(2) # frequency

        # Convert to decimal degrees
        ra = pyasl.hmsToDeg(ra[0], ra[1], ra[2])
        dec = pyasl.dmsToDeg(dec[0], dec[1], dec[2])

        # Convert world coordinates to pixel
        x, y = w.all_world2pix(ra, dec, 1)

        # Round to nearest integer
        xcor = int(np.round(x))
        ycor = int(np.round(y))

        return xcor, ycor
    
    @staticmethod
    def spacedPointsCircle(num_points, circle_radius, centre_coords, minimum_spread_distance):
        points = [centre_coords]
        
        for i in range(num_points-1):
            while True:
                theta = 2 * np.pi * random()
                r = circle_radius * random()

                x = r * np.cos(theta) + centre_coords[0]
                y = r * np.sin(theta) + centre_coords[1]

                distances = cdist([[x,y]], points, 'euclidean')
                min_distance = min(distances[0])
                
                if min_distance >= minimum_spread_distance or len(points) == 1:
                    points.append([x,y])
                    break
        return points
    
    @staticmethod
    def arrayfix(array):
        for i, val in enumerate(array):
            if val == 0:
                array[i] = (array[i-1] + array[i+1])/2
        return array
    
    @staticmethod
    def gaussf(x, x0):
        y = 0
        for i in range(1,11):
            y += (0.26 * np.exp(-((x-i*x0)**2) / (1/5)**2)) # i = 1,2,3 ... 9, 10
        return y
    
    def fits_flux(self, position):

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
            anullusstats = ApertureStats(page, annulus)
            bkg_mean = anullusstats.mean
            aperture_area = aperture.area_overlap(page)
            freq_uncert = bkg_mean * aperture_area

            # Background data
            bkg = Background2D(page, (50, 50)).background

            # Aperture sum of the fits image minus the background
            aphot = aperture_photometry(page - bkg, aperture)
            apsum = aphot['aperture_sum'][0]

            # Calculate corrected flux
            total_flux = apsum*(pix2deg**2)/barea
            fluxes.append(total_flux)
            uncertainties.append(freq_uncert)

        return np.array(fluxes), np.array(uncertainties)

    def circlePointVars(self, minimum_point_distance, num_plots, circle_radius):
        
        self.centre_x, self.centre_y = self.wcs2pix(self.ra, self.dec, self.hdr)
        self.plotColours = []
        self.minimum_point_distance = minimum_point_distance
        self.num_plots = num_plots
        if self.num_plots < 4 and self.num_plots > 1:
            raise ValueError('num_plots must be >= 4 or num_plots == 1')
        self.circle_radius = circle_radius
        self.points = [self.centre_x, self.centre_y]

    def zfind(self, ftransition, zStart=0, dz=0.01, zEnd=10, timer=False):
        self.coordinates = self.spacedPointsCircle(self.num_plots, self.circle_radius, centre_coords=self.points, minimum_spread_distance=self.minimum_point_distance)
        self.ftransition = ftransition 
        self.dz = dz

        freq_start = self.hdr['CRVAL3']/10**9 # GHz
        freq_incr = self.hdr['CDELT3']/10**9 # GHz
        freq_len = np.shape(self.data)[0] # length
        freq_end = freq_start + freq_len * freq_incr # where to stop
        self.xAxisFlux = np.linspace(freq_start, freq_end, freq_len) # axis to plot

        z_n = int((1/dz)*(zEnd-zStart))+1 # number of redshifts to iterate through
        self.z = np.linspace(zStart, zEnd, z_n) # redshift array
        self.allChi2 = []
        self.allFlux = []

        start = time()

        for index, coord in enumerate(self.coordinates):

            # Get fluxes and ucnertainties at each image
            chi2_array = [] # initial 
            y_flux, uncert = self.fits_flux(coord)
            self.allFlux.append(y_flux)
            uncert = self.arrayfix(uncert) # average 0's from values left & right
            y_flux *= 1000 # convert from uJy to mJy
            uncert *= 1000

            # For every redshift, calculate the corresponding chi squared value
            for ddz in self.z:
                loc = ftransition/(1+ddz) # location of the gaussian peak
                parameters, covariance = curve_fit(lambda x, b: self.gaussf(x, x0=loc), self.xAxisFlux, y_flux, absolute_sigma=True) # best fit
                f_exp = self.gaussf(self.xAxisFlux, loc) # expected function
                chi2 = sum(((y_flux - f_exp))**2)
                chi2_array.append(chi2)

            min_plot_chi2 = min(chi2_array)
            
            if index == 0:
                self.plotColours.append('black')
                target_chi2 = min_plot_chi2
            elif min_plot_chi2 <= target_chi2:
                self.plotColours.append('red')
            elif min_plot_chi2 > target_chi2 and min_plot_chi2 <= 1.05*target_chi2:
                self.plotColours.append('gold')
            else:
                self.plotColours.append('green')
            
            self.allChi2.append(chi2_array)

            if timer == True:
                print(f'{index+1}/{len(self.coordinates)} ...')

        end = time()
        if timer == True:
            print(f'Took {round((end-start)/60, 3)} minutes to process data')

    def plotCoords(self, savefile=None):
        circle_points = np.transpose(self.coordinates)
        points_x = circle_points[0, :] # all x coordinates except the first which is the original
        points_y = circle_points[1, :] # all y coordinates except the first which is the original
        circ = plt.Circle((self.centre_x, self.centre_y), self.circle_radius, fill=False, color='blue')
        fig, ax = plt.subplots()
        fig.set_figwidth(7)
        fig.set_figheight(7)
        ax.add_patch(circ)
        plt.title('Distribution of spaced random points')
        plt.scatter(points_x, points_y, color=self.plotColours)
        plt.xlim(-self.circle_radius-1+self.centre_x, self.circle_radius+1+self.centre_x)
        plt.ylim(-self.circle_radius-1+self.centre_y, self.circle_radius+1+self.centre_y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

        if savefile != None:
            plt.savefig(f'{savefile}', dpi=200)
    
    def plotChi2(self, savefile=None):
        if self.num_plots >= 4:
            row_col = int(np.sqrt(self.num_plots))
            fig, axs = plt.subplots(row_col, row_col, tight_layout=True, sharex=True)
            fig.supxlabel('Redshift')
            fig.supylabel('$\chi^2$', x=0.01)
            axs = axs.flatten()
            for index, array in enumerate(self.allChi2):
                lowest_redshift = self.z[np.argmin(self.allChi2[index])]
                axs[index].plot(self.z, array, color=self.plotColours[index])
                axs[index].plot(lowest_redshift, min(array), 'bo', markersize=5)
                coord = np.round(self.coordinates[index], 2)
                axs[index].set_title(f'x,y = {coord}. Min Chi2 = {round(min(array), 2)}')
                axs[index].set_yscale('log')
            plt.yscale('log')
            plt.show()
            if savefile != None:
                fig.savefig(f'{savefile}', dpi=200)
            
        else:
            plt.plot(self.z, self.allChi2[0])
            plt.title('Distribution of spaced random points')
            plt.xlabel('Redshift')
            plt.ylabel('$\chi^2$')
            plt.yscale('log')
            plt.show()
            if savefile != None:
                plt.savefig(f'{savefile}', dpi=200)

    def plotFlux(self, savefile=None):

        d = Decimal(str(self.dz))
        d = abs(d.as_tuple().exponent)
        
        if self.num_plots >= 4:
            row_col = int(np.sqrt(self.num_plots))
            fig, axs = plt.subplots(row_col, row_col, tight_layout=True, sharex=True)
            fig.supxlabel('Frequency $(GHz)$')
            fig.supylabel('Flux $(mJy)$')
            axs = axs.flatten()
            for index, array in enumerate(self.allFlux):
                lowest_redshift = self.z[np.argmin(self.allChi2[index])]
                axs[index].plot(self.xAxisFlux, array, color='black', drawstyle='steps-mid')
                axs[index].plot(self.xAxisFlux, self.gaussf(self.xAxisFlux, self.ftransition/(1+lowest_redshift)), color='red')
                axs[index].margins(x=0)
                axs[index].fill_between(self.xAxisFlux, self.allFlux[index], 0, where=(self.allFlux[index] > 0), color='gold', alpha=0.75)
                axs[index].set_title(f'z={round(lowest_redshift, d)}')

            if savefile != None:
                fig.savefig(f'{savefile}', dpi=200)
            plt.show()

        else:
            lowest_redshift = self.z[np.argmin(self.allChi2[0])]
            plt.figure(figsize=(15,5))
            plt.plot([min(self.xAxisFlux), max(self.xAxisFlux)], [0, 0], color='black', linestyle='--', dashes=(5,5))
            plt.plot(self.xAxisFlux, self.allFlux[0], color='black', drawstyle='steps-mid')
            plt.plot(self.xAxisFlux, self.gaussf(self.xAxisFlux, self.ftransition/(1+lowest_redshift)), color='red')
            plt.title(f'z={round(lowest_redshift, d)}')
            plt.fill_between(self.xAxisFlux, self.allFlux[0], 0, where=(self.allFlux[0] > 0), color='gold', alpha=0.75)
            plt.xlabel('Frequency $(GHz)$')
            plt.ylabel('Flux $(mJy)$')
            plt.margins(x=0)
            plt.savefig('fig1.png', dpi=200)
            plt.show()