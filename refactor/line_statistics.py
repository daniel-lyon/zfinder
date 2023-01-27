import numpy as np
import matplotlib.pyplot as plt

from random import random
from astropy.wcs import WCS
from sslf.sslf import Spectrum
from fits2flux import fits2flux, wcs2pix
from scipy.spatial.distance import cdist
from PyAstronomy.pyasl import degToHMS, degToDMS

def line_stats(flux):
    s = Spectrum(flux)
    s.find_cwt_peaks(scales=np.arange(4,10), snr=3)

    # Calculate the ratio of the snrs and scales
    snrs = [round(i,2) for i in s.peak_snrs] # the snrs of the peaks
    scales = [i[1]-i[0] for i in s.channel_edges] # the scales of the peaks
    return snrs, scales

def flatten_list(input_list: list[list]):
    ''' Turns lists of lists into a single list '''
    flattened_list = []
    for array in input_list:
        for x in array:
            flattened_list.append(x)
    return flattened_list

def spaced_circle_points(num_points, circle_radius, centre_coords, minimum_spread_distance):
    """ Calculate points in a circle with a minimum spread """
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

class line_statistics(fits2flux):
    def __init__(self, image, ra, dec, aperture_radius, bvalue, num_points, min_spread=1):
        """ 
        Calculates statistics on random lines found across random points on the .fits image.
        Finds the signal-to-noise-ratio (snr) and scale of each random position. Takes 
        several minutes to compute.

        Parameters
        ----------
        num_points : int
            The number of points to search for lines in
        
        min_spread : float, optional
            The minimum spread distance that must be between every point (pixels)
            Defualt = 1
        """

        super().__init__(image, ra, dec, aperture_radius, bvalue)
        self.num_points = num_points
        self.min_spread = min_spread

    @staticmethod
    def _fits_circle_radius(data):
        """ Using ra & dec axis from a fits data cube, calculate the radius of non nan values"""
        # Assuming the image is a cube with a circle of non-nan values
        data = data[0] # remove third column
        data_len = len(data[0]) # The total length
        target_row = data[(data_len//2) - 1] # the middle row

        # The true radius is the total length minus the number of nans
        nan_count = sum(np.isnan(x) for x in target_row) 
        diameter = data_len - nan_count
        radius = (diameter // 2) - 7 # minus 7 as a little buffer
        return radius

    @staticmethod
    def plot_snr_scales(snrs, scales):
        """ Plot two histograms, one of snrs and one of scales """
        snrs = flatten_list(snrs)
        scales = flatten_list(scales)

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
        plt.show()
    
    def _pix2wcs(self, x, y):
        """ Convert x,y coordinates to ra, dec"""
        # Drop redundant axis
        w = WCS(self.hdr)
        if self.hdr['NAXIS'] > 2:
            w = w.dropaxis(3) # stokes
            w = w.dropaxis(2) # frequency

        # Convert x,y to degrees
        ra, dec = w.all_pix2world(x, y, 1)

        # Convert degrees to ra, dec
        ra = degToHMS(ra)
        dec = degToDMS(dec)
        return ra, dec
    
    def _get_flux(self):
        return super().get_flux()

    def perform_analysis(self):
        """ Find scales and snr of random points """
        
        # Initialise return arrays
        all_snrs = []
        all_scales = []

        # Find x,y coordinates of the target
        x, y = wcs2pix(self.ra, self.dec, self.hdr)

        # Calculate the coordinates for the number of points
        radius = self._fits_circle_radius(self.data) # Find the radius of the .fits image
        coordinates = spaced_circle_points(self.num_points, radius, [x,y], self.min_spread)

        # Use sslf to find lines on all points
        for i, (x, y) in enumerate(coordinates):
            self.ra, self.dec = self._pix2wcs(x,y) # convert x,y to ra,dec
            flux, unc = self._get_flux()
            snrs, scales = line_stats(flux) # use flux axis to find lines on

            all_snrs.append(snrs)
            all_scales.append(scales)
            print(f'{i+1}/{self.num_points}')
        
        self.plot_snr_scales(all_snrs, all_scales)
        return all_snrs, all_scales

def main():
    image = '0856_cube_c0.4_nat_80MHz_taper3.fits'  
    ra = [8, 56, 14.8]
    dec = [2, 24, 0.6, 1]
    apeture_radius = 3
    bvalue=3
    points = 10
    min_spread = 1

    stats = line_statistics(image, ra, dec, apeture_radius, bvalue, points, min_spread)
    snrs, scales = stats.perform_analysis()

if __name__ == '__main__':
    main()