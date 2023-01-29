import numpy as np
from zflux import zflux
from random import random
from astropy.wcs import WCS
from sslf.sslf import Spectrum
from fits2flux import fits2flux, wcs2pix
from scipy.spatial.distance import cdist
from PyAstronomy.pyasl import degToHMS, degToDMS

def line_stats(flux):
    """ Signal-to-noise ratio and scale resulting from blind line finding """
    s = Spectrum(flux)
    s.find_cwt_peaks(scales=np.arange(4,10), snr=3)

    # Calculate the ratio of the snrs and scales
    snrs = [round(i,2) for i in s.peak_snrs] # the snrs of the peaks
    scales = [i[1]-i[0] for i in s.channel_edges] # the scales of the peaks
    return snrs, scales

def flatten_list(input_list: list[list]):
    """ Turns lists of lists into a single list """
    flattened_list = []
    for array in input_list:
        for x in array:
            flattened_list.append(x)
    return flattened_list

def spaced_circle_points(num_points, circle_radius, centre_coords, minimum_spread_distance):
    """ Calculate points in a circle with a minimum spread """
    
    points = [centre_coords] # centre_coords = [x,y] -> points = [[x,y]]
    
    # Iterate through the number of points.
    for _ in range(num_points-1):
        
        # Keep generating the current point until it is at least the minimum distance away from all 
        i = 0
        while i < 10000:
            theta = 2 * np.pi * random() # choose a random direction
            r = circle_radius * random() # choose a random radius

            # Convert coordinates to cartesian
            x = r * np.cos(theta) + centre_coords[0]
            y = r * np.sin(theta) + centre_coords[1]

            # Find the distance between all the placed points
            distances = cdist([[x,y]], points, 'euclidean')
            min_distance = min(distances[0])

            i += 1
            
            # If the minimum distance is satisfied for all points, go to next point
            if min_distance >= minimum_spread_distance or len(points) == 1:
                points.append([x,y])
                i = 0
                break
    return points

def pix2wcs(x, y, hdr):
    """ Convert x,y positional coordinates to world ra, dec"""
    # Drop redundant axis
    w = WCS(hdr)
    if hdr['NAXIS'] > 2:
        w = w.dropaxis(3) # stokes
        w = w.dropaxis(2) # frequency

    # Convert x,y to degrees
    ra, dec = w.all_pix2world(x, y, 1)

    # Convert degrees to ra, dec
    ra = degToHMS(ra)
    dec = degToDMS(dec)
    return ra, dec

class line_statistics(fits2flux):
    def __init__(self, image, ra, dec, aperture_radius, bvalue, transition):
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
            Default = 1
        """
        super().__init__(image, ra, dec, aperture_radius, bvalue)
        self.transition = transition
    
    def _get_flux(self):
        return super().get_flux()

    def _get_freq(self):
        return super().get_freq()

    def perform_analysis(self, num_points, radius=50, min_spread=1):
        """ 
        Find statistics of: signal-to-noise ratio, scale, and delta-z of random points
        
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
        
        # Initialise return arrays
        all_snrs = []
        all_scales = []
        all_lowest_z = []

        # Find x,y coordinates of the target
        x, y = wcs2pix(self.ra, self.dec, self.hdr)

        # Calculate the coordinates for the number of points
        self.coordinates = spaced_circle_points(num_points, radius, [x,y], min_spread)

        # Use sslf to find lines on all points
        for i, (x, y) in enumerate(self.coordinates):
            self.ra, self.dec = pix2wcs(x,y, self.hdr) # convert x,y to ra,dec
            freq = self._get_freq()
            flux, unc = self._get_flux()
            snrs, scales = line_stats(flux) # use flux axis to find lines on

            zf = zflux(self.transition, freq, flux, unc)
            z, chi2 = zf.zfind()

            all_snrs.append(snrs)
            all_scales.append(scales)
            all_lowest_z.append(z[np.argmin(chi2)])
            print(f'{i+1}/{len(self.coordinates)}')
        
        return all_snrs, all_scales, all_lowest_z

def main():
    image = '0856_cube_c0.4_nat_80MHz_taper3.fits'  
    ra = [8, 56, 14.8]
    dec = [2, 24, 0.6, 1]
    apeture_radius = 3
    bvalue = 3
    transition = 115.2712

    radius = 50
    points = 5
    min_spread = 1

    stats = line_statistics(image, ra, dec, apeture_radius, bvalue, transition)
    snrs, scales, zs = stats.perform_analysis(points, radius, min_spread)

if __name__ == '__main__':
    main()