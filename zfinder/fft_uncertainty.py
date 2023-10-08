import csv

import numpy as np
from scipy.spatial.distance import cdist
from astropy.io import fits
from astropy.wcs import WCS

from fits2flux import Fits2flux, wcs2pix
from fft import fft
from per_pixel import radec2str, get_all_flux

# Generate a list of x, y coordinates inside a circle
# Ensure that the coordinates are a minimum distance apart from all other coordinates
def get_coords(n, radius, centre, min_spread):
    """ Doc string here """
    coords = [centre]
    
    # Iterate through the number of coords.
    for _ in range(n):
        
        # Keep generating the current point until it is at least the minimum distance away from all 
        while True:
            theta = 2 * np.pi * np.random.random() # choose a random direction
            r = radius * np.random.random() # choose a random radius

            # Convert coordinates to cartesian
            x = r * np.cos(theta) + centre[0]
            y = r * np.sin(theta) + centre[1]

            # Find the distance between all the placed coords
            distances = cdist([[x,y]], coords, 'euclidean')
            min_distance = min(distances[0])
            
            # If the minimum distance is satisfied for all coords, go to next point
            if min_distance >= min_spread:
                coords.append([x,y])
                break
    return coords[1:]

def fft_uncertainty(fitsfile, ra, dec, aperture_radius, n=100, radius=50, min_spread=1):
    """ Doc string here """
    # Find x,y coordinates of the target
    hdr = fits.getheader(fitsfile)
    wcs = WCS(hdr, naxis=2)
    center_x, center_y = wcs2pix(ra, dec, hdr)
    coords = get_coords(n, radius, [center_x, center_y], min_spread)
        
    # Get the FFT flux from each coordinate (excluding the target)
    all_ra = []
    all_dec = []
    for x, y in coords:
        ra, dec = wcs.all_pix2world(x, y, 1)
        ra, dec = radec2str(ra, dec)
        all_ra.append(ra)
        all_dec.append(dec)

    # Get the flux values for all ra and dec coordinates
    all_flux, _ = get_all_flux(fitsfile, all_ra, all_dec, aperture_radius)
    freq = Fits2flux(fitsfile, ra, dec, aperture_radius).get_freq()
    
    # Calculate the FFT for each flux
    all_fflux = []
    for flux in all_flux:   
        _, fflux = fft(freq, flux)
        all_fflux.append(fflux)
    
    # Calculate the standard deviation of all the fflux channels
    all_std = []
    all_fflux = np.transpose(all_fflux)
    for array in all_fflux:
        std = np.mean(array)
        all_std.append(std)
        
    # Write the data to the csv
    all_std = [[std] for std in all_std]
    with open('fft_uncertainty.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerows(all_std)
    return all_std

def main():
    fitsfile = 'zfinder/0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = '08:56:14.8'
    dec = '02:24:00.6'

    n = 100
    radius = 50
    min_spread = 1
    aperture_radius = 3
    
    fft_uncertainty(fitsfile, ra, dec, aperture_radius, n, radius, min_spread)
    
if __name__ == '__main__':
    main()