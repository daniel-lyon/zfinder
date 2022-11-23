# Required functions
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.aperture import CircularAperture, aperture_photometry

# Read .fits image
fits_image = fits.open('0856.fits.image.tt0.fits')

# Get data
hdr = fits_image[0].header 
data = fits_image[0].data[0][0] # data was size (400, 400, 1, 1) -> converted to (400, 400)

# Aperture setup
data_dimensions = np.shape(data) # shape of the data sturecture (x,y)
radius = data_dimensions[0]/2 # radius of the circular aperture (200)
position = (radius, radius) # centre of the image (200, 200)
aperture = CircularAperture(position, radius)

# Aperture sum of the fits image
apsum = aperture_photometry(data, aperture)['aperture_sum'][0]

# Calculate beam area
bmaj = hdr['BMAJ']
bmin = hdr['BMIN']
barea = 1.1331 * bmaj * bmin

# Corrected flux
pix2deg = hdr['CDELT1'] # unit conversion of apsum to barea units
total_flux = apsum*(pix2deg**2)/barea
print()
print(f'.fits total image flux = {total_flux}')
print()

