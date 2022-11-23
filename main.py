# Import functions
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.aperture import CircularAperture, aperture_photometry

# Read .fits image
fits_image = fits.open('0856.fits.image.tt0.fits')

# Get data
hdr = fits_image[0].header 
data = fits_image[0].data[0][0] # data was size (400, 400, 1, 1)

# Aperture setup
data_shape = np.shape(data) # shape of the data sturecture (x,y)
radius = data_shape[0]/2 # radius of the circular aperture (200)
position = (radius, radius) # centre of the image (200, 200)
aperture = CircularAperture(position, radius)

# Flux of the fits image
total_flux = aperture_photometry(data, aperture)['aperture_sum'][0]
print(total_flux)

bmaj = hdr['BMAJ']
bmin = hdr['BMIN']
#print(bmaj, bmin)




barea = 1.1331 * bmaj * bmin

