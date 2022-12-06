# Required functions
from astropy.io import fits
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D

# Read .fits image
fits_image = fits.open('0856.fits.image.tt0.fits')

# Get data
hdr = fits_image[0].header
data = fits_image[0].data[0][0] # data was size (400, 400, 1, 1) -> converted to (400, 400)

# Required header values
bmaj = hdr['BMAJ']
bmin = hdr['BMIN']
pix2deg = hdr['CDELT2'] # unit conversion

# Aperture setup
radius = round(1/pix2deg/3600) # number of pixels in one arcsecond = 5
position = (192, 198) # centre of the aperture
aperture = CircularAperture(position, radius)

# Background data
bkg = Background2D(data, (50, 50)).background

# Aperture sum of the fits image minus the background
aphot = aperture_photometry(data - bkg, aperture)
apsum = aphot['aperture_sum'][0]

# Corrected flux
barea = 1.1331 * bmaj * bmin # beam area
total_flux = apsum*(pix2deg**2)/barea
print()
print(f'flux = {total_flux*1000} mJy')