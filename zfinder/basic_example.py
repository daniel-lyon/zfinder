# Import packages
from zfinder import zfinder

# Required variables
image = '0856_cube_c0.4_nat_80MHz_taper3.fits'
ra = [8, 56, 14.8]
dec = [2, 24, 0.6, 1]
transition = 115.2712
aperture_radius = 3
bvalue = 3

source = zfinder(image, ra, dec, transition, aperture_radius, bvalue)
source.zflux()
source.zfft()