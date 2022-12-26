'''
Minimum code needed to find the redshift of a target source.
Also displays a plot of the chi-squared and flux.
'''

# Import zfinder
from zfinder import RedshiftFinder as zf, RedshiftPlotter as zp

# Required variables
image = '0856_cube_c0.4_nat_80MHz_taper3.fits'
ra = [8, 56, 14.8] # Right Ascenion (h, m, s)
dec = [2, 24, 0.6, 1] # Declination (d, m, s, sign)
aperture_radius = 3 # Aperture Radius (pixels)
bvalue = 3 # BMAJ & BMIN (arcseconds)
ftransition = 115.2712 # the first transition in {unit}Hz (this is a CO)

# Find the redshift
zf1 = zf(image, ra, dec, aperture_radius, bvalue)
z = zf1.zfind(ftransition)
print(f'The redshift of this source is z={z[0]}')

zp1 = zp(zf1)
zp1.plot_chi2() # Plot chi-squared vs z
zp1.plot_flux() # Plot flux vs frequency